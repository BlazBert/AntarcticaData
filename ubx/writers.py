"""Sinks: Parquet (per-message) and Zarr (MON-SPAN spectra).

Each Parquet sink keeps an in-memory list of column-chunks, flushes a row
group when ~128 MB of data has accumulated, and writes a single compressed
file per (day × message). MON-SPAN spectra append into a per-day Zarr group
under ``mon_span.zarr/<yyyymmdd>/``.

Per-day rather than global stores so that 32 parallel workers don't trample
each other; the ``analysis/`` modules later open the per-day artefacts as a
``pyarrow.dataset`` (Parquet) or as a virtual Zarr concat (xarray) and treat
them as a single logical table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from numcodecs import Blosc

from ubx.schemas import ALL_SCHEMAS

# Aim for ~128 MB row groups (after compression).  This is well below the
# 512 MB Parquet "row group size" sweet spot for OLAP workloads but large
# enough that we get good compression.
_DEFAULT_ROW_GROUP_BYTES = 128 * 1024 * 1024
_PARQUET_COMPRESSION = "zstd"
_PARQUET_COMPRESSION_LEVEL = 3


class ParquetSink:
    """Append-only Parquet writer for one message type, one day.

    Buffers per-column Python lists (cheap append) and converts to Arrow
    only at flush time. This is dramatically faster than building a
    ``RecordBatch`` per ``append()`` call when each call carries 1 row,
    which is what RXM-SFRBX gives us (~1.6M rows/day).

    Usage:
        sink = ParquetSink(path, schema)
        sink.append({"t_ns": np.array([...]), "lon_1e7": ..., ...})
        ...
        sink.close()
    """

    def __init__(
        self,
        path: str | Path,
        schema: pa.Schema,
        *,
        row_group_target_rows: int = 1_000_000,
        compression: str = _PARQUET_COMPRESSION,
        compression_level: int = _PARQUET_COMPRESSION_LEVEL,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        # Per-column buffer; values are appended (numpy arrays or lists).
        self._buffers: dict[str, list[Any]] = {f.name: [] for f in schema}
        self._buffered_rows = 0
        self._row_group_target_rows = row_group_target_rows
        self._writer: pq.ParquetWriter | None = None
        self._compression = compression
        self._compression_level = compression_level
        self._n_rows_total = 0

    def append(self, columns: dict[str, Any]) -> None:
        """Append a chunk. ``columns`` keys must match ``self.schema.names``."""
        n_rows = -1
        for field in self.schema:
            v = columns.get(field.name)
            if v is None:
                continue
            if hasattr(v, "shape") and len(v.shape) > 0:
                row_count = v.shape[0]
            else:
                row_count = len(v)
            if n_rows < 0:
                n_rows = row_count
            self._buffers[field.name].append(v)
        if n_rows <= 0:
            return
        self._buffered_rows += n_rows
        if self._buffered_rows >= self._row_group_target_rows:
            self._flush()

    def _flush(self) -> None:
        if self._buffered_rows <= 0:
            return
        arrays = []
        for field in self.schema:
            chunks = self._buffers[field.name]
            if not chunks:
                arrays.append(pa.nulls(self._buffered_rows, type=field.type))
                continue
            arrays.append(_concat_to_arrow(chunks, field))
        table = pa.Table.from_arrays(arrays, schema=self.schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                str(self.path),
                self.schema,
                compression=self._compression,
                compression_level=self._compression_level,
                use_dictionary=True,
                write_statistics=True,
            )
        self._writer.write_table(table)
        self._n_rows_total += self._buffered_rows
        for k in self._buffers:
            self._buffers[k] = []
        self._buffered_rows = 0

    def close(self) -> None:
        self._flush()
        if self._writer is None:
            # Ensure the file exists even when no rows were written. This
            # prevents Snakemake from failing with "missing output" on days
            # where a particular message type was absent (e.g. MON-SYS for
            # very short files, or QZSS frames at high latitude).
            self._writer = pq.ParquetWriter(
                str(self.path),
                self.schema,
                compression=self._compression,
                compression_level=self._compression_level,
                use_dictionary=True,
                write_statistics=True,
            )
        self._writer.close()
        self._writer = None

    def __enter__(self) -> ParquetSink:
        return self

    def __exit__(self, *_a: Any) -> None:
        self.close()


def _concat_to_arrow(chunks: list[Any], field: pa.Field) -> pa.Array:
    """Concatenate per-column chunks into a single Arrow array.

    All-numpy fast path: ``np.concatenate`` then ``pa.array``. For binary /
    list types we go via ``pa.array(list_of_lists)`` after flattening one
    level of Python lists.
    """
    t = field.type
    if pa.types.is_binary(t):
        # chunks is list of lists of bytes (each inner list length 1).
        flat: list[bytes] = []
        for c in chunks:
            flat.extend(c)
        return pa.array(flat, type=t)
    if pa.types.is_list(t):
        flat_list: list[Any] = []
        for c in chunks:
            flat_list.extend(c)
        return pa.array(flat_list, type=t)
    # Numpy concat path
    if all(isinstance(c, np.ndarray) for c in chunks):
        cat = np.concatenate(chunks)
        return pa.array(cat, type=t)
    # Fallback — let pyarrow handle it
    flat_obj: list[Any] = []
    for c in chunks:
        flat_obj.extend(c if isinstance(c, list) else c.tolist())
    return pa.array(flat_obj, type=t)


def _to_arrow(value: Any, field: pa.Field) -> pa.Array:
    """Coerce a Python/Numpy column to an Arrow array honouring ``field.type``."""
    t = field.type
    # List columns (e.g. dwrd) are passed as Python lists
    if pa.types.is_list(t):
        return pa.array(value, type=t)
    if isinstance(value, np.ndarray):
        # numpy → arrow zero-copy in many cases
        return pa.array(value, type=t)
    return pa.array(value, type=t)


# ---------------------------------------------------------------------------
# MON-SPAN — Zarr v2 store
# ---------------------------------------------------------------------------

# We use Zarr v2 (the more stable one in `zarr-python<3`) — `zarr-python>=3`
# refactored the API; locking to v2 keeps the writer simple. Layout:
#
#   <root>/
#     <yyyymmdd>/
#       spectrum   shape (T, 2, 256)  uint8     chunks (3600, 1, 256)
#       t_ns       shape (T,)         int64
#       center_hz  shape (T, 2)       uint32
#       span_hz    shape (T, 2)       uint32
#       res_hz     shape (T, 2)       uint32
#       pga_db     shape (T, 2)       int8
#
# The rf_block axis (length 2) is "L1" (block 0) / "L2L5" (block 1).


_BLOSC_ZSTD3 = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


class ZarrSpanSink:
    """Append-only writer for MON-SPAN into a per-day Zarr group."""

    SPECTRUM_BIN_COUNT = 256
    RF_BLOCK_COUNT = 2
    DEFAULT_CHUNK_T = 3600  # ~1 hour at 1 Hz

    def __init__(
        self,
        zarr_root: str | Path,
        day_yyyymmdd: str,
        *,
        chunk_t: int = DEFAULT_CHUNK_T,
    ) -> None:
        self.zarr_root = Path(zarr_root)
        self.zarr_root.mkdir(parents=True, exist_ok=True)
        self.group_path = self.zarr_root / day_yyyymmdd
        self.day = day_yyyymmdd

        self._store = zarr.DirectoryStore(str(self.zarr_root))
        # Open or create the per-day group
        root = zarr.open_group(self._store, mode="a")
        self._g = root.require_group(day_yyyymmdd)

        # Allocate arrays lazily on first append (we need to know chunk sizes)
        self._chunk_t = chunk_t
        self._z_spec = self._g.require_dataset(
            "spectrum",
            shape=(0, self.RF_BLOCK_COUNT, self.SPECTRUM_BIN_COUNT),
            chunks=(chunk_t, 1, self.SPECTRUM_BIN_COUNT),
            dtype="u1",
            compressor=_BLOSC_ZSTD3,
            exact=True,
        )
        self._z_t = self._g.require_dataset(
            "t_ns", shape=(0,), chunks=(chunk_t,), dtype="i8", compressor=_BLOSC_ZSTD3
        )
        self._z_center = self._g.require_dataset(
            "center_hz",
            shape=(0, self.RF_BLOCK_COUNT),
            chunks=(chunk_t, self.RF_BLOCK_COUNT),
            dtype="u4",
            compressor=_BLOSC_ZSTD3,
        )
        self._z_span = self._g.require_dataset(
            "span_hz",
            shape=(0, self.RF_BLOCK_COUNT),
            chunks=(chunk_t, self.RF_BLOCK_COUNT),
            dtype="u4",
            compressor=_BLOSC_ZSTD3,
        )
        self._z_res = self._g.require_dataset(
            "res_hz",
            shape=(0, self.RF_BLOCK_COUNT),
            chunks=(chunk_t, self.RF_BLOCK_COUNT),
            dtype="u4",
            compressor=_BLOSC_ZSTD3,
        )
        self._z_pga = self._g.require_dataset(
            "pga_db",
            shape=(0, self.RF_BLOCK_COUNT),
            chunks=(chunk_t, self.RF_BLOCK_COUNT),
            dtype="i1",
            compressor=_BLOSC_ZSTD3,
        )

        # Buffers — flushed in chunks
        self._buf_spec: list[np.ndarray] = []
        self._buf_t: list[int] = []
        self._buf_center: list[np.ndarray] = []
        self._buf_span: list[np.ndarray] = []
        self._buf_res: list[np.ndarray] = []
        self._buf_pga: list[np.ndarray] = []

    def append(self, decoded: dict[str, Any]) -> None:
        """Append one MON-SPAN message (decoded dict from ``decode_mon_span``)."""
        nb = int(decoded["nBlocks"][0])
        spec = np.asarray(decoded["spectrum"], dtype=np.uint8)
        center = np.asarray(decoded["center_hz"], dtype=np.uint32)
        span = np.asarray(decoded["span_hz"], dtype=np.uint32)
        res = np.asarray(decoded["res_hz"], dtype=np.uint32)
        pga = np.asarray(decoded["pga_db"], dtype=np.int8)
        # Pad to 2 RF blocks if the receiver only emits 1 (rare). Truncate
        # if more than 2 (very rare).
        if nb < self.RF_BLOCK_COUNT:
            spec = np.pad(spec, ((0, self.RF_BLOCK_COUNT - nb), (0, 0)))
            center = np.pad(center, (0, self.RF_BLOCK_COUNT - nb))
            span = np.pad(span, (0, self.RF_BLOCK_COUNT - nb))
            res = np.pad(res, (0, self.RF_BLOCK_COUNT - nb))
            pga = np.pad(pga, (0, self.RF_BLOCK_COUNT - nb))
        elif nb > self.RF_BLOCK_COUNT:
            spec = spec[: self.RF_BLOCK_COUNT]
            center = center[: self.RF_BLOCK_COUNT]
            span = span[: self.RF_BLOCK_COUNT]
            res = res[: self.RF_BLOCK_COUNT]
            pga = pga[: self.RF_BLOCK_COUNT]
        self._buf_spec.append(spec)
        self._buf_t.append(int(decoded["t_ns"][0]))
        self._buf_center.append(center)
        self._buf_span.append(span)
        self._buf_res.append(res)
        self._buf_pga.append(pga)
        if len(self._buf_t) >= self._chunk_t:
            self._flush()

    def _flush(self) -> None:
        if not self._buf_t:
            return
        n = len(self._buf_t)
        spec_arr = np.stack(self._buf_spec, axis=0)
        t_arr = np.array(self._buf_t, dtype=np.int64)
        center_arr = np.stack(self._buf_center, axis=0)
        span_arr = np.stack(self._buf_span, axis=0)
        res_arr = np.stack(self._buf_res, axis=0)
        pga_arr = np.stack(self._buf_pga, axis=0)
        self._z_spec.append(spec_arr, axis=0)
        self._z_t.append(t_arr, axis=0)
        self._z_center.append(center_arr, axis=0)
        self._z_span.append(span_arr, axis=0)
        self._z_res.append(res_arr, axis=0)
        self._z_pga.append(pga_arr, axis=0)
        self._buf_spec.clear()
        self._buf_t.clear()
        self._buf_center.clear()
        self._buf_span.clear()
        self._buf_res.clear()
        self._buf_pga.clear()

    def close(self) -> None:
        self._flush()
        # Persist a small JSON sidecar with day-level metadata
        meta = {
            "day": self.day,
            "n_records": int(self._z_t.shape[0]),
            "rf_blocks": ["L1", "L2L5"],
            "bins": self.SPECTRUM_BIN_COUNT,
        }
        (self.zarr_root / f"{self.day}.span.meta.json").write_text(
            json.dumps(meta, indent=2)
        )

    def __enter__(self) -> ZarrSpanSink:
        return self

    def __exit__(self, *_a: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Multi-message dispatcher used by ubx.parallel — one call site per file
# ---------------------------------------------------------------------------


class DaySinks:
    """Bundle of all per-message sinks for a single day.

    Lifecycle:
        with DaySinks(staging_root, "20250930", spectrum_root) as sinks:
            sinks.add("nav_pvt", decoded_dict)
            sinks.add("rxm_rawx", decoded_dict)
            ...
    """

    def __init__(
        self,
        staging_root: str | Path,
        day_yyyymmdd: str,
        spectrum_zarr_root: str | Path,
    ) -> None:
        self.staging_root = Path(staging_root)
        self.day = day_yyyymmdd
        self.day_dir = self.staging_root / day_yyyymmdd
        self.day_dir.mkdir(parents=True, exist_ok=True)
        # One Parquet sink per message name in ALL_SCHEMAS
        self.parquet: dict[str, ParquetSink] = {
            name: ParquetSink(self.day_dir / f"{name}.parquet", schema)
            for name, schema in ALL_SCHEMAS.items()
        }
        self.zarr = ZarrSpanSink(spectrum_zarr_root, day_yyyymmdd)

    def add(self, msg_name: str, decoded: dict[str, Any]) -> None:
        if msg_name == "mon_span":
            self.zarr.append(decoded)
        else:
            self.parquet[msg_name].append(decoded)

    def close(self) -> None:
        for s in self.parquet.values():
            s.close()
        self.zarr.close()

    def __enter__(self) -> DaySinks:
        return self

    def __exit__(self, *_a: Any) -> None:
        self.close()
