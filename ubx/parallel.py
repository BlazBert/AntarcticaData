"""Per-file parser worker + ``multiprocessing.Pool`` driver.

The worker is :func:`parse_one_day` — given a path to a ``.ubx`` file it:

1. Memory-maps the file
2. Walks frames via ``ubx.parser.iter_frames``
3. Decodes each known message into column arrays
4. Appends to per-day Parquet/Zarr sinks
5. Returns a dict of statistics for the driver to aggregate

The driver is :func:`run_pool` — it builds a ``multiprocessing.Pool`` (default
32 workers, ``maxtasksperchild=4``) and pipes file paths through
``pool.imap_unordered``.

Why ``Pool(32)`` rather than ``Pool(128)`` on a 128-thread server: see plan
section "Stage 1". I/O queue saturates well before 128 workers, parsing is
fairly compute-bound so memory bandwidth (not threads) dominates, and the
parallel ``convbin``/``gfzrnx``/PRIDE PPP-AR stages later need free cores.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import traceback
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from ubx.messages import CLASS_ID_TO_NAME, MSG_DECODERS, MessageName
from ubx.parser import iter_frames, open_mmap
from ubx.writers import DaySinks

log = logging.getLogger(__name__)

_DAY_RE = re.compile(r"(20\d{6})\.ubx$")


@dataclass
class DayStats:
    """Per-file stats returned by :func:`parse_one_day`."""

    day: str
    src: str
    bytes_read: int
    n_frames_total: int
    n_frames_decoded: int
    counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    elapsed_s: float = 0.0


def _day_from_path(path: Path) -> str:
    m = _DAY_RE.search(path.name)
    if not m:
        raise ValueError(
            f"Cannot infer day from filename {path.name}; expected YYYYMMDD.ubx"
        )
    return m.group(1)


def parse_one_day(args: tuple) -> DayStats:
    """Worker entry point.

    ``args`` is ``(ubx_path, staging_root, spectrum_zarr_root, skip_msgs)``
    where ``skip_msgs`` is a frozenset of ``MessageName.value`` strings to
    drop (e.g. ``frozenset({"rxm_sfrbx"})`` for the fast analysis path).
    """
    import time

    if len(args) == 3:
        ubx_path, staging_root, spectrum_root = args
        skip_msgs: frozenset[str] = frozenset()
    else:
        ubx_path, staging_root, spectrum_root, skip_msgs = args
    ubx_path = Path(ubx_path)
    day = _day_from_path(ubx_path)
    stats = DayStats(
        day=day,
        src=str(ubx_path),
        bytes_read=ubx_path.stat().st_size,
        n_frames_total=0,
        n_frames_decoded=0,
    )
    t0 = time.time()
    decoder_failures: dict[str, int] = defaultdict(int)
    try:
        with open_mmap(ubx_path) as mm:
            with DaySinks(staging_root, day, spectrum_root) as sinks:
                ctx: dict[str, Any] = {"t_ns_pvt": 0}
                counts: dict[str, int] = defaultdict(int)
                for fr in iter_frames(mm):
                    stats.n_frames_total += 1
                    key = (fr.class_id, fr.msg_id)
                    name = CLASS_ID_TO_NAME.get(key)
                    if name is None:
                        continue
                    if name.value in skip_msgs:
                        counts[name.value] += 1
                        continue
                    decoder = MSG_DECODERS[name]
                    try:
                        decoded = decoder(fr.payload, ctx)
                    except Exception as exc:  # noqa: BLE001
                        decoder_failures[name.value] += 1
                        if decoder_failures[name.value] <= 3:
                            log.warning(
                                "decoder %s raised on frame at offset %d (len=%d): %s",
                                name.value,
                                fr.payload_offset,
                                fr.payload_length,
                                exc,
                            )
                        continue
                    if decoded is None:
                        continue
                    if name is MessageName.NAV_PVT:
                        try:
                            ctx["t_ns_pvt"] = int(decoded["t_ns"][0])
                        except (KeyError, IndexError):
                            pass
                    try:
                        sinks.add(name.value, decoded)
                    except Exception as exc:  # noqa: BLE001
                        decoder_failures[f"{name.value}_write"] += 1
                        if decoder_failures[f"{name.value}_write"] <= 3:
                            log.warning(
                                "writer %s raised at offset %d: %s",
                                name.value,
                                fr.payload_offset,
                                exc,
                            )
                        continue
                    counts[name.value] += 1
                    stats.n_frames_decoded += 1
                stats.counts = dict(counts)
    except Exception:
        stats.error = traceback.format_exc()
    if decoder_failures:
        log.warning("decoder/writer failure tally for %s: %s", day, dict(decoder_failures))
    stats.elapsed_s = time.time() - t0
    return stats


def run_pool(
    ubx_paths: Iterable[str | Path],
    staging_root: str | Path,
    spectrum_root: str | Path,
    *,
    workers: int = 32,
    maxtasksperchild: int = 4,
    progress: bool = True,
) -> list[DayStats]:
    """Driver — fan-out across ``workers`` processes."""
    paths = [str(Path(p)) for p in ubx_paths]
    if not paths:
        log.warning("run_pool: no input files")
        return []
    args_list = [(p, str(staging_root), str(spectrum_root)) for p in paths]

    # spawn context — clean state per worker, no fork-inherited handles
    ctx = get_context("spawn")
    results: list[DayStats] = []
    log.info("run_pool: %d files, %d workers", len(paths), workers)
    with ctx.Pool(processes=workers, maxtasksperchild=maxtasksperchild) as pool:
        iterator = pool.imap_unordered(parse_one_day, args_list, chunksize=1)
        if progress and sys.stderr.isatty():
            try:
                from tqdm import tqdm  # noqa: PLC0415

                iterator = tqdm(iterator, total=len(paths), unit="file")
            except ImportError:
                pass
        for st in iterator:
            results.append(st)
            if st.error:
                log.error("FAILED %s\n%s", st.src, st.error)
            else:
                log.info(
                    "OK %s frames=%d decoded=%d in %.1fs",
                    st.day,
                    st.n_frames_total,
                    st.n_frames_decoded,
                    st.elapsed_s,
                )
    return results


def discover_ubx_files(ubx_dir: str | Path) -> list[Path]:
    """List ``YYYYMMDD.ubx`` files in ``ubx_dir`` sorted by date."""
    p = Path(ubx_dir)
    found = sorted(p.glob("*.ubx"))
    return [f for f in found if _DAY_RE.search(f.name)]


__all__ = ["DayStats", "parse_one_day", "run_pool", "discover_ubx_files"]
