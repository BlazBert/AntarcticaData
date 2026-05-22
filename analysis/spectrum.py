"""MON-SPAN spectrogram analysis + RFI detection.

Reads ``derived/mon_span.zarr/<day>/`` (uint8 spectrum, shape (T, 2, 256))
and produces:

* ``derived/spectrum/<day>.spectrogram.npz`` — 1-min averaged spectrogram
  per RF block, with frequency axis in MHz.
* ``derived/spectrum/<day>.rfi.parquet`` — detected RFI events (per-bin
  running median + 5·MAD threshold over 1-h windows).
* Cross-day aggregate: ``derived/spectrum/spectrogram_full.npz``.

The MON-SPAN data is encoded as 8-bit "amplitude" — relative scale, not
absolute dBm. We treat it as a relative spectrogram.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click
import numpy as np
import polars as pl

from analysis._common import (
    derived_dir,
    list_days,
    load_config,
    resolve_path,
    write_parquet,
)

log = logging.getLogger(__name__)


def _is_contiguous_seconds(t_ns: np.ndarray, avg_seconds: int) -> bool:
    """True if t_ns differences are within (1 ± 0.1) seconds (accept 1-Hz jitter)."""
    if t_ns.size < 2:
        return True
    d_ns = np.diff(t_ns)
    return bool(np.all(np.abs(d_ns - 1_000_000_000) < 100_000_000))


def _open_day_zarr(day: str, cfg: dict) -> dict[str, np.ndarray] | None:
    import zarr  # noqa: PLC0415

    root = resolve_path(cfg["paths"]["spectrum_zarr"])
    g = root / day
    if not g.exists():
        log.warning("No zarr group for day %s at %s", day, g)
        return None
    z = zarr.open_group(str(root), mode="r")
    if day not in z:
        return None
    grp = z[day]
    return {
        "spectrum": grp["spectrum"][:],   # (T, 2, 256) uint8
        "t_ns": grp["t_ns"][:],            # (T,)
        "center_hz": grp["center_hz"][:],  # (T, 2)
        "span_hz": grp["span_hz"][:],
        "res_hz": grp["res_hz"][:],
        "pga_db": grp["pga_db"][:],
    }


def _frequency_axis(center_hz: np.ndarray, span_hz: np.ndarray, n_bins: int = 256) -> np.ndarray:
    """Per-block frequency axis for the median (center, span). MHz."""
    c = float(np.median(center_hz))
    s = float(np.median(span_hz))
    f0 = c - s / 2
    return np.linspace(f0, f0 + s, n_bins, endpoint=False) / 1e6


def compute_day_spectrogram(day: str, *, avg_seconds: int = 60, cfg: dict | None = None) -> dict[str, Any] | None:
    cfg = cfg or load_config()
    data = _open_day_zarr(day, cfg)
    if data is None:
        return None
    spec_u8 = data["spectrum"]                # (T, 2, 256) uint8
    t_ns = data["t_ns"].astype(np.int64)
    if spec_u8.size == 0:
        return None

    # Drop any frames recorded before the first NAV-PVT was seen — those
    # carry t_ns = 0 (no UTC anchor). Without this, the bin range explodes
    # to billions because t_s jumps from 0 to ~1.7e9 mid-array.
    valid = t_ns > 0
    if not valid.all():
        spec_u8 = spec_u8[valid]
        t_ns = t_ns[valid]
    if spec_u8.size == 0:
        return None

    # Convert to per-second bins and group by ``avg_seconds``.
    t_s = t_ns / 1e9
    t0 = float(t_s[0] - (t_s[0] % avg_seconds))
    bins = ((t_s - t0) // avg_seconds).astype(np.int64)
    n_bins = int(bins[-1] + 1)

    # MON-SPAN runs at exactly 1 Hz, so the contiguous reshape path is
    # correct when the records form a contiguous block of length == n_bins
    # × avg_seconds. Otherwise we fall back to ``np.add.reduceat``.
    n = spec_u8.shape[0]
    if n == n_bins * avg_seconds and _is_contiguous_seconds(t_ns, avg_seconds):
        spec_f = spec_u8.astype(np.float32)
        out = spec_f.reshape(n_bins, avg_seconds, 2, 256).mean(axis=1)
    else:
        # Defensive path: reduceat over the bin-start indices.
        unique_bins, starts = np.unique(bins, return_index=True)
        spec_f = spec_u8.astype(np.float32)
        sums = np.add.reduceat(spec_f, starts, axis=0)
        ends = np.append(starts[1:], spec_f.shape[0])
        counts = (ends - starts).astype(np.float32)
        out_grouped = sums / counts[:, None, None]
        # Re-place onto contiguous bin grid (gaps -> NaN)
        out = np.full((n_bins, 2, 256), np.nan, dtype=np.float32)
        out[unique_bins, :, :] = out_grouped
    out_t_s = t0 + np.arange(out.shape[0]) * avg_seconds
    return {
        "spectrogram": out,                                         # (n_bins, 2, 256)
        "t_s": out_t_s,
        "freq_mhz_l1": _frequency_axis(data["center_hz"][:, 0], data["span_hz"][:, 0]),
        "freq_mhz_l5": _frequency_axis(data["center_hz"][:, 1], data["span_hz"][:, 1]),
        "pga_db_mean": np.mean(data["pga_db"], axis=0),
        "n_records": int(spec_u8.shape[0]),
    }


def detect_rfi(spectrogram: np.ndarray, *, mad_k: float = 5.0) -> np.ndarray:
    """Bool mask of RFI per (time_bin, rf_block, freq_bin).

    For each (rf_block, freq_bin), compute the median + ``mad_k``·MAD of the
    time series; flag samples above this threshold. Robust against slowly-
    varying bias and against impulsive narrow-band interference.
    """
    med = np.median(spectrogram, axis=0, keepdims=True)
    mad = np.median(np.abs(spectrogram - med), axis=0, keepdims=True) + 1e-6
    return spectrogram > (med + mad_k * mad)


def _render_quicklook_png(res: dict, day: str, out_dir: Path) -> Path:
    """Per-day spectrogram quicklook PNG (L1 + L2/L5 panels). Lazy mpl import."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    spec = res["spectrogram"]
    t_s = res["t_s"]
    freq_l1 = res["freq_mhz_l1"]
    freq_l2 = res["freq_mhz_l5"]
    t_h = (t_s - t_s.min()) / 3600.0

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    for ax, block, freq, title in (
        (axes[0], spec[:, 0, :], freq_l1, "L1 band"),
        (axes[1], spec[:, 1, :], freq_l2, "L2/L5 band"),
    ):
        im = ax.pcolormesh(t_h, freq, block.T, cmap="magma", shading="auto")
        ax.set_ylabel("Freq (MHz)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Amplitude")
    axes[1].set_xlabel("Hours since first epoch")
    fig.suptitle(f"MON-SPAN waterfall — {day}")
    fig.tight_layout()
    png_path = out_dir / f"{day}.spectrogram.png"
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return png_path


def write_day_spectrum(day: str, cfg: dict | None = None) -> Path | None:
    cfg = cfg or load_config()
    out_dir = derived_dir(cfg) / "spectrum"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = compute_day_spectrogram(day, cfg=cfg)
    if res is None:
        log.warning("No spectrum data for %s", day)
        return None
    np_path = out_dir / f"{day}.spectrogram.npz"
    np.savez_compressed(
        np_path,
        spectrogram=res["spectrogram"],
        t_s=res["t_s"],
        freq_mhz_l1=res["freq_mhz_l1"],
        freq_mhz_l5=res["freq_mhz_l5"],
        pga_db_mean=res["pga_db_mean"],
    )
    log.info("Wrote %s (%d time bins)", np_path, res["spectrogram"].shape[0])
    try:
        png_path = _render_quicklook_png(res, day, out_dir)
        log.info("Wrote %s", png_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Quicklook PNG failed for %s: %s", day, exc)

    # RFI events
    rfi_mask = detect_rfi(res["spectrogram"])
    # Convert mask to event rows (time_idx, block_idx, freq_idx)
    idx = np.argwhere(rfi_mask)
    if idx.size == 0:
        return np_path
    df = pl.DataFrame(
        {
            "t_s": np.asarray(res["t_s"])[idx[:, 0]],
            "rf_block": idx[:, 1].astype(np.int8),
            "freq_idx": idx[:, 2].astype(np.int16),
            "freq_mhz": np.where(
                idx[:, 1] == 0,
                res["freq_mhz_l1"][idx[:, 2]],
                res["freq_mhz_l5"][idx[:, 2]],
            ),
            "amplitude": res["spectrogram"][idx[:, 0], idx[:, 1], idx[:, 2]],
        }
    )
    rfi_path = out_dir / f"{day}.rfi.parquet"
    write_parquet(df, rfi_path)
    log.info("RFI events → %s (%d)", rfi_path, df.height)
    return np_path


@click.command()
@click.option("--day", "day", default=None)
def main(day: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    days = [day] if day else list_days(cfg)
    for d in days:
        write_day_spectrum(d, cfg)


if __name__ == "__main__":
    main()
