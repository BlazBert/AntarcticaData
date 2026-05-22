"""Carrier-phase scintillation proxy at 2 Hz.

True σ_φ (phase-scintillation index) requires ≥50 Hz amplitude data; at
2 Hz Nyquist is 1 Hz so we cannot compute the canonical S4 / σ_φ over the
full irregularity bandwidth. We instead compute a *low-rate proxy*:

* Detrend the carrier-phase residual by a 6th-order polynomial over a
  60-second sliding window (per (gnssId, svId, sigId) arc).
* σ_φ_proxy = std(residual)·λ converted to radians.

The output is explicitly labelled ``proxy`` and the paper carries the
caveat that it captures only sub-Nyquist phase noise.

Output: ``derived/scint/<day>.scint.parquet`` with columns
``t_ns, gnssId, svId, sigId, sigma_phi_proxy_rad, n_samples``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import polars as pl

from analysis._common import (
    C_M_S,
    derived_dir,
    gnss_signal_freq,
    list_days,
    load_config,
    read_parquet,
    staged_path,
    write_parquet,
)

log = logging.getLogger(__name__)


def _detrended_std(x: np.ndarray, order: int = 6) -> float:
    """std of polynomial-residual; NaN-safe."""
    if x.size < order + 2:
        return float("nan")
    if not np.all(np.isfinite(x)):
        return float("nan")
    t = np.arange(x.size, dtype=np.float64)
    c = np.polyfit(t, x, order)
    res = x - np.polyval(c, t)
    return float(np.std(res))


def compute_day_scint(day: str, *, window_s: float = 60.0, cfg: dict | None = None) -> pl.DataFrame:
    cfg = cfg or load_config()
    rawx = read_parquet(staged_path(day, "rxm_rawx", cfg))
    if rawx.is_empty():
        return pl.DataFrame()
    rawx = rawx.filter((pl.col("trkStat") & 0x02) != 0).sort(
        ["gnssId", "svId", "sigId", "t_ns"]
    )

    rows: list[dict] = []
    # Group iteration gives a DataFrame per (gnssId, svId, sigId)
    for (gnssId, svId, sigId), group in rawx.group_by(["gnssId", "svId", "sigId"]):
        if group.height < 30:
            continue
        t_ns = group["t_ns"].to_numpy()
        cp = group["cpMes"].to_numpy()                # carrier phase in cycles
        freqId = int(group["freqId"][0])
        f = gnss_signal_freq(int(gnssId), int(sigId), freqId)
        if not np.isfinite(f) or f <= 0:
            continue
        lam = C_M_S / f
        # Carrier phase in metres → radians: phi_rad = phi_cyc * 2π
        phi_rad = cp * 2.0 * np.pi

        # Sliding 60-s window
        win_ns = int(window_s * 1e9)
        n = phi_rad.size
        # Walk through windows of fixed time-span (epochs roughly evenly spaced)
        i = 0
        while i < n - 1:
            t0 = t_ns[i]
            j = i
            while j < n and (t_ns[j] - t0) < win_ns:
                j += 1
            chunk = phi_rad[i:j]
            if chunk.size >= 30:
                sigma = _detrended_std(chunk)
                rows.append(
                    {
                        "t_ns": int(t0 + win_ns // 2),
                        "gnssId": int(gnssId),
                        "svId": int(svId),
                        "sigId": int(sigId),
                        "sigma_phi_proxy_rad": sigma,
                        "n_samples": int(chunk.size),
                    }
                )
            i = j
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def write_day_scint(day: str, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    df = compute_day_scint(day, cfg=cfg)
    out = derived_dir(cfg) / "scint" / f"{day}.scint.parquet"
    write_parquet(df, out)
    log.info("Wrote %s (%d rows)", out, df.height)
    return out


@click.command()
@click.option("--day", "day", default=None)
def main(day: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    days = [day] if day else list_days(cfg)
    for d in days:
        write_day_scint(d, cfg)


if __name__ == "__main__":
    main()
