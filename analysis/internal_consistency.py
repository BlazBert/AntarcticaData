"""Per-day NAV-PVT vs NAV-HPPOSLLH internal-consistency table.

Computes the per-day distribution of the difference between the
Kalman-filtered NAV-PVT fix and the high-precision NAV-HPPOSLLH fix.
This is the receiver-internal sanity check referenced in section
"Receiver internal consistency" of the manuscript. Results are
written to ``tables/T_internal_consistency.csv`` so reviewers can
verify the no-systematic-dependence-on-lat-or-speed claim.

Output columns:
  day, n_epochs, lat_mean, abs_lat_mean, day_mean_speed_ms,
  d2d_p50_mm, d2d_p95_mm, d2d_max_mm,
  dz_p50_mm, dz_p95_mm, dz_max_mm

Usage:
    python -m analysis.internal_consistency
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import polars as pl

from analysis._common import (
    list_days,
    load_config,
    read_parquet,
    staged_path,
    tables_dir,
)

log = logging.getLogger(__name__)
R_EARTH_M = 6_371_008.8


def _day_row(day: str, cfg: dict) -> dict | None:
    pvt = staged_path(day, "nav_pvt", cfg)
    hpp = staged_path(day, "nav_hpposllh", cfg)
    if not pvt.exists() or not hpp.exists():
        return None
    p = read_parquet(pvt)
    h = read_parquet(hpp)
    if p.is_empty() or h.is_empty():
        return None
    pvt_pos = p.select([
        "t_ns",
        (pl.col("lat_1e7").cast(pl.Float64) * 1e-7).alias("lat_pvt"),
        (pl.col("lon_1e7").cast(pl.Float64) * 1e-7).alias("lon_pvt"),
        (pl.col("height_mm").cast(pl.Float64) * 1e-3).alias("h_pvt"),
        (pl.col("gSpeed_mm_s").cast(pl.Float64) * 1e-3).alias("gSpeed_m_s"),
    ]).sort("t_ns")
    hpp_pos = h.select([
        "t_ns",
        (pl.col("lat_1e7").cast(pl.Float64) * 1e-7
         + pl.col("latHp_1e9").cast(pl.Float64) * 1e-9).alias("lat_hpp"),
        (pl.col("lon_1e7").cast(pl.Float64) * 1e-7
         + pl.col("lonHp_1e9").cast(pl.Float64) * 1e-9).alias("lon_hpp"),
        (pl.col("height_mm").cast(pl.Float64) * 1e-3
         + pl.col("heightHp_0p1mm").cast(pl.Float64) * 1e-4).alias("h_hpp"),
    ]).sort("t_ns")
    joined = pvt_pos.join(hpp_pos, on="t_ns", how="inner")
    if joined.is_empty():
        return None
    lat0 = float(joined["lat_pvt"].mean() or 0.0)
    cos_lat = math.cos(math.radians(lat0))
    dn = (joined["lat_hpp"].to_numpy() - joined["lat_pvt"].to_numpy()) * math.radians(1) * R_EARTH_M
    de = (joined["lon_hpp"].to_numpy() - joined["lon_pvt"].to_numpy()) * math.radians(1) * R_EARTH_M * cos_lat
    dz = (joined["h_hpp"].to_numpy() - joined["h_pvt"].to_numpy())
    d2d = np.sqrt(de**2 + dn**2) * 1000.0  # mm
    dz_mm = np.abs(dz) * 1000.0
    return {
        "day": day,
        "n_epochs": int(joined.height),
        "lat_mean": lat0,
        "abs_lat_mean": abs(lat0),
        "day_mean_speed_ms": float(joined["gSpeed_m_s"].mean() or 0.0),
        "d2d_p50_mm": float(np.percentile(d2d, 50)),
        "d2d_p95_mm": float(np.percentile(d2d, 95)),
        "d2d_max_mm": float(np.max(d2d)),
        "dz_p50_mm": float(np.percentile(dz_mm, 50)),
        "dz_p95_mm": float(np.percentile(dz_mm, 95)),
        "dz_max_mm": float(np.max(dz_mm)),
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    cfg = load_config()
    rows = []
    for day in list_days(cfg):
        r = _day_row(day, cfg)
        if r:
            rows.append(r)
    df = pl.DataFrame(rows)
    out = tables_dir(cfg) / "T_internal_consistency.csv"
    df.write_csv(out)
    log.info("Wrote %s (%d rows)", out, df.height)
    if df.is_empty():
        return 1
    # Sanity print of dependence
    from scipy import stats
    lat = df["abs_lat_mean"].to_numpy()
    speed = df["day_mean_speed_ms"].to_numpy()
    d2d = df["d2d_p95_mm"].to_numpy()
    rlat, plat = stats.spearmanr(lat, d2d)
    rspd, pspd = stats.spearmanr(speed, d2d)
    print(f"n_days = {df.height}")
    print(f"rho(d2d_p95, |lat|)        = {rlat:+.3f}  (p = {plat:.2e})")
    print(f"rho(d2d_p95, day_mean_speed) = {rspd:+.3f}  (p = {pspd:.2e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
