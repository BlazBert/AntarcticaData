"""Partial-correlation decomposition of L2/L5 jamInd vs |lat| and temperature.

For each cruise day we already have:
  - mean L2/L5 ``jamInd`` (from ``staging/<day>/mon_rf.parquet``)
  - mean receiver temperature (``MON-SYS.tempValue``)
  - day-median absolute geographic latitude (from track aggregate)

We want to know how much of the jamInd~|lat| correlation survives after
controlling for temperature, and vice versa. The cleanest answer is the
partial Spearman correlation in both directions.

Output:
  - prints rho(jam, lat), rho(jam, T), and the two partial correlations
    rho(jam, lat | T) and rho(jam, T | lat)
  - writes ``tables/T_jam_confound.csv`` with the per-day table used.

Usage:
    python -m analysis.jamming_confound
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

from analysis._common import (
    derived_dir,
    list_days,
    load_config,
    read_parquet,
    staged_path,
    tables_dir,
    write_parquet,
)

log = logging.getLogger(__name__)


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Partial Spearman correlation of x and y controlling for z.

    Implementation: convert each to ranks, then compute the residual of
    rank(x) ~ rank(z) and rank(y) ~ rank(z), and Pearson-correlate the
    residuals. Returns (rho_partial, two-sided p-value).
    """
    from scipy import stats

    rx = stats.rankdata(x); ry = stats.rankdata(y); rz = stats.rankdata(z)
    # Linear regress out rank(z) from rank(x) and rank(y).
    Z = np.column_stack([rz, np.ones_like(rz)])
    bx, _ = np.linalg.lstsq(Z, rx, rcond=None)[:2]
    by, _ = np.linalg.lstsq(Z, ry, rcond=None)[:2]
    ex = rx - Z @ bx
    ey = ry - Z @ by
    r, p = stats.pearsonr(ex, ey)
    return float(r), float(p)


def build_table(cfg: dict | None = None) -> pl.DataFrame:
    cfg = cfg or load_config()
    rows = []
    track_path = derived_dir(cfg) / "track" / "track.all.parquet"
    if not track_path.exists():
        raise FileNotFoundError(f"Missing {track_path}")
    track = read_parquet(track_path).filter(
        (pl.col("t_ns") > 1_700_000_000_000_000_000) & (pl.col("lat").abs() <= 90.0)
    )
    # Per-day day-median |lat|
    track = track.with_columns(
        (pl.col("t_ns") // (86400 * 1_000_000_000)).cast(pl.Int64).alias("day_unix")
    )
    daily_lat = track.group_by("day_unix").agg(
        pl.col("lat").median().abs().alias("abs_lat_median")
    )

    for d in list_days(cfg):
        sys_p = staged_path(d, "mon_sys", cfg)
        rf_p = staged_path(d, "mon_rf", cfg)
        if not sys_p.exists() or not rf_p.exists():
            continue
        sys_df = read_parquet(sys_p)
        rf_df = read_parquet(rf_p)
        if sys_df.is_empty() or rf_df.is_empty():
            continue
        l2_jam = rf_df.filter(pl.col("blockId") == 1)["jamInd"]
        if l2_jam.is_empty():
            continue
        mean_jam = float(l2_jam.mean())
        mean_T = float(sys_df["tempValue_C"].cast(pl.Float64).mean())
        # day_unix derived from the date string
        from datetime import datetime, timezone
        day_unix = int(
            datetime(int(d[0:4]), int(d[4:6]), int(d[6:8]),
                     tzinfo=timezone.utc).timestamp() / 86400
        )
        match = daily_lat.filter(pl.col("day_unix") == day_unix)
        if match.is_empty():
            continue
        abs_lat = float(match["abs_lat_median"][0])
        rows.append({
            "day": d,
            "abs_lat_median": abs_lat,
            "mean_temp_C": mean_T,
            "mean_jam_l5": mean_jam,
        })
    return pl.DataFrame(rows)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    cfg = load_config()
    df = build_table(cfg)
    if df.is_empty():
        print("no rows")
        return 1
    out_csv = tables_dir(cfg) / "T_jam_confound.csv"
    df.write_csv(out_csv)
    log.info("Wrote %s (%d rows)", out_csv, df.height)

    from scipy import stats
    x = df["mean_jam_l5"].to_numpy()
    lat = df["abs_lat_median"].to_numpy()
    T = df["mean_temp_C"].to_numpy()

    r_lat, p_lat = stats.spearmanr(x, lat)
    r_T, p_T = stats.spearmanr(x, T)
    pr_lat, pp_lat = _partial_spearman(x, lat, T)
    pr_T, pp_T = _partial_spearman(x, T, lat)

    print(f"n = {x.size}")
    print(f"rho(jam, |lat|)        = {r_lat:+.3f}  (p = {p_lat:.2e})")
    print(f"rho(jam, T)            = {r_T:+.3f}  (p = {p_T:.2e})")
    print(f"rho(jam, |lat| | T)    = {pr_lat:+.3f}  (p = {pp_lat:.2e})  <-- |lat| effect after removing T")
    print(f"rho(jam, T | |lat|)    = {pr_T:+.3f}  (p = {pp_T:.2e})    <-- T effect after removing |lat|")

    # Short interpretive line for the paper:
    if abs(pr_lat) > abs(pr_T):
        print("=> |lat| dominates after controlling for T")
    elif abs(pr_T) > abs(pr_lat):
        print("=> Temperature dominates after controlling for |lat|")
    else:
        print("=> |lat| and T contribute comparably")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
