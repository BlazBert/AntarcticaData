"""Track-quality diagnostic.

Run on the server:

    cd /home/jovyan/Projects/gps-data/data/code
    python -m verify.diagnose_track --day 20250930
    python -m verify.diagnose_track --day 20260127  # Antarctic
    python -m verify.diagnose_track                 # all staged days, summary table

For each day reports:

* Distribution of ``fixType`` (3 = 3D fix, anything < 3 may carry stale
  coordinates).
* ``hAcc`` and ``numSV`` per fix type.
* MON-RF jamming indicator extremes and AGC standard deviation per RF
  block (L1 and L2/L5).
* Count of rows with literal lat=lon=0 ("Null Island" — fix re-acquisition
  artefact).
* Cross-day summary: which days look anomalous?

This script is read-only — it inspects ``staging/<day>/*.parquet`` and
prints findings. Use ``analysis.anomalies`` to *catalogue* anomalies
into a per-day Parquet that can be cited in the paper.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import polars as pl

from analysis._common import list_days, load_config, staged_path

log = logging.getLogger(__name__)


def diagnose_one_day(day: str, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    pvt_p = staged_path(day, "nav_pvt", cfg)
    rf_p = staged_path(day, "mon_rf", cfg)
    if not pvt_p.exists():
        return {"day": day, "status": "missing_nav_pvt"}
    pvt = pl.read_parquet(pvt_p)

    fix_dist = (
        pvt.group_by("fixType").len()
        .sort("fixType")
        .with_columns((pl.col("len") / pvt.height * 100).alias("pct"))
    )
    by_fix = (
        pvt.group_by("fixType").agg([
            (pl.col("hAcc_mm") * 1e-3).mean().alias("hAcc_mean_m"),
            (pl.col("hAcc_mm") * 1e-3).max().alias("hAcc_max_m"),
            pl.col("numSV").mean().alias("numSV_mean"),
            pl.col("numSV").min().alias("numSV_min"),
        ])
        .sort("fixType")
    )
    null_island = pvt.filter((pl.col("lat_1e7") == 0) & (pl.col("lon_1e7") == 0)).height
    early_t = pvt.filter(pl.col("t_ns") < 1_500_000_000_000_000_000).height

    rf_summary: dict = {}
    if rf_p.exists():
        rf = pl.read_parquet(rf_p)
        if rf.height:
            rf_summary = {
                "by_block": rf.group_by("blockId").agg([
                    pl.col("jamInd").max().alias("jam_max"),
                    pl.col("jamInd").mean().alias("jam_mean"),
                    (pl.col("jamInd") > 32).sum().alias("high_jam_n"),
                    pl.col("agcCnt").std().alias("agc_std"),
                ]).sort("blockId").to_dicts(),
            }

    return {
        "day": day,
        "n_epochs": pvt.height,
        "fix_dist": fix_dist.to_dicts(),
        "by_fix": by_fix.to_dicts(),
        "null_island": null_island,
        "early_t_count": early_t,
        "rf_summary": rf_summary,
    }


def _print_one_day(d: dict) -> None:
    print(f"\n=== {d['day']} ===")
    if d.get("status") == "missing_nav_pvt":
        print("  [missing nav_pvt — skip]")
        return
    print(f"  epochs       : {d['n_epochs']}")
    print(f"  null-island  : {d['null_island']}")
    print(f"  pre-2017 t_ns: {d['early_t_count']}")
    print(f"  fix distribution:")
    for r in d["fix_dist"]:
        print(f"    fixType={r['fixType']}  n={r['len']:>7}  ({r['pct']:.2f}%)")
    print(f"  per-fix accuracy / numSV:")
    for r in d["by_fix"]:
        print(
            f"    fixType={r['fixType']}  hAcc_mean={r['hAcc_mean_m']:.2f} m"
            f"  hAcc_max={r['hAcc_max_m']:.2f} m  numSV_mean={r['numSV_mean']:.1f}"
            f"  numSV_min={r['numSV_min']}"
        )
    if d["rf_summary"]:
        print(f"  MON-RF per block (0=L1, 1=L2/L5):")
        for r in d["rf_summary"]["by_block"]:
            print(
                f"    block={r['blockId']}  jam_max={r['jam_max']}"
                f"  jam_mean={r['jam_mean']:.1f}  high_jam(>32)={r['high_jam_n']}"
                f"  agc_std={r['agc_std']:.1f}"
            )


def _print_summary(rows: list[dict]) -> None:
    print("\n========== CROSS-DAY SUMMARY ==========")
    print(f"{'day':<10}{'epochs':>8}{'fix3%':>8}{'null':>6}"
          f"{'preT':>6}{'jam_max_L1':>12}{'jam_max_L2/L5':>15}")
    for d in rows:
        if d.get("status") == "missing_nav_pvt":
            print(f"{d['day']:<10}  --missing--")
            continue
        fix3 = next((r['pct'] for r in d['fix_dist'] if r['fixType'] >= 3), 0.0)
        rfb = d["rf_summary"].get("by_block", []) if d["rf_summary"] else []
        jam_l1 = next((r['jam_max'] for r in rfb if r['blockId'] == 0), '-')
        jam_l5 = next((r['jam_max'] for r in rfb if r['blockId'] == 1), '-')
        print(
            f"{d['day']:<10}{d['n_epochs']:>8}{fix3:>8.2f}{d['null_island']:>6}"
            f"{d['early_t_count']:>6}{str(jam_l1):>12}{str(jam_l5):>15}"
        )


@click.command()
@click.option("--day", default=None, help="YYYYMMDD; if omitted, all staged days")
def main(day: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    days = [day] if day else list_days(cfg)
    if not days:
        raise click.ClickException("No staged days found.")
    rows = []
    for d in days:
        r = diagnose_one_day(d, cfg)
        rows.append(r)
        _print_one_day(r)
    if len(rows) > 1:
        _print_summary(rows)


if __name__ == "__main__":
    main()
