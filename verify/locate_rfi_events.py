"""Map strong-jamming days to ship positions.

Run on the server after the analysis pipeline has produced
``derived/track/<day>.track.parquet`` and ``staging/<day>/mon_rf.parquet``
files.

Usage::

    cd /home/jovyan/Projects/gps-data/data/code

    # Built-in list of strong-jamming days from the cross-day summary:
    python -m verify.locate_rfi_events

    # Or pass your own list:
    python -m verify.locate_rfi_events --days 20251210,20251222,20260308

    # Or scan all staged days, report only those with L1 jam_max above a threshold:
    python -m verify.locate_rfi_events --scan --l1-threshold 60

For each day prints: median lat/lon, average ground speed, L1 + L2/L5
jam_max, count of L1 jamInd > 60 epochs. From the lat/lon you can tell
whether each strong-jamming day is at-port (speed ≈ 0, lat/lon matches a
known harbour), in-transit (speed > 5 m/s), or stationary at sea.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import polars as pl

from analysis._common import (
    derived_dir,
    list_days,
    load_config,
    read_parquet,
    staged_path,
)

log = logging.getLogger(__name__)

# From the May-2026 cross-day summary across 216 days.
DEFAULT_STRONG_DAYS = [
    "20251123", "20251124",          # two-day RFI episode
    "20251209", "20251210",          # 12-09 mild, 12-10 strong (126)
    "20251222",                      # isolated 156
    "20251228", "20251230",          # late-Dec cluster
    "20260228", "20260301", "20260302", "20260303",  # 3-day stationary at 138/136/134
    "20260307", "20260308",          # 03-08 highest of the cruise (192)
    "20260419", "20260428",          # mild Mediterranean episodes
]


def _row_for_day(day: str, cfg: dict) -> dict | None:
    track_p = derived_dir(cfg) / "track" / f"{day}.track.parquet"
    rf_p = staged_path(day, "mon_rf", cfg)
    if not track_p.exists() or not rf_p.exists():
        return None
    tr = read_parquet(track_p)
    rf = read_parquet(rf_p)
    if tr.is_empty() or rf.is_empty():
        return None

    rf_l1 = rf.filter(pl.col("blockId") == 0)
    rf_l5 = rf.filter(pl.col("blockId") == 1)
    return {
        "day": day,
        "lat_med": float(tr["lat"].median() or float("nan")),
        "lon_med": float(tr["lon"].median() or float("nan")),
        "speed_avg_m_s": float(tr["gSpeed_m_s"].mean() or float("nan")),
        "speed_p95_m_s": float(tr["gSpeed_m_s"].quantile(0.95) or float("nan")),
        "L1_jam_max": int(rf_l1["jamInd"].max() or 0) if not rf_l1.is_empty() else 0,
        "L2L5_jam_max": int(rf_l5["jamInd"].max() or 0) if not rf_l5.is_empty() else 0,
        "L1_jam_gt60_n": int((rf_l1["jamInd"] > 60).sum()) if not rf_l1.is_empty() else 0,
        "L1_jam_gt100_n": int((rf_l1["jamInd"] > 100).sum()) if not rf_l1.is_empty() else 0,
        "agc_l5_std": float(rf_l5["agcCnt"].std() or float("nan")) if not rf_l5.is_empty() else float("nan"),
    }


def _print_table(rows: list[dict]) -> None:
    print(f"{'day':<10}{'lat':>9}{'lon':>10}{'speed_m_s':>11}"
          f"{'speed_p95':>11}{'L1_max':>8}{'L1>60':>8}{'L1>100':>9}{'L2/L5_max':>10}")
    for r in rows:
        print(
            f"{r['day']:<10}"
            f"{r['lat_med']:>9.3f}"
            f"{r['lon_med']:>10.3f}"
            f"{r['speed_avg_m_s']:>11.2f}"
            f"{r['speed_p95_m_s']:>11.2f}"
            f"{r['L1_jam_max']:>8}"
            f"{r['L1_jam_gt60_n']:>8}"
            f"{r['L1_jam_gt100_n']:>9}"
            f"{r['L2L5_jam_max']:>10}"
        )


@click.command()
@click.option("--days", default=None,
              help="Comma-separated YYYYMMDD list. Default = built-in strong-RFI list.")
@click.option("--scan", is_flag=True, default=False,
              help="Scan ALL staged days; combine with --l1-threshold.")
@click.option("--l1-threshold", default=60, type=int, show_default=True,
              help="Only print days with L1 jamInd > threshold at any epoch.")
@click.option("--csv-out", default=None, type=click.Path(),
              help="If set, write the table as CSV here (still printed to stdout).")
def main(days: str | None, scan: bool, l1_threshold: int, csv_out: str | None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    if scan:
        candidates = list_days(cfg)
    elif days:
        candidates = [d.strip() for d in days.split(",") if d.strip()]
    else:
        candidates = DEFAULT_STRONG_DAYS

    rows: list[dict] = []
    for d in candidates:
        r = _row_for_day(d, cfg)
        if r is None:
            log.warning("Skipping %s — track or mon_rf parquet missing", d)
            continue
        if scan and r["L1_jam_max"] <= l1_threshold:
            continue
        rows.append(r)

    rows.sort(key=lambda x: x["day"])
    _print_table(rows)

    if csv_out:
        df = pl.DataFrame(rows)
        Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(csv_out)
        log.info("Wrote %s (%d rows)", csv_out, df.height)


if __name__ == "__main__":
    main()
