"""Catalogue trajectory anomalies — preserve, don't filter.

Spoofing, jamming, boot-state coordinate errors and similar artefacts are
*data* — they tell us something about the GNSS environment along the
cruise. The pipeline already drops them from the visualised track to
keep maps clean, but it must not silently delete them. This module
catalogues every anomaly category into ``derived/anomalies/<day>.parquet``
and emits a summary that can be cited in the paper.

Categories produced:

* ``boot_state``   t_ns < 2017 OR t_ns repeated identically across many
                   rows AND fixType < 3 — receiver clock not yet synced.
* ``no_fix``       fixType < 3 alone (without the boot-state pattern).
* ``null_island``  lat = lon = 0 — fix-reacquisition cache miss.
* ``speed_outlier`` consecutive points imply > 100 knots ground speed.
* ``possible_spoof`` fixType ≥ 3, hAcc < 5 m, but speed-outlier *and*
                    MON-RF jamInd within nominal range. Flagged because
                    a coordinate jump that the receiver itself thinks is
                    accurate and not jammed is the canonical spoofing
                    fingerprint.
* ``high_jamming``  MON-RF jamInd > 64 on either RF block (terrestrial
                    interference, may correlate with cycle-slip elevation).

Each row in the output Parquet has:

    t_ns, lat, lon, fixType, numSV, hAcc_m, vAcc_m,
    category, prev_lat, prev_lon, dt_s, dist_km, implied_speed_kn,
    jamInd_L1, jamInd_L2L5, agcCnt_L1, agcCnt_L2L5

CLI:

    python -m analysis.anomalies --day 20250930
    python -m analysis.anomalies                  # all staged days

Aggregate (cross-day summary table):

    python -m analysis.anomalies --aggregate
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
    haversine_km,
    list_days,
    load_config,
    read_parquet,
    staged_path,
    tables_dir,
    write_parquet,
)

log = logging.getLogger(__name__)

T_MIN_NS = 1_735_689_600 * 1_000_000_000   # 2025-01-01 UTC — pre-cruise
T_MAX_NS = 1_798_761_600 * 1_000_000_000   # 2027-01-01 UTC — post-cruise
SPEED_OUTLIER_KN = 100.0
# Hybrid jamming-flag thresholds — three-layer:
#
#   1. ABSOLUTE FLOOR (HIGH_JAM_THRESHOLD_*_ABS) catches sustained loud
#      events regardless of the day's baseline.
#   2. RELATIVE threshold (median + N*MAD on the same block) catches
#      context-relative excursions on otherwise-quiet days.
#   3. PER-DAY MINIMUM (PER_DAY_MIN_*) clamps the relative threshold so
#      it never drops below where real RFI plausibly begins — needed
#      because MAD collapses to 0/1 on the discrete uint8 jamInd
#      distribution and would otherwise flag the baseline itself.
#
# Final per-day threshold for each block:
#    jam_thr = min(absolute_floor, max(relative_threshold, per_day_minimum))
# A row is flagged iff jamInd > jam_thr on that block.
HIGH_JAM_THRESHOLD_L1_ABS = 60
HIGH_JAM_THRESHOLD_L2L5_ABS = 130
JAM_RELATIVE_N_MAD = 5
# Above the L1 baseline 17–30 but below the absolute floor 60. Any
# jamInd above this on the L1 block is the lowest value we still call
# "interesting" on the quietest day.
PER_DAY_MIN_L1 = 40
# Above the L2/L5 baseline 70–100 but below the absolute floor 130.
PER_DAY_MIN_L2L5 = 110
HACC_TIGHT_M = 5.0


def detect_day_anomalies(day: str, cfg: dict | None = None) -> pl.DataFrame:
    cfg = cfg or load_config()
    pvt_p = staged_path(day, "nav_pvt", cfg)
    rf_p = staged_path(day, "mon_rf", cfg)
    if not pvt_p.exists():
        return pl.DataFrame()
    pvt = read_parquet(pvt_p).sort("t_ns")
    if pvt.is_empty():
        return pl.DataFrame()

    rf_l1: dict[int, dict] = {}
    rf_l5: dict[int, dict] = {}
    # Per-day robust thresholds: median + N*MAD on each block.
    jam_thr_l1 = float(HIGH_JAM_THRESHOLD_L1_ABS)
    jam_thr_l5 = float(HIGH_JAM_THRESHOLD_L2L5_ABS)
    if rf_p.exists():
        rf = read_parquet(rf_p)
        for blk_id, target in ((0, rf_l1), (1, rf_l5)):
            sub = rf.filter(pl.col("blockId") == blk_id)
            for row in sub.iter_rows(named=True):
                target[int(row["t_ns"])] = row
            if not sub.is_empty():
                vals = sub["jamInd"].to_numpy().astype(np.float64)
                med = float(np.median(vals))
                mad = float(np.median(np.abs(vals - med)))
                rel_thr = med + JAM_RELATIVE_N_MAD * mad
                # Clamp the relative threshold below the per-day minimum
                # (the MAD-on-discrete-data trap), then take min with the
                # absolute floor.
                if blk_id == 0:
                    jam_thr_l1 = min(HIGH_JAM_THRESHOLD_L1_ABS,
                                       max(rel_thr, PER_DAY_MIN_L1))
                else:
                    jam_thr_l5 = min(HIGH_JAM_THRESHOLD_L2L5_ABS,
                                       max(rel_thr, PER_DAY_MIN_L2L5))
    log.info("day %s: jam thresholds L1=%.1f, L2L5=%.1f", day, jam_thr_l1, jam_thr_l5)

    rows: list[dict[str, Any]] = []
    lat = pvt["lat_1e7"].to_numpy().astype(np.float64) * 1e-7
    lon = pvt["lon_1e7"].to_numpy().astype(np.float64) * 1e-7
    t = pvt["t_ns"].to_numpy()
    fix = pvt["fixType"].to_numpy()
    nSV = pvt["numSV"].to_numpy()
    hAcc = pvt["hAcc_mm"].to_numpy().astype(np.float64) * 1e-3
    vAcc = pvt["vAcc_mm"].to_numpy().astype(np.float64) * 1e-3

    last_good_i = -1
    for i in range(len(t)):
        cats: list[str] = []
        # Time-window guards
        if t[i] < T_MIN_NS or t[i] > T_MAX_NS:
            cats.append("boot_state")
        if fix[i] < 3 and "boot_state" not in cats:
            cats.append("no_fix")
        if int(pvt["lat_1e7"][i]) == 0 and int(pvt["lon_1e7"][i]) == 0:
            cats.append("null_island")
        # Speed outlier and possible spoof
        prev_lat = prev_lon = float("nan")
        dt_s = float("nan")
        dist_km = float("nan")
        speed_kn = float("nan")
        if last_good_i >= 0:
            prev_lat = float(lat[last_good_i])
            prev_lon = float(lon[last_good_i])
            dt_s = max((t[i] - t[last_good_i]) / 1e9, 1e-3)
            dist_km = float(haversine_km(
                np.array([prev_lat]), np.array([prev_lon]),
                np.array([float(lat[i])]), np.array([float(lon[i])]),
            )[0])
            speed_kn = (dist_km / (dt_s / 3600.0)) / 1.852
            if speed_kn > SPEED_OUTLIER_KN:
                cats.append("speed_outlier")
                # Spoof signature: receiver still thinks it's a good fix
                if fix[i] >= 3 and hAcc[i] < HACC_TIGHT_M:
                    cats.append("possible_spoof")
        # MON-RF anomaly merge — split L1 (real RFI) from L2/L5 (baseline).
        # Hybrid threshold: absolute floor OR per-day (median + 5*MAD).
        rf1 = rf_l1.get(int(t[i]))
        rf5 = rf_l5.get(int(t[i]))
        jam_l1 = rf1["jamInd"] if rf1 else None
        jam_l5 = rf5["jamInd"] if rf5 else None
        if jam_l1 is not None and jam_l1 > jam_thr_l1:
            cats.append("high_jamming_l1")
        if jam_l5 is not None and jam_l5 > jam_thr_l5:
            cats.append("high_jamming_l5")

        if cats:
            rows.append({
                "t_ns": int(t[i]),
                "lat": float(lat[i]),
                "lon": float(lon[i]),
                "fixType": int(fix[i]),
                "numSV": int(nSV[i]),
                "hAcc_m": float(hAcc[i]),
                "vAcc_m": float(vAcc[i]),
                "category": ";".join(cats),
                "prev_lat": prev_lat,
                "prev_lon": prev_lon,
                "dt_s": dt_s,
                "dist_km": dist_km,
                "implied_speed_kn": speed_kn,
                "jamInd_L1": int(jam_l1) if jam_l1 is not None else -1,
                "jamInd_L2L5": int(jam_l5) if jam_l5 is not None else -1,
                "agcCnt_L1": int(rf1["agcCnt"]) if rf1 else -1,
                "agcCnt_L2L5": int(rf5["agcCnt"]) if rf5 else -1,
            })

        if "boot_state" not in cats and "no_fix" not in cats and "null_island" not in cats:
            last_good_i = i

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def write_day(day: str, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    df = detect_day_anomalies(day, cfg)
    out = derived_dir(cfg) / "anomalies" / f"{day}.anomalies.parquet"
    write_parquet(df, out)
    log.info("Wrote %s (%d anomalies)", out, df.height)
    return out


def aggregate(cfg: dict | None = None) -> Path:
    """Build a cross-day summary table of anomaly counts per category."""
    cfg = cfg or load_config()
    in_dir = derived_dir(cfg) / "anomalies"
    parts = []
    for p in sorted(in_dir.glob("*.anomalies.parquet")):
        try:
            df = read_parquet(p)
            day = p.stem.split(".")[0]
            if df.is_empty():
                parts.append({"day": day, "n_total": 0})
                continue
            cats: dict[str, int] = {}
            for c in df["category"].to_list():
                for cat in c.split(";"):
                    cats[cat] = cats.get(cat, 0) + 1
            row = {"day": day, "n_total": df.height, **cats}
            parts.append(row)
        except Exception as exc:  # noqa: BLE001
            log.warning("Bad %s: %s", p, exc)
    if not parts:
        log.warning("No anomaly files found in %s", in_dir)
        return tables_dir(cfg) / "T_anomalies.csv"
    df = pl.DataFrame(parts)
    out = tables_dir(cfg) / "T_anomalies.csv"
    df.write_csv(out)
    log.info("Wrote %s (%d days)", out, df.height)
    return out


@click.command()
@click.option("--day", default=None, help="YYYYMMDD; default = all staged days")
@click.option("--aggregate/--no-aggregate", "do_aggregate", default=True)
def main(day: str | None, do_aggregate: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    days = [day] if day else list_days(cfg)
    for d in days:
        write_day(d, cfg)
    if do_aggregate:
        aggregate(cfg)


if __name__ == "__main__":
    main()
