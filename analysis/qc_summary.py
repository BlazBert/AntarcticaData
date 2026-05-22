"""Per-day QC summary statistics — produces one row of Table T4.

Stats per day:
* ``n_epochs_pvt`` — number of NAV-PVT epochs.
* ``fix_rate`` — fraction of epochs with ``fixType ≥ 3`` (3D fix or better).
* ``mean_numSV`` — mean number of SVs used in the navigation solution.
* ``mean_pDOP`` — mean position DOP.
* ``mean_cno_per_signal`` — mean C/N₀ per ``(gnssId, sigId)``.
* ``cycle_slip_count`` — count of `(trkStat & 0x02) == 0` ↗1 transitions
  per ``(gnssId, svId, sigId)``, summed over the day.
* ``unique_sv_count`` — number of distinct ``(gnssId, svId)`` ever observed.
* ``mean_temperature_C`` — receiver temperature (from MON-SYS).
* ``rf_block_jamming_pct`` — fraction of MON-RF samples with `jamInd > 32`.
* ``data_completeness`` — actual / expected sample count for each message at 2 Hz.

Output: ``derived/qc/<day>.qc.json`` and a row appended to
``tables/T4_daily_stats.parquet`` (idempotent — overwritten if rerun).
"""

from __future__ import annotations

import json
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
    read_parquet,
    staged_path,
    tables_dir,
    write_parquet,
)


def _mp_path(day: str, cfg: dict) -> Path:
    return derived_dir(cfg) / "multipath" / f"{day}.multipath.parquet"

log = logging.getLogger(__name__)


def _safe_read(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    try:
        return read_parquet(path)
    except Exception:  # noqa: BLE001
        log.warning("Could not read %s", path)
        return None


def compute_day_qc(day: str, cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    pvt = _safe_read(staged_path(day, "nav_pvt", cfg))
    sat = _safe_read(staged_path(day, "nav_sat", cfg))
    rawx = _safe_read(staged_path(day, "rxm_rawx", cfg))
    sys_ = _safe_read(staged_path(day, "mon_sys", cfg))
    rf = _safe_read(staged_path(day, "mon_rf", cfg))

    out: dict[str, Any] = {"day": day}

    # NAV-PVT
    if pvt is not None and pvt.height:
        out["n_epochs_pvt"] = pvt.height
        out["fix_rate"] = float((pvt["fixType"] >= 3).cast(pl.Float64).mean() or 0.0)
        out["mean_numSV"] = float(pvt["numSV"].cast(pl.Float64).mean() or 0.0)
        out["mean_pDOP"] = float(
            (pvt["pDOP"].cast(pl.Float64) * 0.01).mean() or 0.0
        )
        first_t_ns = int(pvt["t_ns"].min() or 0)
        last_t_ns = int(pvt["t_ns"].max() or 0)
        out["first_t_ns"] = first_t_ns
        out["last_t_ns"] = last_t_ns
        # Expected at 2 Hz over the (last - first) span
        span_s = max(1.0, (last_t_ns - first_t_ns) / 1e9)
        out["expected_pvt_2hz"] = int(span_s * 2)
        out["pvt_completeness"] = (
            float(pvt.height / out["expected_pvt_2hz"]) if out["expected_pvt_2hz"] else 0.0
        )
    else:
        out["n_epochs_pvt"] = 0
        out["fix_rate"] = 0.0

    # NAV-SAT — unique SVs and per-constellation distribution
    if sat is not None and sat.height:
        unique_sv = sat.select(["gnssId", "svId"]).unique().height
        out["unique_sv_count"] = unique_sv
        per_gnss = (
            sat.group_by("gnssId")
            .agg([pl.col("svId").n_unique().alias("n_unique"), pl.col("cno").mean().alias("mean_cno")])
            .sort("gnssId")
        )
        out["per_gnss"] = per_gnss.to_dicts()
    else:
        out["unique_sv_count"] = 0
        out["per_gnss"] = []

    # RXM-RAWX — per-signal mean C/N₀ and cycle slips
    if rawx is not None and rawx.height:
        per_sig = (
            rawx.group_by(["gnssId", "sigId"])
            .agg(
                [
                    pl.col("cno").mean().alias("mean_cno"),
                    pl.col("cno").std().alias("std_cno"),
                    pl.col("cno").count().alias("n_obs"),
                ]
            )
            .sort(["gnssId", "sigId"])
        )
        out["mean_cno_per_signal"] = per_sig.to_dicts()
        # Cycle slips: trkStat bit 0x02 (carrier-phase tracked); we count
        # transitions ↘ (lock lost) per (gnss, sv, sig) arc.
        cs = _count_cycle_slips(rawx)
        out["cycle_slip_count"] = int(cs)
        # Slip ratio = slips / carrier-phase-locked observations (Bosser
        # Table 2 convention). Reported as slips per 1000 obs for legibility.
        n_locked = int(((rawx["trkStat"] & 0x02) != 0).sum() or 0)
        out["n_carrier_obs"] = n_locked
        out["slip_ratio_per_1000"] = (
            float(1000.0 * cs / n_locked) if n_locked else float("nan")
        )
    else:
        out["mean_cno_per_signal"] = []
        out["cycle_slip_count"] = 0
        out["n_carrier_obs"] = 0
        out["slip_ratio_per_1000"] = float("nan")

    # Multipath summary (Estey-Meertens M1/M2). Computed by
    # ``analysis.multipath`` and read here for T4 aggregation.
    mp_path = _mp_path(day, cfg)
    if mp_path.exists():
        try:
            mp = read_parquet(mp_path)
        except Exception:  # noqa: BLE001
            mp = None
        if mp is not None and mp.height:
            m1 = mp["M1"].drop_nulls()
            m2 = mp["M2"].drop_nulls()
            out["mp_n_obs"] = int(mp.height)
            out["mp1_rms_m"] = float(np.sqrt(np.mean(np.square(m1.to_numpy()))))
            out["mp2_rms_m"] = float(np.sqrt(np.mean(np.square(m2.to_numpy()))))
            out["mp1_p50_m"] = float(np.median(np.abs(m1.to_numpy())))
            out["mp1_p95_m"] = float(np.percentile(np.abs(m1.to_numpy()), 95))
            out["mp2_p50_m"] = float(np.median(np.abs(m2.to_numpy())))
            out["mp2_p95_m"] = float(np.percentile(np.abs(m2.to_numpy()), 95))
        else:
            out["mp_n_obs"] = 0
            out["mp1_rms_m"] = float("nan")
            out["mp2_rms_m"] = float("nan")
    else:
        out["mp_n_obs"] = 0
        out["mp1_rms_m"] = float("nan")
        out["mp2_rms_m"] = float("nan")

    # MON-SYS — temperature
    if sys_ is not None and sys_.height:
        out["mean_temperature_C"] = float(
            sys_["tempValue_C"].cast(pl.Float64).mean() or 0.0
        )
        out["max_temperature_C"] = float(sys_["tempValue_C"].max() or 0)
        out["mean_cpu_load"] = float(sys_["cpuLoad"].cast(pl.Float64).mean() or 0.0)
    else:
        out["mean_temperature_C"] = float("nan")

    # MON-RF — jamming / AGC / antenna status
    if rf is not None and rf.height:
        # one row per (epoch, blockId) — compute per-block summaries
        per_block = (
            rf.group_by("blockId")
            .agg(
                [
                    pl.col("agcCnt").mean().alias("mean_agcCnt"),
                    pl.col("agcCnt").max().alias("max_agcCnt"),
                    pl.col("noisePerMS").mean().alias("mean_noisePerMS"),
                    pl.col("jamInd").mean().alias("mean_jamInd"),
                    (pl.col("jamInd") > 32).cast(pl.Float64).mean().alias("jam_pct"),
                    pl.col("antStatus").mode().first().alias("mode_antStatus"),
                ]
            )
            .sort("blockId")
        )
        out["per_rf_block"] = per_block.to_dicts()
    else:
        out["per_rf_block"] = []

    return out


def _count_cycle_slips(rawx: pl.DataFrame) -> int:
    """Total cycle-slip events across (gnss, sv, sig) for one day.

    Definition: a transition where ``trkStat & 0x02`` (carrier phase locked)
    goes from 1 → 0 between consecutive epochs of the same arc.
    """
    if rawx.height < 2:
        return 0
    slips = (
        rawx.sort(["gnssId", "svId", "sigId", "t_ns"])
        .with_columns((pl.col("trkStat") & 0x02 != 0).alias("locked"))
        .with_columns(
            pl.col("locked").shift(1).over(["gnssId", "svId", "sigId"]).alias("prev_locked")
        )
        .filter(pl.col("prev_locked") & (~pl.col("locked")))
    )
    return int(slips.height)


def to_t4_row(day_qc: dict[str, Any]) -> dict[str, Any]:
    """Flatten the rich per-day dict into a single Table T4 row."""
    return {
        "day": day_qc["day"],
        "n_epochs_pvt": int(day_qc.get("n_epochs_pvt", 0)),
        "fix_rate": float(day_qc.get("fix_rate", 0.0)),
        "mean_numSV": float(day_qc.get("mean_numSV", float("nan"))),
        "mean_pDOP": float(day_qc.get("mean_pDOP", float("nan"))),
        "unique_sv_count": int(day_qc.get("unique_sv_count", 0)),
        "cycle_slip_count": int(day_qc.get("cycle_slip_count", 0)),
        "n_carrier_obs": int(day_qc.get("n_carrier_obs", 0)),
        "slip_ratio_per_1000": float(day_qc.get("slip_ratio_per_1000", float("nan"))),
        "mp1_rms_m": float(day_qc.get("mp1_rms_m", float("nan"))),
        "mp2_rms_m": float(day_qc.get("mp2_rms_m", float("nan"))),
        "mp1_p95_m": float(day_qc.get("mp1_p95_m", float("nan"))),
        "mp2_p95_m": float(day_qc.get("mp2_p95_m", float("nan"))),
        "mean_temperature_C": float(day_qc.get("mean_temperature_C", float("nan"))),
        "max_temperature_C": float(day_qc.get("max_temperature_C", float("nan"))),
        "pvt_completeness": float(day_qc.get("pvt_completeness", float("nan"))),
        "first_t_ns": int(day_qc.get("first_t_ns", 0)),
        "last_t_ns": int(day_qc.get("last_t_ns", 0)),
    }


def write_qc(day: str, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    qc = compute_day_qc(day, cfg)
    out_dir = derived_dir(cfg) / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{day}.qc.json"
    out_path.write_text(json.dumps(qc, indent=2, default=str))
    log.info("Wrote %s", out_path)
    return out_path


def aggregate_t4(cfg: dict | None = None) -> Path:
    """Walk all per-day qc.json files and emit Table T4."""
    cfg = cfg or load_config()
    qc_dir = derived_dir(cfg) / "qc"
    rows: list[dict[str, Any]] = []
    for p in sorted(qc_dir.glob("*.qc.json")):
        try:
            qc = json.loads(p.read_text())
            rows.append(to_t4_row(qc))
        except Exception:  # noqa: BLE001
            log.warning("Bad qc file %s", p)
    if not rows:
        log.warning("aggregate_t4: no qc files found in %s", qc_dir)
    df = pl.DataFrame(rows)
    out = tables_dir(cfg) / "T4_daily_stats.parquet"
    write_parquet(df, out)
    csv_out = tables_dir(cfg) / "T4_daily_stats.csv"
    df.write_csv(csv_out)
    log.info("Aggregated T4 → %s (%d rows)", out, df.height)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--day", "day", default=None, help="YYYYMMDD; if omitted, run all staged days")
@click.option("--aggregate/--no-aggregate", default=True, help="Build Table T4 at the end")
def main(day: str | None, aggregate: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    days = [day] if day else list_days(cfg)
    if not days:
        raise click.ClickException("No staged days found.")
    for d in days:
        write_qc(d, cfg)
    if aggregate:
        aggregate_t4(cfg)


if __name__ == "__main__":
    main()
