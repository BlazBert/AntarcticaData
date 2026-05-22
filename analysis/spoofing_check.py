"""Multi-signature GNSS spoofing detection.

Spoofing — broadcasting fake GNSS signals to fool a receiver into reporting
a wrong position — leaves multiple subtle signatures that a single check
won't catch. This module computes seven independent indicators per day,
combines them into a per-epoch suspicion score, and emits a per-day
summary plus the worst-offender epochs to ``derived/spoofing/<day>.parquet``.

## Indicators

1. **AGC drop without RFI** — spoofers transmit much stronger than real
   GPS, causing receiver AGC to reduce while jamInd stays nominal.
   Real RFI raises *both* AGC noise and jamInd. Score: ``agc_drop`` is
   1.0 when AGC drops > 2σ below the daily median while jamInd ≤ 32.

2. **CN0 uniformity** — real SVs span a 20-dB CN0 range driven by
   elevation/multipath. A spoofer with a single antenna transmits all
   SVs at uniform CN0. Score: ``cno_uniformity`` = 1 − std(CN0 of used
   SVs) / 12 (clamped to [0, 1]); high means SVs are suspiciously
   uniform.

3. **CN0 vs elevation flatness** — real CN0 increases ~1.5–2 dB/10°
   elevation. Spoofed signals from a single direction show no such
   trend. Score: ``cno_elev_slope`` = 1 when |slope| < 0.5 dB/10°,
   linearly down to 0 at slope ≥ 1.5 dB/10°.

4. **Per-constellation consistency** — hard to spoof all GNSS at once.
   Score: ``constellation_inconsistency`` = std of per-constellation
   mean prResidual; high when one system's residuals diverge from
   others.

5. **Position vs velocity coherence** — if reported speed is nonzero
   but consecutive positions don't move (or vice versa), the fix is
   internally inconsistent. Score: ``pos_vel_mismatch`` based on
   ``|haversine_speed − gSpeed| / max(gSpeed, 0.1)``.

6. **Time discontinuity** — spoofers occasionally introduce time
   offsets. Score: ``time_jump`` = 1 if `iTOW` jumps non-monotonically
   between consecutive PVT epochs.

7. **High-precision/Kalman divergence** — `NAV-HPPOSLLH` and
   `NAV-PVT` come from different filter chains; spoofing affects them
   differently in some cases. Score: ``hp_pvt_div`` rises when 3D
   distance > 0.5 m.

The weighted sum of indicators is the **suspicion score** ∈ [0, 7].
Score ≥ 3 in steady-state is worth visual inspection; ≥ 5 is a strong
spoofing flag.

## CLI

    python -m analysis.spoofing_check --day 20260308
    python -m analysis.spoofing_check               # all days + aggregate
    python -m analysis.spoofing_check --aggregate   # rebuild summary only

## Output layout

    derived/spoofing/<day>.spoofing.parquet
        t_ns, lat, lon, fixType, numSV, hAcc_m,
        agc_drop, cno_uniformity, cno_elev_slope,
        constellation_inconsistency, pos_vel_mismatch, time_jump,
        hp_pvt_div, suspicion

    tables/T_spoofing.csv
        day, n_epochs, n_suspect_3plus, n_suspect_5plus,
        worst_score, worst_lat, worst_lon, worst_t_ns
"""

from __future__ import annotations

import logging
from pathlib import Path

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


def _agc_drop_score(rf: pl.DataFrame) -> pl.DataFrame:
    """Per-epoch flag: AGC dropped > 2σ below daily median while jamInd ≤ 32."""
    if rf.is_empty():
        return pl.DataFrame({"t_ns": [], "agc_drop": []}).with_columns(
            pl.col("t_ns").cast(pl.Int64), pl.col("agc_drop").cast(pl.Float32)
        )
    # Use L1 block (0); spoofing primarily targets GPS L1
    rf0 = rf.filter(pl.col("blockId") == 0)
    if rf0.is_empty():
        return pl.DataFrame({"t_ns": [], "agc_drop": []}).with_columns(
            pl.col("t_ns").cast(pl.Int64), pl.col("agc_drop").cast(pl.Float32)
        )
    agc = rf0["agcCnt"].cast(pl.Float64)
    med = float(agc.median() or 0.0)
    std = float(agc.std() or 1.0)
    score = ((med - rf0["agcCnt"]) > 2 * std) & (rf0["jamInd"] <= 32)
    return rf0.select([pl.col("t_ns"), score.cast(pl.Float32).alias("agc_drop")])


def _per_epoch_cno_metrics(sat: pl.DataFrame) -> pl.DataFrame:
    """Per-epoch CN0 uniformity and CN0-vs-elev slope."""
    if sat.is_empty():
        return pl.DataFrame({
            "t_ns": [], "cno_uniformity": [], "cno_elev_slope": [],
            "constellation_inconsistency": [],
        })
    used = sat.filter(pl.col("svUsed") & (pl.col("cno") > 0) & (pl.col("elev") > 5))
    if used.is_empty():
        return pl.DataFrame({
            "t_ns": [], "cno_uniformity": [], "cno_elev_slope": [],
            "constellation_inconsistency": [],
        })
    by_t = used.group_by("t_ns").agg([
        pl.col("cno").std().alias("cno_std"),
        # Compute CN0/elev slope via covariance / variance
        pl.cov("cno", "elev").alias("_cov"),
        pl.col("elev").var().alias("_elev_var"),
        pl.col("prRes_0p1m").mean().over(["gnssId"]).std().alias("prres_const_std"),
    ]).with_columns([
        (1.0 - pl.col("cno_std") / 12.0).clip(0.0, 1.0).cast(pl.Float32).alias("cno_uniformity"),
        (pl.col("_cov") / pl.col("_elev_var")).abs().alias("_slope_abs"),
    ]).with_columns(
        # slope=0 ⇒ score 1.0; slope=1.5 dB/10° (= 0.15 dB/°) ⇒ score 0
        (1.0 - pl.col("_slope_abs") / 0.15).clip(0.0, 1.0).cast(pl.Float32).alias("cno_elev_slope")
    ).with_columns(
        # Normalise per-constellation residual std into [0,1]
        (pl.col("prres_const_std").fill_null(0.0) / 100.0).clip(0.0, 1.0).cast(pl.Float32)
        .alias("constellation_inconsistency")
    ).select([
        "t_ns", "cno_uniformity", "cno_elev_slope", "constellation_inconsistency",
    ])
    return by_t


def _pvt_metrics(pvt: pl.DataFrame) -> pl.DataFrame:
    """Position-velocity mismatch + time-jump indicator."""
    if pvt.is_empty():
        return pl.DataFrame({
            "t_ns": [], "pos_vel_mismatch": [], "time_jump": [],
        })
    pvt = pvt.sort("t_ns")
    t = pvt["t_ns"].to_numpy()
    lat = (pvt["lat_1e7"].to_numpy().astype(np.float64) * 1e-7)
    lon = (pvt["lon_1e7"].to_numpy().astype(np.float64) * 1e-7)
    g = (pvt["gSpeed_mm_s"].to_numpy().astype(np.float64) * 1e-3)
    iTOW = pvt["iTOW"].to_numpy().astype(np.int64)
    n = len(t)
    pv = np.zeros(n, dtype=np.float32)
    tj = np.zeros(n, dtype=np.float32)
    if n >= 2:
        dt_h = np.maximum(np.diff(t) / 1e9 / 3600.0, 1e-6)
        dist_km = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
        haver_speed = (dist_km * 1000.0) / (dt_h * 3600.0)        # m/s
        denom = np.maximum(g[1:], 0.1)
        mismatch = np.abs(haver_speed - g[1:]) / denom
        pv[1:] = np.clip(mismatch / 5.0, 0.0, 1.0).astype(np.float32)
        tj[1:] = ((np.diff(iTOW) <= 0).astype(np.float32))
    return pl.DataFrame({
        "t_ns": pvt["t_ns"],
        "pos_vel_mismatch": pv,
        "time_jump": tj,
    })


def _hp_pvt_div(pvt: pl.DataFrame, hpp: pl.DataFrame) -> pl.DataFrame:
    if pvt.is_empty() or hpp.is_empty():
        return pl.DataFrame({"t_ns": [], "hp_pvt_div": []})
    a = pvt.select([
        "t_ns",
        (pl.col("lat_1e7").cast(pl.Float64) * 1e-7).alias("lat_p"),
        (pl.col("lon_1e7").cast(pl.Float64) * 1e-7).alias("lon_p"),
    ])
    b = hpp.select([
        "t_ns",
        (pl.col("lat_1e7").cast(pl.Float64) * 1e-7
         + pl.col("latHp_1e9").cast(pl.Float64) * 1e-9).alias("lat_h"),
        (pl.col("lon_1e7").cast(pl.Float64) * 1e-7
         + pl.col("lonHp_1e9").cast(pl.Float64) * 1e-9).alias("lon_h"),
    ])
    j = a.join(b, on="t_ns", how="inner")
    if j.is_empty():
        return pl.DataFrame({"t_ns": [], "hp_pvt_div": []})
    d_km = haversine_km(j["lat_p"].to_numpy(), j["lon_p"].to_numpy(),
                          j["lat_h"].to_numpy(), j["lon_h"].to_numpy())
    div = (d_km * 1000.0).clip(0.0, None) / 0.5     # 0.5 m → score 1
    return pl.DataFrame({
        "t_ns": j["t_ns"],
        "hp_pvt_div": np.clip(div, 0.0, 1.0).astype(np.float32),
    })


def detect_day(day: str, cfg: dict | None = None) -> pl.DataFrame:
    cfg = cfg or load_config()
    pvt = read_parquet(staged_path(day, "nav_pvt", cfg))
    sat = read_parquet(staged_path(day, "nav_sat", cfg))
    rf = read_parquet(staged_path(day, "mon_rf", cfg))
    hpp = read_parquet(staged_path(day, "nav_hpposllh", cfg))
    if pvt.is_empty():
        return pl.DataFrame()

    base = pvt.select([
        pl.col("t_ns"),
        (pl.col("lat_1e7").cast(pl.Float64) * 1e-7).alias("lat"),
        (pl.col("lon_1e7").cast(pl.Float64) * 1e-7).alias("lon"),
        pl.col("fixType").cast(pl.UInt8),
        pl.col("numSV").cast(pl.UInt8),
        (pl.col("hAcc_mm").cast(pl.Float64) * 1e-3).alias("hAcc_m"),
    ])
    df = (
        base
        .join(_agc_drop_score(rf), on="t_ns", how="left")
        .join(_per_epoch_cno_metrics(sat), on="t_ns", how="left")
        .join(_pvt_metrics(pvt), on="t_ns", how="left")
        .join(_hp_pvt_div(pvt, hpp), on="t_ns", how="left")
    )
    score_cols = [
        "agc_drop", "cno_uniformity", "cno_elev_slope",
        "constellation_inconsistency", "pos_vel_mismatch",
        "time_jump", "hp_pvt_div",
    ]
    for c in score_cols:
        if c not in df.columns:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float32).alias(c))
        df = df.with_columns(pl.col(c).fill_null(0.0))
    df = df.with_columns(sum(pl.col(c) for c in score_cols).alias("suspicion"))
    return df


def _summarise_day(day: str, df: pl.DataFrame) -> dict:
    if df.is_empty():
        return {"day": day, "n_epochs": 0}
    n = df.height
    suspect3 = int((df["suspicion"] >= 3.0).sum())
    suspect5 = int((df["suspicion"] >= 5.0).sum())
    worst_idx = int(df["suspicion"].arg_max() or 0)
    return {
        "day": day,
        "n_epochs": n,
        "n_suspect_3plus": suspect3,
        "n_suspect_5plus": suspect5,
        "worst_score": float(df["suspicion"][worst_idx]),
        "worst_lat": float(df["lat"][worst_idx]),
        "worst_lon": float(df["lon"][worst_idx]),
        "worst_t_ns": int(df["t_ns"][worst_idx]),
    }


def write_day(day: str, cfg: dict | None = None) -> tuple[Path, dict]:
    cfg = cfg or load_config()
    df = detect_day(day, cfg)
    out = derived_dir(cfg) / "spoofing" / f"{day}.spoofing.parquet"
    write_parquet(df, out)
    log.info("Wrote %s (%d epochs)", out, df.height)
    return out, _summarise_day(day, df)


def aggregate(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    in_dir = derived_dir(cfg) / "spoofing"
    rows = []
    for p in sorted(in_dir.glob("*.spoofing.parquet")):
        day = p.stem.split(".")[0]
        try:
            df = read_parquet(p)
            rows.append(_summarise_day(day, df))
        except Exception as exc:  # noqa: BLE001
            log.warning("bad %s: %s", p, exc)
    if not rows:
        log.warning("no spoofing parquet files found")
        return tables_dir(cfg) / "T_spoofing.csv"
    out = tables_dir(cfg) / "T_spoofing.csv"
    pl.DataFrame(rows).sort("worst_score", descending=True).write_csv(out)
    log.info("Wrote %s (%d days)", out, len(rows))
    return out


@click.command()
@click.option("--day", default=None, help="YYYYMMDD; default = all staged days")
@click.option("--aggregate-only", is_flag=True, default=False,
              help="Only rebuild T_spoofing.csv from existing per-day parquets")
def main(day: str | None, aggregate_only: bool) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    if not aggregate_only:
        days = [day] if day else list_days(cfg)
        summaries = []
        for d in days:
            _, s = write_day(d, cfg)
            summaries.append(s)
    aggregate(cfg)


if __name__ == "__main__":
    main()
