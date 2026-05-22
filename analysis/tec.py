"""Slant TEC and ROTI from L1−L2 geometry-free combination (validation only).

Per ``(gnssId, svId)`` we form the geometry-free carrier-phase combination

    Φ_GF (m) = Φ1·λ1 − Φ2·λ2

and the corresponding code combination

    P_GF  (m) = P2 − P1

so that ``Φ_GF − bias_arc = STEC · k`` where ``k = 40.308·(1/f1²−1/f2²)``.
We level the carrier with ``mean(P_GF − Φ_GF)`` per arc to remove the
ambiguity. STEC is then converted to vTEC via the thin-shell mapping
function ``M(ε) = 1/cos(arcsin(R_E sin(zenith)/(R_E+H)))``.

ROTI = std(dTEC/dt) over a 5-min sliding window — the standard
ionospheric-irregularity index.

Output: ``derived/tec/<day>.tec.parquet`` with columns
``t_ns, gnssId, svId, elev, stec, vtec, dTECdt, roti``.

NOTE: this module is for *validation* of the dataset, not a headline
science product. Single-receiver geometry-free has unknown DCB.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import polars as pl

from analysis._common import (
    C_M_S,
    R_EARTH_KM,
    derived_dir,
    gnss_signal_freq,
    list_days,
    load_config,
    read_parquet,
    staged_path,
    write_parquet,
)
from analysis.multipath import L1L2_SIG_PAIRS

log = logging.getLogger(__name__)

# 40.308 m·Hz²·m / TECU  (from `STEC[m] = 40.308/f² · TEC[el/m²]`, with TEC in TECU=1e16 el/m²)
TEC_K = 40.308e16  # × (1/f_low² − 1/f_high²) gives metres per TECU (positive)


def _shell_mapping(elev_deg: np.ndarray, h_km: float = 350.0) -> np.ndarray:
    """Klobuchar single-layer obliquity factor, ε in degrees."""
    z = np.deg2rad(90.0 - np.maximum(elev_deg, 1.0))
    sin_z_prime = (R_EARTH_KM / (R_EARTH_KM + h_km)) * np.sin(z)
    return 1.0 / np.cos(np.arcsin(sin_z_prime))


def compute_day_tec(day: str, cfg: dict | None = None) -> pl.DataFrame:
    cfg = cfg or load_config()
    rawx = read_parquet(staged_path(day, "rxm_rawx", cfg))
    sat = read_parquet(staged_path(day, "nav_sat", cfg))
    if rawx.is_empty():
        return pl.DataFrame()

    sat_small = sat.select(["t_ns", "gnssId", "svId", "elev"]).unique(
        subset=["t_ns", "gnssId", "svId"]
    )
    rawx = rawx.join(sat_small, on=["t_ns", "gnssId", "svId"], how="left")
    rawx = rawx.filter((pl.col("trkStat") & 0x02) != 0)

    out_frames: list[pl.DataFrame] = []
    for gnss_id, (sig1, sig2) in L1L2_SIG_PAIRS.items():
        sub = rawx.filter(pl.col("gnssId") == gnss_id)
        if sub.is_empty():
            continue
        l1 = sub.filter(pl.col("sigId") == sig1).select(
            ["t_ns", "svId", "freqId", "prMes", "cpMes", "elev"]
        ).rename({"prMes": "P1", "cpMes": "Phi1"})
        l2 = sub.filter(pl.col("sigId") == sig2).select(
            ["t_ns", "svId", "freqId", "prMes", "cpMes"]
        ).rename({"prMes": "P2", "cpMes": "Phi2"})
        joined = l1.join(l2, on=["t_ns", "svId", "freqId"], how="inner")
        if joined.is_empty():
            continue

        n = joined.height
        gnss_arr = np.full(n, gnss_id, dtype=np.uint8)
        if gnss_id == 6:
            freqId_arr = joined["freqId"].to_numpy().astype(np.int32)
            n_chan = freqId_arr - 7
            f1 = (1_602_000_000.0 + n_chan * 562_500.0).astype(np.float64) if sig1 == 0 \
                 else (1_246_000_000.0 + n_chan * 437_500.0).astype(np.float64)
            f2 = (1_602_000_000.0 + n_chan * 562_500.0).astype(np.float64) if sig2 == 0 \
                 else (1_246_000_000.0 + n_chan * 437_500.0).astype(np.float64)
        else:
            f1 = np.full(n, gnss_signal_freq(gnss_id, sig1, 0), dtype=np.float64)
            f2 = np.full(n, gnss_signal_freq(gnss_id, sig2, 0), dtype=np.float64)
        lam1 = C_M_S / f1
        lam2 = C_M_S / f2

        Phi1m = joined["Phi1"].to_numpy() * lam1
        Phi2m = joined["Phi2"].to_numpy() * lam2
        P1 = joined["P1"].to_numpy()
        P2 = joined["P2"].to_numpy()
        elev = joined["elev"].to_numpy().astype(np.float64)

        Phi_GF = Phi1m - Phi2m            # (m)
        P_GF = P2 - P1                     # (m)
        # k = 40.308·(1/f2² − 1/f1²) [m / TECU, positive for f1 > f2]
        k = TEC_K * (1.0 / f2**2 - 1.0 / f1**2)

        # Build the per-row frame, then segment into continuous arcs per
        # (gnssId, svId) on time gaps and cycle slips. Per-arc levelling
        # uses the carrier-code mean (P_GF - Phi_GF). A new arc starts on
        # >2.5 s gap or |ΔΦ_GF| > 0.5 m between consecutive epochs.
        df = pl.DataFrame(
            {
                "t_ns": joined["t_ns"],
                "gnssId": gnss_arr,
                "svId": joined["svId"].cast(pl.UInt8),
                "elev": np.where(np.isfinite(elev), elev, 0).astype(np.int16),
                "Phi_GF": Phi_GF,
                "P_GF": P_GF,
                "k": k,
            }
        ).sort(["gnssId", "svId", "t_ns"])
        GAP_NS = int(2.5e9)
        SLIP_M = 0.5
        df = df.with_columns([
            (pl.col("t_ns").diff().over(["gnssId", "svId"]).fill_null(0) > GAP_NS).alias("_gap"),
            (pl.col("Phi_GF").diff().over(["gnssId", "svId"]).abs().fill_null(0) > SLIP_M).alias("_slip"),
        ])
        df = df.with_columns(
            (pl.col("_gap") | pl.col("_slip")).cum_sum().over(["gnssId", "svId"]).alias("_arc")
        )
        # Filter out near-degenerate k values (numerical safety) and short
        # arcs (< 30 epochs ≈ 15 s at 2 Hz). Short arcs don't carry useful
        # TEC information — the per-arc mean for slip-leveling is unstable
        # there, producing extreme outliers that dominate the day's σ.
        K_MIN = 5e-22
        MIN_ARC_LEN = 30
        df = df.with_columns(
            pl.len().over(["gnssId", "svId", "_arc"]).alias("_arc_len")
        ).filter((pl.col("k").abs() > K_MIN) & (pl.col("_arc_len") >= MIN_ARC_LEN))
        df = df.with_columns(
            (pl.col("P_GF") - pl.col("Phi_GF")).mean().over(["gnssId", "svId", "_arc"]).alias("bias")
        )
        df = df.with_columns(
            ((pl.col("Phi_GF") + pl.col("bias")) / pl.col("k")).alias("stec")
        ).drop(["_gap", "_slip", "_arc_len"])
        # Vertical TEC via shell mapping
        elev_arr = df["elev"].to_numpy().astype(np.float64)
        m_factor = _shell_mapping(elev_arr)
        df = df.with_columns(pl.Series("vtec", df["stec"].to_numpy() / m_factor))
        # dTEC/dt in TECU/s
        df = df.sort(["svId", "t_ns"])
        df = df.with_columns(
            ((pl.col("stec").diff().over("svId"))
             / (pl.col("t_ns").diff().over("svId").cast(pl.Float64) * 1e-9)).alias("dTECdt")
        )
        out_frames.append(df.select(["t_ns", "gnssId", "svId", "elev", "stec", "vtec", "dTECdt"]))
    if not out_frames:
        return pl.DataFrame()
    return pl.concat(out_frames, how="vertical")


def write_day_tec(day: str, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    df = compute_day_tec(day, cfg)
    out = derived_dir(cfg) / "tec" / f"{day}.tec.parquet"
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
        write_day_tec(d, cfg)


if __name__ == "__main__":
    main()
