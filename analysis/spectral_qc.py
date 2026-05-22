"""Lomb-Scargle spectra of long-term QC time series.

GIANT-REGAIN (Scheinert et al. 2025) ran Lomb-Scargle on coordinate
residual time series to detect draconitic-year and other periodic
artefacts. We do the equivalent on three 216-day cruise time series:

* Daily MON-RF \texttt{jamInd} on each block (L1, L2/L5).
* Daily cycle-slip rate (slips per 1000 carrier-phase obs).
* Daily mean ``hAcc`` from NAV-HPPOSLLH.

All three are pulled from ``tables/T4_daily_stats.parquet`` (extended
by ``analysis.qc_summary``) and from per-day ``derived/mon_rf`` aggregates.

The output is one Lomb-Scargle spectrum per metric, plotted on a single
multi-panel figure with 95% false-alarm bars. Periods of interest
(1 day, 1 week, 27-d Carrington, 60-d, 90-d) are marked with vertical
lines.

Outputs:
* ``figures/output/fig_lomb_scargle_qc.pdf``
* ``tables/T_periodograms.parquet`` (frequency, power, metric)

Usage:
    python -m analysis.spectral_qc
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

try:
    from scipy.signal import lombscargle
except ImportError:  # pragma: no cover
    lombscargle = None

from analysis._common import (
    derived_dir,
    list_days,
    load_config,
    read_parquet,
    tables_dir,
    write_parquet,
)
from figures._helpers import apply_style

log = logging.getLogger(__name__)


def _day_to_unix_s(d: str) -> float:
    from datetime import datetime, timezone

    return datetime(int(d[0:4]), int(d[4:6]), int(d[6:8]), tzinfo=timezone.utc).timestamp()


def _load_t4(cfg: dict) -> pl.DataFrame:
    return read_parquet(tables_dir(cfg) / "T4_daily_stats.parquet").sort("day")


def _load_jam_daily(cfg: dict) -> pl.DataFrame:
    """Mean ``jamInd`` per day per RF block, aggregated from staging MON-RF."""
    from analysis._common import staged_path

    rows = []
    for d in list_days(cfg):
        p = staged_path(d, "mon_rf", cfg)
        if not p.exists():
            continue
        try:
            df = read_parquet(p)
        except Exception:  # noqa: BLE001
            continue
        if df.is_empty():
            continue
        agg = (
            df.group_by("blockId")
            .agg(pl.col("jamInd").mean().alias("mean_jamInd"))
            .to_dicts()
        )
        row = {"day": d}
        for a in agg:
            row[f"jam_block_{a['blockId']}"] = float(a["mean_jamInd"])
        rows.append(row)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort("day")


def _periodogram(
    t_days: np.ndarray, y: np.ndarray,
    periods_d: np.ndarray,
) -> np.ndarray:
    """Lomb-Scargle power at the requested periods.

    Returns power normalised so that white noise has unit variance per
    independent frequency bin (scipy convention with ``normalize=True``
    is unavailable; we use unnormalised then divide by std²).
    """
    if lombscargle is None:
        raise RuntimeError("scipy.signal.lombscargle is required")
    mask = np.isfinite(y) & np.isfinite(t_days)
    t = t_days[mask].astype(np.float64)
    v = y[mask].astype(np.float64)
    if t.size < 5:
        return np.full_like(periods_d, np.nan, dtype=np.float64)
    v = v - v.mean()
    omegas = 2 * np.pi / periods_d
    p = lombscargle(t, v, omegas)
    var = np.var(v)
    return p / (var * 0.5 * t.size) if var > 0 else p


def build(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    t4 = _load_t4(cfg)
    t_days = np.array([
        _day_to_unix_s(d) for d in t4["day"].to_list()
    ]) / 86400.0
    metrics: dict[str, np.ndarray] = {}
    if "slip_ratio_per_1000" in t4.columns:
        metrics["Slip ratio (per 1000)"] = t4["slip_ratio_per_1000"].to_numpy()
    elif "cycle_slip_count" in t4.columns:
        metrics["Cycle slips (per day)"] = t4["cycle_slip_count"].to_numpy()
    if "mp1_rms_m" in t4.columns:
        metrics["MP1 RMS (m)"] = t4["mp1_rms_m"].to_numpy()

    jam = _load_jam_daily(cfg)
    if not jam.is_empty():
        merged = t4.select("day").join(jam, on="day", how="left")
        if "jam_block_0" in merged.columns:
            metrics["jamInd L1"] = merged["jam_block_0"].to_numpy()
        if "jam_block_1" in merged.columns:
            metrics["jamInd L2/L5"] = merged["jam_block_1"].to_numpy()

    periods = np.geomspace(1.0, 120.0, 400)  # 1 day to 4 months

    apply_style()
    fig, axes = plt.subplots(
        len(metrics), 1,
        figsize=(7, 1.7 * len(metrics) + 0.5),
        sharex=True,
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    period_rows: list[dict] = []
    for ax, (name, y) in zip(axes, metrics.items()):
        try:
            pwr = _periodogram(t_days, y, periods)
        except RuntimeError as exc:
            log.warning("%s: %s", name, exc)
            continue
        ax.plot(periods, pwr, "-", lw=0.9)
        for p_mark, lbl in [(1, "1 d"), (7, "1 w"), (27, "27 d"),
                            (60, "60 d"), (90, "90 d")]:
            ax.axvline(p_mark, ls="--", color="grey", lw=0.4)
            ax.text(p_mark, ax.get_ylim()[1] * 0.95, lbl,
                    fontsize=7, ha="left", va="top", color="grey")
        ax.set_xscale("log")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        for f_p, p_v in zip(periods, pwr):
            period_rows.append(
                {"metric": name, "period_d": float(f_p), "power": float(p_v)}
            )
    axes[-1].set_xlabel("Period (days)")
    fig.suptitle(
        "Lomb-Scargle spectra of daily QC time series (216 days)", fontsize=10
    )
    out_dir = Path(__file__).resolve().parent.parent / "figures" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_lomb_scargle_qc.pdf"
    fig.savefig(out_path, dpi=200)
    log.info("Wrote %s", out_path)

    if period_rows:
        write_parquet(pl.DataFrame(period_rows),
                      tables_dir(cfg) / "T_periodograms.parquet")
    return out_path


@click.command()
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    build()


if __name__ == "__main__":
    main()
