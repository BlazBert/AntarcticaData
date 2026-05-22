"""Emit ready-to-paste LaTeX snippets for the remaining `[TBD]` slots.

Run once on the server after the analysis pipeline + the corrected
``analysis.anomalies`` (L1/L2L5 split) + ``verify.track_outliers`` are
all up to date. Prints, in order:

  1. Two Spearman correlations for §5.5 (L2/L5 baseline vs |lat| and vs
     receiver temperature).
  2. LaTeX rows for the strong-RFI events table (§5.5).
  3. LaTeX rows for the outliers-by-region table (§5.7).

Just paste each section into the corresponding ``[TBD]`` slot in
``paper/manuscript/main.tex``.

CLI::

    cd /home/jovyan/Projects/gps-data/data/code
    python -m verify.finish_paper
    python -m verify.finish_paper > /tmp/paper_snippets.tex     # if you'd
                                                                 # rather pipe
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

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
)

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# §5.5 — Spearman ρ for L2/L5 baseline vs |lat| and vs receiver temperature
# -----------------------------------------------------------------------------


def _per_day_l5_lat_temp(cfg: dict) -> pl.DataFrame:
    """For each staged day, the day-median (jamInd_L2L5, |lat|, temperature)."""
    days = list_days(cfg)
    rows: list[dict] = []
    for d in days:
        rf_p = staged_path(d, "mon_rf", cfg)
        sys_p = staged_path(d, "mon_sys", cfg)
        track_p = derived_dir(cfg) / "track" / f"{d}.track.parquet"
        if not rf_p.exists() or not track_p.exists():
            continue
        rf = read_parquet(rf_p)
        if rf.is_empty():
            continue
        l5 = rf.filter(pl.col("blockId") == 1)
        if l5.is_empty():
            continue
        tr = read_parquet(track_p)
        if tr.is_empty():
            continue
        temp_med = float("nan")
        if sys_p.exists():
            sys_df = read_parquet(sys_p)
            if not sys_df.is_empty() and "tempValue_C" in sys_df.columns:
                temp_med = float(sys_df["tempValue_C"].median() or float("nan"))
        rows.append({
            "day": d,
            "jam_l5_median": float(l5["jamInd"].median() or float("nan")),
            "abs_lat_median": float(abs(tr["lat"].median() or 0.0)),
            "temp_median_C": temp_med,
        })
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Spearman ρ + p-value using scipy; defensive against NaN."""
    from scipy.stats import spearmanr
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan"), float("nan"), int(mask.sum())
    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p), int(mask.sum())


def emit_l5_correlations(cfg: dict) -> None:
    df = _per_day_l5_lat_temp(cfg)
    if df.is_empty():
        print("% (no MON-RF / track data available — skipping §5.5 correlations)")
        return
    jam = df["jam_l5_median"].to_numpy()
    lat = df["abs_lat_median"].to_numpy()
    temp = df["temp_median_C"].to_numpy()
    rho_lat, p_lat, n_lat = _spearman(lat, jam)
    rho_temp, p_temp, n_temp = _spearman(temp, jam)
    print()
    print("% ----- §5.5: paste these two values into the [TBD: ρ] slots -----")
    print(f"% L2/L5 baseline (day-median jamInd) vs |latitude|: "
           f"Spearman rho = {rho_lat:+.3f}, p = {p_lat:.2e}, n = {n_lat}")
    print(f"% L2/L5 baseline (day-median jamInd) vs receiver temperature: "
           f"Spearman rho = {rho_temp:+.3f}, p = {p_temp:.2e}, n = {n_temp}")
    print()
    sig_lat = "is" if p_lat < 0.05 else "is not"
    sig_temp = "is" if p_temp < 0.05 else "is not"
    print("% Suggested replacement text in §5.5 (verb selected by p-value):")
    print(f"% \"Across the 216 days the L2/L5 baseline {sig_lat} statistically")
    print(f"%  correlated with absolute latitude (Spearman")
    print(f"%  $\\rho = {rho_lat:+.2f}$, $p = {p_lat:.2g}$) and {sig_temp}")
    print(f"%  with receiver temperature (Spearman $\\rho = {rho_temp:+.2f}$,")
    print(f"%  $p = {p_temp:.2g}$).\"")
    print(f"% NOTE: review whether the lat / temp correlations are independent")
    print(f"% or confounded (polar days are colder); your paper should mention this.")


# -----------------------------------------------------------------------------
# §5.5 — Strong-RFI events table (T_rfi_events.csv → LaTeX rows)
# -----------------------------------------------------------------------------


def emit_rfi_events_rows(cfg: dict) -> None:
    p = tables_dir(cfg) / "T_rfi_events.csv"
    if not p.exists():
        print("% (T_rfi_events.csv missing — run verify.locate_rfi_events first)")
        return
    df = pl.read_csv(p).sort("day")
    print()
    print("% ----- §5.5 / tab:rfi-events — paste these rows into the table -----")
    for r in df.iter_rows(named=True):
        day = str(r["day"])
        day_iso = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
        lat = float(r["lat_med"])
        lon = float(r["lon_med"])
        speed = float(r["speed_avg_m_s"])
        jam = int(r["L1_jam_max"])
        gt60 = int(r["L1_jam_gt60_n"])
        print(f"{day_iso} & {lat:+.3f} & {lon:+.3f} & {speed:.2f} & {jam} & {gt60} \\\\")


# -----------------------------------------------------------------------------
# §5.7 — Outliers-by-region table (T_track_outliers_by_region.csv → LaTeX rows)
# -----------------------------------------------------------------------------


def emit_outliers_by_region_rows(cfg: dict) -> None:
    p = tables_dir(cfg) / "T_track_outliers_by_region.csv"
    if not p.exists():
        print("% (T_track_outliers_by_region.csv missing — run verify.track_outliers first)")
        return
    df = pl.read_csv(p)
    # Build LaTeX rows with the same column order as the template in main.tex:
    #   Region | cold_start | rfi_correl. | scint_cand. | low_geom. | no_fix | total
    # The CSV columns are whichever classes actually appear; map them in.
    cols_in_csv = [c for c in df.columns if c not in ("region", "total")]
    print()
    print("% ----- §5.7 / tab:outliers-region — replace the table header AND rows -----")
    print(f"% Classes present in your run: {cols_in_csv}")
    print("% (Update the table column headers in main.tex to match these.)")
    print()
    for r in df.iter_rows(named=True):
        parts = [str(r["region"])]
        for c in cols_in_csv:
            parts.append(str(r[c]))
        total = r.get("total", sum(r[c] for c in cols_in_csv))
        parts.append(str(total))
        print(" & ".join(parts) + " \\\\")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


@click.command()
def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    cfg = load_config()
    print("% =======================================================")
    print("% Auto-generated paper-table snippets for main.tex")
    print("% Generated by: python -m verify.finish_paper")
    print("% =======================================================")
    emit_l5_correlations(cfg)
    emit_rfi_events_rows(cfg)
    emit_outliers_by_region_rows(cfg)
    print()
    print("% End of snippets")


if __name__ == "__main__":
    main()
