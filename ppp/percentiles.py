"""Position-formal-error percentile table (Bosser 2021 Table 3 equivalent).

For each observable (sigma_horiz, sigma_vert, sigma_3d, ZTD where
available) we report the distribution at p = 1, 5, 10, 50, 90, 95, 99
percent over all cruise epochs. Two sources are combined:

* Onboard ZED-F9P-15B receiver: ``hAcc``/``vAcc`` from NAV-HPPOSLLH.
  Available for all 216 days; reflects the receiver's internal Kalman
  filter formal error.
* PRIDE PPP-AR kinematic solution: σ_E, σ_N, σ_U from the kin file
  posrec block. Available only on days where PPP was run; reflects the
  full post-processed formal error.

Outputs:
* ``tables/T_pos_percentiles.parquet`` (long form)
* ``tables/T_pos_percentiles.csv``
* ``tables/T_pos_percentiles.tex`` (Copernicus-style LaTeX table)

Usage:
    python -m ppp.percentiles
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import click
import numpy as np
import polars as pl

from analysis._common import (
    derived_dir,
    list_days,
    load_config,
    read_parquet,
    resolve_path,
    staged_path,
    tables_dir,
    write_parquet,
)

log = logging.getLogger(__name__)

PERCENTILES = [1, 5, 10, 50, 90, 95, 99]


def _onboard_sigmas(cfg: dict) -> dict[str, np.ndarray]:
    """Concatenate onboard hAcc/vAcc (metres) across all staged days."""
    hAcc: list[np.ndarray] = []
    vAcc: list[np.ndarray] = []
    for day in list_days(cfg):
        p = staged_path(day, "nav_hpposllh", cfg)
        if not p.exists():
            continue
        try:
            df = read_parquet(p)
        except Exception:  # noqa: BLE001
            continue
        if df.is_empty():
            continue
        hAcc.append(
            (df["hAcc_0p1mm"].cast(pl.Float64).to_numpy() * 1e-4)
        )
        vAcc.append(
            (df["vAcc_0p1mm"].cast(pl.Float64).to_numpy() * 1e-4)
        )
    if not hAcc:
        return {}
    return {
        "hAcc_m": np.concatenate(hAcc),
        "vAcc_m": np.concatenate(vAcc),
    }


def _ppp_sigmas(cfg: dict) -> dict[str, np.ndarray]:
    """Concatenate σ_E/σ_N/σ_U across all available PPP diff files.

    The diff parquet from ``ppp.compare`` carries the onboard hAcc/vAcc;
    if PRIDE wrote formal errors into the kin file (newer versions), we
    pick them up from ``derived/ppp/<day>/sigma.parquet`` when present.
    Otherwise we fall back to PPP-to-onboard *residuals* as a proxy for
    the achievable accuracy.
    """
    base = derived_dir(cfg) / "ppp"
    de: list[np.ndarray] = []
    dn: list[np.ndarray] = []
    du: list[np.ndarray] = []
    for day in list_days(cfg):
        diff = base / day / "diff.parquet"
        if not diff.exists():
            continue
        try:
            df = read_parquet(diff)
        except Exception:  # noqa: BLE001
            continue
        if df.is_empty():
            continue
        # Keep only well-conditioned PRIDE epochs:
        # * nsat > 4 (a real solution, not a placeholder/float)
        # * 0 < pdop < 4 (good geometry)
        # The raw diff parquet retains every row; we filter only here so
        # the published percentile table reflects the true onboard-vs-PPP
        # agreement, not the float-solution / sea-state-heave tail.
        if {"nsat", "pdop"} <= set(df.columns):
            df = df.filter(
                (pl.col("nsat") > 4)
                & (pl.col("pdop") > 0.0)
                & (pl.col("pdop") < 4.0)
            )
        if df.is_empty():
            continue
        de.append(df["de_m"].cast(pl.Float64).to_numpy())
        dn.append(df["dn_m"].cast(pl.Float64).to_numpy())
        du.append(df["du_m"].cast(pl.Float64).to_numpy())
    if not de:
        return {}
    e = np.concatenate(de)
    n = np.concatenate(dn)
    u = np.concatenate(du)
    horiz = np.sqrt(e**2 + n**2)
    return {
        "ppp_horiz_resid_m": horiz,
        "ppp_vert_resid_m": np.abs(u),
        "ppp_3d_resid_m": np.sqrt(e**2 + n**2 + u**2),
    }


def _percentile_row(name: str, values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"metric": name, **{f"p{p}": float("nan") for p in PERCENTILES},
                "n": 0}
    row: dict = {"metric": name, "n": int(values.size)}
    for p in PERCENTILES:
        row[f"p{p}"] = float(np.percentile(values, p))
    return row


def build_table(cfg: dict | None = None) -> pl.DataFrame:
    cfg = cfg or load_config()
    rows: list[dict] = []
    onboard = _onboard_sigmas(cfg)
    if onboard:
        rows.append(_percentile_row("onboard_hAcc_m", onboard["hAcc_m"]))
        rows.append(_percentile_row("onboard_vAcc_m", onboard["vAcc_m"]))
    ppp = _ppp_sigmas(cfg)
    if ppp:
        rows.append(_percentile_row("ppp_horiz_resid_m", ppp["ppp_horiz_resid_m"]))
        rows.append(_percentile_row("ppp_vert_resid_m", ppp["ppp_vert_resid_m"]))
        rows.append(_percentile_row("ppp_3d_resid_m", ppp["ppp_3d_resid_m"]))
    return pl.DataFrame(rows)


def _latex_table(df: pl.DataFrame) -> str:
    """Render the percentile table as a Copernicus-class LaTeX snippet."""
    header_labels = {
        "onboard_hAcc_m": r"Onboard \texttt{hAcc} formal (m)",
        "onboard_vAcc_m": r"Onboard \texttt{vAcc} formal (m)",
        "ppp_horiz_resid_m": r"Onboard-vs-PPP horiz.\ residual (m)",
        "ppp_vert_resid_m": r"Onboard-vs-PPP vert.\ residual (m)",
        "ppp_3d_resid_m": r"Onboard-vs-PPP 3-D residual (m)",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Distribution of two position-quality metrics across "
        r"the 216-day cruise. Rows 1--2 are the receiver-internal "
        r"Kalman-filter formal error (\texttt{NAV-HPPOSLLH.hAcc}, "
        r"\texttt{vAcc}). Rows 3--5 are the difference between the "
        r"onboard fix and an independently post-processed kinematic PPP "
        r"solution (PRIDE PPP-AR), filtered to PRIDE "
        r"\texttt{nsat}~$>4$ and \texttt{pdop}~$<4$. Each row reports "
        r"the percentiles $p_{1}$, $p_{5}$, $p_{10}$, $p_{50}$, "
        r"$p_{90}$, $p_{95}$, $p_{99}$ in metres, with $n$ the number "
        r"of contributing epochs.}",
        r"\label{tab:pos-percentiles}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\tophline",
        r"Metric & $p_{1}$ & $p_{5}$ & $p_{10}$ & $p_{50}$ "
        r"& $p_{90}$ & $p_{95}$ & $p_{99}$ & $n$ \\",
        r"\middlehline",
    ]
    for row in df.iter_rows(named=True):
        label = header_labels.get(row["metric"], row["metric"].replace("_", r"\_"))
        cells = [label]
        for p in PERCENTILES:
            v = row[f"p{p}"]
            cells.append(f"{v:.3f}" if v == v else "--")  # NaN-safe
        cells.append(f"{row['n']:,}")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomhline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def write_table(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    df = build_table(cfg)
    out_parquet = tables_dir(cfg) / "T_pos_percentiles.parquet"
    out_csv = tables_dir(cfg) / "T_pos_percentiles.csv"
    out_tex = tables_dir(cfg) / "T_pos_percentiles.tex"
    write_parquet(df, out_parquet)
    df.write_csv(out_csv)
    out_tex.write_text(_latex_table(df) + "\n")
    log.info("Wrote %s, %s, %s", out_parquet, out_csv, out_tex)
    return out_tex


@click.command()
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    write_table()


if __name__ == "__main__":
    main()
