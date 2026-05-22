"""Generate Tables T1..T5 in LaTeX (booktabs) and CSV form.

* T1 — receiver/antenna config (from ``config/receiver.yaml``).
* T2 — constellation/signal coverage (from ``rxm_rawx`` aggregated counts).
* T3 — daily file inventory.
* T4 — produced by ``analysis.qc_summary.aggregate_t4``.
* T5 — produced by ``analysis.trajectory.aggregate_track``.

This module mostly *renders* the existing CSV/Parquet outputs into LaTeX.
It does not recompute the underlying statistics.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import click
import polars as pl

from analysis._common import (
    list_days,
    load_config,
    read_parquet,
    resolve_path,
    staged_path,
    tables_dir,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# T1 — receiver/antenna config
# ---------------------------------------------------------------------------


def t1_receiver_config(cfg: dict | None = None) -> tuple[Path, Path]:
    cfg = cfg or load_config()
    rx = load_config("receiver")
    rows = [
        ("Manufacturer", rx["receiver"]["manufacturer"]),
        ("Model", rx["receiver"]["model"]),
        ("Firmware", rx["receiver"]["firmware"]),
        ("Marker name / number", f"{rx['receiver']['marker_name']} / {rx['receiver']['marker_number']}"),
        ("Marker type", rx["receiver"]["marker_type"]),
        ("Antenna type", rx["antenna"]["type"]),
        ("Antenna serial", rx["antenna"]["serial"]),
        ("Antenna offset (E, N, U) [m]",
         f"({rx['antenna']['delta_e_m']}, {rx['antenna']['delta_n_m']}, {rx['antenna']['delta_h_m']})"),
        ("Constellations", "GPS · GLONASS · Galileo · BeiDou · SBAS · QZSS · NavIC"),
        ("Sample rate (NAV/RAWX)", "2 Hz"),
        ("Sample rate (MON-SPAN)", "1 Hz"),
        ("RF bands", "L1 (1583.5 MHz centre, 128 MHz span); L2/L5 (1191.5 MHz, 128 MHz)"),
    ]
    df = pl.DataFrame(rows, schema=["parameter", "value"], orient="row")
    out_csv = tables_dir(cfg) / "T1_receiver.csv"
    df.write_csv(out_csv)
    out_tex = _to_latex(df, "T1: Receiver and antenna configuration", out_csv.with_suffix(".tex"))
    return out_csv, out_tex


# ---------------------------------------------------------------------------
# T2 — constellation/signal coverage
# ---------------------------------------------------------------------------

GNSS_NAME = {0: "GPS", 1: "SBAS", 2: "GAL", 3: "BDS", 5: "QZSS", 6: "GLO", 7: "NavIC"}


def t2_signal_coverage(cfg: dict | None = None) -> tuple[Path, Path]:
    cfg = cfg or load_config()
    days = list_days(cfg)
    parts = []
    for d in days:
        p = staged_path(d, "rxm_rawx", cfg)
        if not p.exists():
            continue
        parts.append(read_parquet(p).group_by(["gnssId", "sigId"]).agg([
            pl.len().alias("n_obs"),
            pl.col("cno").mean().alias("mean_cno"),
        ]))
    if not parts:
        log.warning("T2: no rxm_rawx data")
        empty = pl.DataFrame({"GNSS": [], "sigId": [], "n_obs": [], "mean_cno": []})
        return _csv_tex_empty(empty, "T2: Constellation/signal coverage", "T2_signal_coverage", cfg)
    df = pl.concat(parts).group_by(["gnssId", "sigId"]).agg([
        pl.col("n_obs").sum().alias("n_obs"),
        pl.col("mean_cno").mean().alias("mean_cno"),
    ]).with_columns(
        pl.col("gnssId").map_elements(lambda x: GNSS_NAME.get(int(x), str(x)),
                                       return_dtype=pl.Utf8).alias("GNSS")
    ).sort(["gnssId", "sigId"]).select(["GNSS", "sigId", "n_obs", "mean_cno"])
    out_csv = tables_dir(cfg) / "T2_signal_coverage.csv"
    df.write_csv(out_csv)
    out_tex = _to_latex(df, "T2: Constellation/signal coverage and mean C/N₀", out_csv.with_suffix(".tex"))
    return out_csv, out_tex


# ---------------------------------------------------------------------------
# T3 — daily file inventory
# ---------------------------------------------------------------------------


def t3_file_inventory(cfg: dict | None = None) -> tuple[Path, Path]:
    cfg = cfg or load_config()
    ubx_dir = resolve_path(cfg["paths"]["ubx_dir"])
    rows = []
    for p in sorted(ubx_dir.glob("*.ubx")):
        sz = p.stat().st_size
        # cheap sha256 of head+tail+size; not a full hash, but enough for inventory
        h = hashlib.sha256()
        with p.open("rb") as fh:
            h.update(fh.read(64 * 1024))
            fh.seek(-64 * 1024, 2)
            h.update(fh.read(64 * 1024))
        h.update(str(sz).encode())
        rows.append({
            "filename": p.name,
            "bytes": sz,
            "sha256_head_tail": h.hexdigest(),
            "mtime": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
        })
    df = pl.DataFrame(rows) if rows else pl.DataFrame({"filename": [], "bytes": [], "sha256_head_tail": [], "mtime": []})
    out_csv = tables_dir(cfg) / "T3_file_inventory.csv"
    df.write_csv(out_csv)
    out_tex = _to_latex(df, "T3: Daily UBX file inventory", out_csv.with_suffix(".tex"))
    return out_csv, out_tex


# ---------------------------------------------------------------------------
# Render to LaTeX
# ---------------------------------------------------------------------------


def _to_latex(df: pl.DataFrame, caption: str, out_path: Path) -> Path:
    """Minimal booktabs LaTeX writer (no extra deps)."""
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\caption{" + caption + "}",
        "\\begin{tabular}{" + "l" * df.width + "}",
        "\\toprule",
        " & ".join(df.columns) + " \\\\",
        "\\midrule",
    ]
    for row in df.iter_rows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:.3f}" if abs(v) < 1e6 else f"{v:.3e}")
            else:
                s = str(v).replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")
                cells.append(s)
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    out_path.write_text("\n".join(lines))
    return out_path


def _csv_tex_empty(df: pl.DataFrame, caption: str, stem: str, cfg: dict) -> tuple[Path, Path]:
    out_csv = tables_dir(cfg) / f"{stem}.csv"
    df.write_csv(out_csv)
    out_tex = _to_latex(df, caption, out_csv.with_suffix(".tex"))
    return out_csv, out_tex


@click.command()
@click.option("--only", default=None, help="Comma-separated table list, e.g. 'T1,T3'")
def main(only: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    todo = {"T1", "T2", "T3"} if not only else set(s.strip() for s in only.split(","))
    if "T1" in todo:
        t1_receiver_config(cfg)
    if "T2" in todo:
        t2_signal_coverage(cfg)
    if "T3" in todo:
        t3_file_inventory(cfg)


if __name__ == "__main__":
    main()
