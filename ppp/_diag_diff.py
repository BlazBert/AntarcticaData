"""Diagnose PPP-vs-onboard residuals from one or more days.

Reads ``<derived>/ppp/<day>/diff.parquet`` files and reports:
  - raw d3d quantiles (every matched row)
  - filtered d3d quantiles (well-conditioned rows only: nsat>4, 0<pdop<4)
  - vertical / horizontal split (du and sqrt(de^2+dn^2))
  - per-day summary if multiple days are listed

Usage:
    python3 -m ppp._diag_diff 20250930
    python3 -m ppp._diag_diff 20250930 20260115 20260315
    python3 -m ppp._diag_diff --all       # every day with a diff.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from analysis._common import derived_dir, load_config


def _quantiles(v: np.ndarray) -> dict:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, [50, 95, 99]).round(3).tolist()
    return {"n": int(v.size), "p50": q[0], "p95": q[1], "p99": q[2],
            "max": float(v.max().round(3))}


def _summarise(diff: pl.DataFrame, label: str) -> None:
    n_raw = diff.height
    clean = diff.filter(
        (pl.col("nsat") > 4)
        & (pl.col("pdop") > 0.0)
        & (pl.col("pdop") < 4.0)
    )
    print(f"\n=== {label} ===")
    print(f"  rows raw:   {n_raw:>7}")
    print(f"  rows clean: {clean.height:>7}  (nsat>4, 0<pdop<4)")

    for name, df in (("raw", diff), ("clean", clean)):
        if df.is_empty():
            continue
        d3d = df["d3d_m"].to_numpy()
        dh = np.sqrt(df["de_m"].to_numpy() ** 2
                     + df["dn_m"].to_numpy() ** 2)
        du = np.abs(df["du_m"].to_numpy())
        print(f"  {name} d3d:    {_quantiles(d3d)}")
        print(f"  {name} horiz:  {_quantiles(dh)}")
        print(f"  {name} vert:   {_quantiles(du)}")


def main(argv: list[str]) -> int:
    cfg = load_config()
    base = derived_dir(cfg) / "ppp"
    if len(argv) < 2:
        print(__doc__)
        return 2
    if argv[1] == "--all":
        days = sorted(
            p.name for p in base.iterdir()
            if p.is_dir() and (p / "diff.parquet").exists()
        )
    else:
        days = argv[1:]

    if not days:
        print("no days with diff.parquet found")
        return 1

    all_clean_d3d: list[np.ndarray] = []
    for day in days:
        p = base / day / "diff.parquet"
        if not p.exists():
            print(f"missing: {p}")
            continue
        diff = pl.read_parquet(p)
        _summarise(diff, day)
        clean = diff.filter(
            (pl.col("nsat") > 4) & (pl.col("pdop") > 0.0) & (pl.col("pdop") < 4.0)
        )
        if not clean.is_empty():
            all_clean_d3d.append(clean["d3d_m"].to_numpy())

    if len(days) > 1 and all_clean_d3d:
        pooled = np.concatenate(all_clean_d3d)
        print(f"\n=== POOLED ({len(days)} days, clean) ===")
        print(f"  rows: {pooled.size:,}")
        print(f"  d3d quantiles: {_quantiles(pooled)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
