"""End-to-end smoke tests for the analysis modules on a single staged day.

These tests are gated by the ``RUN_BIG`` env var because they require the
2025-09-30 reference file already parsed into ``staging/20250930/``.
"""

from __future__ import annotations

import os

import pytest

from analysis._common import staged_path

REQUIRES_BIG = os.environ.get("RUN_BIG") != "1"
DAY = "20250930"


pytestmark = pytest.mark.skipif(
    REQUIRES_BIG, reason="set RUN_BIG=1 to enable end-to-end tests against staged data"
)


def test_nav_pvt_has_expected_row_count():
    p = staged_path(DAY, "nav_pvt")
    if not p.exists():
        pytest.skip(f"{p} not staged")
    import polars as pl

    df = pl.read_parquet(p)
    # 2 Hz × 86400 s ≈ 172800 epochs (allow ±1%)
    assert 170000 <= df.height <= 175000, df.height


def test_qc_runs_on_day():
    from analysis.qc_summary import compute_day_qc

    qc = compute_day_qc(DAY)
    assert qc["n_epochs_pvt"] > 0
    assert 0.95 <= qc["fix_rate"] <= 1.0
    assert qc["mean_numSV"] > 20


def test_trajectory_runs_on_day():
    from analysis.trajectory import build_day_track

    df = build_day_track(DAY)
    assert df.height > 0
    assert "lat" in df.columns and "lon" in df.columns
    # Trieste latitude ~ 45.6°N
    lat = df["lat"].mean()
    assert 30 <= lat <= 60, f"unexpected mean latitude {lat}"
