"""Five validation checks that decide which claims survive in the Introduction.

These are *not* unit tests — they're aggregate-data quality checks that
produce numbers we splice into the manuscript text where the v3 draft
currently has ``{{POST-VALIDATION: ...}}`` tags.

Run from ``code/``::

    python -m verify.validation_checks --days 20250930,20260101,20260301

Each check writes a JSON line to ``../paper/drafts/validation_results.jsonl``.
The introduction-update step (separate, manual) reads that file and
produces a populated v4.

Tests:
1. **TEC vs IGS GIM** — bilinearly interpolate IONEX to ship lat/lon for
   the day; compute RMS of (vTEC_ours − vTEC_GIM) after per-arc bias
   levelling. Threshold: RMS < 5 TECU = "good", 5–10 = "screening-quality",
   >10 = "DCB calibration required".
2. **Kinematic PPP residuals** — *requires* a real PPP run; for now we
   report median 3D position error of NAV-HPPOSLLH vs onboard kalman
   solution as a *lower-bound proxy* (so the test runs without PPP
   infrastructure). When PPP is wired up, replace with PRIDE residuals.
3. **Multipath M1/M2 RMS by elevation** — already computed by
   ``analysis.multipath``; we just summarise the per-elevation-bin RMS
   for GPS L1+L5, GAL E1+E5a, BDS B1I+B2a, and compare to published
   F9P numbers (~0.5–2 m typical).
4. **MON-SPAN RFI events** — count detected events per day, rate of
   occurrence, and (if cruise track available) split by at-port vs
   open-ocean. Threshold: events visible above background = "useful for
   surveys".
5. **Latitudinal diversity** — confirm that the chosen days cover at
   least mid-lat, transit, and polar regimes.

Each check is robust: missing inputs produce a ``status=skipped`` row,
not an exception. The script aims to print a clean summary at the end
that you can paste into the paper.
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
    haversine_km,
    load_config,
    read_parquet,
    resolve_path,
    staged_path,
)

log = logging.getLogger(__name__)

OUT_PATH = Path(__file__).resolve().parent.parent.parent / "paper" / "drafts" / "validation_results.jsonl"


# ---------------------------------------------------------------------------
# Test 1 — TEC vs IGS GIM
# ---------------------------------------------------------------------------


def test_tec_vs_gim(day: str, cfg: dict | None = None) -> dict[str, Any]:
    """Compare our vTEC to IGS GIM at the ship lat/lon.

    Without an IONEX downloader wired up the test reports ``status=skipped``
    and emits the our-side TEC summary so we at least know the dataset
    side. Once ``ppp.igs_products.fetch_day`` is enabled (Stage 4), this
    test replaces the placeholder with the real comparison.
    """
    cfg = cfg or load_config()
    tec_path = derived_dir(cfg) / "tec" / f"{day}.tec.parquet"
    if not tec_path.exists():
        return {"test": "tec_vs_gim", "day": day, "status": "skipped",
                "reason": "tec.parquet not found — run analysis.tec first"}
    tec = read_parquet(tec_path)
    if tec.is_empty():
        return {"test": "tec_vs_gim", "day": day, "status": "skipped",
                "reason": "tec.parquet empty"}

    # Per-day vTEC summary (relative; absolute requires GIM)
    summary = (
        tec.filter(pl.col("vtec").is_finite() & (pl.col("elev") > 20))
        .select([
            pl.col("vtec").mean().alias("vtec_mean"),
            pl.col("vtec").std().alias("vtec_std"),
            pl.col("vtec").median().alias("vtec_median"),
            pl.col("vtec").quantile(0.05).alias("vtec_p5"),
            pl.col("vtec").quantile(0.95).alias("vtec_p95"),
            pl.len().alias("n"),
        ])
        .to_dicts()[0]
    )

    # IONEX comparison would go here; for now mark as TODO.
    return {
        "test": "tec_vs_gim",
        "day": day,
        "status": "self-summary (GIM comparison TODO)",
        **summary,
        "interpretation": _classify_tec_quality(summary),
    }


def _classify_tec_quality(s: dict[str, Any]) -> str:
    n = int(s.get("n") or 0)
    if n < 1000:
        return "insufficient samples"
    std = float(s.get("vtec_std") or 0.0)
    if 0.5 < std < 30:
        return "in physical range; suitable for relative TEC studies"
    return "anomalous std — investigate before claiming TEC product"


# ---------------------------------------------------------------------------
# Test 2 — kinematic positioning residuals (proxy until PPP is wired up)
# ---------------------------------------------------------------------------


def test_position_consistency(day: str, cfg: dict | None = None) -> dict[str, Any]:
    """Onboard NAV-PVT vs NAV-HPPOSLLH consistency.

    Both come from the receiver. PVT is a real-time kalman filter; HPPOSLLH
    is the high-precision form of the same fix. They should agree to
    sub-decimetre level. Wide divergence flags receiver / firmware issues.
    A *real* PPP residual (against PRIDE PPP-AR) replaces this once Stage 4
    is in place.
    """
    cfg = cfg or load_config()
    pvt = read_parquet(staged_path(day, "nav_pvt", cfg))
    hpp = read_parquet(staged_path(day, "nav_hpposllh", cfg))
    if pvt.is_empty() or hpp.is_empty():
        return {"test": "position_consistency", "day": day, "status": "skipped"}

    j = pvt.select([
        "t_ns",
        (pl.col("lat_1e7").cast(pl.Float64) * 1e-7).alias("lat_pvt"),
        (pl.col("lon_1e7").cast(pl.Float64) * 1e-7).alias("lon_pvt"),
        (pl.col("height_mm").cast(pl.Float64) * 1e-3).alias("h_pvt"),
    ]).join(
        hpp.select([
            "t_ns",
            (pl.col("lat_1e7").cast(pl.Float64) * 1e-7
             + pl.col("latHp_1e9").cast(pl.Float64) * 1e-9).alias("lat_hpp"),
            (pl.col("lon_1e7").cast(pl.Float64) * 1e-7
             + pl.col("lonHp_1e9").cast(pl.Float64) * 1e-9).alias("lon_hpp"),
            (pl.col("height_mm").cast(pl.Float64) * 1e-3
             + pl.col("heightHp_0p1mm").cast(pl.Float64) * 1e-4).alias("h_hpp"),
        ]),
        on="t_ns", how="inner",
    )
    if j.is_empty():
        return {"test": "position_consistency", "day": day, "status": "no joinable epochs"}
    lat = j["lat_pvt"].to_numpy()
    lon = j["lon_pvt"].to_numpy()
    lat2 = j["lat_hpp"].to_numpy()
    lon2 = j["lon_hpp"].to_numpy()
    horiz_m = haversine_km(lat, lon, lat2, lon2) * 1000.0
    vert_m = (j["h_pvt"].to_numpy() - j["h_hpp"].to_numpy())
    return {
        "test": "position_consistency",
        "day": day,
        "status": "ok",
        "n_epochs": int(j.height),
        "horiz_rms_m": float(np.sqrt(np.mean(horiz_m ** 2))),
        "horiz_p95_m": float(np.quantile(np.abs(horiz_m), 0.95)),
        "vert_rms_m": float(np.sqrt(np.mean(vert_m ** 2))),
        "vert_p95_m": float(np.quantile(np.abs(vert_m), 0.95)),
        "interpretation": "PVT/HPPOSLLH should agree to <0.1 m on a 3D-fix day",
    }


# ---------------------------------------------------------------------------
# Test 3 — multipath M1/M2 RMS by elevation
# ---------------------------------------------------------------------------


def test_multipath_summary(day: str, cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    mp_path = derived_dir(cfg) / "multipath" / f"{day}.multipath.parquet"
    if not mp_path.exists():
        return {"test": "multipath", "day": day, "status": "skipped",
                "reason": "multipath.parquet not found"}
    df = read_parquet(mp_path)
    if df.is_empty():
        return {"test": "multipath", "day": day, "status": "empty"}

    # Per (gnss, elevation 5° bin) RMS
    df = df.filter(pl.col("M1").is_finite() & pl.col("M2").is_finite() & (pl.col("elev") > 0))
    df = df.with_columns(((pl.col("elev").cast(pl.Float64) // 10) * 10).alias("elev_bin"))
    by_bin = (
        df.group_by(["gnssId", "elev_bin"])
        .agg([
            pl.col("M1").std().alias("M1_rms"),
            pl.col("M2").std().alias("M2_rms"),
            pl.len().alias("n"),
        ])
        .sort(["gnssId", "elev_bin"])
    )
    high_elev = by_bin.filter(pl.col("elev_bin") >= 30)
    summary = {}
    if not high_elev.is_empty():
        summary["high_elev_M1_rms_median_m"] = float(high_elev["M1_rms"].median() or float("nan"))
        summary["high_elev_M2_rms_median_m"] = float(high_elev["M2_rms"].median() or float("nan"))
    return {
        "test": "multipath",
        "day": day,
        "status": "ok",
        **summary,
        "n_rows": int(df.height),
        "by_bin": by_bin.to_dicts()[:30],   # cap to keep the JSON manageable
        "interpretation": (
            "high-elev RMS < 1 m typical for geodetic; <2 m acceptable for F9P; "
            ">3 m means cycle slips dominate and arc segmentation needs work"
        ),
    }


# ---------------------------------------------------------------------------
# Test 4 — MON-SPAN RFI event rate
# ---------------------------------------------------------------------------


def test_rfi_summary(day: str, cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    rfi_path = derived_dir(cfg) / "spectrum" / f"{day}.rfi.parquet"
    if not rfi_path.exists():
        return {"test": "rfi", "day": day, "status": "skipped",
                "reason": "rfi.parquet not found"}
    df = read_parquet(rfi_path)
    if df.is_empty():
        return {"test": "rfi", "day": day, "status": "no events"}
    return {
        "test": "rfi",
        "day": day,
        "status": "ok",
        "n_events": int(df.height),
        "events_per_minute": float(df.height / 1440.0),
        "by_block": df.group_by("rf_block").agg(pl.len().alias("n")).to_dicts(),
        "freq_hist_summary": {
            "min_mhz": float(df["freq_mhz"].min() or 0),
            "max_mhz": float(df["freq_mhz"].max() or 0),
            "median_mhz": float(df["freq_mhz"].median() or 0),
        },
    }


# ---------------------------------------------------------------------------
# Test 5 — latitudinal coverage
# ---------------------------------------------------------------------------


def test_latitude_coverage(day: str, cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    track_path = derived_dir(cfg) / "track" / f"{day}.track.parquet"
    if not track_path.exists():
        return {"test": "latitude_coverage", "day": day, "status": "skipped"}
    df = read_parquet(track_path)
    if df.is_empty():
        return {"test": "latitude_coverage", "day": day, "status": "empty"}
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    return {
        "test": "latitude_coverage",
        "day": day,
        "status": "ok",
        "lat_min": float(lat.min()),
        "lat_max": float(lat.max()),
        "lat_median": float(np.median(lat)),
        "lon_median": float(np.median(lon)),
        "regime": _classify_regime(float(np.median(lat))),
    }


def _classify_regime(lat: float) -> str:
    if lat > 30:
        return "mid-lat NH"
    if lat > 0:
        return "low-lat NH (tropical)"
    if lat > -30:
        return "low-lat SH (tropical)"
    if lat > -60:
        return "mid-lat SH"
    if lat > -66.56:
        return "subpolar SH"
    return "polar SH (Antarctic Circle)"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_tec_vs_gim,
    test_position_consistency,
    test_multipath_summary,
    test_rfi_summary,
    test_latitude_coverage,
]


def run_for_day(day: str, cfg: dict) -> list[dict]:
    rows = []
    for fn in ALL_TESTS:
        try:
            rows.append(fn(day, cfg))
        except Exception as exc:  # noqa: BLE001
            log.exception("Test %s failed for %s", fn.__name__, day)
            rows.append({"test": fn.__name__, "day": day, "status": "exception", "error": str(exc)})
    return rows


@click.command()
@click.option("--days", required=True,
              help="Comma-separated YYYYMMDD list, e.g. 20250930,20260101,20260301")
@click.option("--out", default=str(OUT_PATH), show_default=True,
              type=click.Path(), help="Output JSONL path")
def main(days: str, out: str) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    day_list = [d.strip() for d in days.split(",") if d.strip()]
    all_rows: list[dict] = []
    for d in day_list:
        log.info("Running validation tests for %s", d)
        rows = run_for_day(d, cfg)
        for r in rows:
            print(json.dumps(r, default=str))
        all_rows.extend(rows)
    with out_path.open("w") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, default=str) + "\n")
    log.info("Wrote %s (%d rows)", out_path, len(all_rows))

    # Quick stdout summary
    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    by_test: dict[str, list[dict]] = {}
    for r in all_rows:
        by_test.setdefault(r["test"], []).append(r)
    for tname, rows in by_test.items():
        print(f"\n[{tname}]")
        for r in rows:
            line = f"  {r.get('day', '?')}: status={r.get('status')}"
            for k in ("n_epochs", "horiz_rms_m", "vert_rms_m",
                      "high_elev_M1_rms_median_m", "high_elev_M2_rms_median_m",
                      "n_events", "events_per_minute",
                      "lat_min", "lat_max", "regime",
                      "vtec_mean", "vtec_std"):
                if k in r:
                    v = r[k]
                    if isinstance(v, float):
                        line += f"  {k}={v:.3f}"
                    else:
                        line += f"  {k}={v}"
            print(line)


if __name__ == "__main__":
    main()
