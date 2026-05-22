"""Diagnose why one validation day has unexpectedly low GIM agreement.

Pulls geomagnetic context (Kp), daily QC counters, and the day's TEC
arc-level stats for the target day and its +-2 day window. Used to
decide whether a low Spearman rho against IGS GIM reflects real
storm-time ionospheric structure (paper-positive framing) or a local
data-quality artifact (paper-negative framing).

Usage:
    python -m verify.diagnose_tec_outlier              # default: 20260215
    python -m verify.diagnose_tec_outlier --day 20260315
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import click
import polars as pl

log = logging.getLogger(__name__)

WORK = Path(__file__).resolve().parent.parent.parent / "work"


def _window(day: str, halfwidth: int = 2) -> list[str]:
    d0 = datetime.strptime(day, "%Y%m%d")
    return [(d0 + timedelta(days=k)).strftime("%Y%m%d")
            for k in range(-halfwidth, halfwidth + 1)]


def _print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


@click.command()
@click.option("--day", default="20260215",
              help="YYYYMMDD target day. Default 20260215 (the polar-departure "
                   "day flagged as ρ=+0.16 in T_tec_vs_gim.csv).")
def main(day: str) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    days = _window(day, halfwidth=2)
    print(f"Diagnostic window: {days[0]} ... {days[-1]} (target: {day})")

    # ------------------------------------------------------------------
    # 1. Kp / RFI / position. Prefer the cached scint_vs_kp.csv if present;
    #    otherwise fall back to a direct GFZ Kp fetch for just this window
    #    so the diagnostic is self-contained.
    # ------------------------------------------------------------------
    _print_section("1. Geomagnetic context (Kp)")
    sk_path = WORK / "derived" / "scint_vs_kp.csv"
    kp_max_target = None
    used_cache = False
    if sk_path.exists():
        sk = pl.read_csv(sk_path).with_columns(pl.col("day").cast(pl.Utf8))
        cols_wanted = [
            "day", "lat", "lon", "speed_m_s", "in_antarctic_circle",
            "Kp_max", "Kp_mean", "Kp_n_storm_lvl",
            "L1_jam_max", "L1_jam_gt60_n", "rfi_events_total",
        ]
        cols = [c for c in cols_wanted if c in sk.columns]
        sub = sk.filter(pl.col("day").is_in(days)).select(cols)
        if not sub.is_empty():
            used_cache = True
            print("Source: cached scint_vs_kp.csv")
            with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=180):
                print(sub)
            if "Kp_max" in cols:
                tgt = sub.filter(pl.col("day") == day)
                if not tgt.is_empty():
                    kp_max_target = float(tgt["Kp_max"][0])

    if not used_cache:
        if sk_path.exists():
            print(f"(scint_vs_kp.csv exists but has no rows for "
                  f"{days[0]}..{days[-1]} — falling back to direct GFZ fetch)")
        else:
            print(f"scint_vs_kp.csv not found at {sk_path} "
                  "— falling back to direct GFZ Kp fetch")
        try:
            from verify.correlate_kp import _fetch_kp  # noqa: PLC0415
            d0 = datetime.strptime(days[0], "%Y%m%d").strftime("%Y-%m-%d")
            d1 = datetime.strptime(days[-1], "%Y%m%d").strftime("%Y-%m-%d")
            kp = _fetch_kp(d0, d1)
            if kp.is_empty():
                print("Kp fetch returned empty — check internet access from the server")
            else:
                with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=140):
                    print(kp)
                tgt = kp.filter(
                    pl.col("date")
                    == datetime.strptime(day, "%Y%m%d").strftime("%Y-%m-%d"))
                if not tgt.is_empty():
                    kp_max_target = float(tgt["Kp_max"][0])
        except Exception as e:
            print(f"Kp fetch failed: {e!r}")
            print("(install requests, or check outbound HTTPS to kp.gfz-potsdam.de)")

    if kp_max_target is not None:
        if kp_max_target >= 5:
            print(f"\n  -> Target day {day} Kp_max={kp_max_target} reached storm "
                  "level (>=5): supports 'real ionospheric activity' framing.")
        elif kp_max_target >= 4:
            print(f"\n  -> Target day {day} Kp_max={kp_max_target} is elevated "
                  "(>=4, active level) but below storm threshold: weak support "
                  "for ionospheric activity.")
        else:
            print(f"\n  -> Target day {day} Kp_max={kp_max_target} is quiet: "
                  "Kp alone does NOT explain the divergence.")

    # ------------------------------------------------------------------
    # 2. Daily QC (T4_daily_stats.parquet)
    # ------------------------------------------------------------------
    _print_section("2. Daily QC counters (T4_daily_stats.parquet)")
    t4_path = WORK / "tables" / "T4_daily_stats.parquet"
    if not t4_path.exists():
        print(f"NOT FOUND: {t4_path}")
        print("(run analysis.qc_summary if you want QC context)")
    else:
        t4 = pl.read_parquet(t4_path).sort("day")
        # Show window + cruise-wide median for comparison
        cols_wanted = [
            "day", "n_epochs_pvt", "fix_rate", "mean_numSV",
            "cycle_slip_count", "slip_ratio_per_1000",
            "mp1_rms_m", "mp2_rms_m",
        ]
        cols = [c for c in cols_wanted if c in t4.columns]
        sub = t4.filter(pl.col("day").is_in(days)).select(cols)
        if sub.is_empty():
            print(f"(no T4 rows for window {days[0]}..{days[-1]})")
        else:
            with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=180):
                print(sub)
            # Cruise-wide percentiles for comparison
            print("\n  cruise-wide percentiles for comparison:")
            for c in cols:
                if c == "day":
                    continue
                series = t4[c]
                p50, p95 = series.median(), series.quantile(0.95)
                print(f"    {c:<22} median={p50!s:>10}  p95={p95!s:>10}")

    # ------------------------------------------------------------------
    # 3. TEC arc-level stats for the target day
    # ------------------------------------------------------------------
    _print_section(f"3. TEC arc-level stats for {day}")
    tec_path = WORK / "derived" / "tec" / f"{day}.tec.parquet"
    if not tec_path.exists():
        print(f"NOT FOUND: {tec_path}")
        print("(run analysis.tec --day {day} first)")
    else:
        tec = pl.read_parquet(tec_path)
        n_all = len(tec)
        hi = tec.filter(pl.col("elev") >= 40)
        n_hi = len(hi)
        print(f"rows total: {n_all:,}   high-elev (>=40 deg): {n_hi:,}")
        if n_hi > 0:
            print(f"  vtec median (hi-elev) : {hi['vtec'].median():+.2f} TECU")
            print(f"  vtec p10/p90 (hi-elev): {hi['vtec'].quantile(0.1):+.2f}"
                  f" / {hi['vtec'].quantile(0.9):+.2f} TECU")
            print(f"  vtec std (hi-elev)    : {hi['vtec'].std():.2f} TECU")
            print(f"  |dTEC/dt| p50 (hi-elev): "
                  f"{hi['dTECdt'].abs().median():.4f} TECU/s")
            print(f"  |dTEC/dt| p95 (hi-elev): "
                  f"{hi['dTECdt'].abs().quantile(0.95):.4f} TECU/s")

            # Compare against the other polar days we already have
            print("\n  Reference: same metrics for other polar/Antarctic days")
            for d in ["20260101", "20260127"]:
                p = WORK / "derived" / "tec" / f"{d}.tec.parquet"
                if not p.exists():
                    continue
                ref = pl.read_parquet(p).filter(pl.col("elev") >= 40)
                if ref.is_empty():
                    continue
                print(f"    {d}: median vtec {ref['vtec'].median():+.2f} TECU, "
                      f"std {ref['vtec'].std():.2f}, "
                      f"|dTEC/dt| p95 {ref['dTECdt'].abs().quantile(0.95):.4f} TECU/s")

    _print_section("Interpretation hints")
    print(
        "  - If Kp_max >= 5 OR Kp_n_storm_lvl > 0 on the target day:\n"
        "      → real geomagnetic activity; low ρ is paper-POSITIVE\n"
        "        (dataset resolves storm-time structure GIM cannot).\n"
        "  - Else if fix_rate << cruise median, cycle_slip_count >> p95,\n"
        "    or mp2_rms_m >> p95:\n"
        "      → local data-quality artifact; low ρ is paper-NEGATIVE\n"
        "        (hedge the polar finding in §5.5).\n"
        "  - Else if |dTEC/dt| p95 on target day >> reference polar days:\n"
        "      → real ionospheric variability not flagged by Kp;\n"
        "        could be polar-cap patches / TIDs.\n"
        "  - If none of the above: send me the raw numbers anyway —\n"
        "    we may need to inspect the per-arc level individually."
    )


if __name__ == "__main__":
    main()
