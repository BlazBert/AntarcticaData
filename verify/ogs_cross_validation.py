"""Cross-validate the F9P cruise track against the OGS ship-reporting feed.

The R/V Laura Bassi exposes its own position track via a JSON API at
``https://laurabassi.ogs.it/json/lb_track<campaign_id>.json`` (per-campaign,
12-minute cadence). The reporting chain is the ship's Trimble GPS + Octans
gyro broadcast as NMEA — a sensor and acquisition path entirely independent
of our u-blox F9P + Tersus AX4E02 setup, so it provides a genuine third-party
sanity check on the published cruise track.

For each OGS timestamp we take the *nearest* F9P sample (at 1 Hz the gap is
always <=0.5 s) and compute the haversine horizontal residual. Output:

    work/tables/T_ogs_vs_f9p.csv          — per-fix residuals (one row per OGS fix)
    work/tables/T_ogs_vs_f9p_summary.csv  — overall and per-campaign rollup
    work/figures/output/fig_ogs_vs_f9p.pdf — residual time series + CDF

Cached OGS JSON lives in ``work/derived/ogs_tracks/`` so re-runs do not
re-hit the OGS server. Use ``--refetch`` to force a refresh.

Cruise coverage (as of 2026-05-17): OGS tracks 7+8 cover 2025-10-03 → 2026-02-28
(~149 / 216 cruise days). The first week (Sep 26 → Oct 2) and the return leg
(Mar 1 → Apr 29) are not yet published as OGS campaigns; this script reports
on whatever overlap is available.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import polars as pl

from analysis._common import (
    derived_dir,
    figures_dir,
    haversine_km,
    load_config,
    read_parquet,
    tables_dir,
)

log = logging.getLogger(__name__)

OGS_CAMPAIGNS: dict[int, str] = {
    7: "Transfer ITA-NZ 2025",
    8: "41th PNRA Antarctic Campaign",
}
OGS_TRACK_URL = "https://laurabassi.ogs.it/json/lb_track{cid}.json"
MATCH_TOL_S = 60.0  # reject pairs where the nearest F9P sample is > this away


def _fetch_track(cid: int, cache_dir: Path, refetch: bool) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"lb_track{cid}.json"
    if path.exists() and not refetch:
        log.info("cache hit  campaign %d -> %s", cid, path)
        return path
    url = OGS_TRACK_URL.format(cid=cid)
    log.info("fetching   campaign %d <- %s", cid, url)
    with urllib.request.urlopen(url, timeout=30) as resp:
        path.write_bytes(resp.read())
    return path


def _load_ogs(path: Path, cid: int) -> pl.DataFrame:
    raw = json.loads(path.read_text())
    rows = raw["result"]
    # utc is naive ISO ("2025-11-21 23:59:11") and is UTC per the dashboard.
    t_ns = np.fromiter(
        (
            int(datetime.strptime(r["utc"], "%Y-%m-%d %H:%M:%S")
                .replace(tzinfo=timezone.utc).timestamp() * 1e9)
            for r in rows
        ),
        dtype=np.int64,
        count=len(rows),
    )
    return pl.DataFrame(
        {
            "t_ns": t_ns,
            "lat_ogs": np.fromiter((r["lat"] for r in rows), float, len(rows)),
            "lon_ogs": np.fromiter((r["lon"] for r in rows), float, len(rows)),
            "campaign_id": np.full(len(rows), cid, dtype=np.int16),
        }
    ).sort("t_ns")


def _nearest_match(ogs: pl.DataFrame, f9p: pl.DataFrame) -> pl.DataFrame:
    """For every OGS timestamp, pick the nearest F9P sample."""
    if ogs.is_empty() or f9p.is_empty():
        return pl.DataFrame()
    t_f9p = f9p["t_ns"].to_numpy()
    t_ogs = ogs["t_ns"].to_numpy()
    idx = np.searchsorted(t_f9p, t_ogs)
    idx_lo = np.clip(idx - 1, 0, len(t_f9p) - 1)
    idx_hi = np.clip(idx, 0, len(t_f9p) - 1)
    dt_lo = np.abs(t_ogs - t_f9p[idx_lo])
    dt_hi = np.abs(t_ogs - t_f9p[idx_hi])
    use_hi = dt_hi < dt_lo
    pick = np.where(use_hi, idx_hi, idx_lo)
    dt_s = np.minimum(dt_lo, dt_hi) / 1e9
    has_speed = "gSpeed_m_s" in f9p.columns
    extra_cols = [
        pl.Series("lat_f9p", f9p["lat"].to_numpy()[pick]),
        pl.Series("lon_f9p", f9p["lon"].to_numpy()[pick]),
        pl.Series("match_gap_s", dt_s),
    ]
    if has_speed:
        extra_cols.append(pl.Series("gSpeed_m_s", f9p["gSpeed_m_s"].to_numpy()[pick]))
    matched = ogs.with_columns(extra_cols).filter(pl.col("match_gap_s") <= MATCH_TOL_S)
    if matched.is_empty():
        return matched
    horiz_km = haversine_km(
        matched["lat_ogs"].to_numpy(),
        matched["lon_ogs"].to_numpy(),
        matched["lat_f9p"].to_numpy(),
        matched["lon_f9p"].to_numpy(),
    )
    return matched.with_columns(pl.Series("horiz_m", horiz_km * 1000.0))


def _add_inter_fix_speed(matched: pl.DataFrame) -> pl.DataFrame:
    """Add ``speed_m_s`` derived from consecutive OGS fixes (haversine / dt).

    Fallback when ``gSpeed_m_s`` from the F9P PVT isn't joined. At the OGS
    12-minute cadence the centred-difference speed is averaged over ~12 min
    of motion — fine for the gross transit-vs-dwell distinction we need.
    """
    if matched.is_empty():
        return matched
    m = matched.sort("t_ns")
    t = m["t_ns"].to_numpy()
    lat = m["lat_ogs"].to_numpy()
    lon = m["lon_ogs"].to_numpy()
    R = 6371008.8
    rl = np.deg2rad(lat)
    dlat = np.deg2rad(np.diff(lat))
    dlon = np.deg2rad(np.diff(lon))
    a = np.sin(dlat / 2) ** 2 + np.cos(rl[:-1]) * np.cos(rl[1:]) * np.sin(dlon / 2) ** 2
    d_m = 2 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    dt_s = np.maximum(np.diff(t) / 1e9, 1.0)
    sp_int = d_m / dt_s
    # Reject intervals crossing campaign boundaries (>1h) to keep speed finite
    sp_int = np.where(np.diff(t) / 1e9 < 3600.0, sp_int, np.nan)
    sp = np.concatenate(([sp_int[0]], (sp_int[:-1] + sp_int[1:]) / 2, [sp_int[-1]]))
    return m.with_columns(pl.Series("speed_m_s", sp))


def _speed_decomposition(matched: pl.DataFrame) -> dict[str, float]:
    """Linear fit ``horiz = slope * speed + intercept`` on moving fixes.

    ``slope`` ≈ OGS timestamp lag (s); ``intercept`` ≈ static inter-antenna
    baseline (m). Returns Pearson r and the dwell-only p50.
    """
    if "speed_m_s" not in matched.columns:
        return {}
    df = matched.filter(pl.col("speed_m_s").is_finite() & (pl.col("speed_m_s") < 20.0))
    if df.height < 100:
        return {}
    sp = df["speed_m_s"].to_numpy()
    h = df["horiz_m"].to_numpy()
    r = float(np.corrcoef(sp, h)[0, 1])
    mov = sp > 1.0
    slope, intercept = (float(x) for x in np.polyfit(sp[mov], h[mov], 1))
    dwell = sp < 0.2
    p50_dwell = float(np.median(h[dwell])) if dwell.any() else float("nan")
    return {
        "pearson_r": r,
        "lag_s_estimate": slope,
        "static_offset_m_estimate": intercept,
        "p50_dwell_m": p50_dwell,
        "n_dwell": int(dwell.sum()),
        "n_moving": int(mov.sum()),
    }


def _per_day_summary(matched: pl.DataFrame) -> pl.DataFrame:
    return matched.with_columns(
        (pl.col("t_ns") // 86_400_000_000_000).alias("_day_bin")
    ).group_by("_day_bin").agg([
        pl.col("campaign_id").first(),
        pl.len().alias("n"),
        pl.col("horiz_m").median().alias("p50_m"),
        pl.col("horiz_m").quantile(0.90).alias("p90_m"),
        pl.col("horiz_m").quantile(0.95).alias("p95_m"),
        pl.col("horiz_m").max().alias("max_m"),
        pl.col("t_ns").min().alias("t_start_ns"),
    ]).sort("t_start_ns").with_columns(
        pl.col("t_start_ns").map_elements(
            lambda x: datetime.fromtimestamp(x / 1e9, tz=timezone.utc).strftime("%Y-%m-%d"),
            return_dtype=pl.Utf8,
        ).alias("day"),
    ).select(["day", "campaign_id", "n", "p50_m", "p90_m", "p95_m", "max_m"])


def _build_figure(matched: pl.DataFrame, decomp: dict[str, float],
                  out_path: Path) -> None:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.dates as mdates  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    t_dt = np.array([
        datetime.fromtimestamp(int(x) / 1e9, tz=timezone.utc).replace(tzinfo=None)
        for x in matched["t_ns"].to_numpy()
    ])
    horiz = matched["horiz_m"].to_numpy()
    cid = matched["campaign_id"].to_numpy()
    has_speed = "speed_m_s" in matched.columns

    n_panels = 3 if (has_speed and decomp) else 2
    fig = plt.figure(figsize=(8.5, 3.0 * n_panels + 0.5))
    gs = fig.add_gridspec(n_panels, 1,
                          height_ratios=[1, 1, 1.1][:n_panels])

    # Panel 1: residual time series, by campaign
    ax_ts = fig.add_subplot(gs[0])
    for c, color in [(7, "C0"), (8, "C3")]:
        mask = cid == c
        if not mask.any():
            continue
        ax_ts.scatter(t_dt[mask], horiz[mask], s=3, color=color, alpha=0.5,
                      linewidths=0,
                      label=f"Camp. {c} {OGS_CAMPAIGNS.get(c,'?')} (n={int(mask.sum()):,})")
    ax_ts.set_ylabel("Horizontal residual (m)")
    ax_ts.set_yscale("log")
    ax_ts.set_title("F9P cruise track vs OGS Trimble+Octans ship reporting")
    ax_ts.xaxis.set_major_locator(mdates.MonthLocator())
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_ts.grid(True, which="both", alpha=0.3)
    ax_ts.legend(loc="upper right", fontsize=8)

    # Panel 2: CDF (all + dwell-only when available)
    ax_cdf = fig.add_subplot(gs[1])
    hs = np.sort(horiz)
    ax_cdf.plot(hs, np.arange(1, len(hs) + 1) / len(hs),
                color="black", lw=1.2, label="all fixes")
    if has_speed:
        sp = matched["speed_m_s"].to_numpy()
        dwell_mask = np.isfinite(sp) & (sp < 0.2)
        if dwell_mask.any():
            hd = np.sort(horiz[dwell_mask])
            ax_cdf.plot(hd, np.arange(1, len(hd) + 1) / len(hd),
                        color="C2", lw=1.2,
                        label=f"port dwell only (speed<0.2 m/s, n={int(dwell_mask.sum()):,})")
    for p, lbl in [(0.5, "p50"), (0.9, "p90"), (0.95, "p95")]:
        v = np.quantile(horiz, p)
        ax_cdf.axvline(v, color="gray", ls="--", lw=0.6)
        ax_cdf.text(v * 1.05, 0.10, f"{lbl}={v:.1f} m",
                    fontsize=8, color="gray", rotation=90)
    ax_cdf.set_xscale("log")
    ax_cdf.set_xlabel("Horizontal residual (m)")
    ax_cdf.set_ylabel("Cumulative fraction")
    ax_cdf.set_title(f"CDF (n={len(horiz):,} matched OGS fixes)")
    ax_cdf.legend(loc="lower right", fontsize=8)
    ax_cdf.grid(True, which="both", alpha=0.3)

    # Panel 3: residual vs speed with linear fit (only when speed available)
    if n_panels == 3:
        ax_sp = fig.add_subplot(gs[2])
        sp = matched["speed_m_s"].to_numpy()
        v_ok = np.isfinite(sp) & (sp >= 0) & (sp < 20)
        hb = ax_sp.hexbin(sp[v_ok], horiz[v_ok], gridsize=60, cmap="viridis",
                          mincnt=1, bins="log", extent=(0, 10, 0, 120))
        fig.colorbar(hb, ax=ax_sp, label="log10(fixes per bin)", shrink=0.85)
        xx = np.linspace(0, 10, 50)
        slope = decomp["lag_s_estimate"]
        intercept = decomp["static_offset_m_estimate"]
        ax_sp.plot(xx, slope * xx + intercept, "r-", lw=1.6,
                   label=f"linear fit: residual = {slope:.2f}·speed + {intercept:.2f} m")
        ax_sp.set_xlabel("Ship ground speed from OGS inter-fix interval (m/s)")
        ax_sp.set_ylabel("Horizontal residual (m)")
        ax_sp.set_title(f"Residual scales with ship speed (Pearson r = {decomp['pearson_r']:.2f})")
        ax_sp.set_xlim(0, 10)
        ax_sp.set_ylim(0, 120)
        ax_sp.legend(loc="upper left", fontsize=9)
        ax_sp.text(0.98, 0.05,
                   f"intercept ≈ {intercept:.1f} m  →  inter-antenna baseline\n"
                   f"slope ≈ {slope:.2f} s  →  OGS timestamp lag",
                   transform=ax_sp.transAxes, fontsize=8, va="bottom", ha="right",
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                             alpha=0.85, edgecolor="gray"))
        ax_sp.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


@click.command()
@click.option("--refetch", is_flag=True, default=False,
              help="Force re-download of OGS track JSONs (default: use cache)")
@click.option("--campaigns", default="7,8",
              help="Comma-separated OGS campaign IDs to validate against (default: 7,8)")
def main(refetch: bool, campaigns: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    cfg = load_config()
    cids = [int(c) for c in campaigns.split(",") if c.strip()]

    cache = derived_dir(cfg) / "ogs_tracks"
    f9p_path = derived_dir(cfg) / "track" / "track.all.parquet"
    if not f9p_path.exists():
        raise SystemExit(
            f"F9P aggregated track not found: {f9p_path}\n"
            "Run `python -m analysis.trajectory` first."
        )
    f9p_cols = read_parquet(f9p_path).columns
    keep = ["t_ns", "lat", "lon"] + (["gSpeed_m_s"] if "gSpeed_m_s" in f9p_cols else [])
    f9p = read_parquet(f9p_path).select(keep).sort("t_ns")
    log.info("F9P track: %d samples, %s .. %s",
             f9p.height,
             datetime.fromtimestamp(f9p["t_ns"].min() / 1e9, tz=timezone.utc),
             datetime.fromtimestamp(f9p["t_ns"].max() / 1e9, tz=timezone.utc))

    matched_parts: list[pl.DataFrame] = []
    for cid in cids:
        path = _fetch_track(cid, cache, refetch)
        ogs = _load_ogs(path, cid)
        log.info("OGS  campaign %d (%s): %d fixes  %s .. %s",
                 cid, OGS_CAMPAIGNS.get(cid, "?"), ogs.height,
                 datetime.fromtimestamp(ogs["t_ns"].min() / 1e9, tz=timezone.utc),
                 datetime.fromtimestamp(ogs["t_ns"].max() / 1e9, tz=timezone.utc))
        m = _nearest_match(ogs, f9p)
        log.info("  matched %d / %d fixes (within %.0f s of an F9P sample)",
                 m.height, ogs.height, MATCH_TOL_S)
        if not m.is_empty():
            matched_parts.append(m)

    if not matched_parts:
        log.warning("No matched fixes — nothing to write.")
        return
    matched = pl.concat(matched_parts, how="vertical").sort("t_ns")
    # Add inter-fix speed (or alias the F9P PVT speed if available)
    if "gSpeed_m_s" in matched.columns:
        matched = matched.rename({"gSpeed_m_s": "speed_m_s"})
    else:
        matched = _add_inter_fix_speed(matched)

    per_fix_cols = ["t_ns", "campaign_id", "lat_ogs", "lon_ogs",
                    "lat_f9p", "lon_f9p", "match_gap_s", "horiz_m"]
    if "speed_m_s" in matched.columns:
        per_fix_cols.append("speed_m_s")
    per_fix_path = tables_dir(cfg) / "T_ogs_vs_f9p.csv"
    matched.select(per_fix_cols).write_csv(per_fix_path)
    log.info("wrote %s (%d rows)", per_fix_path, matched.height)

    daily = _per_day_summary(matched)
    daily_path = tables_dir(cfg) / "T_ogs_vs_f9p_daily.csv"
    daily.write_csv(daily_path)
    log.info("wrote %s (%d days)", daily_path, daily.height)

    h = matched["horiz_m"].to_numpy()
    decomp = _speed_decomposition(matched)
    metrics: list[str] = ["n_matched", "p50_m", "p90_m", "p95_m", "p99_m", "max_m", "mean_m"]
    values: list[float] = [
        float(len(h)),
        float(np.median(h)), float(np.quantile(h, 0.9)),
        float(np.quantile(h, 0.95)), float(np.quantile(h, 0.99)),
        float(np.max(h)), float(np.mean(h)),
    ]
    for k in ("p50_dwell_m", "n_dwell", "n_moving", "pearson_r",
              "lag_s_estimate", "static_offset_m_estimate"):
        if k in decomp:
            metrics.append(k)
            values.append(float(decomp[k]))
    summary = pl.DataFrame({"metric": metrics, "value": values})
    summary_path = tables_dir(cfg) / "T_ogs_vs_f9p_summary.csv"
    summary.write_csv(summary_path)
    log.info("wrote %s", summary_path)

    print("\n--- F9P vs OGS horizontal residual ---")
    with pl.Config(tbl_rows=30, tbl_width_chars=120):
        print(summary)
        print(daily)

    fig_path = figures_dir(cfg) / "output" / "fig_ogs_vs_f9p.pdf"
    _build_figure(matched, decomp, fig_path)


if __name__ == "__main__":
    main()
