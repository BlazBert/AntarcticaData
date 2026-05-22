"""Diagnose cruise-track outliers — investigate before filtering.

The pipeline already drops fixes that fail the speed / fixType / null-island
filters from the *displayed* track (``analysis.trajectory``) and catalogues
them in ``derived/anomalies/<day>.anomalies.parquet``. This script answers
a different question: **what are those dropped fixes actually doing**, and
in particular, are some of them real ionospheric / RFI / spoofing events
worth keeping (especially in the Antarctic legs)?

Inputs (all produced by the existing pipeline):

* ``derived/track/track.all.parquet``    — clean post-filter cruise track
* ``derived/anomalies/<day>.anomalies.parquet``  — raw outliers + context
  (jamInd L1/L2L5, AGC, prev_lat/lon, dt_s, implied_speed_kn, ...)
* ``derived/spoofing/<day>.spoofing.parquet``    — 7-indicator suspicion score
* ``tables/T4_daily_stats.parquet``       — per-day cycle-slip counts
* ``staging/<day>/nav_pvt.parquet``       — pDOP at outlier epochs (optional)

Outputs (under ``work/tables/`` and ``work/figures/output/``):

* ``T_track_clean_downsampled.csv``   1 fix per N minutes of the clean track
* ``T_track_outliers.csv``            every outlier + final class + context
* ``T_track_outliers_by_region.csv``  region × class histogram
* ``track_outliers_map.pdf``          2-panel cartopy: clean | outliers-by-class
* ``track_outliers_map.html``         folium interactive (one layer per class)

Classes (precedence, top wins):

  possible_spoof          — speed-outlier + tight hAcc + nominal jamInd
                            (canonical spoofing fingerprint)
  spoofing_correlated     — spoofing_check suspicion ≥ 3
  rfi_correlated          — high_jamming category OR same-epoch jamInd > 64
  scintillation_candidate — anomaly at lat ≤ −60 AND the day's cycle-slip
                            count is in the upper quartile (sub-Antarctic
                            ionospheric burst — keep for paper 2)
  cold_start              — boot_state (receiver clock not yet synced)
  null_island             — lat = lon = 0
  low_geometry            — passed the clean filter but pDOP > 5.0
  no_fix                  — fixType < 3
  speed_outlier           — pure jump, none of the above
  unclassified            — anomaly tag set but nothing matched

Region bands (decided from lat alone — coarse on purpose):

  antarctic            lat ≤ −60
  drake_subantarctic   −60 < lat ≤ −45
  mid_southern         −45 < lat ≤ −23
  tropics              −23 < lat ≤  23
  mid_northern          23 < lat ≤  36
  mediterranean_basin  lat >  36

CLI::

    python -m verify.track_outliers                       # all defaults
    python -m verify.track_outliers --downsample-min 10
    python -m verify.track_outliers --no-map              # CSVs only
    python -m verify.track_outliers --pdop-threshold 4.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click
import numpy as np
import polars as pl

from analysis._common import (
    derived_dir,
    figures_dir,
    list_days,
    load_config,
    read_parquet,
    staged_path,
    tables_dir,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Class taxonomy
# ---------------------------------------------------------------------------

CLASS_PRECEDENCE: list[str] = [
    "possible_spoof",
    "spoofing_correlated",
    "rfi_l1",
    "rfi_l5",
    "scintillation_candidate",
    "cold_start",
    "null_island",
    "low_geometry",
    "no_fix",
    "speed_outlier",
    "unclassified",
]

CLASS_COLOR: dict[str, str] = {
    "possible_spoof":          "#FF1493",
    "spoofing_correlated":     "#9400D3",
    "rfi_l1":                  "#FF4500",
    "rfi_l5":                "#FFA07A",
    "scintillation_candidate": "#1E90FF",
    "cold_start":              "#808080",
    "null_island":             "#000000",
    "low_geometry":            "#FFA500",
    "no_fix":                  "#A9A9A9",
    "speed_outlier":           "#DC143C",
    "unclassified":            "#FFD700",
}

REGION_BANDS: list[tuple[str, float, float]] = [
    ("antarctic",           -91.0,  -60.0),
    ("drake_subantarctic",  -60.0,  -45.0),
    ("mid_southern",        -45.0,  -23.0),
    ("tropics",             -23.0,   23.0),
    ("mid_northern",         23.0,   36.0),
    ("mediterranean_basin",  36.0,   91.0),
]


def _region_of(lat: float) -> str:
    for name, lo, hi in REGION_BANDS:
        if lo < lat <= hi:
            return name
    return "unknown"


# ---------------------------------------------------------------------------
# Input loaders
# ---------------------------------------------------------------------------


def _load_clean_track(cfg: dict) -> pl.DataFrame:
    p = derived_dir(cfg) / "track" / "track.all.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — run analysis.trajectory --aggregate first."
        )
    df = read_parquet(p)
    # Defensive: drop NaN/inf coords (track.all.parquet should already be clean)
    df = df.filter(
        pl.col("lat").is_finite() & pl.col("lon").is_finite()
        & (pl.col("lat").abs() <= 90.0) & (pl.col("lon").abs() <= 180.0)
    )
    if df.is_empty():
        raise RuntimeError("Clean track is empty after final guards.")
    return df


def _load_anomalies(cfg: dict, days: list[str]) -> pl.DataFrame:
    in_dir = derived_dir(cfg) / "anomalies"
    parts: list[pl.DataFrame] = []
    for d in days:
        p = in_dir / f"{d}.anomalies.parquet"
        if not p.exists():
            continue
        df = read_parquet(p)
        if df.is_empty():
            continue
        parts.append(df.with_columns(pl.lit(d).alias("day")))
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="vertical_relaxed").sort("t_ns")


def _load_spoofing(cfg: dict, days: list[str]) -> pl.DataFrame:
    in_dir = derived_dir(cfg) / "spoofing"
    parts: list[pl.DataFrame] = []
    for d in days:
        p = in_dir / f"{d}.spoofing.parquet"
        if not p.exists():
            continue
        df = read_parquet(p)
        if df.is_empty() or "suspicion" not in df.columns:
            continue
        parts.append(df.select(["t_ns", "suspicion"]))
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="vertical_relaxed").sort("t_ns")


def _load_daily_cycle_slips(cfg: dict) -> dict[str, int]:
    t4 = tables_dir(cfg) / "T4_daily_stats.parquet"
    if not t4.exists():
        return {}
    df = read_parquet(t4)
    if "cycle_slip_count" not in df.columns or "day" not in df.columns:
        return {}
    return dict(zip(df["day"].to_list(), df["cycle_slip_count"].to_list()))


def _load_pdop(cfg: dict, days: list[str]) -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    for d in days:
        p = staged_path(d, "nav_pvt", cfg)
        if not p.exists():
            continue
        try:
            df = read_parquet(p).select([
                pl.col("t_ns"),
                (pl.col("pDOP").cast(pl.Float64) * 0.01).alias("pDOP_value"),
            ])
            parts.append(df)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load nav_pvt for %s: %s", d, exc)
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="vertical_relaxed").sort("t_ns")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _scintillation_slip_threshold(daily_slips: dict[str, int]) -> int:
    """Upper-quartile cycle-slip count across the cruise — days above this
    are 'slip-burst' days. If no T4 data, return a very high sentinel so
    nothing gets flagged as scintillation_candidate."""
    if not daily_slips:
        return 10**12
    arr = np.asarray(list(daily_slips.values()), dtype=np.int64)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 10**12
    return int(np.quantile(arr, 0.75))


def _classify(
    anomalies: pl.DataFrame,
    spoofing: pl.DataFrame,
    daily_slips: dict[str, int],
    slip_burst_threshold: int,
) -> pl.DataFrame:
    """Build the final outliers dataframe with one class per row."""
    if anomalies.is_empty():
        return pl.DataFrame()

    # Join spoofing suspicion at the same t_ns (left join — missing → 0)
    if not spoofing.is_empty():
        anomalies = anomalies.join(spoofing, on="t_ns", how="left")
    if "suspicion" not in anomalies.columns:
        anomalies = anomalies.with_columns(pl.lit(0.0).alias("suspicion"))
    anomalies = anomalies.with_columns(pl.col("suspicion").fill_null(0.0))

    # Add day-level cycle-slip count to each row for scintillation check
    slip_map_keys = list(daily_slips.keys())
    slip_map_vals = [int(daily_slips[k]) for k in slip_map_keys]
    slip_lookup = pl.DataFrame({
        "day": slip_map_keys,
        "_day_slips": slip_map_vals,
    }) if slip_map_keys else None
    if slip_lookup is not None and "day" in anomalies.columns:
        anomalies = anomalies.join(slip_lookup, on="day", how="left")
    if "_day_slips" not in anomalies.columns:
        anomalies = anomalies.with_columns(pl.lit(0).alias("_day_slips"))
    anomalies = anomalies.with_columns(pl.col("_day_slips").fill_null(0))

    # Region tag
    anomalies = anomalies.with_columns(
        pl.col("lat").map_elements(_region_of, return_dtype=pl.Utf8).alias("region")
    )

    # Per-row class — precedence as documented
    cats_col = anomalies["category"].to_list()
    susp_col = anomalies["suspicion"].to_list()
    jam_l1_col = anomalies["jamInd_L1"].to_list() if "jamInd_L1" in anomalies.columns else [-1] * anomalies.height
    jam_l5_col = anomalies["jamInd_L2L5"].to_list() if "jamInd_L2L5" in anomalies.columns else [-1] * anomalies.height
    lat_col = anomalies["lat"].to_list()
    slips_col = anomalies["_day_slips"].to_list()

    classes: list[str] = []
    for tags_s, susp, jam1, jam5, lat, dslips in zip(
        cats_col, susp_col, jam_l1_col, jam_l5_col, lat_col, slips_col
    ):
        tagset = set(tags_s.split(";")) if tags_s else set()
        is_high_lat = (lat is not None) and (lat <= -60.0)
        is_slip_burst = int(dslips or 0) > slip_burst_threshold
        same_epoch_jam = max(jam1 or -1, jam5 or -1) > 64

        if "possible_spoof" in tagset:
            classes.append("possible_spoof")
            continue
        if float(susp or 0.0) >= 3.0:
            classes.append("spoofing_correlated")
            continue
        if "high_jamming_l1" in tagset or (jam1 or -1) > 60:
            classes.append("rfi_l1")
            continue
        if "high_jamming_l5" in tagset or (jam5 or -1) > 130:
            classes.append("rfi_l5")
            continue
        if is_high_lat and is_slip_burst and (
            "speed_outlier" in tagset or "no_fix" in tagset
        ):
            classes.append("scintillation_candidate")
            continue
        if "boot_state" in tagset:
            classes.append("cold_start")
            continue
        if "null_island" in tagset:
            classes.append("null_island")
            continue
        if "no_fix" in tagset:
            classes.append("no_fix")
            continue
        if "speed_outlier" in tagset:
            classes.append("speed_outlier")
            continue
        classes.append("unclassified")

    return anomalies.with_columns(pl.Series("class", classes))


def _detect_low_geometry(
    clean: pl.DataFrame, pdop: pl.DataFrame, *, pdop_threshold: float
) -> pl.DataFrame:
    """Clean-track fixes (so they passed every filter) but whose pDOP is high.

    These are NOT in derived/anomalies/* — they're a fresh category. We
    keep them small (one row per offending epoch) to add to the outliers
    table with class='low_geometry'.
    """
    if clean.is_empty() or pdop.is_empty():
        return pl.DataFrame()
    j = clean.join(pdop, on="t_ns", how="inner").filter(
        pl.col("pDOP_value") > pdop_threshold
    )
    if j.is_empty():
        return pl.DataFrame()
    out = j.select([
        pl.col("t_ns"),
        pl.col("lat"),
        pl.col("lon"),
        pl.col("fixType"),
        pl.col("numSV"),
        (pl.col("hAcc_m")).alias("hAcc_m") if "hAcc_m" in j.columns else pl.lit(float("nan")).alias("hAcc_m"),
        pl.col("pDOP_value"),
    ])
    out = out.with_columns([
        pl.lit("low_geometry").alias("category"),
        pl.lit(0.0).alias("suspicion"),
        pl.lit(-1).alias("jamInd_L1"),
        pl.lit(-1).alias("jamInd_L2L5"),
        pl.lit("").alias("day"),
        pl.col("lat").map_elements(_region_of, return_dtype=pl.Utf8).alias("region"),
        pl.lit("low_geometry").alias("class"),
    ])
    return out


# ---------------------------------------------------------------------------
# Downsampling for the clean-track map
# ---------------------------------------------------------------------------


def _downsample_clean(df: pl.DataFrame, minutes: int) -> pl.DataFrame:
    bin_ns = max(minutes, 1) * 60 * 1_000_000_000
    cols_to_keep = ["lat", "lon"]
    agg = [pl.col("t_ns").first(), pl.col("lat").mean(), pl.col("lon").mean()]
    for opt in ("hMSL_m", "height_m", "gSpeed_m_s", "numSV", "hAcc_m"):
        if opt in df.columns:
            agg.append(pl.col(opt).mean().alias(opt))
            cols_to_keep.append(opt)
    if "fixType" in df.columns:
        agg.append(pl.col("fixType").last())
    return (
        df.with_columns((pl.col("t_ns") // bin_ns).alias("_bin"))
        .group_by("_bin")
        .agg(agg)
        .drop("_bin")
        .sort("t_ns")
    )


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _region_summary(outliers: pl.DataFrame) -> pl.DataFrame:
    if outliers.is_empty():
        return pl.DataFrame()
    pivot = (
        outliers.group_by(["region", "class"])
        .agg(pl.len().alias("n"))
        .pivot(values="n", index="region", on="class", aggregate_function="sum")
        .fill_null(0)
        .sort("region")
    )
    # Add a total column
    class_cols = [c for c in pivot.columns if c != "region"]
    if class_cols:
        pivot = pivot.with_columns(
            sum(pl.col(c) for c in class_cols).alias("total")
        )
    return pivot


# ---------------------------------------------------------------------------
# Map rendering
# ---------------------------------------------------------------------------


def _cluster_centroids(outliers: pl.DataFrame, *, eps_deg: float = 1.5) -> list[dict]:
    """Greedy 2-D clustering of outlier positions for map annotations.

    Returns a list of clusters, each ``{lat, lon, n, classes: dict[class,n]}``,
    so we can label the dominant geographic concentrations on the map.
    """
    if outliers.is_empty():
        return []
    pts = np.column_stack([
        outliers["lat"].to_numpy(), outliers["lon"].to_numpy()
    ])
    cls = outliers["class"].to_list()
    centroids: list[list[float]] = []
    counts: list[int] = []
    members: list[dict[str, int]] = []
    for p, c in zip(pts, cls):
        if centroids:
            d = np.linalg.norm(np.asarray(centroids) - p, axis=1)
            j = int(np.argmin(d))
            if d[j] < eps_deg:
                counts[j] += 1
                centroids[j][0] += (p[0] - centroids[j][0]) / counts[j]
                centroids[j][1] += (p[1] - centroids[j][1]) / counts[j]
                members[j][c] = members[j].get(c, 0) + 1
                continue
        centroids.append([float(p[0]), float(p[1])])
        counts.append(1)
        members.append({c: 1})
    return [
        {"lat": centroids[i][0], "lon": centroids[i][1],
         "n": counts[i], "classes": members[i]}
        for i in range(len(centroids))
    ]


def _label_cluster(c: dict) -> str:
    """Best-effort geographic label for a cluster centroid."""
    lat, lon = c["lat"], c["lon"]
    if -45 <= lat <= -43 and 172 <= lon <= 174:
        return "Lyttelton (NZ)"
    if -75.5 <= lat <= -74 and 163 <= lon <= 165:
        return "Mario Zucchelli"
    if 45 <= lat <= 46 and 13 <= lon <= 14:
        return "Trieste"
    if 8 <= lat <= 10 and -80 <= lon <= -78:
        return "Panama Canal"
    if 35 <= lat <= 45 and -10 <= lon <= 20:
        return "Mediterranean"
    if lat <= -60:
        return f"Antarctic ({lat:.1f}°, {lon:.1f}°)"
    return f"{lat:.1f}°, {lon:.1f}°"


def _render_static_map(
    clean: pl.DataFrame,
    outliers: pl.DataFrame,
    out_path: Path,
) -> Path:
    """Three-panel layout: clean track + outliers map + region/class bar."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.gridspec import GridSpec  # noqa: PLC0415

    try:
        import cartopy.crs as ccrs  # noqa: PLC0415
        import cartopy.feature as cfeature  # noqa: PLC0415
        have_cartopy = True
    except ImportError:
        have_cartopy = False

    fig = plt.figure(figsize=(14, 8.5))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2.0, 1.0], hspace=0.18, wspace=0.05)
    proj = None
    if have_cartopy:
        proj = ccrs.Robinson(central_longitude=0)
        ax1 = fig.add_subplot(gs[0, 0], projection=proj)
        ax2 = fig.add_subplot(gs[0, 1], projection=proj)
        for ax in (ax1, ax2):
            try:
                ax.add_feature(cfeature.LAND, facecolor="0.92",
                               edgecolor="0.55", linewidth=0.3)
                ax.add_feature(cfeature.OCEAN, facecolor="#F4F8FB")
                ax.coastlines(linewidth=0.4, color="0.35")
            except Exception:  # noqa: BLE001
                pass
            ax.gridlines(linewidth=0.2, color="0.7")
    else:
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        for ax in (ax1, ax2):
            ax.set_xlabel("Lon (°E)")
            ax.set_ylabel("Lat (°N)")
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.grid(linewidth=0.2, color="0.7")
            ax.set_aspect("equal")
    ax3 = fig.add_subplot(gs[1, :])

    # Panel A — clean track, time-graded
    if not clean.is_empty():
        lon = clean["lon"].to_numpy()
        lat = clean["lat"].to_numpy()
        t_ns = clean["t_ns"].to_numpy().astype(np.float64)
        t_lo = float(np.quantile(t_ns, 0.01))
        t_hi = float(np.quantile(t_ns, 0.99))
        denom = max(1.0, t_hi - t_lo)
        t_norm = np.clip((t_ns - t_lo) / denom, 0.0, 1.0)
        plot_kwargs = {"transform": ccrs.PlateCarree()} if have_cartopy else {}
        sc = ax1.scatter(lon, lat, c=t_norm, cmap="viridis", s=1.5,
                          vmin=0.0, vmax=1.0, **plot_kwargs)
        cb = fig.colorbar(sc, ax=ax1, orientation="horizontal", pad=0.04, shrink=0.7)
        cb.set_label("cruise time (1st–99th percentile normalised)")
    ax1.set_title(f"Clean cruise track  (n={clean.height})")

    # Panel B — outliers map with LARGE markers + cluster annotations.
    classes_in_data = []
    if not outliers.is_empty():
        classes_in_data = [c for c in CLASS_PRECEDENCE
                            if (outliers["class"] == c).any()]
        for cls in classes_in_data:
            sub = outliers.filter(pl.col("class") == cls)
            lat = sub["lat"].to_numpy()
            lon = sub["lon"].to_numpy()
            plot_kwargs = {"transform": ccrs.PlateCarree()} if have_cartopy else {}
            ax2.scatter(
                lon, lat,
                s=80,
                color=CLASS_COLOR[cls],
                label=f"{cls} (n={sub.height})",
                alpha=0.78,
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
                **plot_kwargs,
            )
        # Cluster annotations
        for c in _cluster_centroids(outliers, eps_deg=2.0):
            if c["n"] < max(2, int(0.01 * outliers.height)):
                continue   # skip very small clusters
            label = f"{_label_cluster(c)}\n(n={c['n']})"
            xy_kwargs = ({"xycoords": ccrs.PlateCarree()._as_mpl_transform(ax2)}
                          if have_cartopy else {})
            ax2.annotate(
                label,
                xy=(c["lon"], c["lat"]),
                xytext=(c["lon"] + 8, c["lat"] + 8),
                fontsize=7,
                ha="left",
                va="bottom",
                arrowprops={"arrowstyle": "->", "lw": 0.4, "color": "0.2"},
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                       "edgecolor": "0.6", "alpha": 0.85},
                zorder=6,
                **xy_kwargs,
            )
        ax2.legend(loc="lower left", fontsize=7, markerscale=0.7,
                    framealpha=0.85)
    ax2.set_title(f"Outliers map  (n_total={outliers.height})")

    # Panel C — stacked horizontal bar chart by region.
    if not outliers.is_empty():
        summary = (outliers.group_by(["region", "class"])
                   .agg(pl.len().alias("n"))
                   .pivot(values="n", index="region", on="class",
                            aggregate_function="sum")
                   .fill_null(0)
                   .sort("region"))
        regions = summary["region"].to_list()
        # Sum across classes for ordering
        class_cols = [c for c in summary.columns if c != "region"]
        totals = np.array([
            sum(int(r[c]) for c in class_cols)
            for r in summary.iter_rows(named=True)
        ])
        order = np.argsort(-totals)
        regions = [regions[i] for i in order]
        totals = totals[order]
        per_class = {
            c: np.array([int(summary.row(i, named=True)[c]) for i in order])
            for c in class_cols
        }
        y = np.arange(len(regions))
        left = np.zeros(len(regions))
        for cls in CLASS_PRECEDENCE:
            if cls not in per_class:
                continue
            counts = per_class[cls]
            if counts.sum() == 0:
                continue
            ax3.barh(y, counts, left=left, color=CLASS_COLOR[cls],
                      edgecolor="black", linewidth=0.4, label=cls,
                      height=0.7)
            left += counts
        for i, (yt, tot) in enumerate(zip(y, totals)):
            ax3.text(tot + max(1, 0.01 * totals.max()), yt,
                      f"  {int(tot)}", va="center", fontsize=8)
        ax3.set_yticks(y)
        ax3.set_yticklabels(regions, fontsize=9)
        ax3.invert_yaxis()
        ax3.set_xlabel("Flagged-epoch count")
        ax3.set_title("Outliers by region and class (stacked counts)")
        ax3.grid(axis="x", linewidth=0.3, alpha=0.5)
        ax3.set_axisbelow(True)
        ax3.legend(loc="lower right", fontsize=7, ncol=1, framealpha=0.85)
    else:
        ax3.text(0.5, 0.5, "No outliers", ha="center", va="center",
                  transform=ax3.transAxes, color="0.5")
        ax3.set_axis_off()

    fig.suptitle("Cruise track quality diagnostic", fontsize=12, y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s (cartopy=%s)", out_path, have_cartopy)
    return out_path


def _render_folium_map(
    clean: pl.DataFrame,
    outliers: pl.DataFrame,
    out_path: Path,
) -> Path | None:
    try:
        import folium  # noqa: PLC0415
        from folium.plugins import MarkerCluster  # noqa: PLC0415
    except ImportError:
        log.warning("folium not installed — skipping HTML map.")
        return None

    if clean.is_empty():
        log.warning("No clean track — skipping HTML map.")
        return None

    lat0 = float(np.median(clean["lat"].to_numpy()))
    lon0 = float(np.median(clean["lon"].to_numpy()))
    m = folium.Map(location=[lat0, lon0], zoom_start=2, tiles="OpenStreetMap",
                   prefer_canvas=True)

    # Clean track as a faint polyline; split on time gaps + antimeridian
    coords = list(zip(clean["lat"].to_list(), clean["lon"].to_list()))
    t_ns = clean["t_ns"].to_numpy()
    seg: list[list[tuple[float, float]]] = [[coords[0]]] if coords else []
    GAP_NS = 30 * 60 * 1_000_000_000
    for i in range(1, len(coords)):
        big_gap = (t_ns[i] - t_ns[i - 1]) > GAP_NS
        anti = abs(coords[i][1] - coords[i - 1][1]) > 180.0
        if big_gap or anti:
            seg.append([])
        seg[-1].append(coords[i])
    fg_track = folium.FeatureGroup(name="Clean track", show=True)
    for s in seg:
        if len(s) >= 2:
            folium.PolyLine(s, color="#1F77B4", weight=1.5, opacity=0.55).add_to(fg_track)
    fg_track.add_to(m)

    # Outliers as per-class layers
    if not outliers.is_empty():
        for cls in CLASS_PRECEDENCE:
            sub = outliers.filter(pl.col("class") == cls)
            if sub.is_empty():
                continue
            fg = folium.FeatureGroup(name=f"{cls} (n={sub.height})", show=True)
            for r in sub.iter_rows(named=True):
                tooltip = (
                    f"<b>{cls}</b><br>"
                    f"day {r.get('day','?')}<br>"
                    f"lat {r['lat']:.4f}, lon {r['lon']:.4f}<br>"
                    f"fixType {r.get('fixType','?')}, "
                    f"numSV {r.get('numSV','?')}<br>"
                    f"hAcc {r.get('hAcc_m','?')} m<br>"
                    f"suspicion {r.get('suspicion', 0.0):.2f}<br>"
                    f"jamInd L1 {r.get('jamInd_L1','?')}, "
                    f"L2/L5 {r.get('jamInd_L2L5','?')}<br>"
                    f"implied speed {r.get('implied_speed_kn','?')} kn"
                )
                folium.CircleMarker(
                    [r["lat"], r["lon"]],
                    radius=4,
                    color=CLASS_COLOR[cls],
                    fill=True,
                    fill_color=CLASS_COLOR[cls],
                    fill_opacity=0.85,
                    weight=0.8,
                    popup=folium.Popup(tooltip, max_width=320),
                ).add_to(fg)
            fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    log.info("Wrote %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    cfg: dict,
    *,
    downsample_min: int = 5,
    pdop_threshold: float = 5.0,
    make_static: bool = True,
    make_html: bool = True,
) -> dict[str, Path]:
    days = list_days(cfg)
    if not days:
        raise click.ClickException("No staged days found.")
    log.info("Loading inputs for %d days", len(days))

    clean = _load_clean_track(cfg)
    anomalies = _load_anomalies(cfg, days)
    spoofing = _load_spoofing(cfg, days)
    daily_slips = _load_daily_cycle_slips(cfg)
    slip_burst_threshold = _scintillation_slip_threshold(daily_slips)
    pdop = _load_pdop(cfg, days)

    log.info(
        "Inputs — clean=%d anomalies=%d spoofing_epochs=%d days_with_slips=%d "
        "pDOP_rows=%d slip_burst_threshold=%d",
        clean.height, anomalies.height, spoofing.height,
        len(daily_slips), pdop.height, slip_burst_threshold,
    )

    # Classify the pre-catalogued outliers
    classified = _classify(anomalies, spoofing, daily_slips, slip_burst_threshold)

    # Add fresh low_geometry rows (clean-track fixes with pDOP > threshold)
    low_geom = _detect_low_geometry(clean, pdop, pdop_threshold=pdop_threshold)
    if not low_geom.is_empty():
        # Align columns by taking the intersection
        common = sorted(set(classified.columns) & set(low_geom.columns))
        if not classified.is_empty():
            outliers = pl.concat([classified.select(common), low_geom.select(common)],
                                  how="vertical_relaxed").sort("t_ns")
        else:
            outliers = low_geom
    else:
        outliers = classified

    log.info("Outliers total: %d", outliers.height)
    if not outliers.is_empty():
        class_hist = (
            outliers.group_by("class")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
        )
        for row in class_hist.iter_rows(named=True):
            log.info("  %-26s %d", row["class"], row["n"])

    # Downsample the clean track for the map + CSV
    clean_ds = _downsample_clean(clean, downsample_min)
    log.info(
        "Clean track downsampled %d → %d at %d min", clean.height, clean_ds.height, downsample_min,
    )

    # Write CSVs
    out_paths: dict[str, Path] = {}
    tdir = tables_dir(cfg)
    tdir.mkdir(parents=True, exist_ok=True)

    ds_csv = tdir / "T_track_clean_downsampled.csv"
    clean_ds.write_csv(ds_csv)
    log.info("Wrote %s (%d rows)", ds_csv, clean_ds.height)
    out_paths["clean_csv"] = ds_csv

    out_csv = tdir / "T_track_outliers.csv"
    if outliers.is_empty():
        # Write an empty file with the expected columns so downstream tools don't fail
        pl.DataFrame({c: [] for c in ["t_ns", "lat", "lon", "class", "region"]}).write_csv(out_csv)
    else:
        # Drop internal helper columns
        drop_cols = [c for c in ("_day_slips",) if c in outliers.columns]
        outliers.drop(drop_cols).write_csv(out_csv)
    log.info("Wrote %s (%d rows)", out_csv, outliers.height)
    out_paths["outliers_csv"] = out_csv

    summary = _region_summary(outliers)
    sum_csv = tdir / "T_track_outliers_by_region.csv"
    if summary.is_empty():
        pl.DataFrame({"region": [], "total": []}).write_csv(sum_csv)
    else:
        summary.write_csv(sum_csv)
    log.info("Wrote %s", sum_csv)
    out_paths["region_csv"] = sum_csv

    # Maps
    if make_static:
        fig_dir = figures_dir(cfg) / "output"
        out_paths["pdf"] = _render_static_map(
            clean_ds, outliers, fig_dir / "track_outliers_map.pdf"
        )
    if make_html:
        fig_dir = figures_dir(cfg) / "output"
        html_path = _render_folium_map(
            clean_ds, outliers, fig_dir / "track_outliers_map.html"
        )
        if html_path:
            out_paths["html"] = html_path

    return out_paths


@click.command()
@click.option("--downsample-min", default=5, type=int, show_default=True,
              help="Downsample minutes for the clean-track map and CSV")
@click.option("--pdop-threshold", default=5.0, type=float, show_default=True,
              help="Fixes that passed the speed filter but have pDOP above "
                   "this value are flagged as low_geometry")
@click.option("--no-map", is_flag=True, default=False,
              help="Skip the static PDF map")
@click.option("--no-html", is_flag=True, default=False,
              help="Skip the folium HTML map")
def main(downsample_min: int, pdop_threshold: float, no_map: bool, no_html: bool) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    paths = run(
        cfg,
        downsample_min=downsample_min,
        pdop_threshold=pdop_threshold,
        make_static=not no_map,
        make_html=not no_html,
    )
    print("\nOutputs:")
    for k, p in paths.items():
        print(f"  {k:<12} {p}")


if __name__ == "__main__":
    main()
