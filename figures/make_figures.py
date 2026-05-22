"""Generate ESSD-ready figures.

CLI:
    python -m figures.make_figures --quick              # smoke-test on one day
    python -m figures.make_figures                      # all 12 figures, full cruise
    python -m figures.make_figures --only fig11,fig12   # specific subset

Output: ``work/figures/figXX_*.pdf`` (also PNG copies for the README).

The figure functions try to be defensive: if the input file isn't there,
they emit a small placeholder PDF and log a warning rather than crashing
the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import click
import numpy as np
import polars as pl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis._common import (
    derived_dir,
    figures_dir,
    list_days,
    load_config,
    read_parquet,
    tables_dir,
)
from figures._helpers import GNSS_COLORS, GNSS_LABEL, apply_style, gnss_color

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# fig01 — cruise track map
# ---------------------------------------------------------------------------


def _load_track_segments(cfg: dict) -> list[tuple[str, np.ndarray, np.ndarray]] | None:
    """Continuous (label, lon[], lat[]) segments for the cruise track.

    Builds a single 1-min-decimated cruise track from
    ``derived/track/track.all.parquet``, then splits it into segments at:

    * Time gaps larger than ``GAP_MINUTES`` (data outages — drawing a
      polyline across them would imply travel where there was none).
    * Antimeridian crossings (consecutive longitude jumps $>180^{\circ}$,
      e.g. Lyttelton at $+172^{\circ}$\,E to Antarctic operations near
      $-178^{\circ}$\,E) — without this Leaflet / cartopy draw a
      horizontal line straight across the map.
    * Implausibly fast steps ($>$``MAX_SPEED_KN``\,kn between consecutive
      decimated samples), which catch the few stale-cache fixes that
      pass the ``hAcc`` filter in ``analysis.trajectory``.

    Each surviving sub-segment is labelled by the UTC date of its first
    point (for the day-coloured cmap). Returns ``None`` if
    ``track.all.parquet`` is missing or empty.
    """
    track_path = derived_dir(cfg) / "track" / "track.all.parquet"
    if not track_path.exists():
        return None
    df = read_parquet(track_path)
    df = df.filter(
        pl.col("lat").is_finite() & pl.col("lon").is_finite()
        & (pl.col("lat").abs() <= 90.0) & (pl.col("lon").abs() <= 180.0)
        & (pl.col("t_ns") > 0)
    ).sort("t_ns")
    if df.is_empty():
        return None
    # 1-min decimation across the whole cruise
    df = (df.with_columns((pl.col("t_ns") // 60_000_000_000).alias("_min"))
            .group_by("_min").agg([
                pl.col("t_ns").first(),
                pl.col("lat").mean(),
                pl.col("lon").mean(),
            ]).sort("_min"))
    if df.height < 2:
        return None
    t = df["t_ns"].to_numpy()
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()

    # Identify break indices (where a NEW segment must start)
    GAP_MINUTES = 30
    MAX_SPEED_KN = 60.0
    gap_ns = GAP_MINUTES * 60 * 1_000_000_000
    # Haversine distance between consecutive points (km)
    rlat1, rlat2 = np.deg2rad(lat[:-1]), np.deg2rad(lat[1:])
    dlat = rlat2 - rlat1
    dlon = np.deg2rad(lon[1:] - lon[:-1])
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    dist_km = 2 * 6371.0088 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    dt_h = np.maximum(np.diff(t) / 1e9 / 3600.0, 1e-6)
    speed_kn = (dist_km / dt_h) / 1.852
    big_lon_jump = np.abs(np.diff(lon)) > 180.0
    breaks = np.where(
        (np.diff(t) > gap_ns) | (speed_kn > MAX_SPEED_KN) | big_lon_jump
    )[0] + 1
    cut = set(breaks.tolist())

    # Build per-segment labels using the UTC date of the first point
    from datetime import datetime, timezone
    segments: list[tuple[str, np.ndarray, np.ndarray]] = []
    current_lon: list[float] = []
    current_lat: list[float] = []
    current_label = ""
    for i in range(len(t)):
        if i in cut and current_lon:
            if len(current_lon) >= 2:
                segments.append((current_label,
                                  np.asarray(current_lon),
                                  np.asarray(current_lat)))
            current_lon = []
            current_lat = []
        if not current_lon:
            current_label = datetime.fromtimestamp(
                int(t[i]) / 1e9, tz=timezone.utc).date().isoformat()
        current_lon.append(float(lon[i]))
        current_lat.append(float(lat[i]))
    if len(current_lon) >= 2:
        segments.append((current_label,
                          np.asarray(current_lon),
                          np.asarray(current_lat)))
    return segments


def _load_track_dots(cfg: dict, *, decimate_min: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str] | None:
    """Single (lon, lat, t_ns, t_lo_iso, t_hi_iso) point cloud for the
    cruise track. Returns ``None`` if input is missing.

    Each output point is the per-``decimate_min`` mean position; no
    segmentation, no polyline. Drawing as scatter avoids any
    antimeridian / data-gap artefacts that polylines exhibit.
    """
    track_path = derived_dir(cfg) / "track" / "track.all.parquet"
    if not track_path.exists():
        return None
    df = read_parquet(track_path)
    df = df.filter(
        pl.col("lat").is_finite() & pl.col("lon").is_finite()
        & (pl.col("lat").abs() <= 90.0) & (pl.col("lon").abs() <= 180.0)
        & (pl.col("t_ns") > 0)
    ).sort("t_ns")
    if df.is_empty():
        return None
    bin_ns = max(decimate_min, 1) * 60 * 1_000_000_000
    df = (df.with_columns((pl.col("t_ns") // bin_ns).alias("_bin"))
            .group_by("_bin").agg([
                pl.col("t_ns").first(),
                pl.col("lat").mean(),
                pl.col("lon").mean(),
            ]).sort("_bin"))
    from datetime import datetime, timezone
    t_ns = df["t_ns"].to_numpy()
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    t_lo = datetime.fromtimestamp(int(t_ns.min()) / 1e9, tz=timezone.utc).date().isoformat()
    t_hi = datetime.fromtimestamp(int(t_ns.max()) / 1e9, tz=timezone.utc).date().isoformat()
    return lon, lat, t_ns, t_lo, t_hi


def fig01_cruise_track(
    out_dir: Path,
    cfg: dict,
    *,
    decimate_min: int = 5,
    central_lon: float = 160.0,
) -> Path:
    """Static cruise-track map rendered as a scatter point cloud (one dot
    per ``decimate_min`` minutes) coloured by UTC date.

    ``central_lon`` shifts the projection so the cruise activity is
    moved away from the map edges. Default 160°E puts Lyttelton /
    Mario Zucchelli near the centre and the Atlantic descent on the
    left half of the map; set to 0 for Greenwich-centred, to 180 for
    dateline-centred, or to ``-90`` to put the Americas in the
    centre. Drawing dots rather than a polyline sidesteps the
    antimeridian-crossing and data-gap artefacts that produce
    spurious straight lines on a global projection. Tries cartopy
    → geopandas fallback → plain matplotlib.
    """
    fig_path = out_dir / "fig01_cruise_track_map.pdf"
    data = _load_track_dots(cfg, decimate_min=decimate_min)
    if data is None:
        return _placeholder(fig_path, "fig01: no track — run analysis.trajectory --aggregate")
    lon, lat, t_ns, t_lo, t_hi = data
    fig = plt.figure(figsize=(11, 5.8))
    cmap = plt.get_cmap("viridis")

    # Normalised colour by cruise time
    t_arr = t_ns.astype(np.float64)
    t_min = float(t_arr.min())
    t_max = float(t_arr.max())
    denom = max(1.0, t_max - t_min)
    t_norm = np.clip((t_arr - t_min) / denom, 0.0, 1.0)

    used = "plain"
    try:
        import cartopy.crs as ccrs  # noqa: PLC0415
        import cartopy.feature as cfeature  # noqa: PLC0415

        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=central_lon))
        try:
            ax.add_feature(cfeature.LAND, facecolor="0.92",
                           edgecolor="0.55", linewidth=0.3)
            ax.add_feature(cfeature.OCEAN, facecolor="#F4F8FB")
        except Exception as feat_exc:  # noqa: BLE001
            log.warning("cartopy features unavailable (%s); plotting coastlines only", feat_exc)
        try:
            ax.coastlines(linewidth=0.4, color="0.35")
        except Exception:  # noqa: BLE001
            pass
        ax.gridlines(linewidth=0.2, color="0.7")
        ax.scatter(lon, lat, c=t_norm, cmap=cmap, s=1.2, alpha=0.85,
                    edgecolors="none", transform=ccrs.PlateCarree(), zorder=4)
        for plon, plat, ptxt in [
            (13.809, 45.613, "Trieste"),
            (-79.54, 8.87, "Panama"),
            (172.72, -43.61, "Lyttelton"),
            (164.11, -74.69, "MZS"),
        ]:
            ax.plot(plon, plat, marker="o", color="black", markersize=4,
                    transform=ccrs.PlateCarree(), zorder=5)
            ax.text(plon, plat, "  " + ptxt,
                    transform=ccrs.PlateCarree(), fontsize=7, zorder=5)
        ax.set_global()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.03)
        cb.set_ticks([0.0, 0.5, 1.0])
        # Midpoint label
        from datetime import datetime, timezone
        t_mid = datetime.fromtimestamp((t_min + t_max) / 2e9, tz=timezone.utc).date().isoformat()
        cb.set_ticklabels([t_lo, t_mid, t_hi])
        cb.set_label("UTC date")
        used = "cartopy"
    except Exception as exc:  # noqa: BLE001
        log.warning("cartopy path failed (%s); trying geopandas fallback", exc)
        fig.clear()
        try:
            import geopandas as gpd  # noqa: PLC0415
            world = None
            for src in ("naturalearth_lowres", None):
                try:
                    world = gpd.read_file(gpd.datasets.get_path(src)) if src else None
                    if world is not None:
                        break
                except Exception:  # noqa: BLE001
                    world = None
            ax = fig.add_subplot(1, 1, 1)
            if world is not None:
                world.plot(ax=ax, color="0.92", edgecolor="0.55", linewidth=0.3)
            ax.scatter(lon, lat, c=t_norm, cmap=cmap, s=1.2, alpha=0.85,
                        edgecolors="none")
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.set_aspect("equal")
            used = "geopandas" if world is not None else "plain"
        except Exception as exc2:  # noqa: BLE001
            log.warning("geopandas fallback failed (%s); plotting plain", exc2)
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(lon, lat, c=t_norm, cmap=cmap, s=1.2, alpha=0.85,
                        edgecolors="none")
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.set_aspect("equal")
            used = "plain"

    ax.set_title(
        f"R/V Laura Bassi cruise track ({decimate_min}-min dots, n={len(lon):,})"
    )
    # TODO(inset): the Antarctic dwell + Lyttelton approach legs overlap
    # visually on the global Robinson view. Add a regional inset (e.g.
    # NearsidePerspective central_lon=170, central_lat=-60, satellite_h
    # ~3000 km) covering lon 150°E–180°E, lat 75°S–40°S so the four
    # repeated approach lines (Lyttelton → Ross Sea → Lyttelton ×2) read
    # as distinct paths. Needs visual iteration — leave commented for now.
    log.info("fig01 backend: %s, n_points: %d", used, len(lon))
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


# ---------------------------------------------------------------------------
# fig05 — multipath M1/M2 vs elevation
# ---------------------------------------------------------------------------


def fig05_multipath(out_dir: Path, cfg: dict, day: str | None = None) -> Path:
    fig_path = out_dir / "fig05_multipath_M1M2_vs_elev_lat.pdf"
    days = [day] if day else list_days(cfg)
    parts = []
    for d in days:
        p = derived_dir(cfg) / "multipath" / f"{d}.multipath.parquet"
        if p.exists():
            parts.append(read_parquet(p))
    if not parts:
        return _placeholder(fig_path, "fig05: no multipath data")
    df = pl.concat(parts, how="vertical").filter(pl.col("elev") > 0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, col, title in zip(axes, ("M1", "M2"), ("L1 multipath M1", "L5/E5a/B2a multipath M2")):
        # Reject |M| > 10 m before computing per-elev std — residual cycle
        # slips that survive the SLIP_M arc segmentation are 100 m+ and
        # blow up the std otherwise.
        clean = df.filter(pl.col(col).abs() < 10.0)
        for gid in sorted(clean["gnssId"].unique().to_list()):
            sub = clean.filter(pl.col("gnssId") == gid)
            elev_bins = np.arange(0, 91, 5)
            grouped = sub.with_columns(
                ((pl.col("elev").cast(pl.Float64) // 5) * 5).alias("elev_bin")
            ).group_by("elev_bin").agg([pl.col(col).std().alias("rms")])
            grouped = grouped.sort("elev_bin")
            ax.plot(
                grouped["elev_bin"].to_numpy(),
                grouped["rms"].to_numpy(),
                marker="o",
                label=GNSS_LABEL.get(gid, gid),
                color=gnss_color(gid),
            )
        ax.set_xlabel("Elevation (°)")
        ax.set_title(title)
        ax.set_xlim(0, 90)
    axes[0].set_ylabel("Multipath RMS (m)")
    axes[0].legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


# ---------------------------------------------------------------------------
# fig11 — daily-averaged RF spectrum waterfall (L1 + L2/L5)
# ---------------------------------------------------------------------------


def fig11_rf_waterfall(out_dir: Path, cfg: dict, day: str | None = None) -> Path:
    fig_path = out_dir / "fig11_daily_rf_waterfall.pdf"
    days = [day] if day else list_days(cfg)
    # The first cruise day (20250926) covers only ~15 h after cold-start
    # and is not representative for the published waterfall; same for the
    # closing arrival day (20260429). Prefer 2025-09-30 (the established
    # reference day used elsewhere in the paper); fall back to the first
    # full day otherwise.
    PARTIAL_DAYS = {"20250926", "20260429"}
    PREFERRED = "20250930"
    src = None
    day_used: str | None = None
    if day is None and PREFERRED in days:
        p = derived_dir(cfg) / "spectrum" / f"{PREFERRED}.spectrogram.npz"
        if p.exists():
            src = p
            day_used = PREFERRED
    if src is None:
        for d in days:
            if day is None and d in PARTIAL_DAYS:
                continue
            p = derived_dir(cfg) / "spectrum" / f"{d}.spectrogram.npz"
            if p.exists():
                src = p
                day_used = d
                break
    if src is None:
        return _placeholder(fig_path, "fig11: no spectrogram available")
    arr = np.load(src, allow_pickle=False)
    spec = arr["spectrogram"]   # (n_t, 2, 256)
    t_s = arr["t_s"]
    freq_l1 = arr["freq_mhz_l1"]
    # Second RF block's frequency axis has been renamed twice in this
    # codebase: `freq_mhz_l2` (very early), `freq_mhz_l2l5` (legacy on
    # the existing .npz cache), and `freq_mhz_l5` (current spectrum.py
    # output). Accept any of them so cached .npz files don't force a
    # 216-day spectrum re-run.
    for _k in ("freq_mhz_l5", "freq_mhz_l2l5", "freq_mhz_l2"):
        if _k in arr.files:
            freq_l2 = arr[_k]
            break
    else:
        raise KeyError(
            f"No L5/L2 frequency axis in {src} (available keys: {list(arr.files)})"
        )
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # t_s is unix seconds as float64 (spectrum.py emits t_ns/1e9).
    # NumPy refuses to cast float→datetime64 directly, so go via int64.
    t_dt = np.asarray(t_s, dtype=np.int64).astype("datetime64[s]")
    import matplotlib.dates as mdates  # noqa: PLC0415
    for ax, block, freq, title in (
        (axes[0], spec[:, 0, :], freq_l1, "L1 band"),
        (axes[1], spec[:, 1, :], freq_l2, "L5 band"),
    ):
        im = ax.pcolormesh(t_dt, freq, block.T, cmap="magma", shading="auto")
        ax.set_ylabel("Freq (MHz)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax,
                     label="Spectrum amplitude (UBX-MON-SPAN raw, 0–255)")
    axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[1].set_xlabel(f"UTC time on {day_used[:4]}-{day_used[4:6]}-{day_used[6:8]}")
    fig.suptitle(f"MON-SPAN RF spectrum — day {day_used}")
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


# ---------------------------------------------------------------------------
# fig12 — receiver temperature + AGC + jamming
# ---------------------------------------------------------------------------


def fig12_temp_agc(out_dir: Path, cfg: dict, day: str | None = None) -> Path:
    fig_path = out_dir / "fig12_temp_agc_timeseries.pdf"
    from analysis._common import staged_path  # noqa: PLC0415

    days = [day] if day else list_days(cfg)
    sys_parts = []
    rf_parts = []
    for d in days:
        s = staged_path(d, "mon_sys", cfg)
        r = staged_path(d, "mon_rf", cfg)
        if s.exists():
            sys_parts.append(read_parquet(s))
        if r.exists():
            rf_parts.append(read_parquet(r))
    if not sys_parts or not rf_parts:
        return _placeholder(fig_path, "fig12: missing mon_sys or mon_rf")
    sys_df = pl.concat(sys_parts, how="vertical").sort("t_ns")
    rf_df = pl.concat(rf_parts, how="vertical").sort("t_ns")
    # Cruise-window filter: MON-SYS and MON-RF occasionally carry rows
    # with t_ns=0 (cold-start / corrupt frames) that stretch the x-axis
    # from 1970 to 2030+. Drop anything outside [2025-01-01, 2027-01-01).
    T_MIN_NS = 1_735_689_600_000_000_000
    T_MAX_NS = 1_798_761_600_000_000_000
    sys_df = sys_df.filter(
        (pl.col("t_ns") >= T_MIN_NS) & (pl.col("t_ns") < T_MAX_NS)
    )
    rf_df = rf_df.filter(
        (pl.col("t_ns") >= T_MIN_NS) & (pl.col("t_ns") < T_MAX_NS)
    )
    t_sys = sys_df["t_ns"].to_numpy().astype("datetime64[ns]")
    fig, axes = plt.subplots(
        4, 1, figsize=(10, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 3, 0.9]},
    )
    axes[0].plot(t_sys, sys_df["tempValue_C"].to_numpy(),
                 color="black", linewidth=0.6)
    axes[0].set_ylabel("Receiver T (°C)\n(tempState=unknown)")
    axes[0].set_ylim(25, 55)
    axes[0].set_title("Receiver temperature, AGC, and jamming")
    for blk_id, color, label in ((0, "#377EB8", "L1 RF block"), (1, "#E41A1C", "L2/L5 RF block")):
        sub = rf_df.filter(pl.col("blockId") == blk_id)
        if sub.is_empty():
            continue
        t = sub["t_ns"].to_numpy().astype("datetime64[ns]")
        axes[1].plot(t, sub["agcCnt"].to_numpy(), label=label, color=color, linewidth=0.6)
        axes[2].plot(t, sub["jamInd"].to_numpy(), label=label, color=color, linewidth=0.6)
    axes[1].set_ylabel("AGC counts")
    axes[1].legend(fontsize=7, loc="upper right")
    axes[2].set_ylabel("Jamming indicator")

    # Cruise-phase strip — separate panel at the bottom, doesn't overlap
    # any data line. Each phase is a coloured rectangle with a short label.
    # (label, start ISO, end ISO, hex colour)
    PHASES = [
        ("Trieste",      "2025-09-26", "2025-10-04", "#1f77b4"),
        ("Lyttelton",    "2025-11-20", "2025-11-28", "#2ca02c"),
        ("Antarctic 1",  "2025-12-02", "2025-12-17", "#d62728"),
        ("Lyttelton",    "2025-12-21", "2026-01-02", "#2ca02c"),
        ("Antarctic 2",  "2026-01-06", "2026-02-09", "#d62728"),
        ("Lyttelton",    "2026-02-28", "2026-03-09", "#2ca02c"),
        ("Trieste",      "2026-04-19", "2026-04-29", "#1f77b4"),
    ]
    phase_ax = axes[3]
    phase_ax.set_yticks([])
    phase_ax.set_ylim(0, 1)
    phase_ax.set_ylabel("Phase", fontsize=8)
    phase_ax.set_xlabel("UTC")
    for label, t0, t1, colour in PHASES:
        x0 = np.datetime64(t0)
        x1 = np.datetime64(t1)
        phase_ax.axvspan(x0, x1, color=colour, alpha=0.75, zorder=0)
        mid = x0 + (x1 - x0) // 2
        phase_ax.text(
            mid, 0.5, label,
            ha="center", va="center", fontsize=6.5, color="white",
            fontweight="bold",
            clip_on=True,
        )
    # Light grey background for the strip so unshaded ("transit") gaps
    # are visually distinct from the panels above.
    phase_ax.set_facecolor("#f5f5f5")
    # Suppress duplicate x-tick labels on the data panels.
    for ax in axes[:3]:
        ax.tick_params(labelbottom=False)

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    return fig_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _placeholder(path: Path, message: str) -> Path:
    log.warning(message)
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10, color="red")
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

FIGURES: dict[str, Callable[..., Path]] = {
    "fig01": fig01_cruise_track,
    "fig05": fig05_multipath,
    "fig11": fig11_rf_waterfall,
    "fig12": fig12_temp_agc,
}


@click.command()
@click.option("--day", default=None, help="Restrict per-day figures to this YYYYMMDD")
@click.option("--only", default=None, help="Comma-separated figXX list")
@click.option("--quick", is_flag=True, default=False, help="Smoke-test mode (per-day figures only)")
@click.option("--central-lon", default=160.0, type=float, show_default=True,
              help="Robinson projection central longitude for fig01 "
                    "(160=Pacific-centred, 0=Greenwich, -90=Americas, 180=dateline)")
@click.option("--decimate-min", default=5, type=int, show_default=True,
              help="Per-N-minute dot decimation for fig01")
def main(day: str | None, only: str | None, quick: bool,
          central_lon: float, decimate_min: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    apply_style()
    cfg = load_config()
    out = figures_dir(cfg) / "output"
    out.mkdir(parents=True, exist_ok=True)
    keys = list(FIGURES.keys())
    if only:
        keys = [k.strip() for k in only.split(",")]
    if quick:
        # Skip the cruise-wide figures
        keys = [k for k in keys if k != "fig01"]
    for k in keys:
        fn = FIGURES.get(k)
        if not fn:
            log.warning("Unknown figure %s", k)
            continue
        try:
            if k in {"fig03", "fig04", "fig05", "fig11", "fig12"}:
                p = fn(out, cfg, day=day)
            elif k == "fig01":
                p = fn(out, cfg, decimate_min=decimate_min, central_lon=central_lon)
            else:
                p = fn(out, cfg)
            log.info("%s -> %s", k, p)
        except Exception as exc:  # noqa: BLE001
            log.exception("Figure %s failed: %s", k, exc)


if __name__ == "__main__":
    main()
