"""Interactive HTML cruise-track visualisation.

Folium-based, works without an internet connection (uses the OSM tile
URL by default but degrades to a blank tile if the server is offline).
Per-day segments are coloured separately so missing days are visible
as gaps instead of being painted as straight lines across the globe.

CLI:
    python -m figures.cruise_track_html
    python -m figures.cruise_track_html --out /tmp/cruise.html

Outputs:
    work/figures/output/cruise_track.html
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import polars as pl

from analysis._common import figures_dir, load_config, read_parquet, resolve_path

log = logging.getLogger(__name__)


def _split_on_antimeridian(coords: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    """Split a (lat, lon) sequence wherever consecutive longitudes jump by
    >180° (i.e. the track crosses the antimeridian). Without this, Leaflet
    draws a single straight polyline from +179° back to −179° across the
    whole globe.
    """
    if len(coords) < 2:
        return [coords] if coords else []
    parts: list[list[tuple[float, float]]] = [[coords[0]]]
    for i in range(1, len(coords)):
        prev_lat, prev_lon = coords[i - 1]
        lat, lon = coords[i]
        if abs(lon - prev_lon) > 180.0:
            parts.append([])
        parts[-1].append((lat, lon))
    return [p for p in parts if len(p) >= 2]


def _decimated_track(df: pl.DataFrame) -> pl.DataFrame:
    """1-point-per-minute decimation of the full track.

    Defensive against per-day files that contain rows with bogus t_ns
    (≤ 2017 or ≥ 2030 — typically pre-fix epochs that slipped past earlier
    filters), or with all-zero coordinates.
    """
    # Cruise window guard: anything outside 2017–2030 is bogus. The
    # u-blox receiver clock is sometimes left at 1970 / week-rollover
    # values until first GPS-time sync; those rows produce huge time
    # gaps and decimation collisions.
    T_MIN = 1_735_689_600 * 1_000_000_000   # 2025-01-01 UTC — pre-cruise
    T_MAX = 1_798_761_600 * 1_000_000_000   # 2027-01-01 UTC — post-cruise
    df = df.filter(
        (pl.col("t_ns") > T_MIN) & (pl.col("t_ns") < T_MAX)
        & ~((pl.col("lat") == 0.0) & (pl.col("lon") == 0.0))
    )
    return (
        df.with_columns((pl.col("t_ns") // 60_000_000_000).alias("_min"))
        .group_by("_min")
        .agg([
            pl.col("lat").mean(),
            pl.col("lon").mean(),
            pl.col("t_ns").first(),
        ])
        .sort("_min")
    )


def _continuous_segments(
    df: pl.DataFrame,
    *,
    gap_minutes: int = 30,
    max_speed_kn: float = 50.0,
) -> list[list[tuple[float, float]]]:
    """Split the whole track into continuous segments.

    A new segment starts whenever:
    * the next minute-bin is more than ``gap_minutes`` after the previous one
      (real data-loss gap), OR
    * the implied speed between consecutive points exceeds ``max_speed_kn``
      knots — no surface vessel exceeds this, so the jump must be a stale
      / out-of-order coordinate.

    Each continuous segment is then further split on antimeridian crossings
    (>180° longitude jumps), which would otherwise draw a straight line
    across the entire map in Leaflet.
    """
    import numpy as np  # noqa: PLC0415

    if df.is_empty():
        return []
    decim = _decimated_track(df)
    t = decim["t_ns"].to_numpy()
    lats = decim["lat"].to_numpy()
    lons = decim["lon"].to_numpy()
    if len(t) < 2:
        return []
    gap_ns = gap_minutes * 60 * 1_000_000_000
    # Haversine distance for each consecutive pair (km)
    rlat1, rlat2 = np.deg2rad(lats[:-1]), np.deg2rad(lats[1:])
    dlat = rlat2 - rlat1
    dlon = np.deg2rad(lons[1:] - lons[:-1])
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    dist_km = 2 * 6371.0088 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    dt_h = np.maximum(np.diff(t) / 1e9 / 3600.0, 1e-6)
    speed_kn = (dist_km / dt_h) / 1.852
    breaks = np.where(
        (np.diff(t) > gap_ns) | (speed_kn > max_speed_kn)
    )[0] + 1

    coarse: list[list[tuple[float, float]]] = [[]]
    cut = set(breaks.tolist())
    for i in range(len(t)):
        if i in cut:
            coarse.append([])
        coarse[-1].append((float(lats[i]), float(lons[i])))
    out: list[list[tuple[float, float]]] = []
    for seg in coarse:
        if len(seg) >= 2:
            out.extend(_split_on_antimeridian(seg))
    return out


def render(out_path: Path, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    track_path = resolve_path(cfg["paths"]["derived"]) / "track" / "track.all.parquet"
    if not track_path.exists():
        raise FileNotFoundError(f"Missing {track_path} — run analysis.trajectory --aggregate first")
    df = read_parquet(track_path)
    if df.is_empty():
        raise RuntimeError("track.all.parquet is empty")

    # Defensive: drop any NaN/inf coordinates and obvious outliers
    df = df.filter(
        pl.col("lat").is_finite() & pl.col("lon").is_finite()
        & (pl.col("lat").abs() <= 90.0) & (pl.col("lon").abs() <= 180.0)
    )
    if df.is_empty():
        raise RuntimeError("No finite coordinates in track.all.parquet")

    try:
        import folium  # noqa: PLC0415
        from folium.plugins import MarkerCluster, AntPath  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "folium not installed. Add to deps: pip install folium"
        ) from exc

    # Centre on the median position
    lat0 = float(np.median(df["lat"].to_numpy()))
    lon0 = float(np.median(df["lon"].to_numpy()))
    m = folium.Map(
        location=[lat0, lon0],
        zoom_start=2,
        tiles="OpenStreetMap",
        prefer_canvas=True,
    )

    # Build continuous segments split only on real time gaps + antimeridian.
    # The track stays as one polyline through midnight UTC, so the cruise
    # looks continuous even though daily files boundary-split the data.
    segments = _continuous_segments(df, gap_minutes=30)
    if not segments:
        raise RuntimeError("No track segments built")
    log.info("Drawing %d continuous segments (gap-split on >30min gaps + antimeridian)", len(segments))
    fg_segments = folium.FeatureGroup(name="Cruise track")
    for j, coords in enumerate(segments):
        folium.PolyLine(
            coords,
            color="#D62728",
            weight=2.0,
            opacity=0.9,
            no_clip=True,
            popup=f"Segment {j + 1} — {len(coords)} pts",
        ).add_to(fg_segments)
    fg_segments.add_to(m)

    # Mark first and last points
    first_lat, first_lon = segments[0][0]
    last_lat, last_lon = segments[-1][-1]
    folium.Marker(
        [first_lat, first_lon],
        popup="Start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)
    folium.Marker(
        [last_lat, last_lon],
        popup="End",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)

    # Day-start markers (clustered to keep the map fast). One per UTC date.
    cluster = MarkerCluster(name="Day starts").add_to(m)
    daily_starts = (
        df.with_columns(pl.from_epoch(pl.col("t_ns"), time_unit="ns").dt.date().alias("_d"))
        .group_by("_d", maintain_order=True)
        .agg([pl.col("lat").first(), pl.col("lon").first()])
        .sort("_d")
    )
    for d_, lat, lon in zip(daily_starts["_d"].to_list(),
                              daily_starts["lat"].to_list(),
                              daily_starts["lon"].to_list()):
        folium.CircleMarker(
            [lat, lon],
            radius=2,
            color="black",
            fill=True,
            fill_opacity=0.7,
            popup=str(d_),
        ).add_to(cluster)

    folium.LayerControl().add_to(m)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    log.info("Wrote %s", out_path)
    return out_path


@click.command()
@click.option("--out", default=None, type=click.Path(),
              help="Output HTML path; default work/figures/output/cruise_track.html")
def main(out: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    out_path = Path(out) if out else (figures_dir(cfg) / "output" / "cruise_track.html")
    render(out_path, cfg)


if __name__ == "__main__":
    main()
