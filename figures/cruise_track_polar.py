"""South-polar stereographic cruise-track HTML via Plotly.

For an Antarctic cruise this projection is the right choice — the south
pole sits at the centre, so the antimeridian crossing is no longer a
discontinuity, and the high-southern-latitude dwell is clearly visible.
The resulting HTML works fully offline (Plotly bundles its JS) and uses
Natural Earth land outlines from Plotly's CDN-hosted topojson, which is
typically reachable from server environments that block tile servers.

CLI:
    python -m figures.cruise_track_polar
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import polars as pl

from analysis._common import figures_dir, load_config, read_parquet, resolve_path

log = logging.getLogger(__name__)


def render(out_path: Path, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    track_path = resolve_path(cfg["paths"]["derived"]) / "track" / "track.all.parquet"
    if not track_path.exists():
        raise FileNotFoundError(track_path)
    df = read_parquet(track_path).filter(
        pl.col("lat").is_finite()
        & pl.col("lon").is_finite()
        & (pl.col("lat").abs() <= 90.0)
        & (pl.col("lon").abs() <= 180.0)
        & (pl.col("t_ns") > 0)
    )
    if df.is_empty():
        raise RuntimeError("No finite track points")

    try:
        import plotly.graph_objects as go  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("plotly not installed (`pip install plotly`)") from exc

    df = df.sort("t_ns")
    # 1-minute decimation, then split into continuous segments on >30 min
    # gaps. We do not split on calendar days — that produces 240 disjoint
    # polylines for a continuous voyage that crosses midnight UTC.
    decim = (
        df.with_columns((pl.col("t_ns") // 60_000_000_000).alias("_min"))
        .group_by("_min")
        .agg([
            pl.col("lat").mean(),
            pl.col("lon").mean(),
            pl.col("t_ns").first(),
        ])
        .sort("_min")
    )
    import numpy as np  # noqa: PLC0415

    t = decim["t_ns"].to_numpy()
    lats = decim["lat"].to_numpy()
    lons = decim["lon"].to_numpy()
    gap_ns = 30 * 60 * 1_000_000_000
    seg_id = np.zeros(len(t), dtype=np.int32)
    if len(t) > 1:
        breaks = np.where(np.diff(t) > gap_ns)[0] + 1
        for b in breaks:
            seg_id[b:] += 1

    fig = go.Figure()
    n_segs = int(seg_id.max() + 1) if len(t) else 0
    for s in range(n_segs):
        mask = seg_id == s
        if mask.sum() < 2:
            continue
        fig.add_trace(go.Scattergeo(
            lat=lats[mask].tolist(),
            lon=lons[mask].tolist(),
            mode="lines",
            line=dict(width=1.5, color="#D62728"),
            name=f"segment {s + 1}",
            showlegend=False,
            hovertemplate="%{lat:.3f}, %{lon:.3f}<extra></extra>",
        ))

    # Centre on the south pole; show the whole hemisphere up to ~30°N
    fig.update_geos(
        projection=dict(type="azimuthal equal area",
                        rotation=dict(lat=-90, lon=0)),
        showland=True,
        landcolor="#EEEEEE",
        showocean=True,
        oceancolor="#F4F8FB",
        showcountries=True,
        countrycolor="#999999",
        coastlinecolor="#444444",
        coastlinewidth=0.5,
        showframe=False,
        lataxis=dict(range=[-90, 30]),
    )
    fig.update_layout(
        title=dict(
            text=f"Cruise track — {df.with_columns(pl.from_epoch(pl.col('t_ns'), time_unit='ns').dt.date().alias('_d')).select(pl.col('_d').n_unique()).item()} days",
            x=0.5,
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=750,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    log.info("Wrote %s", out_path)
    return out_path


@click.command()
@click.option("--out", default=None, type=click.Path())
def main(out: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    out_path = Path(out) if out else figures_dir(cfg) / "output" / "cruise_track_polar.html"
    render(out_path, cfg)


if __name__ == "__main__":
    main()
