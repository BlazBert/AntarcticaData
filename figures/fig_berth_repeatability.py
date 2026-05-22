"""Within-berth horizontal repeatability bar chart.

Reads ``work/tables/T_external_validation.csv`` and emits
``work/figures/output/fig_berth_repeatability.pdf``.

The CSV contains one row per *port-day-window* — a port-stop window
where the ship was within ``radius_km`` of a reference coordinate and
stationary. Each port in the CSV is generally **not** a single pier:
"Trieste" contains both the departure and return piers (1.5 km apart);
"Mario Zucchelli" contains multiple anchorages over ~10 km. A naive
``std()`` over all visits to one port yields a spread of the port
*area* rather than receiver repeatability at a fixed berth.

This script sub-clusters each port's visits into physical berths using
a greedy distance-threshold rule (``cluster_eps_m``, default 50 m) on
the per-visit median position, then reports the cross-visit σ within
each berth that has ``min_visits >= 3``.

CLI::

    python -m figures.fig_berth_repeatability
    python -m figures.fig_berth_repeatability --cluster-eps-m 100
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis._common import figures_dir, load_config, tables_dir

log = logging.getLogger(__name__)

R_EARTH_M = 6_371_008.8


def _enu_metres(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    """Return (n, 2) array of (north, east) metres from a reference point."""
    dlat = np.deg2rad(lat - lat0)
    dlon = np.deg2rad(lon - lon0)
    n = dlat * R_EARTH_M
    e = dlon * R_EARTH_M * np.cos(np.deg2rad(lat0))
    return np.column_stack([n, e])


def _greedy_cluster(points_m: np.ndarray, eps_m: float) -> np.ndarray:
    """Greedy single-pass clustering: assign each point to the nearest
    existing cluster centroid within ``eps_m``, otherwise start a new
    cluster. Returns an int label per point."""
    labels = np.full(points_m.shape[0], -1, dtype=np.int64)
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    for i, p in enumerate(points_m):
        if not centroids:
            centroids.append(p.copy()); counts.append(1); labels[i] = 0; continue
        dists = np.linalg.norm(np.asarray(centroids) - p, axis=1)
        j = int(np.argmin(dists))
        if dists[j] <= eps_m:
            # Online mean update
            counts[j] += 1
            centroids[j] += (p - centroids[j]) / counts[j]
            labels[i] = j
        else:
            centroids.append(p.copy()); counts.append(1); labels[i] = len(centroids) - 1
    return labels


def compute_berths(df: pl.DataFrame, *, eps_m: float, min_visits: int) -> pl.DataFrame:
    """One row per (port, sub-berth) with n_visits, σ_horiz_m, bias to ref."""
    out_rows = []
    for port, sub in df.group_by("port", maintain_order=True):
        port = port[0] if isinstance(port, tuple) else port
        lat = sub["ship_med_lat"].to_numpy()
        lon = sub["ship_med_lon"].to_numpy()
        # Cluster around the port-mean local frame
        lat0 = float(np.mean(lat)); lon0 = float(np.mean(lon))
        pts = _enu_metres(lat, lon, lat0, lon0)
        labels = _greedy_cluster(pts, eps_m)
        for c in np.unique(labels):
            mask = labels == c
            if int(mask.sum()) < min_visits:
                continue
            cl_pts = pts[mask]
            sigma = float(np.sqrt(cl_pts[:, 0].var() + cl_pts[:, 1].var()))
            bias_med = float(np.median(sub["bias_horiz_m"].to_numpy()[mask]))
            cl_lat = float(np.mean(lat[mask])); cl_lon = float(np.mean(lon[mask]))
            out_rows.append({
                "port": port,
                "berth_lat": cl_lat,
                "berth_lon": cl_lon,
                "n_visits": int(mask.sum()),
                "sigma_horiz_m": sigma,
                "bias_horiz_median_m": bias_med,
            })
    if not out_rows:
        return pl.DataFrame()
    return pl.DataFrame(out_rows).sort("sigma_horiz_m")


def _classify_region(port: str) -> str:
    if "Trieste" in port:
        return "Mediterranean"
    if "Lyttelton" in port or "Wellington" in port:
        return "Lyttelton (NZ)"
    if "Zucchelli" in port or "Antarctic" in port or "Ross" in port:
        return "Antarctic"
    if "Panama" in port:
        return "Tropics"
    return "Other"


REGION_COLOR = {
    "Mediterranean":   "#1F77B4",
    "Lyttelton (NZ)":  "#2CA02C",
    "Antarctic":       "#D62728",
    "Tropics":         "#FF7F0E",
    "Other":           "#7F7F7F",
}


def render(out_path: Path, *, eps_m: float, min_visits: int, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    csv_path = tables_dir(cfg) / "T_external_validation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} missing — run verify.igs_compare first")
    df = pl.read_csv(csv_path)
    if df.is_empty():
        raise RuntimeError(f"{csv_path} is empty")

    per_berth = compute_berths(df, eps_m=eps_m, min_visits=min_visits)
    if per_berth.is_empty():
        raise RuntimeError(
            f"No sub-berth clusters with ≥{min_visits} visits at eps_m={eps_m}. "
            "Lower min-visits, or raise --cluster-eps-m, to allow more lumping."
        )
    log.info("Sub-berth clusters (eps=%.0f m, min_visits=%d):", eps_m, min_visits)
    for r in per_berth.iter_rows(named=True):
        log.info("  %-25s lat=%.4f lon=%.4f  n=%2d  σ=%.3f m",
                  r["port"], r["berth_lat"], r["berth_lon"], r["n_visits"], r["sigma_horiz_m"])

    def _fmt_latlon(lat: float, lon: float) -> str:
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        return f"{abs(lat):.3f}°{ns}, {abs(lon):.3f}°{ew}"

    labels = [f"{r['port']} ({_fmt_latlon(r['berth_lat'], r['berth_lon'])})"
              for r in per_berth.iter_rows(named=True)]
    sigma = per_berth["sigma_horiz_m"].to_numpy()
    nvis = per_berth["n_visits"].to_numpy()
    regions = [_classify_region(r["port"]) for r in per_berth.iter_rows(named=True)]
    colors = [REGION_COLOR[r] for r in regions]

    fig, ax = plt.subplots(figsize=(9.5, max(3.8, 0.35 * len(labels))))
    y = np.arange(len(labels))
    ax.barh(y, sigma, color=colors, edgecolor="black", linewidth=0.4)
    for i, (s, n) in enumerate(zip(sigma, nvis)):
        ax.text(s + max(0.005, 0.01 * sigma.max()), i,
                 f"  σ={s:.3f} m  (n={int(n)})",
                 va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Horizontal repeatability σ (m)")
    ax.set_title(f"Within-berth horizontal repeatability of receiver position\n"
                  f"(σ of per-visit median fix; greedy {eps_m:.0f} m sub-clusters, "
                  f"≥{min_visits} visits per cluster)")
    ax.invert_yaxis()
    ax.grid(axis="x", linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in REGION_COLOR.values()]
    # Legend pinned outside the axes (bottom-centre) so it cannot overlap
    # the long per-bar σ/n annotations on the right of the chart.
    ax.legend(
        handles, list(REGION_COLOR.keys()),
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncols=len(REGION_COLOR), fontsize=8, framealpha=0.9,
        title="Region", title_fontsize=8,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s (%d berths)", out_path, len(labels))
    return out_path


@click.command()
@click.option("--out", default=None, type=click.Path(),
              help="Output PDF path; default work/figures/output/fig_berth_repeatability.pdf")
@click.option("--cluster-eps-m", default=50.0, type=float, show_default=True,
              help="Greedy clustering radius (m). Visits within this distance "
                   "of an existing cluster centroid are assigned to it.")
@click.option("--min-visits", default=3, type=int, show_default=True,
              help="Drop clusters with fewer than this many visits.")
def main(out: str | None, cluster_eps_m: float, min_visits: int) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    out_path = Path(out) if out else (figures_dir(cfg) / "output" / "fig_berth_repeatability.pdf")
    render(out_path, eps_m=cluster_eps_m, min_visits=min_visits, cfg=cfg)


if __name__ == "__main__":
    main()
