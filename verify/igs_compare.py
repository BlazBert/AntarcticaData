"""Compare ship-receiver position at port stops against the published
location of the nearest IGS station.

ESSD requires *external* validation, not just internal consistency.
This script picks every port-stop window (ship stationary, gSpeed < 0.1 m/s
for ≥ 30 min within ``radius_km`` of a known port), computes the median
position over that window, and compares to the **published port-dock
coordinate** (or to a nearby IGS station coordinate, if you provide one).

Output:
    tables/T_external_validation.csv
        port, day, n_epochs, ref_lat, ref_lon, ship_med_lat, ship_med_lon,
        bias_horiz_m, bias_north_m, bias_east_m, ship_height_med_m,
        ref_label

Usage::

    cd /home/jovyan/Projects/gps-data/data/code

    # Default: use the port coordinates from config/ports.yaml
    python -m verify.igs_compare

    # Or: explicitly compare to IGS reference points (provide a YAML
    # extending ports.yaml with the keys below; see ``EXAMPLE_REFS`` block).
    python -m verify.igs_compare --refs config/igs_refs.yaml

The "reference" coordinates are user-supplied because:

* IGS station ECEF coordinates change daily by a few mm; we only need
  cm-level reference. The published station "approximate" coordinates
  from the IGS site log are sufficient.
* The nearest IGS site to Lyttelton (MQZG, Mt John, NZ) is ~250 km away
  and not co-located with any port-stop. For Lyttelton we use the
  *port-dock* coordinate as the reference instead.
* Trieste has multiple geodetic markers in the harbour; pick the one
  closest to the antenna mount point.

Reference points should be provided as a YAML list of dicts with at
least ``name``, ``lat``, ``lon``, ``radius_km``, ``label`` (e.g.
``"IGS:TRIE"`` or ``"chart:Lyttelton-pier-3"``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click
import numpy as np
import polars as pl
import yaml

from analysis._common import (
    derived_dir,
    haversine_km,
    list_days,
    load_config,
    read_parquet,
    resolve_path,
    tables_dir,
)

log = logging.getLogger(__name__)

R_EARTH_M = 6371008.8
DEFAULT_RADIUS_KM = 5.0
DEFAULT_DWELL_MIN = 30
DEFAULT_SPEED_THR = 0.1


# Reference list. Replace ``lat``/``lon`` with the actual published port
# dock coordinate or IGS site approximate-position from the SINEX
# header.  ``label`` should describe the reference origin; this lands in
# the CSV ``ref_label`` column for traceability.
EXAMPLE_REFS: list[dict] = [
    {
        "name": "Trieste (Punto Franco)",
        "lat": 45.6133,
        "lon": 13.8090,
        "radius_km": 5.0,
        "label": "chart:Trieste-PuntoFranco",
    },
    {
        "name": "Lyttelton (NZ)",
        "lat": -43.6062,
        "lon": 172.7164,
        "radius_km": 5.0,
        "label": "chart:Lyttelton-port",
    },
    {
        "name": "Mario Zucchelli",
        "lat": -74.6906,
        "lon": 164.1144,
        "radius_km": 3.0,
        "label": "site_log:MZS1",
    },
]


def _enu_offset(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    """Local-tangent-plane (north, east) metres of (lat, lon) from (lat0, lon0)."""
    dlat = np.deg2rad(lat - lat0)
    dlon = np.deg2rad(lon - lon0)
    n = dlat * R_EARTH_M
    e = dlon * R_EARTH_M * np.cos(np.deg2rad(lat0))
    return float(n), float(e)


def _port_windows(track: pl.DataFrame, ref: dict) -> list[tuple[int, int]]:
    """Find contiguous (start_idx, end_idx) windows where the ship is
    within ``radius_km`` of the reference and stationary."""
    if track.is_empty():
        return []
    lat = track["lat"].to_numpy()
    lon = track["lon"].to_numpy()
    speed = track["gSpeed_m_s"].fill_null(np.nan).to_numpy()
    t_ns = track["t_ns"].to_numpy()
    dist = haversine_km(
        np.full_like(lat, ref["lat"]),
        np.full_like(lon, ref["lon"]),
        lat, lon,
    )
    in_window = (dist <= float(ref.get("radius_km", DEFAULT_RADIUS_KM))) & (
        np.nan_to_num(speed, nan=1e9) < DEFAULT_SPEED_THR
    )
    if not in_window.any():
        return []
    edges = np.diff(in_window.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0])
    if in_window[0]:
        starts.insert(0, 0)
    if in_window[-1]:
        ends.append(in_window.size - 1)
    out = []
    min_dwell_ns = DEFAULT_DWELL_MIN * 60 * 1_000_000_000
    for s, e in zip(starts, ends):
        if t_ns[e] - t_ns[s] >= min_dwell_ns:
            out.append((int(s), int(e)))
    return out


def compare_one_day(day: str, refs: list[dict], cfg: dict) -> list[dict]:
    track_path = derived_dir(cfg) / "track" / f"{day}.track.parquet"
    if not track_path.exists():
        return []
    track = read_parquet(track_path)
    if track.is_empty():
        return []
    rows: list[dict] = []
    for ref in refs:
        windows = _port_windows(track, ref)
        for s, e in windows:
            sub = track.slice(s, e - s + 1)
            ship_lat = float(sub["lat"].median() or float("nan"))
            ship_lon = float(sub["lon"].median() or float("nan"))
            ship_h = float(sub["hMSL_m"].median() or float("nan"))
            n, eo = _enu_offset(ref["lat"], ref["lon"], ship_lat, ship_lon)
            horiz = float(np.hypot(n, eo))
            rows.append({
                "port": ref["name"],
                "day": day,
                "n_epochs": int(sub.height),
                "ref_lat": float(ref["lat"]),
                "ref_lon": float(ref["lon"]),
                "ship_med_lat": ship_lat,
                "ship_med_lon": ship_lon,
                "bias_north_m": n,
                "bias_east_m": eo,
                "bias_horiz_m": horiz,
                "ship_height_med_m": ship_h,
                "ref_label": ref.get("label", ""),
            })
    return rows


@click.command()
@click.option("--refs", default=None, type=click.Path(),
              help="YAML file with list[dict(name, lat, lon, radius_km, label)]; "
                   "default uses a built-in 3-port list.")
@click.option("--out", default=None, type=click.Path(),
              help="Output CSV path; default work/tables/T_external_validation.csv")
def main(refs: str | None, out: str | None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    if refs:
        ref_list = yaml.safe_load(Path(refs).read_text())
    else:
        ref_list = EXAMPLE_REFS
    log.info("Reference points: %s", [r["name"] for r in ref_list])
    days = list_days(cfg)
    all_rows: list[dict] = []
    for d in days:
        all_rows.extend(compare_one_day(d, ref_list, cfg))
    out_path = Path(out) if out else (tables_dir(cfg) / "T_external_validation.csv")
    if not all_rows:
        log.warning("No port-stop windows matched the reference list. "
                    "Check radius_km and that the ship was actually at these ports.")
        out_path.write_text("port,day,n_epochs,ref_lat,ref_lon,ship_med_lat,ship_med_lon,"
                            "bias_north_m,bias_east_m,bias_horiz_m,ship_height_med_m,ref_label\n")
        return
    df = pl.DataFrame(all_rows).sort(["port", "day"])
    df.write_csv(out_path)
    log.info("Wrote %s (%d port-day rows)", out_path, df.height)
    summary = df.group_by("port").agg([
        pl.len().alias("n_days"),
        pl.col("bias_horiz_m").median().alias("bias_horiz_med_m"),
        pl.col("bias_horiz_m").max().alias("bias_horiz_max_m"),
        pl.col("bias_north_m").median().alias("bias_north_med_m"),
        pl.col("bias_east_m").median().alias("bias_east_med_m"),
    ]).sort("port")
    print()
    print(summary)


if __name__ == "__main__":
    main()
