"""Antarctic-stations focused analysis.

The cruise spent multiple days inside the Antarctic Circle (lat < −66.56°)
and likely passed near several research stations. This script:

1. Lists every staged day where median latitude < −60° (Sub-Antarctic)
   or < −66.56° (Antarctic Circle), and reports nearby known stations.
2. Cross-references with MON-SPAN event counts and L1/L2/L5 jamInd
   peaks for each Antarctic day.
3. Highlights days where MON-SPAN events spike *without* matching jamInd
   excursion — likely candidates for ionospheric/scintillation effects
   distinct from terrestrial RFI.

Known Antarctic stations within plausible reach of the cruise (edit
``ANTARCTIC_STATIONS`` to add more):

* Mario Zucchelli — Italian, Terra Nova Bay (Ross Sea)
* Concordia (Dome C) — French/Italian, inland — unlikely from a ship
* McMurdo / Scott Base — US/NZ, Ross Sea
* Casey — Australian, East Antarctica
* Davis — Australian, East Antarctica
* Mawson — Australian, East Antarctica
* Syowa — Japanese, East Antarctica
* SANAE — South African, Queen Maud Land
* Neumayer III — German, Ekström Ice Shelf
* Halley VI — British, Brunt Ice Shelf

CLI:

    python -m verify.antarctic_focus
    python -m verify.antarctic_focus --csv-out ../work/tables/T_antarctic.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import polars as pl

from analysis._common import (
    derived_dir,
    haversine_km,
    list_days,
    load_config,
    read_parquet,
    staged_path,
)

log = logging.getLogger(__name__)


ANTARCTIC_STATIONS = [
    {"name": "Mario Zucchelli",  "lat": -74.6906, "lon": 164.1144, "country": "IT"},
    {"name": "Concordia (Dome C)","lat": -75.1000, "lon": 123.3500, "country": "FR/IT"},
    {"name": "McMurdo",          "lat": -77.8419, "lon": 166.6863, "country": "US"},
    {"name": "Scott Base",       "lat": -77.8485, "lon": 166.7676, "country": "NZ"},
    {"name": "Casey",            "lat": -66.2818, "lon": 110.5276, "country": "AU"},
    {"name": "Davis",            "lat": -68.5764, "lon": 77.9689,  "country": "AU"},
    {"name": "Mawson",           "lat": -67.6028, "lon": 62.8714,  "country": "AU"},
    {"name": "Syowa",            "lat": -69.0061, "lon": 39.5908,  "country": "JP"},
    {"name": "SANAE IV",         "lat": -71.6735, "lon":  -2.8500, "country": "ZA"},
    {"name": "Neumayer III",     "lat": -70.6667, "lon":  -8.2667, "country": "DE"},
    {"name": "Halley VI",        "lat": -75.5792, "lon": -26.6543, "country": "UK"},
    {"name": "Belgrano II",      "lat": -77.8742, "lon": -34.6258, "country": "AR"},
    {"name": "Rothera",          "lat": -67.5667, "lon": -68.1333, "country": "UK"},
    {"name": "Frei (Pres. F.)",  "lat": -62.2014, "lon": -58.9619, "country": "CL"},
    {"name": "Esperanza",        "lat": -63.3982, "lon": -56.9967, "country": "AR"},
]


def _nearest_station(lat: float, lon: float, max_km: float = 200.0) -> tuple[str, float]:
    best_name = ""
    best_d = float("inf")
    for s in ANTARCTIC_STATIONS:
        d = float(haversine_km(
            np.array([lat]), np.array([lon]),
            np.array([s["lat"]]), np.array([s["lon"]]),
        )[0])
        if d < best_d:
            best_d = d
            best_name = s["name"]
    return (best_name, best_d) if best_d < max_km else ("", best_d)


def _row_for_day(day: str, cfg: dict) -> dict | None:
    track_p = derived_dir(cfg) / "track" / f"{day}.track.parquet"
    rf_p = staged_path(day, "mon_rf", cfg)
    rfi_p = derived_dir(cfg) / "spectrum" / f"{day}.rfi.parquet"
    if not track_p.exists() or not rf_p.exists():
        return None
    tr = read_parquet(track_p)
    if tr.is_empty():
        return None
    lat_med = float(tr["lat"].median() or 0.0)
    if lat_med > -60.0:
        return None     # not Sub-Antarctic
    lon_med = float(tr["lon"].median() or 0.0)
    speed_avg = float(tr["gSpeed_m_s"].mean() or 0.0)
    nearest, d_km = _nearest_station(lat_med, lon_med)

    rf = read_parquet(rf_p)
    rfl1 = rf.filter(pl.col("blockId") == 0)
    rfl5 = rf.filter(pl.col("blockId") == 1)

    rfi_count = 0
    rfi_l1 = 0
    rfi_l5 = 0
    if rfi_p.exists():
        rfi = read_parquet(rfi_p)
        rfi_count = rfi.height
        rfi_l1 = int((rfi["rf_block"] == 0).sum())
        rfi_l5 = int((rfi["rf_block"] == 1).sum())

    return {
        "day": day,
        "lat": lat_med,
        "lon": lon_med,
        "speed_m_s": speed_avg,
        "in_antarctic_circle": lat_med < -66.56361,
        "nearest_station": nearest,
        "station_dist_km": round(d_km, 1) if nearest else None,
        "L1_jam_max": int(rfl1["jamInd"].max() or 0) if not rfl1.is_empty() else 0,
        "L2L5_jam_max": int(rfl5["jamInd"].max() or 0) if not rfl5.is_empty() else 0,
        "L1_jam_gt60_n": int((rfl1["jamInd"] > 60).sum()) if not rfl1.is_empty() else 0,
        "L1_jam_gt100_n": int((rfl1["jamInd"] > 100).sum()) if not rfl1.is_empty() else 0,
        "rfi_events_total": rfi_count,
        "rfi_events_L1": rfi_l1,
        "rfi_events_L2L5": rfi_l5,
    }


def _scintillation_candidate(r: dict) -> bool:
    """Heuristic: high MON-SPAN event count without matching jamInd excursion."""
    return (
        r["rfi_events_total"] > 5000
        and r["L1_jam_max"] < 60
        and r["L2L5_jam_max"] < 110
    )


def _print(rows: list[dict]) -> None:
    print(f"{'day':<10}{'lat':>9}{'lon':>10}{'spd':>6}{'AntCirc':>8}"
          f"{'station':<22}{'dist_km':>8}"
          f"{'L1max':>7}{'L1>60':>7}{'L1>100':>8}{'L2L5max':>9}"
          f"{'rfi_evt':>9}{'note':>14}")
    for r in rows:
        note = "scint?" if _scintillation_candidate(r) else ""
        print(
            f"{r['day']:<10}{r['lat']:>9.3f}{r['lon']:>10.3f}{r['speed_m_s']:>6.2f}"
            f"{('Y' if r['in_antarctic_circle'] else 'N'):>8}"
            f"{(r['nearest_station'] or '-'):<22}"
            f"{(r['station_dist_km'] if r['station_dist_km'] is not None else '-'):>8}"
            f"{r['L1_jam_max']:>7}{r['L1_jam_gt60_n']:>7}{r['L1_jam_gt100_n']:>8}"
            f"{r['L2L5_jam_max']:>9}{r['rfi_events_total']:>9}{note:>14}"
        )


@click.command()
@click.option("--csv-out", default=None, type=click.Path(),
              help="Optional CSV destination")
def main(csv_out: str | None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    days = list_days(cfg)
    rows = []
    for d in days:
        r = _row_for_day(d, cfg)
        if r is not None:
            rows.append(r)
    rows.sort(key=lambda x: x["day"])
    print(f"# {len(rows)} sub-Antarctic days (lat < −60°)\n")
    _print(rows)
    n_circle = sum(1 for r in rows if r["in_antarctic_circle"])
    n_scint = sum(1 for r in rows if _scintillation_candidate(r))
    print(f"\n# Summary: {n_circle} days inside Antarctic Circle, "
          f"{n_scint} days with possible-scintillation pattern (high MON-SPAN events, low jamInd)")
    if csv_out:
        df = pl.DataFrame(rows)
        Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(csv_out)
        log.info("Wrote %s (%d rows)", csv_out, df.height)


if __name__ == "__main__":
    main()
