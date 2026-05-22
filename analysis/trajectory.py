"""Cruise track + crossings.

Reads ``staging/<day>/nav_hpposllh.parquet`` (or falls back to ``nav_pvt.parquet``),
joins NAV-PVT for ``gSpeed_mm_s`` / ``fixType``, and produces:

* ``derived/track/<day>.track.parquet`` — per-day decimated track (1-Hz default).
* ``derived/track/track.geojson`` — full-cruise GeoJSON LineString (after
  ``aggregate``).
* ``derived/track/track.nc`` — NetCDF (time, lat, lon, height, gSpeed) for
  ``xarray`` users.
* ``tables/T5_crossings.csv`` — equator (×2), Antarctic Circle (±66.5°)
  crossings, port stops.

Port detection: speed < ``ports.detect.speed_threshold_m_s`` for at least
``ports.detect.min_dwell_minutes`` minutes within ``radius_km`` of any
``ports.yaml`` entry.
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
    cumulative_distance_km,
    derived_dir,
    haversine_km,
    list_days,
    load_config,
    read_parquet,
    staged_path,
    tables_dir,
    write_parquet,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-day track
# ---------------------------------------------------------------------------


def build_day_track(day: str, *, decimate_hz: float = 1.0, cfg: dict | None = None) -> pl.DataFrame:
    """Per-day decimated trajectory dataframe.

    Columns: ``t_ns, lat, lon, height_m, hMSL_m, hAcc_m, vAcc_m, gSpeed_m_s,
    fixType``. Decimated to ``decimate_hz`` (default 1 Hz; raw is 2 Hz).

    Defensive filters (applied here, in the data layer, so all downstream
    consumers see clean tracks):

    1. ``t_ns`` outside the cruise window (2017–2030) — receiver clock is
       sometimes 1970 / week-rollover until first GPS-time sync.
    2. ``fixType < 3`` — no-fix or 2D-only rows carry stale coordinates.
    3. Lat=0 ∧ Lon=0 (a "Null Island" jump that occasionally appears
       during fix re-acquisition).
    4. Time-sorted *after* the above, then a haversine speed-outlier pass
       drops rows that imply > 100 knots ground speed against the previous
       sample (no surface vessel exceeds this).
    """
    cfg = cfg or load_config()
    pvt_path = staged_path(day, "nav_pvt", cfg)
    hp_path = staged_path(day, "nav_hpposllh", cfg)
    pvt = read_parquet(pvt_path)
    if pvt.is_empty():
        return pl.DataFrame()

    T_MIN = 1_735_689_600 * 1_000_000_000   # 2025-01-01 UTC — pre-cruise
    T_MAX = 1_798_761_600 * 1_000_000_000   # 2027-01-01 UTC — post-cruise
    # Reject stale-cache coordinates that pass the fixType filter but
    # carry large positional uncertainty. Typical at-sea hAcc is <1 m;
    # an hAcc above ~50 m at fixType ≥ 3 is almost always a re-acquisition
    # artefact (the receiver still reports "fixed" while phase-lock catches
    # up with the new geometry). Catches the orphan dots that produced the
    # speckled appearance in the cruise-track map.
    HACC_MAX_MM = 50_000   # 50 m horizontal accuracy
    pvt = pvt.filter(
        (pl.col("t_ns") > T_MIN) & (pl.col("t_ns") < T_MAX)
        & (pl.col("fixType") >= 3)
        & ~((pl.col("lat_1e7") == 0) & (pl.col("lon_1e7") == 0))
        & (pl.col("hAcc_mm") <= HACC_MAX_MM)
    ).sort("t_ns")
    if pvt.is_empty():
        return pl.DataFrame()

    # Use HPPOSLLH for sub-mm precision when available; otherwise PVT.
    if hp_path.exists():
        hp = read_parquet(hp_path)
        # Build high-precision lat/lon: lat_deg = lat_1e7*1e-7 + latHp_1e9*1e-9
        track = hp.select(
            [
                pl.col("t_ns"),
                (pl.col("lat_1e7").cast(pl.Float64) * 1e-7
                 + pl.col("latHp_1e9").cast(pl.Float64) * 1e-9).alias("lat"),
                (pl.col("lon_1e7").cast(pl.Float64) * 1e-7
                 + pl.col("lonHp_1e9").cast(pl.Float64) * 1e-9).alias("lon"),
                (pl.col("height_mm").cast(pl.Float64) * 1e-3
                 + pl.col("heightHp_0p1mm").cast(pl.Float64) * 1e-4).alias("height_m"),
                (pl.col("hMSL_mm").cast(pl.Float64) * 1e-3
                 + pl.col("hMSLHp_0p1mm").cast(pl.Float64) * 1e-4).alias("hMSL_m"),
                (pl.col("hAcc_0p1mm").cast(pl.Float64) * 1e-4).alias("hAcc_m"),
                (pl.col("vAcc_0p1mm").cast(pl.Float64) * 1e-4).alias("vAcc_m"),
            ]
        )
    else:
        track = pvt.select(
            [
                pl.col("t_ns"),
                (pl.col("lat_1e7").cast(pl.Float64) * 1e-7).alias("lat"),
                (pl.col("lon_1e7").cast(pl.Float64) * 1e-7).alias("lon"),
                (pl.col("height_mm").cast(pl.Float64) * 1e-3).alias("height_m"),
                (pl.col("hMSL_mm").cast(pl.Float64) * 1e-3).alias("hMSL_m"),
                (pl.col("hAcc_mm").cast(pl.Float64) * 1e-3).alias("hAcc_m"),
                (pl.col("vAcc_mm").cast(pl.Float64) * 1e-3).alias("vAcc_m"),
            ]
        )

    # Join speed and fix type from PVT (time-aligned: HPPOSLLH iTOW matches PVT iTOW)
    pvt_small = pvt.select(
        [
            pl.col("t_ns"),
            (pl.col("gSpeed_mm_s").cast(pl.Float64) * 1e-3).alias("gSpeed_m_s"),
            pl.col("fixType"),
            pl.col("numSV"),
        ]
    )
    track = track.join(pvt_small, on="t_ns", how="left").sort("t_ns")

    # Decimate
    decim_ns = int(1e9 / decimate_hz)
    if decim_ns > 0 and track.height > 0:
        track = track.with_columns((pl.col("t_ns") // decim_ns).alias("_bin")).group_by("_bin").agg(
            [
                pl.col("t_ns").first(),
                pl.col("lat").mean(),
                pl.col("lon").mean(),
                pl.col("height_m").mean(),
                pl.col("hMSL_m").mean(),
                pl.col("hAcc_m").mean(),
                pl.col("vAcc_m").mean(),
                pl.col("gSpeed_m_s").mean(),
                pl.col("fixType").last(),
                pl.col("numSV").last(),
            ]
        ).drop("_bin").sort("t_ns")

    # Speed-outlier pass: drop rows whose haversine speed against the
    # previous *kept* sample exceeds 100 knots (~51 m/s). Iterative single
    # pass — for the per-day track this is sufficient since legit large
    # jumps don't happen.
    if track.height >= 2:
        track = _speed_filter(track, max_speed_kn=100.0)

    return track


def _speed_filter(track: pl.DataFrame, *, max_speed_kn: float = 100.0) -> pl.DataFrame:
    """Drop rows that imply > ``max_speed_kn`` from the previous row.

    Uses a single forward pass — sufficient for cleaning up
    once-per-day pre-fix coordinate stragglers. Not a Kalman filter.
    """
    lat = track["lat"].to_numpy().astype(np.float64)
    lon = track["lon"].to_numpy().astype(np.float64)
    t = track["t_ns"].to_numpy()
    n = len(t)
    keep = np.ones(n, dtype=bool)
    last_i = 0
    max_kmh = max_speed_kn * 1.852
    for i in range(1, n):
        dt_h = max((t[i] - t[last_i]) / 1e9 / 3600.0, 1e-6)
        # haversine
        rlat1 = np.deg2rad(lat[last_i]); rlat2 = np.deg2rad(lat[i])
        dlat = rlat2 - rlat1
        dlon = np.deg2rad(lon[i] - lon[last_i])
        a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
        dist_km = 2 * 6371.0088 * np.arcsin(min(1.0, np.sqrt(a)))
        if dist_km / dt_h > max_kmh:
            keep[i] = False
        else:
            last_i = i
    if keep.all():
        return track
    return track.filter(pl.Series(keep))


def _render_day_track_png(df: pl.DataFrame, day: str, out_dir: Path) -> Path | None:
    if df.is_empty():
        return None
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    h_msl = df["hMSL_m"].to_numpy()
    speed = df["gSpeed_m_s"].fill_null(0.0).to_numpy()
    t_h = (df["t_ns"].to_numpy() - df["t_ns"].min()) / 1e9 / 3600.0
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    sc = axes[0].scatter(lon, lat, c=t_h, s=2, cmap="viridis")
    axes[0].set_xlabel("Lon (°E)")
    axes[0].set_ylabel("Lat (°N)")
    axes[0].set_title(f"Track {day}")
    axes[0].set_aspect("equal")
    fig.colorbar(sc, ax=axes[0], label="hours")
    axes[1].plot(t_h, h_msl, color="black", linewidth=0.8)
    axes[1].set_xlabel("Hours since first epoch")
    axes[1].set_ylabel("Height MSL (m)")
    axes[1].set_title("Height MSL")
    axes[2].plot(t_h, speed, color="C1", linewidth=0.8)
    axes[2].set_xlabel("Hours since first epoch")
    axes[2].set_ylabel("Ground speed (m/s)")
    axes[2].set_title("Speed")
    fig.tight_layout()
    png = out_dir / f"{day}.track.png"
    fig.savefig(png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return png


def write_day_track(day: str, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    df = build_day_track(day, cfg=cfg)
    out_dir = derived_dir(cfg) / "track"
    out = out_dir / f"{day}.track.parquet"
    write_parquet(df, out)
    log.info("Wrote %s (%d rows)", out, df.height)
    try:
        png = _render_day_track_png(df, day, out_dir)
        if png:
            log.info("Wrote %s", png)
    except Exception as exc:  # noqa: BLE001
        log.warning("Track quicklook PNG failed for %s: %s", day, exc)
    return out


# ---------------------------------------------------------------------------
# Cross-day aggregation: GeoJSON, NetCDF, crossings
# ---------------------------------------------------------------------------


def _load_all_tracks(cfg: dict) -> pl.DataFrame:
    track_dir = derived_dir(cfg) / "track"
    files = sorted(track_dir.glob("*.track.parquet"))
    if not files:
        return pl.DataFrame()
    parts = [read_parquet(f) for f in files]
    return pl.concat(parts, how="vertical").sort("t_ns")


def _detect_crossings(track: pl.DataFrame) -> list[dict[str, Any]]:
    """Equator and Antarctic-Circle crossings (lat sign changes / threshold).

    Restricted to the cruise window (2025-01-01 to 2027-01-01); rows
    with bogus ``t_ns`` outside that window (pre-fix-lock cached
    coordinates that slipped through the trajectory filter) produced
    nonsensical "crossings" — those are now dropped.
    """
    if track.is_empty():
        return []
    T_MIN = 1_735_689_600 * 1_000_000_000
    T_MAX = 1_798_761_600 * 1_000_000_000
    track = track.filter(
        (pl.col("t_ns") > T_MIN) & (pl.col("t_ns") < T_MAX)
    ).sort("t_ns")
    if track.is_empty():
        return []
    lat = track["lat"].to_numpy()
    lon = track["lon"].to_numpy()
    t_ns = track["t_ns"].to_numpy()
    events: list[dict[str, Any]] = []

    for label, threshold in (("equator", 0.0), ("antarctic_circle", -66.56361)):
        sign = np.sign(lat - threshold)
        # Indices where sign changes
        diff = np.diff(sign)
        idx = np.where(diff != 0)[0]
        for i in idx:
            t_iso = np.datetime64(int(t_ns[i]), "ns").astype(str)
            events.append(
                {
                    "type": label,
                    "t_iso": t_iso,
                    "lat": float(lat[i]),
                    "lon": float(lon[i]),
                    "direction": "south" if (lat[i + 1] < lat[i]) else "north",
                }
            )
    events.sort(key=lambda e: e["t_iso"])
    return events


def _detect_port_stops(track: pl.DataFrame, ports_cfg: dict) -> list[dict[str, Any]]:
    """Sliding-window detection of port stops."""
    if track.is_empty():
        return []
    ports = ports_cfg["ports"]
    speed_thr = float(ports_cfg["detect"]["speed_threshold_m_s"])
    dwell_min = float(ports_cfg["detect"]["min_dwell_minutes"])

    lat = track["lat"].to_numpy()
    lon = track["lon"].to_numpy()
    t_ns = track["t_ns"].to_numpy()
    speed = track["gSpeed_m_s"].fill_null(np.nan).to_numpy()

    events: list[dict[str, Any]] = []
    for port in ports:
        port_lat = float(port["lat"])
        port_lon = float(port["lon"])
        radius = float(port["radius_km"])
        dist = haversine_km(
            np.full_like(lat, port_lat), np.full_like(lon, port_lon), lat, lon
        )
        within = (dist <= radius) & (np.nan_to_num(speed, nan=1e9) < speed_thr)
        if not within.any():
            continue
        # Find contiguous within-windows of length ≥ dwell_min
        runs = _contiguous_runs(within)
        for s, e in runs:
            duration_min = (t_ns[e] - t_ns[s]) / 1e9 / 60.0
            if duration_min < dwell_min:
                continue
            events.append(
                {
                    "type": "port_stop",
                    "port": port["name"],
                    "t_start": np.datetime64(int(t_ns[s]), "ns").astype(str),
                    "t_end": np.datetime64(int(t_ns[e]), "ns").astype(str),
                    "duration_min": float(duration_min),
                    "lat": float(np.median(lat[s : e + 1])),
                    "lon": float(np.median(lon[s : e + 1])),
                }
            )
    events.sort(key=lambda e: e["t_start"])
    return events


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0 or not mask.any():
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0])
    if mask[0]:
        starts = [0, *starts]
    if mask[-1]:
        ends = [*ends, mask.size - 1]
    return list(zip(starts, ends))


def aggregate_track(cfg: dict | None = None) -> dict[str, Path]:
    cfg = cfg or load_config()
    ports_cfg = load_config("ports")
    track = _load_all_tracks(cfg)
    out_dir = derived_dir(cfg) / "track"
    outputs: dict[str, Path] = {}

    # Per-day coords + cumulative distance
    if not track.is_empty():
        cum = cumulative_distance_km(track["lat"].to_numpy(), track["lon"].to_numpy())
        track = track.with_columns(pl.Series("cum_km", cum))
        all_path = out_dir / "track.all.parquet"
        write_parquet(track, all_path)
        outputs["all_parquet"] = all_path

        # GeoJSON MultiLineString — break into segments on time gaps (>1 h)
        # so non-contiguous days don't render as straight lines across the
        # globe. Decimated to ~1 point per minute for file size.
        line_track = track.with_columns((pl.col("t_ns") // 60_000_000_000).alias("_min")).group_by("_min").agg(
            [pl.col("lat").mean(), pl.col("lon").mean(), pl.col("t_ns").first()]
        ).sort("_min")
        t_arr = line_track["t_ns"].to_numpy()
        lons = line_track["lon"].to_list()
        lats = line_track["lat"].to_list()
        segments: list[list[tuple[float, float]]] = [[]]
        GAP_NS = 60 * 60 * 1_000_000_000  # 1 hour
        for i in range(len(t_arr)):
            if i > 0 and (t_arr[i] - t_arr[i - 1]) > GAP_NS:
                segments.append([])
            segments[-1].append((float(lons[i]), float(lats[i])))
        # Drop empty/single-point segments
        segments = [s for s in segments if len(s) >= 2]
        gj = {
            "type": "Feature",
            "geometry": {"type": "MultiLineString", "coordinates": segments},
            "properties": {
                "name": "Antarctica 2025/26 cruise",
                "n_segments": len(segments),
                "n_points": sum(len(s) for s in segments),
                "first_t": np.datetime64(int(track["t_ns"].min()), "ns").astype(str),
                "last_t": np.datetime64(int(track["t_ns"].max()), "ns").astype(str),
                "total_distance_km": float(cum[-1]) if cum.size else 0.0,
            },
        }
        gj_path = out_dir / "track.geojson"
        gj_path.write_text(json.dumps(gj))
        outputs["geojson"] = gj_path

        # NetCDF (xarray) — best-effort, skip cleanly if netcdf4 missing
        try:
            import xarray as xr  # noqa: PLC0415

            t_dt = track["t_ns"].to_numpy().astype("datetime64[ns]")
            ds = xr.Dataset(
                {
                    "lat": ("time", track["lat"].to_numpy()),
                    "lon": ("time", track["lon"].to_numpy()),
                    "height_m": ("time", track["height_m"].to_numpy()),
                    "hMSL_m": ("time", track["hMSL_m"].to_numpy()),
                    "gSpeed_m_s": ("time", track["gSpeed_m_s"].to_numpy()),
                    "cum_km": ("time", track["cum_km"].to_numpy()),
                    "fixType": ("time", track["fixType"].to_numpy()),
                    "numSV": ("time", track["numSV"].to_numpy()),
                },
                coords={"time": t_dt},
                attrs={
                    "title": "Antarctica 2025/26 ship-borne GNSS cruise track",
                    "instrument": "u-blox ZED-F9P-15B",
                    "decimation_hz": 1.0,
                    "source": "NAV-HPPOSLLH (or NAV-PVT fallback)",
                },
            )
            nc_path = out_dir / "track.nc"
            encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
            # Force the netCDF4 backend (supports int64) and encode time as
            # seconds-since-epoch float64, which round-trips cleanly without
            # losing the date (we don't need ns precision in this product).
            ds = ds.assign_coords(time=ds["time"].astype("datetime64[s]"))
            ds.to_netcdf(nc_path, engine="netcdf4", encoding=encoding)
            outputs["netcdf"] = nc_path
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping NetCDF: %s", exc)

    # Crossings (T5)
    crossings = _detect_crossings(track)
    ports = _detect_port_stops(track, ports_cfg)
    rows: list[dict[str, Any]] = [*crossings, *ports]
    df = pl.DataFrame(rows) if rows else pl.DataFrame()
    t5 = tables_dir(cfg) / "T5_crossings.csv"
    if not df.is_empty():
        df.write_csv(t5)
        outputs["t5_csv"] = t5
        log.info("Crossings → %s (%d events)", t5, df.height)

    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--day", "day", default=None, help="YYYYMMDD")
@click.option("--aggregate/--no-aggregate", default=True)
def main(day: str | None, aggregate: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    days = [day] if day else list_days(cfg)
    for d in days:
        write_day_track(d, cfg)
    if aggregate:
        aggregate_track(cfg)


if __name__ == "__main__":
    main()
