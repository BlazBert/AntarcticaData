"""TEC validation against IGS Global Ionosphere Maps (GIM).

For one or more representative days, fetch the IGS combined IONEX GIM,
bilinearly interpolate it at the ship position and 1-hour aggregation
epochs, and compare against the dataset's high-elevation vertical TEC.

Single-receiver geometry-free TEC has an unknown DCB constant; we
report (a) the RMS of the residual after subtracting a single per-day
constant offset (relative agreement) and (b) the Spearman correlation
of the two series (shape agreement, DCB-invariant).

Output:
    work/tables/T_tec_vs_gim.csv
    figures/output/fig_tec_vs_gim.pdf

Usage:
    python -m verify.tec_vs_gim --day 20250930 --day 20260127 --day 20260415
"""

from __future__ import annotations

import gzip
import io
import logging
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import click
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from analysis._common import (
    derived_dir,
    list_days,
    load_config,
    read_parquet,
    staged_path,
    tables_dir,
)

log = logging.getLogger(__name__)


# Ten well-spread days covering the cruise legs visible in the paper
# (Trieste departure dock, Mediterranean / Atlantic transit out, Lyttelton
# berth, Southern-Ocean approach, Antarctic dwell × 2, polar departure,
# Atlantic return transit × 2, Mediterranean return approach).
DEFAULT_DAYS: tuple[str, ...] = (
    "20250930",  # Trieste dock, pre-departure baseline (31 SVs visible)
    "20251020",  # Mediterranean -> Atlantic transit
    "20251115",  # Lyttelton (NZ) berth (mid-November)
    "20251210",  # Southern-Ocean approach to Antarctica
    "20260101",  # Antarctic dwell (New Year)
    "20260127",  # Mario Zucchelli anchorage
    "20260215",  # Polar departure, re-transit Southern Ocean
    "20260315",  # Atlantic return-leg (mid-March)
    "20260415",  # Mid-Atlantic return (April)
    "20260425",  # Mediterranean return, approaching Trieste
)


# ----------------------------------------------------------------------
# IONEX download
# ----------------------------------------------------------------------

IONEX_MIRRORS = [
    # CODE Analysis Centre at AIUB Bern: 1-hour final GIM, open FTP,
    # long-name convention.
    "ftp://ftp.aiub.unibe.ch/CODE/{yyyy}/COD0OPSFIN_{yyyy}{ddd}0000_01D_01H_GIM.INX.gz",
    # Same, rapid (lower latency, used if final isn't yet available)
    "ftp://ftp.aiub.unibe.ch/CODE/{yyyy}/COD0OPSRAP_{yyyy}{ddd}0000_01D_01H_GIM.INX.gz",
]


def _day_to_doy(day: str) -> tuple[int, int, int]:
    """YYYYMMDD -> (yyyy, doy, yy)."""
    d = datetime.strptime(day, "%Y%m%d")
    yyyy = d.year
    doy = (d - datetime(yyyy, 1, 1)).days + 1
    yy = yyyy % 100
    return yyyy, doy, yy


def fetch_ionex(day: str, cache_dir: Path) -> Path | None:
    """Download IGS GIM IONEX for one day; cache locally; return path or None."""
    yyyy, doy, yy = _day_to_doy(day)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"igsg{doy:03d}0.{yy:02d}i"
    if cache_path.exists() and cache_path.stat().st_size > 1000:
        log.info("Using cached IONEX %s", cache_path)
        return cache_path
    for tmpl in IONEX_MIRRORS:
        url = tmpl.format(yyyy=yyyy, ddd=f"{doy:03d}", yy=f"{yy:02d}")
        log.info("Trying IONEX from %s", url)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read()
            # The .Z file is Unix compress format; try gzip first, then
            # raw text fallback (some mirrors serve uncompressed).
            try:
                text = gzip.decompress(raw).decode("ascii", "replace")
            except OSError:
                try:
                    text = raw.decode("ascii", "replace")
                except UnicodeDecodeError:
                    log.warning("Cannot decode IONEX from %s", url)
                    continue
            if "END OF FILE" in text:
                cache_path.write_text(text)
                log.info("Cached %s (%d bytes)", cache_path, cache_path.stat().st_size)
                return cache_path
        except Exception as exc:  # noqa: BLE001
            log.info("Mirror failed (%s): %s", url, str(exc)[:80])
            continue
    log.error("All IONEX mirrors failed for %s", day)
    return None


# ----------------------------------------------------------------------
# IONEX parsing — minimal IONEX 1 reader
# ----------------------------------------------------------------------


def parse_ionex(path: Path) -> dict:
    """Parse IONEX → {epochs (datetime list), lats, lons, vtec[n_t,n_lat,n_lon],
       exponent}."""
    text = path.read_text()
    in_header = True
    lat1 = lat2 = dlat = None
    lon1 = lon2 = dlon = None
    exponent = -1
    epochs: list[datetime] = []
    maps: list[np.ndarray] = []
    cur_map: list[list[float]] = []
    cur_lat_row: list[float] = []
    n_lat = n_lon = 0
    parse_map = False
    expect_data = False

    for line in text.splitlines():
        label = line[60:].rstrip() if len(line) >= 60 else ""
        body = line[:60]

        if in_header:
            if label == "EXPONENT":
                exponent = int(body.split()[0])
            elif label == "LAT1 / LAT2 / DLAT":
                parts = body.split()
                lat1, lat2, dlat = float(parts[0]), float(parts[1]), float(parts[2])
                n_lat = int(round((lat2 - lat1) / dlat)) + 1
            elif label == "LON1 / LON2 / DLON":
                parts = body.split()
                lon1, lon2, dlon = float(parts[0]), float(parts[1]), float(parts[2])
                n_lon = int(round((lon2 - lon1) / dlon)) + 1
            elif label == "END OF HEADER":
                in_header = False
            continue

        if label == "START OF TEC MAP":
            parse_map = True
            cur_map = []
            cur_lat_row = []
            continue
        if label == "EPOCH OF CURRENT MAP":
            parts = body.split()
            ep = datetime(int(parts[0]), int(parts[1]), int(parts[2]),
                          int(parts[3]), int(parts[4]), int(parts[5]),
                          tzinfo=timezone.utc)
            epochs.append(ep)
            continue
        if label == "LAT/LON1/LON2/DLON/H":
            if cur_lat_row:
                cur_map.append(cur_lat_row)
                cur_lat_row = []
            expect_data = True
            continue
        if label == "END OF TEC MAP":
            if cur_lat_row:
                cur_map.append(cur_lat_row)
                cur_lat_row = []
            maps.append(np.array(cur_map, dtype=np.float64))
            parse_map = False
            expect_data = False
            continue
        if label == "START OF RMS MAP" or label == "END OF FILE":
            break

        if parse_map and expect_data:
            # Numeric data row: integer TEC values 5 chars each.
            # Data rows have no label and extend the full line width,
            # so use the FULL line (not body=line[:60]) to avoid
            # truncating the row at column 60.
            vals = line.split()
            for v in vals:
                try:
                    cur_lat_row.append(int(v))
                except ValueError:
                    pass
            # When the row reaches n_lon, push it
            if len(cur_lat_row) >= n_lon:
                cur_map.append(cur_lat_row[:n_lon])
                cur_lat_row = cur_lat_row[n_lon:]

    lats = np.linspace(lat1, lat2, n_lat)
    lons = np.linspace(lon1, lon2, n_lon)
    vtec = np.stack(maps, axis=0) * (10.0 ** exponent)   # → TECU
    # Sentinel value 9999 (in raw integer units) → NaN
    vtec[vtec >= 9999 * (10.0 ** exponent) * 0.99] = np.nan
    return {
        "epochs": epochs,
        "lats": lats,
        "lons": lons,
        "vtec": vtec,
    }


def gim_interp(gim: dict, t: datetime, lat: float, lon: float) -> float:
    """Bilinear (lat, lon) + linear-in-time interpolation of GIM vTEC."""
    eps = gim["epochs"]
    # bracket time
    times = [e.timestamp() for e in eps]
    ts = t.timestamp()
    if ts <= times[0]:
        i0, i1, w = 0, 0, 0.0
    elif ts >= times[-1]:
        i0, i1, w = len(eps) - 1, len(eps) - 1, 0.0
    else:
        i1 = next(i for i, x in enumerate(times) if x > ts)
        i0 = i1 - 1
        w = (ts - times[i0]) / (times[i1] - times[i0])
    lats = gim["lats"]; lons = gim["lons"]
    lon_w = ((lon + 180.0) % 360.0) - 180.0     # wrap
    # bilinear lat/lon
    def bilin(grid):
        if lats[0] > lats[-1]:
            # lat array is descending in IONEX (typical)
            lat_idx = np.searchsorted(-lats, -lat) - 1
            lat_idx = max(0, min(len(lats) - 2, lat_idx))
            la0, la1 = lats[lat_idx], lats[lat_idx + 1]
            fl = (la0 - lat) / (la0 - la1) if la0 != la1 else 0.0
        else:
            lat_idx = np.searchsorted(lats, lat) - 1
            lat_idx = max(0, min(len(lats) - 2, lat_idx))
            la0, la1 = lats[lat_idx], lats[lat_idx + 1]
            fl = (lat - la0) / (la1 - la0) if la1 != la0 else 0.0
        lon_idx = np.searchsorted(lons, lon_w) - 1
        lon_idx = max(0, min(len(lons) - 2, lon_idx))
        lo0, lo1 = lons[lon_idx], lons[lon_idx + 1]
        fn = (lon_w - lo0) / (lo1 - lo0) if lo1 != lo0 else 0.0
        v00 = grid[lat_idx, lon_idx]
        v01 = grid[lat_idx, lon_idx + 1]
        v10 = grid[lat_idx + 1, lon_idx]
        v11 = grid[lat_idx + 1, lon_idx + 1]
        v0 = v00 * (1 - fn) + v01 * fn
        v1 = v10 * (1 - fn) + v11 * fn
        return v0 * (1 - fl) + v1 * fl
    g0 = bilin(gim["vtec"][i0])
    g1 = bilin(gim["vtec"][i1])
    return float(g0 * (1 - w) + g1 * w)


# ----------------------------------------------------------------------
# Per-day comparison
# ----------------------------------------------------------------------


def ensure_tec_parquet(day: str, cfg: dict) -> Path | None:
    """If derived/tec/{day}.tec.parquet is missing, try to regenerate from staging.

    Returns the path if available (existing or freshly built), else None.
    """
    out = derived_dir(cfg) / "tec" / f"{day}.tec.parquet"
    if out.exists():
        return out
    staged = staged_path(day, "rxm_rawx", cfg)
    if not Path(staged).exists():
        log.warning("No staged UBX data for %s (looked for %s); skipping", day, staged)
        return None
    log.info("Regenerating TEC parquet for %s ...", day)
    rc = subprocess.run(
        [sys.executable, "-m", "analysis.tec", "--day", day],
        cwd=str(Path(__file__).resolve().parent.parent),
    ).returncode
    if rc != 0 or not out.exists():
        log.warning("analysis.tec failed for %s (rc=%s); skipping", day, rc)
        return None
    return out


def compare_day(day: str, cfg: dict, gim_cache: Path,
                elev_cutoff: float = 40.0,
                bin_hr: float = 1.0) -> dict | None:
    """Return dict of arrays for this day, or None if data missing."""
    tec_path = ensure_tec_parquet(day, cfg)
    if tec_path is None:
        return None
    tec = read_parquet(tec_path)
    if tec.is_empty():
        return None
    # Receiver position: use nav_hpposllh at same epochs (joined coarsely)
    hp = read_parquet(staged_path(day, "nav_hpposllh", cfg)).select(
        ["t_ns", "lat_1e7", "lon_1e7"]
    )
    hp = hp.with_columns([
        (pl.col("lat_1e7").cast(pl.Float64) * 1e-7).alias("rx_lat"),
        (pl.col("lon_1e7").cast(pl.Float64) * 1e-7).alias("rx_lon"),
    ]).select(["t_ns", "rx_lat", "rx_lon"]).sort("t_ns")
    if hp.is_empty():
        return None

    # Aggregate our vTEC into hourly bins, using high-elevation observations.
    high = tec.filter(pl.col("elev") >= int(elev_cutoff))
    if high.is_empty():
        log.warning("No high-elev TEC for %s", day)
        return None
    bin_ns = int(bin_hr * 3600 * 1e9)
    high = high.with_columns(((pl.col("t_ns") // bin_ns) * bin_ns).alias("t_bin"))
    ours = (high.group_by("t_bin")
            .agg(pl.col("vtec").median().alias("our_vtec_med"),
                 pl.col("vtec").count().alias("n")))
    # Receiver position at bin centre
    hp = hp.with_columns(((pl.col("t_ns") // bin_ns) * bin_ns).alias("t_bin"))
    pos = (hp.group_by("t_bin")
           .agg(pl.col("rx_lat").median().alias("rx_lat"),
                pl.col("rx_lon").median().alias("rx_lon")))
    df = ours.join(pos, on="t_bin").sort("t_bin")
    df = df.filter(pl.col("n") >= 50)
    if df.is_empty():
        return None

    # GIM
    gim_path = fetch_ionex(day, gim_cache)
    if gim_path is None:
        return None
    gim = parse_ionex(gim_path)

    t_centres = [datetime.fromtimestamp(t / 1e9, tz=timezone.utc)
                 for t in (df["t_bin"].to_numpy() + bin_ns // 2)]
    gim_vtec = np.array(
        [gim_interp(gim, t, lat, lon) for t, lat, lon in
         zip(t_centres, df["rx_lat"].to_numpy(), df["rx_lon"].to_numpy())]
    )

    our_v = df["our_vtec_med"].to_numpy()
    mask = np.isfinite(our_v) & np.isfinite(gim_vtec)
    our_v, gim_v = our_v[mask], gim_vtec[mask]
    times = np.array(t_centres)[mask]
    if our_v.size < 5:
        return None

    # Relative comparison: subtract per-day mean offset, report RMS of residual
    offset = np.median(our_v - gim_v)
    our_corrected = our_v - offset
    residual = our_corrected - gim_v
    rms = float(np.sqrt(np.mean(residual ** 2)))
    rho, _ = spearmanr(our_v, gim_v)

    return {
        "day": day,
        "n_bins": int(our_v.size),
        "offset_TECU": float(offset),
        "rms_after_offset_TECU": rms,
        "spearman_rho": float(rho),
        "times": times,
        "our_vtec_corrected": our_corrected,
        "gim_vtec": gim_v,
        "rx_lat": df.filter(mask if isinstance(mask, pl.Series) else pl.Series(mask))["rx_lat"].to_numpy(),
    }


def build_figure(results: list[dict], out_path: Path) -> None:
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.4 * n), constrained_layout=True)
    if n == 1:
        axes = np.array([axes])
    for i, r in enumerate(results):
        ax = axes[i]
        ax.plot(r["times"], r["our_vtec_corrected"], "-o", ms=3, lw=0.9,
                color="#1f77b4", label="this dataset (DCB-corrected)")
        ax.plot(r["times"], r["gim_vtec"], "-s", ms=3, lw=0.9,
                color="#d62728", label="IGS GIM")
        ax.set_ylabel("vTEC (TECU)")
        ax.set_title(f"{r['day']}  (n={r['n_bins']} hourly bins, "
                     f"rms={r['rms_after_offset_TECU']:.1f} TECU, "
                     f"ρ={r['spearman_rho']:+.2f})", fontsize=9)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("UTC time")
    fig.suptitle("Vertical-TEC validation against IGS GIM", fontsize=11)
    fig.savefig(out_path, dpi=150)
    log.info("Wrote %s", out_path)


def build_summary_figure(results: list[dict], out_path: Path) -> None:
    """Three-panel cruise-wide summary for paper inclusion (used when N >= 6):

    (a) Spearman rho per day
    (b) Post-DCB RMS per day
    (c) Per-day DCB offset
    All three share an x-axis of cruise date so legs read left-to-right.
    """
    days = [datetime.strptime(r["day"], "%Y%m%d").replace(tzinfo=timezone.utc)
            for r in results]
    rho = np.array([r["spearman_rho"] for r in results])
    rms = np.array([r["rms_after_offset_TECU"] for r in results])
    off = np.array([r["offset_TECU"] for r in results])

    fig, axes = plt.subplots(3, 1, figsize=(11, 7),
                             constrained_layout=True, sharex=True)
    axes[0].bar(days, rho, width=8, color="#1f77b4", edgecolor="k", lw=0.4)
    axes[0].axhline(0.5, color="grey", ls="--", lw=0.6, label="moderate (0.5)")
    axes[0].axhline(0.7, color="green", ls="--", lw=0.6, label="strong (0.7)")
    axes[0].set_ylabel("Spearman ρ")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].bar(days, rms, width=8, color="#d62728", edgecolor="k", lw=0.4)
    axes[1].set_ylabel("Post-DCB RMS\n(TECU)")
    axes[1].grid(alpha=0.3, axis="y")

    axes[2].bar(days, off, width=8, color="#9467bd", edgecolor="k", lw=0.4)
    axes[2].set_ylabel("Per-day DCB\noffset (TECU)")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[2].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    axes[2].set_xlabel("UTC date")
    axes[2].grid(alpha=0.3, axis="y")
    fig.autofmt_xdate(rotation=30, ha="right")

    fig.suptitle(
        f"Vertical-TEC cross-validation against IGS GIM "
        f"(N={len(results)} days spanning the cruise)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=150)
    log.info("Wrote %s", out_path)


@click.command()
@click.option("--day", "days", multiple=True,
              help=f"YYYYMMDD; repeat for multiple. Default: {len(DEFAULT_DAYS)} "
                   f"days spanning the cruise (see DEFAULT_DAYS).")
@click.option("--gim-cache", default=None,
              help="Cache dir for IONEX files. Default ../work/derived/ionex_cache.")
def main(days: tuple[str, ...], gim_cache: str | None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    if not days:
        days = DEFAULT_DAYS
    cache = Path(gim_cache) if gim_cache else (
        derived_dir(cfg).parent / "derived" / "ionex_cache"
    )

    results: list[dict] = []
    for d in days:
        log.info("=== %s ===", d)
        r = compare_day(d, cfg, cache)
        if r is not None:
            results.append(r)
            log.info("  n_bins=%d  rms=%.2f TECU  rho=%+.2f  offset=%.1f TECU",
                     r["n_bins"], r["rms_after_offset_TECU"],
                     r["spearman_rho"], r["offset_TECU"])

    if not results:
        log.error("No results — check IONEX availability and local TEC data")
        return

    # CSV summary
    out_csv = tables_dir(cfg) / "T_tec_vs_gim.csv"
    pl.DataFrame([{k: v for k, v in r.items()
                   if k in ("day", "n_bins", "offset_TECU",
                            "rms_after_offset_TECU", "spearman_rho")}
                  for r in results]).write_csv(out_csv)
    log.info("Wrote %s", out_csv)

    # Figures. With N >= 6 days, the per-day stack is too tall for the
    # paper, so write a paper-friendly summary as fig_tec_vs_gim.pdf and
    # ship the per-day detail as fig_tec_vs_gim_detail.pdf for
    # supplementary inclusion. With N < 6 (e.g. a quick local check),
    # the per-day stack is small enough to be the headline figure.
    out_dir = Path(__file__).resolve().parent.parent / "figures" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(results) >= 6:
        build_summary_figure(results, out_dir / "fig_tec_vs_gim.pdf")
        build_figure(results, out_dir / "fig_tec_vs_gim_detail.pdf")
    else:
        build_figure(results, out_dir / "fig_tec_vs_gim.pdf")


if __name__ == "__main__":
    main()
