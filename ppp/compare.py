"""Compare onboard u-blox solution vs PRIDE PPP-AR kinematic solution.

For a given day, read:

* ``staging/<day>/nav_hpposllh.parquet`` — onboard high-precision positions.
* ``derived/ppp/<day>/kin_YYYYDDD`` — PRIDE PPP-AR kinematic output
  (the default ``pdp3 -m K`` filename; the legacy ``kin.pos`` filename
  is also accepted).

Compute time-aligned ENU residuals, write ``derived/ppp/<day>/diff.parquet``,
and append a row to ``tables/T_ppp_residuals.parquet`` containing median,
95th percentile, 99th percentile and max 3-D residual together with the
day-median ZTD.

PRIDE PPP-AR kin output (column layout from the PRIDE user manual):
    MJD  SoD  X(m)  Y(m)  Z(m)  Lat(deg)  Lon(deg)  H(m)  PDOP  Nsat  Q
Lines starting with ``*`` are comments. The format is whitespace
separated.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import polars as pl

from analysis._common import (
    derived_dir,
    list_days,
    load_config,
    read_parquet,
    resolve_path,
    staged_path,
    tables_dir,
    write_parquet,
)

log = logging.getLogger(__name__)


# MJD 0 = 1858-11-17 00:00 UTC; Unix epoch 1970-01-01 = MJD 40587.
_MJD_UNIX_EPOCH_DAYS = 40587

# PRIDE PPP-AR writes timestamps in GPS time. The onboard NAV-HPPOSLLH
# stream is in UTC. GPS time is 18 s ahead of UTC as of the most recent
# leap-second insertion (2017-01-01). Subtract this offset from PRIDE
# timestamps to align with onboard.
_GPS_UTC_LEAP_S = 18


def _find_pride_kin(day: str, cfg: dict) -> Path | None:
    """Locate the PRIDE kin output across naming/layout variants:
      * legacy:                   <out>/kin.pos
      * PRIDE 2.x flat:           <out>/kin_YYYYDDD
      * PRIDE 3.x single DOY:     <out>/YYYY/DDD/kin_YYYYDDD_<site>
      * PRIDE 3.x DOY range:      <out>/YYYY/DDDa-DDDb/kin_YYYYDDDa_<site>
        (happens when the RINEX obs spans midnight UTC: GPS time is 18 s
        ahead, so observations near 00:00 UTC end up tagged with the
        previous DOY by PRIDE.)

    Strategy: recursively glob any ``kin_*`` file under ``<out>/<YYYY>/``
    (excluding the ``product/`` subtree), then fall back to legacy names.
    Returns the first match.
    """
    out_dir = resolve_path(cfg["paths"]["ppp"]) / day
    if not out_dir.exists():
        return None
    yyyy = day[0:4]
    from datetime import date

    doy = (date(int(yyyy), int(day[4:6]), int(day[6:8]))
           - date(int(yyyy), 1, 1)).days + 1
    stamp = f"{yyyy}{doy:03d}"

    # 1) Glob anywhere under <out>/<yyyy>/, skipping the products dir
    year_root = out_dir / yyyy
    if year_root.exists():
        for path in sorted(year_root.rglob("kin_*")):
            if "product" in path.parts:
                continue
            if path.is_file():
                return path

    # 2) Legacy / flat fallbacks
    for c in (out_dir / f"kin_{stamp}", out_dir / "kin.pos"):
        if c.exists():
            return c
    return None


def _read_pride_kin(path: Path) -> pl.DataFrame:
    """Parse a PRIDE ``kin_*`` file into a polars DataFrame.

    The file format is auto-detected up front:

    * PRIDE 3.x: free-form metadata header terminated by ``END OF HEADER``,
      then one record per epoch:
          MJD  SoD  <flag>  X  Y  Z  lat  lon  h_ell  Nsat  6*sys_status  pdop
      Where <flag> is a non-numeric character (``*`` = fixed, ``+`` = float,
      etc.). Any flag is accepted; rows with bad floats are skipped.

    * PRIDE 2.x: no header marker; lines start directly with MJD:
          MJD  SoD  X  Y  Z  lat  lon  h_ell  pdop  nsat  qual

    Returns columns ``t_ns, lat, lon, h_ell, pdop, nsat``.
    """
    if not path.exists():
        return pl.DataFrame()
    text = path.read_text().splitlines()
    is_3x = any("END OF HEADER" in line for line in text)
    rows: list[dict] = []
    in_data = not is_3x
    for raw in text:
        if not in_data:
            if "END OF HEADER" in raw:
                in_data = True
            continue
        line = raw.strip()
        if not line or line.startswith(("*", "#")):
            continue
        parts = line.split()
        if is_3x:
            # 3.x: PRIDE writes two interleaved row layouts in the same file.
            #   17-field (flag present):
            #       MJD SoD * X Y Z lat lon h Nsat 6*flags PDOP
            #   16-field (no flag):
            #       MJD SoD   X Y Z lat lon h Nsat 6*flags PDOP
            # Detect per-row by whether parts[2] is numeric.
            if len(parts) < 9:
                continue
            try:
                float(parts[2])
                has_flag = False  # parts[2] is X (ECEF), no flag column
            except ValueError:
                has_flag = True   # parts[2] is the * / + / - flag character
            off = 1 if has_flag else 0
            try:
                mjd = float(parts[0]); sod = float(parts[1])
                lat = float(parts[5 + off])
                lon = float(parts[6 + off])
                h = float(parts[7 + off])
                nsat = int(parts[8 + off])
                pdop = float(parts[-1])
            except (ValueError, IndexError):
                continue
        else:
            # 2.x: <mjd> <sod> <X> <Y> <Z> <lat> <lon> <h> <pdop> <nsat> <qual>
            if len(parts) < 11:
                continue
            try:
                mjd = float(parts[0]); sod = float(parts[1])
                lat = float(parts[5]); lon = float(parts[6])
                h = float(parts[7]); pdop = float(parts[8])
                nsat = int(parts[9])
            except (ValueError, IndexError):
                continue
        # Normalise lon to [-180, 180] — PRIDE writes [0, 360] east-positive
        # in some sessions (e.g. mid-Pacific), [-180, 180] in others.
        if lon > 180.0:
            lon -= 360.0
        # Sanity-clip: PRIDE sometimes emits unconverged rows with absurd
        # latitudes (e.g. raw ECEF Z in the lat slot due to a missing field).
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        t_unix_s = (mjd - _MJD_UNIX_EPOCH_DAYS) * 86400.0 + sod - _GPS_UTC_LEAP_S
        rows.append(
            {
                "t_ns": int(round(t_unix_s * 1e9)),
                "lat": lat,
                "lon": lon,
                "h_ell": h,
                "pdop": pdop,
                "nsat": nsat,
            }
        )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort("t_ns")


def _read_pride_ztd(day: str, cfg: dict) -> pl.DataFrame:
    """Best-effort reader for the PRIDE ZTD output. Searches the day's
    PRIDE output tree recursively for any ``ztd_*`` file (handles single-
    DOY and DOY-range subdirectory layouts) before falling back to the
    legacy ``ztd_YYYYDDD`` and ``ztd.txt`` names. Columns: ``t_ns, ztd_m``."""
    out_dir = resolve_path(cfg["paths"]["ppp"]) / day
    from datetime import date

    yyyy = day[0:4]
    doy = (date(int(yyyy), int(day[4:6]), int(day[6:8]))
           - date(int(yyyy), 1, 1)).days + 1
    stamp = f"{yyyy}{doy:03d}"
    path: Path | None = None
    year_root = out_dir / yyyy
    if year_root.exists():
        for cand in sorted(year_root.rglob("ztd_*")):
            if "product" in cand.parts:
                continue
            if cand.is_file():
                path = cand
                break
    if path is None:
        for c in (out_dir / f"ztd_{stamp}", out_dir / "ztd.txt"):
            if c.exists():
                path = c
                break
    if path is None:
        return pl.DataFrame()
    text = path.read_text().splitlines()
    is_3x = any("END OF HEADER" in line for line in text)
    rows = []
    in_data = not is_3x
    for raw in text:
        if not in_data:
            if "END OF HEADER" in raw:
                in_data = True
            continue
        line = raw.strip()
        if not line or line.startswith(("*", "#")):
            continue
        parts = line.split()
        if is_3x:
            # 3.x: Year Mon Day Hour Min Sec ZDD ZWDini ZWDcor
            #      0    1   2   3    4   5   6   7      8
            # Total ZTD = ZDD + ZWDini + ZWDcor (m).
            if len(parts) < 9:
                continue
            try:
                from datetime import datetime, timezone
                y, mo, d, hh, mm = (int(parts[i]) for i in range(5))
                sec = float(parts[5])
                ts = datetime(y, mo, d, hh, mm, tzinfo=timezone.utc).timestamp()
                ts += sec - _GPS_UTC_LEAP_S  # PRIDE writes GPS time, align to UTC
                ztd = float(parts[6]) + float(parts[7]) + float(parts[8])
            except (ValueError, IndexError):
                continue
            rows.append({"t_ns": int(round(ts * 1e9)), "ztd_m": ztd})
        else:
            # 2.x: MJD SoD ZTD ...
            if len(parts) < 3:
                continue
            try:
                mjd = float(parts[0]); sod = float(parts[1])
                ztd = float(parts[2])
            except ValueError:
                continue
            t_unix_s = (mjd - _MJD_UNIX_EPOCH_DAYS) * 86400.0 + sod - _GPS_UTC_LEAP_S
            rows.append({"t_ns": int(round(t_unix_s * 1e9)), "ztd_m": ztd})
    return pl.DataFrame(rows).sort("t_ns") if rows else pl.DataFrame()


def _llh_to_enu(
    lat_ref: float, lon_ref: float, lat: np.ndarray, lon: np.ndarray, h: np.ndarray,
    h_ref: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local ENU residuals (m) about (lat_ref, lon_ref, h_ref).

    Adequate for residuals — small-angle approximation; we don't need a
    full ECEF rotation because the per-epoch reference moves with the
    onboard fix.
    """
    R = 6_378_137.0
    # Wrap lat/lon deltas to the shortest signed difference. Without this,
    # a crossing of the antimeridian (lon ~ +/-180) produces ~40000 km of
    # spurious east displacement; without lat-wrap we'd never see it but
    # the symmetry is cheap insurance.
    dlon_deg = ((lon - lon_ref + 180.0) % 360.0) - 180.0
    dlat_deg = ((lat - lat_ref + 90.0) % 180.0) - 90.0
    dlat = np.deg2rad(dlat_deg)
    dlon = np.deg2rad(dlon_deg)
    n = R * dlat
    e = R * np.cos(np.deg2rad(lat_ref)) * dlon
    u = h - h_ref
    return e, n, u


def compare_day(day: str, cfg: dict | None = None) -> dict | None:
    cfg = cfg or load_config()
    onboard = read_parquet(staged_path(day, "nav_hpposllh", cfg))
    if onboard.is_empty():
        return None
    kin_path = _find_pride_kin(day, cfg)
    if kin_path is None:
        log.warning("No PRIDE kin output for %s", day)
        return None
    pride = _read_pride_kin(kin_path)
    if pride.is_empty():
        log.warning("Empty / unparsable PRIDE kin file %s", kin_path)
        return None
    out_dir = derived_dir(cfg) / "ppp" / day
    out_dir.mkdir(parents=True, exist_ok=True)

    onboard = onboard.with_columns(
        [
            (pl.col("lat_1e7").cast(pl.Float64) * 1e-7
             + pl.col("latHp_1e9").cast(pl.Float64) * 1e-9).alias("lat_ob"),
            (pl.col("lon_1e7").cast(pl.Float64) * 1e-7
             + pl.col("lonHp_1e9").cast(pl.Float64) * 1e-9).alias("lon_ob"),
            (pl.col("height_mm").cast(pl.Float64) * 1e-3
             + pl.col("heightHp_0p1mm").cast(pl.Float64) * 1e-4).alias("h_ob"),
            (pl.col("hAcc_0p1mm").cast(pl.Float64) * 1e-4).alias("hAcc_m"),
            (pl.col("vAcc_0p1mm").cast(pl.Float64) * 1e-4).alias("vAcc_m"),
        ]
    ).select(["t_ns", "lat_ob", "lon_ob", "h_ob", "hAcc_m", "vAcc_m"]).sort("t_ns")

    # Nearest-neighbour merge within ±0.5 s (PRIDE is 1 Hz, onboard 2 Hz).
    pride_sel = (
        pride.rename({"lat": "lat_pp", "lon": "lon_pp", "h_ell": "h_pp"})
             .select(["t_ns", "lat_pp", "lon_pp", "h_pp", "pdop", "nsat"])
             .sort("t_ns")
    )
    joined = onboard.join_asof(
        pride_sel,
        on="t_ns",
        strategy="nearest",
        tolerance=int(0.5e9),
    ).drop_nulls(subset=["lat_pp", "lon_pp", "h_pp"])

    if joined.is_empty():
        log.warning("No time-overlap matches for %s", day)
        return None

    e, n, u = _llh_to_enu(
        lat_ref=joined["lat_ob"].to_numpy()[0],
        lon_ref=joined["lon_ob"].to_numpy()[0],
        lat=joined["lat_pp"].to_numpy(),
        lon=joined["lon_pp"].to_numpy(),
        h=joined["h_pp"].to_numpy(),
        h_ref=joined["h_ob"].to_numpy()[0],
    )
    e_ob, n_ob, u_ob = _llh_to_enu(
        lat_ref=joined["lat_ob"].to_numpy()[0],
        lon_ref=joined["lon_ob"].to_numpy()[0],
        lat=joined["lat_ob"].to_numpy(),
        lon=joined["lon_ob"].to_numpy(),
        h=joined["h_ob"].to_numpy(),
        h_ref=joined["h_ob"].to_numpy()[0],
    )
    de = e - e_ob
    dn = n - n_ob
    du = u - u_ob
    d3d = np.sqrt(de**2 + dn**2 + du**2)

    diff = joined.with_columns(
        [
            pl.Series("de_m", de),
            pl.Series("dn_m", dn),
            pl.Series("du_m", du),
            pl.Series("d3d_m", d3d),
        ]
    )
    diff_path = out_dir / "diff.parquet"
    write_parquet(diff, diff_path)

    # ZTD median for the day (optional)
    ztd = _read_pride_ztd(day, cfg)
    ztd_med = float(ztd["ztd_m"].median()) if not ztd.is_empty() else float("nan")

    summary = {
        "day": day,
        "n_matched": diff.height,
        "d3d_p50_m": float(np.median(d3d)),
        "d3d_p95_m": float(np.percentile(d3d, 95)),
        "d3d_p99_m": float(np.percentile(d3d, 99)),
        "d3d_max_m": float(np.max(d3d)),
        "ztd_median_m": ztd_med,
    }
    log.info("compare %s: %s", day, summary)
    return summary


def aggregate_residuals(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    rows: list[dict] = []
    for day in list_days(cfg):
        try:
            s = compare_day(day, cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("compare_day(%s) failed: %s", day, exc)
            continue
        if s is not None:
            rows.append(s)
    df = pl.DataFrame(rows)
    out = tables_dir(cfg) / "T_ppp_residuals.parquet"
    write_parquet(df, out)
    df.write_csv(tables_dir(cfg) / "T_ppp_residuals.csv")
    log.info("Aggregated PPP residuals → %s (%d rows)", out, df.height)
    return out


@click.command()
@click.option("--day", default=None)
@click.option("--aggregate/--no-aggregate", default=True)
def main(day: str | None, aggregate: bool) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    cfg = load_config()
    if day:
        compare_day(day, cfg)
    if aggregate:
        aggregate_residuals(cfg)


if __name__ == "__main__":
    main()
