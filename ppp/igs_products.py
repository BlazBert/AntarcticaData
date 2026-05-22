"""Download IGS final products needed by PRIDE PPP-AR.

Per day we need:

* SP3 — orbits
* CLK — clocks (30 s sampling for "final"; 5 s for "rapid")
* OSB / BIA — observation-specific signal biases
* IONEX — ionosphere maps
* EOP / ERP — Earth orientation parameters
* ATX — antenna phase-center model

CDDIS is the canonical source. Anonymous ftp/https works, but a NASA
Earthdata login is now required for some products (curl -n / .netrc).
This module is a thin wrapper — pass the date and it builds the right
URL, downloads with retry, and stages the file under
``${ppp_dir}/products/<yyyy_doy>/``.

NOTE: This module does not run in the immediate analysis path; PPP is a
secondary product. Call from ``snakemake ppp_all`` only.
"""

from __future__ import annotations

import datetime as dt
import logging
import shutil
from pathlib import Path

import click
import requests

log = logging.getLogger(__name__)

CDDIS_BASE = "https://cddis.nasa.gov/archive/gnss/products"
IGS_AC = "IGS"  # final IGS combined; alternatives: WUM, COD, ESA, GFZ


def _gpsweek_dow(date: dt.date) -> tuple[int, int]:
    """GPS week + day-of-week (0=Sun..6=Sat) for ``date``."""
    gps_epoch = dt.date(1980, 1, 6)
    days = (date - gps_epoch).days
    return divmod(days, 7)


def _doy(date: dt.date) -> int:
    return date.timetuple().tm_yday


def product_urls(date: dt.date, ac: str = IGS_AC) -> dict[str, str]:
    """Return canonical CDDIS URLs for the per-day products of ``date``.

    Uses the long IGS filename convention (``IGS0OPSFIN_YYYYDDD0000_01D_05M_ORB.SP3.gz``)
    introduced 2022-11-27. For older dates this would need the short name.
    """
    week, _ = _gpsweek_dow(date)
    doy = _doy(date)
    yyyy = date.year
    base = f"{CDDIS_BASE}/{week:04d}"
    long_stub = f"{ac}0OPSFIN_{yyyy:04d}{doy:03d}0000_01D"
    return {
        "sp3": f"{base}/{long_stub}_05M_ORB.SP3.gz",
        "clk": f"{base}/{long_stub}_30S_CLK.CLK.gz",
        "obx": f"{base}/{long_stub}_05M_ATT.OBX.gz",
        "bia": f"{base}/{long_stub}_01D_OSB.BIA.gz",
        "erp": f"{base}/{long_stub}_01D_ERP.ERP.gz",
        # IONEX (ionosphere) — per-day, separate path
        "ionex": (
            f"{CDDIS_BASE.replace('products', 'ionex')}/{yyyy:04d}/{doy:03d}/"
            f"COD0OPSFIN_{yyyy:04d}{doy:03d}0000_01D_01H_GIM.INX.gz"
        ),
    }


def static_urls() -> dict[str, str]:
    """Static products needed once (not per day)."""
    return {
        "atx": "https://files.igs.org/pub/station/general/igs20.atx",
    }


def download(url: str, dest: Path, *, timeout: int = 60, retries: int = 3) -> Path:
    """Stream a URL to ``dest`` with simple retry. Skips if dest exists."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            log.info("GET %s -> %s (attempt %d)", url, dest, attempt)
            with requests.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                with dest.open("wb") as fh:
                    shutil.copyfileobj(resp.raw, fh)
            return dest
        except Exception as exc:  # noqa: BLE001
            log.warning("attempt %d failed: %s", attempt, exc)
            if dest.exists():
                dest.unlink()
            if attempt == retries:
                raise


def fetch_day(day: str, dest_root: Path) -> dict[str, Path]:
    date = dt.date(int(day[0:4]), int(day[4:6]), int(day[6:8]))
    urls = product_urls(date)
    out_dir = dest_root / day
    out: dict[str, Path] = {}
    for kind, url in urls.items():
        suffix = url.rsplit("/", 1)[-1]
        out[kind] = download(url, out_dir / suffix)
    # Static products — keep one copy at the root
    for kind, url in static_urls().items():
        out[kind] = download(url, dest_root / url.rsplit("/", 1)[-1])
    return out


@click.command()
@click.option("--day", required=True)
@click.option("--dest", default="../work/ppp/products")
def main(day: str, dest: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    paths = fetch_day(day, Path(dest))
    for k, p in paths.items():
        print(f"{k}: {p}")


if __name__ == "__main__":
    main()
