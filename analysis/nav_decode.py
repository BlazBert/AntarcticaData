"""Decode RXM-SFRBX subframes → broadcast ephemeris (skeleton).

This module is a placeholder/skeleton for the optional self-PPP path. It
reads ``staging/<day>/rxm_sfrbx.parquet`` (with ``dwrd_bytes`` blobs) and
emits decoded ephemerides per (gnssId, svId, IODC) into
``derived/eph/<day>.eph.parquet``.

For the immediate ESSD pipeline we don't actually need this — IGS broadcast
nav files (``brdc<DOY>0.YYn``) are higher quality and available from CDDIS.
This skeleton documents the conversion path; the heavy decoding (LNAV,
CNAV, FNAV, I/NAV, BeiDou D1/D2) is a non-trivial task and is best done
via a proven library (e.g. ``georinex``) once we convert UBX → RINEX.
"""

from __future__ import annotations

import logging

import click

from analysis._common import list_days, load_config

log = logging.getLogger(__name__)


def decode_day(day: str, cfg: dict | None = None) -> None:
    log.warning(
        "nav_decode.decode_day(%s) is a skeleton — for ESSD/PPP use IGS BRDC files."
        " The convbin RINEX-NAV output (Stage 5) covers this path.",
        day,
    )


@click.command()
@click.option("--day", "day", default=None)
def main(day: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    days = [day] if day else list_days(cfg)
    for d in days:
        decode_day(d, cfg)


if __name__ == "__main__":
    main()
