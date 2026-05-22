"""Wrap convbin (RTKLIB demo5 fork) to produce RINEX 3.04 from .ubx.

Per day, runs ``convbin`` against ``${ubx_dir}/<day>.ubx`` and writes the
observation and navigation files under
``${derived}/rinex/<day>/{obs.rnx, nav.rnx}``.

We use the rtklibexplorer demo5 fork rather than upstream RTKLIB because
upstream has F9P signal-mapping bugs (B2a misrouted as B2I, NavIC L5
unsupported on some firmware revisions, etc.). ``convbin`` must be on
PATH.

Usage:
    python -m rinex.convbin_runner --day 20250930
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import click

from analysis._common import derived_dir, list_days, load_config, resolve_path

log = logging.getLogger(__name__)


def _have_convbin() -> bool:
    return shutil.which("convbin") is not None


def run_one_day(day: str, cfg: dict | None = None,
                *, marker: str = "JSI1",
                receiver_type: str = "UBLOX ZED-F9P-15B",
                receiver_firmware: str = "HPG L1L5 1.40",
                antenna_type: str = "TRSAX4E02",
                radome: str = "NONE") -> dict[str, Path]:
    cfg = cfg or load_config()
    if not _have_convbin():
        raise RuntimeError(
            "convbin not in PATH. Build the RTKLIB demo5 fork from "
            "https://github.com/rtklibexplorer/RTKLIB (see "
            "app/consapp/convbin/gcc/) and copy the binary to PATH."
        )
    ubx_path = resolve_path(cfg["paths"]["ubx_dir"]) / f"{day}.ubx"
    if not ubx_path.exists():
        raise FileNotFoundError(f"Missing UBX log for {day}: {ubx_path}")
    out_dir = derived_dir(cfg) / "rinex" / day
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_out = out_dir / "obs.rnx"
    nav_out = out_dir / "nav.rnx"

    # convbin flags:
    #   -r ubx      input format (u-blox UBX binary)
    #   -v 3.04     RINEX version
    #   -od         include Doppler observations
    #   -os         include SNR observations
    #   -oi         include ionospheric corrections in nav file
    #   -ot         include time-system corrections in nav file
    #   -f 5        five frequencies (L1 + L5/E5a/B2a multi-constellation)
    #   -y 2 -y 3 -y 4 -y 5 -y 6 -y 7
    #               enable GPS, GLO, GAL, QZS, SBS, BDS, IRN systems
    #   -hm MARKER  marker name
    #   -ht TYPE    marker type
    #   -hr SER/TYPE/VER     receiver serial / type / firmware version
    #   -ha SER/TYPE/RADOME  antenna serial / type / radome
    #
    # Both -hr and -ha take ONE argument, a slash-separated string. Earlier
    # versions of this script passed three separate space-separated tokens,
    # which convbin silently truncated to the first token (the serial),
    # leaving the type and radome fields blank in the RINEX header. PRIDE
    # PPP-AR then looked up an empty antenna type, fell back to NONE, and
    # applied zero PCO/PCV across all bands. The antenna type must use the
    # IGS-standard 16-char manufacturer-prefix code (TRSAX4E02) to match
    # the entry in IGS20.atx; the marketing name "AX4E02" does not match.
    #
    # We deliberately do NOT pass -y (system exclude). The demo5 fork's
    # default is to emit every constellation the receiver tracks; passing
    # -y in the past was a no-op that confused readers.
    cmd = [
        "convbin",
        "-r", "ubx",
        "-v", "3.04",
        "-od", "-os", "-oi", "-ot",
        "-f", "5",
        "-hm", marker,
        "-ht", "NON_GEODETIC",
        "-hr", f"0/{receiver_type}/{receiver_firmware}",
        "-ha", f"0/{antenna_type}/{radome}",
        "-o", str(obs_out),
        "-n", str(nav_out),
        str(ubx_path),
    ]
    log.info("$ %s", " ".join(cmd))
    log_file = out_dir / "convbin.log"
    with log_file.open("w") as fh:
        rc = subprocess.call(cmd, cwd=out_dir, stdout=fh, stderr=fh)
    if rc != 0:
        raise RuntimeError(
            f"convbin failed for {day} (rc={rc}). See {log_file}"
        )
    if not obs_out.exists():
        raise RuntimeError(
            f"convbin returned 0 but {obs_out} missing. See {log_file}"
        )
    return {"day": day, "obs": obs_out, "nav": nav_out, "log": log_file}


@click.command()
@click.option("--day", default=None,
              help="YYYYMMDD; if omitted, run all staged days")
@click.option("--marker", default="JSI1",
              help="4-char IGS marker name (default JSI1)")
def main(day: str | None, marker: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    cfg = load_config()
    days = [day] if day else list_days(cfg)
    if not days:
        raise click.ClickException("No days found.")
    for d in days:
        try:
            res = run_one_day(d, cfg, marker=marker)
            for k, v in res.items():
                print(f"{k}: {v}")
        except Exception as exc:  # noqa: BLE001
            log.error("RINEX conversion failed for %s: %s", d, exc)


if __name__ == "__main__":
    main()
