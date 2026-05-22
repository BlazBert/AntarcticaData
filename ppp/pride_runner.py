"""Wrap PRIDE PPP-AR (`pdp3`) for kinematic per-day processing.

Inputs (per day):
    rinex_obs   ../work/rinex/<day>/obs.rnx
    rinex_nav   ../work/rinex/<day>/nav.rnx
    products    ../work/ppp/products/<day>/* (from ``ppp.igs_products``)

Outputs:
    ../work/ppp/<day>/kin.pos
    ../work/ppp/<day>/ztd.txt
    ../work/ppp/<day>/residuals.txt
    ../work/ppp/<day>/run.log

Usage:
    python -m ppp.pride_runner --day 20250930

PPP runs in *kinematic* mode with troposphere ZTD as a random-walk
parameter (default σ = 5 mm/√h). Receiver position σ at each epoch is
``inf`` (fully kinematic). PPP-AR is enabled when integer ambiguity
resolution succeeds (typical for IGS final products with multi-GNSS).

This module shells out to ``pdp3`` (the PRIDE PPP-AR CLI). It does not
embed PPP logic — PRIDE is a 70k-line Fortran package; building our own
would be misguided.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import click

from analysis._common import load_config, resolve_path

log = logging.getLogger(__name__)


def _have_pdp3() -> bool:
    return shutil.which("pdp3") is not None


def run_one_day(day: str, cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    if not _have_pdp3():
        raise RuntimeError(
            "pdp3 (PRIDE PPP-AR) not found in PATH. "
            "Install from https://github.com/PrideLab/PRIDE-PPPAR"
        )
    rinex_dir = resolve_path(cfg["paths"]["rinex"]) / day
    obs = rinex_dir / "obs.rnx"
    if not obs.exists():
        raise FileNotFoundError(
            f"Missing RINEX OBS for {day}: {obs}. Run Stage 5 (rinex) first."
        )
    out_dir = resolve_path(cfg["paths"]["ppp"]) / day
    out_dir.mkdir(parents=True, exist_ok=True)
    # PRIDE PPP-AR 3.x CLI. No products-dir flag — pdp3 auto-finds /
    # auto-downloads under $HOME/.PRIDE_PPPAR/data/ unless a -cfg file
    # overrides. Inter-system biases on by default for selected systems
    # via -isb. AR is on by default; -f disables it.
    cmd = [
        "pdp3",
        "-m", "K",            # kinematic
        "-sys", "GREC",       # GPS, GLONASS, Galileo, BeiDou
        "-isb", "REC",        # estimate ISB for non-GPS systems
        "-i", "1",            # 1 Hz output
        str(obs),
    ]
    log.info("$ %s", " ".join(cmd))
    log_file = out_dir / "run.log"
    with log_file.open("w") as fh:
        rc = subprocess.call(cmd, cwd=out_dir, stdout=fh, stderr=fh)
    if rc != 0:
        raise RuntimeError(f"pdp3 failed for {day} (rc={rc}). See {log_file}")
    return {
        "day": day,
        "kin_pos": out_dir / "kin.pos",
        "ztd": out_dir / "ztd.txt",
        "residuals": out_dir / "residuals.txt",
        "log": log_file,
    }


@click.command()
@click.option("--day", required=True)
def main(day: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    res = run_one_day(day)
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
