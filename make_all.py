"""Thin CLI wrapper around Snakemake.

    python make_all.py --stage all      --cores 32
    python make_all.py --stage parse    --cores 32
    python make_all.py --stage analysis --cores 32
    python make_all.py --stage figures  --cores 8
    python make_all.py --stage ppp      --cores 32
    python make_all.py --stage rinex    --cores 16
    python make_all.py --dry-run

The ``Snakefile`` (next to this file) carries the actual DAG. This script
just maps a friendlier ``--stage`` name to a snakemake target.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import click

CODE_DIR = Path(__file__).resolve().parent

STAGE_TARGETS = {
    "all": ["all"],
    "parse": ["parse_all"],
    "analysis": ["analysis_all"],
    "figures": ["figures"],
    "tables": ["t1", "t2", "t3"],
    "aggregate": ["aggregate_t4", "aggregate_track"],
    "ppp": ["ppp_all"],
    "rinex": ["rinex_all"],
}


@click.command()
@click.option(
    "--stage",
    type=click.Choice(list(STAGE_TARGETS.keys())),
    default="all",
    show_default=True,
)
@click.option("--cores", default=32, type=int, show_default=True)
@click.option("--dry-run", "dry_run", is_flag=True, default=False)
@click.option(
    "--days",
    default=None,
    help="Comma-separated YYYYMMDD list to restrict the run",
)
@click.option("--keepgoing/--no-keepgoing", default=True, show_default=True)
def cli(stage: str, cores: int, dry_run: bool, days: str | None, keepgoing: bool) -> None:
    targets = STAGE_TARGETS[stage]
    cmd = ["snakemake", "--cores", str(cores), "-p", *targets]
    if dry_run:
        cmd.append("-n")
    if keepgoing:
        cmd.append("--keep-going")
    if days:
        cmd.extend(["--config", f"days={days}"])
    print("$", " ".join(shlex.quote(c) for c in cmd))
    rc = subprocess.call(cmd, cwd=CODE_DIR)
    sys.exit(rc)


if __name__ == "__main__":
    cli()
