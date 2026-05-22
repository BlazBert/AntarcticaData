"""Command-line entry point: ``python -m ubx.cli``.

Subcommands:

* ``parse <ubx_path>`` — parse a single .ubx file to ``staging/<day>/``.
* ``parse-all`` — parse every ``YYYYMMDD.ubx`` in the configured ``ubx_dir``.
* ``count <ubx_path>`` — quick frame-count summary (no Parquet/Zarr writes).
* ``decode-sample <ubx_path>`` — decode one of each known message and pretty-print.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import yaml

from ubx.messages import CLASS_ID_TO_NAME, MSG_DECODERS, MessageName
from ubx.parallel import discover_ubx_files, parse_one_day, run_pool
from ubx.parser import iter_frames, open_mmap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("ubx.cli")

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "pipeline.yaml"


def _load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else _DEFAULT_CONFIG
    with p.open() as fh:
        return yaml.safe_load(fh)


def _resolve(path_str: str, base: Path) -> Path:
    """Resolve a path that may be absolute or relative to ``base``."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base / p).resolve()


@click.group()
@click.option(
    "--config",
    "config_path",
    default=str(_DEFAULT_CONFIG),
    show_default=True,
    type=click.Path(),
    help="Path to pipeline.yaml",
)
@click.pass_context
def main(ctx: click.Context, config_path: str) -> None:
    """UBX binary parser & pipeline driver."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["config"] = _load_config(config_path)
    ctx.obj["base"] = Path(config_path).resolve().parent.parent  # code/


@main.command()
@click.argument("ubx_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--skip",
    "skip_msgs",
    multiple=True,
    type=click.Choice(
        ["nav_pvt", "nav_hpposllh", "nav_sat", "rxm_rawx", "rxm_sfrbx",
         "rxm_measx", "mon_rf", "mon_sys", "mon_span"]
    ),
    help="Message types to skip (counted, not written). Repeatable.",
)
@click.pass_context
def parse(ctx: click.Context, ubx_path: str, skip_msgs: tuple[str, ...]) -> None:
    """Parse one .ubx file → Parquet + Zarr in staging/<day>/."""
    cfg = ctx.obj["config"]
    base = ctx.obj["base"]
    staging = _resolve(cfg["paths"]["staging"], base)
    spectrum = _resolve(cfg["paths"]["spectrum_zarr"], base)
    log.info("parse %s -> staging=%s spectrum=%s skip=%s", ubx_path, staging, spectrum, skip_msgs)
    st = parse_one_day((str(ubx_path), str(staging), str(spectrum), frozenset(skip_msgs)))
    print(json.dumps(_stats_to_dict(st), indent=2, default=str))
    if st.error:
        sys.exit(1)


@main.command(name="parse-all")
@click.option("--workers", default=None, type=int, help="Override worker count")
@click.option(
    "--days",
    default=None,
    help="Comma-separated YYYYMMDD list (filters files)",
)
@click.pass_context
def parse_all(ctx: click.Context, workers: int | None, days: str | None) -> None:
    """Parse every YYYYMMDD.ubx in the configured ubx_dir."""
    cfg = ctx.obj["config"]
    base = ctx.obj["base"]
    ubx_dir = _resolve(cfg["paths"]["ubx_dir"], base)
    staging = _resolve(cfg["paths"]["staging"], base)
    spectrum = _resolve(cfg["paths"]["spectrum_zarr"], base)
    files = discover_ubx_files(ubx_dir)
    if days:
        wanted = set(days.split(","))
        files = [f for f in files if any(d in f.name for d in wanted)]
    if not files:
        log.error("No .ubx files matched in %s", ubx_dir)
        sys.exit(2)
    n_workers = workers or int(cfg["parallel"]["workers"])
    maxtask = int(cfg["parallel"]["maxtasksperchild"])
    results = run_pool(
        files,
        staging_root=staging,
        spectrum_root=spectrum,
        workers=n_workers,
        maxtasksperchild=maxtask,
    )
    summary = {st.day: _stats_to_dict(st) for st in results}
    out = staging.parent / "parse-all.summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    failures = [st.day for st in results if st.error]
    log.info("Wrote summary -> %s", out)
    if failures:
        log.error("Failures: %s", failures)
        sys.exit(1)


@main.command()
@click.argument("ubx_path", type=click.Path(exists=True, dir_okay=False))
def count(ubx_path: str) -> None:
    """Frame-count summary (no writes)."""
    counts: dict[tuple[int, int], int] = {}
    n = 0
    with open_mmap(ubx_path) as mm:
        for fr in iter_frames(mm):
            counts[(fr.class_id, fr.msg_id)] = counts.get((fr.class_id, fr.msg_id), 0) + 1
            n += 1
    rows = []
    for (c, m), k in sorted(counts.items()):
        name = CLASS_ID_TO_NAME.get((c, m))
        rows.append((f"0x{c:02X}-0x{m:02X}", name.value if name else "?", k))
    print(f"{'Class-ID':<12}{'Name':<16}{'Count':>10}")
    for r in rows:
        print(f"{r[0]:<12}{r[1]:<16}{r[2]:>10}")
    print(f"\nTotal: {n}")


@main.command(name="decode-sample")
@click.argument("ubx_path", type=click.Path(exists=True, dir_okay=False))
def decode_sample(ubx_path: str) -> None:
    """Decode one of each known message and pretty-print."""
    seen: set[MessageName] = set()
    ctx_d: dict[str, int] = {"t_ns_pvt": 0}
    samples: dict[str, dict] = {}
    with open_mmap(ubx_path) as mm:
        for fr in iter_frames(mm):
            key = (fr.class_id, fr.msg_id)
            name = CLASS_ID_TO_NAME.get(key)
            if name is None or name in seen:
                continue
            decoder = MSG_DECODERS[name]
            d = decoder(fr.payload, ctx_d)
            if d is None:
                continue
            seen.add(name)
            if name is MessageName.NAV_PVT:
                ctx_d["t_ns_pvt"] = int(d["t_ns"][0])
            samples[name.value] = {
                k: (
                    v.tolist() if hasattr(v, "tolist") else v
                )
                if not (hasattr(v, "shape") and len(getattr(v, "shape", ())) > 1)
                else f"<ndarray shape={v.shape} dtype={v.dtype}>"
                for k, v in d.items()
            }
            if len(seen) >= len(MSG_DECODERS):
                break
    print(json.dumps(samples, indent=2, default=str))


def _stats_to_dict(st) -> dict:
    return {
        "day": st.day,
        "src": st.src,
        "bytes_read": st.bytes_read,
        "n_frames_total": st.n_frames_total,
        "n_frames_decoded": st.n_frames_decoded,
        "counts": st.counts,
        "elapsed_s": st.elapsed_s,
        "error": st.error,
    }


if __name__ == "__main__":
    main(obj={})
