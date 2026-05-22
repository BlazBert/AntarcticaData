"""Correlate MON-SPAN event counts with the planetary Kp index.

If polar MON-SPAN excursions are real ionospheric scintillation, they
should correlate with geomagnetic activity. This script:

1. Loads ``T_antarctic.csv`` (per-day event counts, lat, lon).
2. Downloads the daily-mean Kp index from the GFZ Potsdam SWE archive
   for the cruise window (no auth required).
3. Joins on date, computes Spearman correlation between
   ``rfi_events_total`` and ``Kp_max`` for sub-Antarctic days.
4. Emits ``derived/scint_vs_kp.csv`` and a quick scatter plot.

The Kp index is a 3-hour planetary geomagnetic activity index ranging
0–9. ``Kp_max`` per day = daily peak; ``Kp_mean`` = daily mean.

CLI:
    python -m verify.correlate_kp
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import polars as pl

from analysis._common import derived_dir, load_config, resolve_path

log = logging.getLogger(__name__)

# GFZ Kp/Ap archive (no auth needed, plain text)
GFZ_KP_URL = (
    "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
)


def _fetch_kp(start: str, end: str) -> pl.DataFrame:
    """Daily Kp summary for [start, end] (YYYY-MM-DD strings)."""
    import requests  # noqa: PLC0415

    log.info("Fetching Kp from %s", GFZ_KP_URL)
    text = requests.get(GFZ_KP_URL, timeout=60).text
    rows = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # GFZ Kp_ap_Ap_SN_F107 format columns (space-separated):
        #   YYYY MM DD days days_m Bsr dB ... Kp1..Kp8 ap1..ap8 Ap SN F107obs F107adj D
        toks = line.split()
        if len(toks) < 19:
            continue
        try:
            yyyy, mm, dd = int(toks[0]), int(toks[1]), int(toks[2])
        except ValueError:
            continue
        date = f"{yyyy:04d}-{mm:02d}-{dd:02d}"
        if date < start or date > end:
            continue
        try:
            kp_vals = [float(t) for t in toks[7:15]]   # Kp1..Kp8
        except (ValueError, IndexError):
            continue
        rows.append({
            "date": date,
            "Kp_max": max(kp_vals),
            "Kp_mean": sum(kp_vals) / len(kp_vals),
            "Kp_n_storm_lvl": sum(1 for v in kp_vals if v >= 5.0),
        })
    return pl.DataFrame(rows)


def _spearman(x: pl.Series, y: pl.Series) -> float:
    a = x.to_numpy().astype(float)
    b = y.to_numpy().astype(float)
    if len(a) < 3:
        return float("nan")
    import numpy as np  # noqa: PLC0415

    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.std() * rb.std() * len(ra))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


@click.command()
@click.option("--ant-csv", default=None, type=click.Path(),
              help="Path to T_antarctic.csv (default: tables/T_antarctic.csv)")
@click.option("--out-csv", default=None, type=click.Path(),
              help="Output joined CSV (default: derived/scint_vs_kp.csv)")
@click.option("--out-png", default=None, type=click.Path(),
              help="Optional scatter PNG (default: figures/output/fig_scint_vs_kp.png)")
def main(ant_csv: str | None, out_csv: str | None, out_png: str | None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    cfg = load_config()
    ant_path = Path(ant_csv) if ant_csv else (
        resolve_path(cfg["paths"]["tables"]) / "T_antarctic.csv"
    )
    if not ant_path.exists():
        raise click.ClickException(
            f"{ant_path} not found — run `python -m verify.antarctic_focus "
            "--csv-out ../work/tables/T_antarctic.csv` first"
        )
    ant = pl.read_csv(ant_path)
    if ant.is_empty():
        raise click.ClickException("T_antarctic.csv is empty")
    ant = ant.with_columns(
        (pl.col("day").cast(pl.Utf8).str.slice(0, 4) + "-"
         + pl.col("day").cast(pl.Utf8).str.slice(4, 2) + "-"
         + pl.col("day").cast(pl.Utf8).str.slice(6, 2)).alias("date")
    )
    start = ant["date"].min()
    end = ant["date"].max()
    log.info("Cruise window: %s to %s", start, end)
    kp = _fetch_kp(start, end)
    if kp.is_empty():
        raise click.ClickException("Kp fetch returned empty — check internet access")

    joined = ant.join(kp, on="date", how="left")
    out_path = Path(out_csv) if out_csv else (derived_dir(cfg) / "scint_vs_kp.csv")
    joined.write_csv(out_path)
    log.info("Wrote %s", out_path)

    j = joined.filter(
        pl.col("Kp_max").is_finite() & pl.col("rfi_events_total").is_finite()
    )
    sp = _spearman(j["rfi_events_total"], j["Kp_max"])
    print(f"\nSpearman ρ(rfi_events_total, Kp_max) = {sp:.3f}  (n = {j.height})")
    print("\nTop 10 days by event count:")
    print(j.sort("rfi_events_total", descending=True).select([
        "date", "lat", "lon", "rfi_events_total", "Kp_max", "Kp_mean", "Kp_n_storm_lvl"
    ]).head(10))

    # Optional scatter
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        png_path = Path(out_png) if out_png else (
            resolve_path(cfg["paths"]["figures"]) / "output" / "fig_scint_vs_kp.png"
        )
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            j["Kp_max"].to_numpy(),
            j["rfi_events_total"].to_numpy(),
            c=j["lat"].to_numpy(),
            cmap="viridis_r",
            s=20,
            edgecolor="black", linewidth=0.3,
        )
        cb = fig.colorbar(ax.collections[0], ax=ax)
        cb.set_label("Latitude (°N)")
        ax.set_xlabel("Daily peak Kp index")
        ax.set_ylabel("MON-SPAN events / day")
        ax.set_yscale("log")
        ax.set_title(f"Polar MON-SPAN events vs Kp  "
                     f"(Spearman ρ = {sp:.3f}, n = {j.height})")
        fig.tight_layout()
        fig.savefig(png_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        log.info("Wrote %s", png_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Scatter PNG skipped: %s", exc)


if __name__ == "__main__":
    main()
