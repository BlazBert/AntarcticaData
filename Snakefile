# Antarctica 2025/26 GNSS pipeline — Snakemake DAG.
#
# Run from the code/ directory:
#
#     snakemake --cores 32 -p
#     snakemake --cores 32 -p analysis_all   # only the per-day analysis
#     snakemake --cores 32 -p figures        # only figure rendering
#     snakemake --cores 32 -p --until parse  # stop at the parse stage
#
# Day discovery: every YYYYMMDD.ubx in config/pipeline.yaml: paths.ubx_dir.
# Re-run is idempotent (mtime-based).
#
# RINEX (Stage 5) and PPP (Stage 4) are defined but not part of `default`;
# invoke them with `snakemake --cores 32 rinex_all` and `ppp_all`.

import re
from pathlib import Path

import yaml

# ---- Config -----------------------------------------------------------------

CFG = yaml.safe_load(Path("config/pipeline.yaml").read_text())

UBX_DIR = Path(CFG["paths"]["ubx_dir"]).resolve()
STAGING = Path(CFG["paths"]["staging"]).resolve()
DERIVED = Path(CFG["paths"]["derived"]).resolve()
FIGURES_DIR = Path(CFG["paths"]["figures"]).resolve()
TABLES_DIR = Path(CFG["paths"]["tables"]).resolve()
SPECTRUM_ZARR = Path(CFG["paths"]["spectrum_zarr"]).resolve()
LOGS = Path(CFG["paths"]["logs"]).resolve()

PER_DAY_PARQUET = ["nav_pvt", "nav_hpposllh", "nav_sat", "rxm_rawx",
                   "rxm_sfrbx", "rxm_measx", "mon_rf", "mon_sys"]


def _discover_days():
    days = []
    for p in sorted(UBX_DIR.glob("*.ubx")):
        m = re.match(r"(20\d{6})\.ubx$", p.name)
        if m:
            days.append(m.group(1))
    return days


DAYS = _discover_days()
if not DAYS:
    raise WorkflowError(f"No YYYYMMDD.ubx files in {UBX_DIR}")


# ---- Targets ----------------------------------------------------------------


rule all:
    input:
        expand(STAGING / "{day}/nav_pvt.parquet", day=DAYS),
        expand(DERIVED / "qc/{day}.qc.json", day=DAYS),
        expand(DERIVED / "track/{day}.track.parquet", day=DAYS),
        expand(DERIVED / "multipath/{day}.multipath.parquet", day=DAYS),
        expand(DERIVED / "spectrum/{day}.spectrogram.npz", day=DAYS),
        TABLES_DIR / "T1_receiver.csv",
        TABLES_DIR / "T2_signal_coverage.csv",
        TABLES_DIR / "T3_file_inventory.csv",
        TABLES_DIR / "T4_daily_stats.parquet",
        TABLES_DIR / "T5_crossings.csv",
        FIGURES_DIR / "output/fig01_cruise_track_map.pdf",
        FIGURES_DIR / "output/fig11_daily_rf_waterfall.pdf",


rule parse_all:
    """All per-day .ubx -> staging Parquet/Zarr."""
    input:
        expand(STAGING / "{day}/nav_pvt.parquet", day=DAYS)


rule analysis_all:
    """All per-day analysis outputs."""
    input:
        expand(DERIVED / "qc/{day}.qc.json", day=DAYS),
        expand(DERIVED / "track/{day}.track.parquet", day=DAYS),
        expand(DERIVED / "multipath/{day}.multipath.parquet", day=DAYS),
        expand(DERIVED / "spectrum/{day}.spectrogram.npz", day=DAYS),


rule figures:
    """Render all figXX_*.pdf."""
    input:
        FIGURES_DIR / "output/fig01_cruise_track_map.pdf",
        FIGURES_DIR / "output/fig05_multipath_M1M2_vs_elev_lat.pdf",
        FIGURES_DIR / "output/fig11_daily_rf_waterfall.pdf",
        FIGURES_DIR / "output/fig12_temp_agc_timeseries.pdf",


# ---- Stage 1: parse ---------------------------------------------------------


rule parse:
    """One .ubx -> 8 Parquet + 1 Zarr group."""
    input:
        UBX_DIR / "{day}.ubx"
    output:
        nav_pvt=STAGING / "{day}/nav_pvt.parquet",
        nav_hpposllh=STAGING / "{day}/nav_hpposllh.parquet",
        nav_sat=STAGING / "{day}/nav_sat.parquet",
        rxm_rawx=STAGING / "{day}/rxm_rawx.parquet",
        rxm_measx=STAGING / "{day}/rxm_measx.parquet",
        mon_rf=STAGING / "{day}/mon_rf.parquet",
        mon_sys=STAGING / "{day}/mon_sys.parquet",
    log:
        LOGS / "parse/{day}.log"
    threads: 1
    resources:
        mem_mb=8000,
        io_workers=1
    shell:
        """
        mkdir -p $(dirname {log})
        python -m ubx.cli parse {input} > {log} 2>&1
        """


# ---- Stage 2: analysis ------------------------------------------------------


rule qc:
    input:
        STAGING / "{day}/nav_pvt.parquet",
        STAGING / "{day}/nav_sat.parquet",
        STAGING / "{day}/rxm_rawx.parquet",
        STAGING / "{day}/mon_sys.parquet",
        STAGING / "{day}/mon_rf.parquet"
    output:
        DERIVED / "qc/{day}.qc.json"
    threads: 1
    log: LOGS / "qc/{day}.log"
    shell:
        "python -m analysis.qc_summary --day {wildcards.day} --no-aggregate > {log} 2>&1"


rule trajectory:
    input:
        STAGING / "{day}/nav_pvt.parquet",
        STAGING / "{day}/nav_hpposllh.parquet"
    output:
        DERIVED / "track/{day}.track.parquet"
    threads: 1
    log: LOGS / "trajectory/{day}.log"
    shell:
        "python -m analysis.trajectory --day {wildcards.day} --no-aggregate > {log} 2>&1"


rule multipath:
    input:
        STAGING / "{day}/rxm_rawx.parquet",
        STAGING / "{day}/nav_sat.parquet"
    output:
        DERIVED / "multipath/{day}.multipath.parquet"
    threads: 1
    log: LOGS / "multipath/{day}.log"
    shell:
        "python -m analysis.multipath --day {wildcards.day} > {log} 2>&1"


rule spectrum:
    # MON-SPAN is written to Zarr alongside Parquet by the parse rule. We
    # depend on nav_pvt.parquet as a proxy "the day was parsed" signal —
    # Snakemake doesn't have first-class Zarr-directory outputs.
    input:
        STAGING / "{day}/nav_pvt.parquet"
    output:
        DERIVED / "spectrum/{day}.spectrogram.npz"
    threads: 1
    log: LOGS / "spectrum/{day}.log"
    shell:
        "python -m analysis.spectrum --day {wildcards.day} > {log} 2>&1"


rule tec:
    input:
        STAGING / "{day}/rxm_rawx.parquet",
        STAGING / "{day}/nav_sat.parquet"
    output:
        DERIVED / "tec/{day}.tec.parquet"
    threads: 1
    log: LOGS / "tec/{day}.log"
    shell:
        "python -m analysis.tec --day {wildcards.day} > {log} 2>&1"


rule scintillation:
    input:
        STAGING / "{day}/rxm_rawx.parquet"
    output:
        DERIVED / "scint/{day}.scint.parquet"
    threads: 2
    log: LOGS / "scint/{day}.log"
    shell:
        "python -m analysis.scintillation_proxy --day {wildcards.day} > {log} 2>&1"


# ---- Cross-day aggregations -------------------------------------------------


rule aggregate_t4:
    input:
        expand(DERIVED / "qc/{day}.qc.json", day=DAYS)
    output:
        TABLES_DIR / "T4_daily_stats.parquet"
    log: LOGS / "agg/T4.log"
    shell:
        "python -m analysis.qc_summary --aggregate > {log} 2>&1"


rule aggregate_track:
    input:
        expand(DERIVED / "track/{day}.track.parquet", day=DAYS)
    output:
        DERIVED / "track/track.all.parquet",
        DERIVED / "track/track.geojson",
        TABLES_DIR / "T5_crossings.csv"
    log: LOGS / "agg/track.log"
    shell:
        "python -m analysis.trajectory --aggregate > {log} 2>&1"


# ---- Stage 3: figures + tables ----------------------------------------------


rule fig_cruise_track:
    input:
        DERIVED / "track/track.all.parquet"
    output:
        FIGURES_DIR / "output/fig01_cruise_track_map.pdf"
    shell:
        "python -m figures.make_figures --only fig01"


rule fig_multipath:
    input:
        expand(DERIVED / "multipath/{day}.multipath.parquet", day=DAYS)
    output:
        FIGURES_DIR / "output/fig05_multipath_M1M2_vs_elev_lat.pdf"
    shell:
        "python -m figures.make_figures --only fig05"


rule fig_spectrum:
    input:
        expand(DERIVED / "spectrum/{day}.spectrogram.npz", day=DAYS)
    output:
        FIGURES_DIR / "output/fig11_daily_rf_waterfall.pdf"
    shell:
        "python -m figures.make_figures --only fig11"


rule fig_temp_agc:
    input:
        expand(STAGING / "{day}/mon_sys.parquet", day=DAYS),
        expand(STAGING / "{day}/mon_rf.parquet", day=DAYS)
    output:
        FIGURES_DIR / "output/fig12_temp_agc_timeseries.pdf"
    shell:
        "python -m figures.make_figures --only fig12"


rule t1:
    output: TABLES_DIR / "T1_receiver.csv"
    shell: "python -m tables.make_tables --only T1"


rule t2:
    input: expand(STAGING / "{day}/rxm_rawx.parquet", day=DAYS)
    output: TABLES_DIR / "T2_signal_coverage.csv"
    shell: "python -m tables.make_tables --only T2"


rule t3:
    output: TABLES_DIR / "T3_file_inventory.csv"
    shell: "python -m tables.make_tables --only T3"


# ---- Stage 4: PPP (parallel — heavyweight, opt-in) --------------------------


rule ppp_all:
    """Per-day kinematic PPP-AR via PRIDE PPP-AR. Opt-in target."""
    input:
        expand(DERIVED / "ppp/{day}/kin.pos", day=DAYS)


rule ppp_one:
    input:
        rinex_obs=DERIVED / "rinex/{day}/obs.rnx",
        rinex_nav=DERIVED / "rinex/{day}/nav.rnx",
    output:
        kin=DERIVED / "ppp/{day}/kin.pos"
    threads: 2
    log: LOGS / "ppp/{day}.log"
    shell:
        "python -m ppp.pride_runner --day {wildcards.day} > {log} 2>&1"


# ---- Stage 5: RINEX (deferred archival) -------------------------------------


rule rinex_all:
    input:
        expand(DERIVED / "rinex/{day}/obs.rnx", day=DAYS)


rule rinex_convbin:
    input:
        UBX_DIR / "{day}.ubx"
    output:
        DERIVED / "rinex/{day}/obs.rnx",
        DERIVED / "rinex/{day}/nav.rnx",
    threads: 1
    log: LOGS / "rinex/{day}.log"
    shell:
        "python -m rinex.convbin_runner --day {wildcards.day} > {log} 2>&1"
