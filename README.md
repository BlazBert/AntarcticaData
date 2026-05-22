# AntarcticaData

Processing pipeline for a multi-band, multi-constellation GNSS dataset recorded
continuously aboard **R/V Laura Bassi** on a 216-day transit
**Trieste → Antarctica → Trieste** (2025-09-26 to 2026-04-29). A u-blox
**ZED-F9P-15B** receiver logged raw observables, broadcast navigation, the full
RF spectrum (`MON-SPAN`), and receiver health telemetry to one daily `.ubx`
file (~700–900 MB; ~180 GB total). Code accompanies a data-description paper
for **Earth System Science Data** (ESSD, Copernicus) and the companion Zenodo
record (DOI: *pending*).

## Layout

```
.
├── ubx/         Binary UBX parser — mmap + structured-dtype, vectorised. No pyubx2.
├── analysis/    Trajectory, QC, multipath, spectrum, TEC, scintillation, anomalies
├── ppp/         PRIDE PPP-AR wrapper + IGS product downloader
├── rinex/       convbin wrapper (RINEX 3.04 conversion)
├── figures/     ESSD figures (fig01_cruise_track, fig05_multipath, fig11_rf_waterfall, …)
├── tables/      T1–T5 in LaTeX + CSV
├── verify/      Tests, validation scripts, parser fixtures
├── config/      pipeline.yaml, receiver.yaml, ports.yaml
└── Snakefile    Idempotent DAG over the 216 days
```

## Install

Python ≥ 3.11. Cartopy needs GEOS/PROJ system libraries
(macOS: `brew install geos proj`; Debian/Ubuntu: `apt install libgeos-dev libproj-dev`).

```sh
git clone https://github.com/<user>/AntarcticaData.git
cd AntarcticaData
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### External binaries

Only needed for the RINEX (`rinex_all`) and PPP (`ppp_all`) Snakemake targets;
the core parse + analysis + figures path runs with Python alone.

| Tool | Used for | Source | Notes |
|---|---|---|---|
| `convbin` | UBX → RINEX 3.04 (`rinex/convbin_runner.py`) | [RTKLIB demo5 fork (rtklibexplorer)](https://github.com/rtklibexplorer/RTKLIB) | Vanilla RTKLIB has F9P signal-mapping bugs (B2a misrouted as B2I, etc.). Build the demo5 fork from source — no binary release. |
| `pdp3` | Kinematic PPP-AR (`ppp/pride_runner.py`) | [PRIDE PPP-AR](https://github.com/PrideLab/PRIDE-PPPAR) ([Geng et al., 2019](https://doi.org/10.1007/s10291-019-0888-1)) | Fortran package; build via the upstream `install.sh`. Tested against PRIDE PPP-AR ≥ 3.0. |

Both must be on `PATH`. Quick check:

```sh
which convbin pdp3
convbin -?         # should print the RTKLIB demo5 usage banner
pdp3 -h            # should print the PRIDE PPP-AR usage banner
```

#### IGS products + CDDIS authentication

`ppp/igs_products.py` downloads per-day IGS final products (SP3, CLK, OSB/BIA,
IONEX, EOP, ATX) from [NASA CDDIS](https://cddis.nasa.gov/archive/gnss/products/).
CDDIS requires a free [NASA Earthdata](https://urs.earthdata.nasa.gov) account
since 2020. Configure once via `~/.netrc`:

```
machine urs.earthdata.nasa.gov login YOUR_USER password YOUR_PASSWORD
```

(`chmod 600 ~/.netrc`). Alternative analysis centres — WUM, COD, ESA, GFZ — can
be selected via the `IGS_AC` constant in `ppp/igs_products.py`.

## Configure

Edit `config/pipeline.yaml`:

```yaml
paths:
  ubx_dir: "/path/to/raw/ubx"      # one YYYYMMDD.ubx per day
  work_dir: "./work"               # all derived artefacts go here
```

`config/receiver.yaml` describes the receiver/antenna setup;
`config/ports.yaml` lists known port stops (used for berth-repeatability QC).

### Data access

The 216 daily `.ubx` files (~180 GB) are archived on Zenodo (DOI: *pending*) —
download them and point `paths.ubx_dir` at the unpacked directory.

## Quickstart — single day

```sh
# Parse one .ubx → 8 Parquet + 1 Zarr block in work/staging/<yyyymmdd>/
python -m ubx.cli parse /path/to/20250930.ubx

# Per-day analysis
python -m analysis.qc_summary  --day 20250930
python -m analysis.trajectory  --day 20250930
python -m analysis.multipath   --day 20250930
python -m analysis.spectrum    --day 20250930
```

## Full pipeline (all 216 days, parallel)

```sh
snakemake --cores 32 -p                  # default target: parse + analysis + figures
snakemake --cores 32 -p analysis_all     # only per-day analysis
snakemake --cores 32 -p figures          # only figure rendering
snakemake --cores 32 -p rinex_all        # RINEX 3.04 conversion (opt-in)
snakemake --cores 32 -p ppp_all          # kinematic PPP-AR (opt-in, heavy)
```

The DAG is idempotent (mtime-based). Day discovery scans
`config.paths.ubx_dir` for `YYYYMMDD.ubx`.

Reference hardware target: ~1 TB RAM / 128 threads (CPU-bound stage is parsing;
PPP is the bottleneck for the full archive). A 32-thread laptop will still
complete a single-day run end-to-end in a few minutes.

## Stages

1. **parse** — `.ubx` → per-message Parquet (`zstd-3`, 128 MB row groups) + Zarr for MON-SPAN spectra.
2. **analysis** — trajectory · QC · multipath (M1/M2, Estey & Meertens) · RF spectrum + RFI · TEC (validation) · scintillation σ_φ proxy.
3. **figures + tables** — ESSD-ready PDFs + T1..T5.
4. **PPP** — kinematic PRIDE PPP-AR per day, compared against onboard `nav_hpposllh`.
5. **RINEX** — `convbin` (RTKLIB demo5) → `obs.rnx` + `nav.rnx` (RINEX 3.04).

## Recorded UBX messages

| Class-ID | Name | Purpose | Rate |
|---|---|---|---|
| 0x02-0x15 | `RXM-RAWX` | Raw observables: pseudorange, carrier phase, Doppler, C/N₀, lockTime, trkStat per signal (≈63 measurements/epoch, 5 constellations) | 2 Hz |
| 0x02-0x13 | `RXM-SFRBX` | Broadcast nav subframes (ephemeris/almanac source) | ~19 Hz total |
| 0x02-0x14 | `RXM-MEASX` | Pre-PVT multi-band measurements | 2 Hz |
| 0x01-0x07 | `NAV-PVT` | Position / velocity / time fix | 2 Hz |
| 0x01-0x14 | `NAV-HPPOSLLH` | High-precision lat / lon / h | 2 Hz |
| 0x01-0x35 | `NAV-SAT` | Per-satellite C/N₀, elevation, azimuth, used-flags | 2 Hz |
| 0x0A-0x38 | `MON-RF` | AGC, noise floor, jamming flag, antenna status per RF block | 1 Hz |
| 0x0A-0x31 | `MON-SPAN` | RF spectrum, 2 RF blocks × 256 bins, 128 MHz span, 500 kHz/bin (RF0 @ 1583.5 MHz = L1; RF1 @ 1191.5 MHz = L2/L5) | 1 Hz |
| 0x0A-0x39 | `MON-SYS` | CPU load, runtime, receiver temperature | ~0.5 Hz |

Five constellations are tracked simultaneously: GPS, Galileo, GLONASS, BeiDou, SBAS.

## Tests

```sh
pytest verify/
```

`verify/fixtures/sample_60s.ubx` is a 60-second clip used for parser
round-trip and schema-invariant tests. Larger validation scripts
(`verify/tec_vs_gim.py`, `verify/ogs_cross_validation.py`,
`verify/track_outliers.py`) need the full Parquet/Zarr outputs and
are not part of the default `pytest` run.


## References

Methods and external tools used by the pipeline:

- **RTKLIB demo5** — Everett, T. (rtklibexplorer fork). RINEX conversion (`convbin`).
  <https://github.com/rtklibexplorer/RTKLIB>
- **PRIDE PPP-AR** — Geng J., Chen X., Pan Y., Mao S., Li C., Zhou J., Zhang K.
  *PRIDE PPP-AR: an open-source software for GNSS PPP ambiguity resolution.*
  GPS Solutions 23:91 (2019). <https://doi.org/10.1007/s10291-019-0888-1>
- **u-blox ZED-F9P interface description** — u-blox AG, *ZED-F9P-15B Data Sheet
  (UBX-22021920)* and *u-blox F9 HPG 1.40 Interface Description*.
- **Code multipath M1/M2** — Estey, L.H. & Meertens, C.M.
  *TEQC: The Multi-Purpose Toolkit for GPS/GLONASS Data.* GPS Solutions 3:42–49 (1999).
- **IGS final products** — Johnston G., Riddell A., Hausler G. *The International
  GNSS Service.* In: *Springer Handbook of Global Navigation Satellite Systems*
  (2017), pp. 967–982. Distributed via NASA CDDIS: <https://www.earthdata.nasa.gov/centers/cddis-daac>

## Citation

> Bertalanic, B. (2026). *R/V Laura Bassi shipborne GNSS dataset,
> Trieste–Antarctica–Trieste (2025-09-26 to 2026-04-29) — raw UBX +
> RINEX 3.04*. Zenodo. DOI: **pending**.

The accompanying ESSD paper DOI will be added on acceptance.

## License

MIT — see `LICENSE`.
