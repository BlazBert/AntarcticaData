# `verify.track_outliers` — run notes

Diagnostic on top of the existing pipeline outputs. Reads-only — does
not modify any staged or derived parquet.

## Prerequisites (already in the pipeline)

Make sure these have run for the days you care about:

```
python -m analysis.trajectory --aggregate          # → derived/track/track.all.parquet
python -m analysis.anomalies                       # → derived/anomalies/<day>.anomalies.parquet
python -m analysis.spoofing_check                  # → derived/spoofing/<day>.spoofing.parquet
python -m analysis.qc_summary                      # → tables/T4_daily_stats.parquet (for cycle slips)
```

## Run

```
python -m verify.track_outliers
```

Other useful invocations:

```
# Coarser downsampling (1 fix per 15 min) for a smaller map / CSV
python -m verify.track_outliers --downsample-min 15

# Tighter pDOP threshold (flags more clean-track fixes as low_geometry)
python -m verify.track_outliers --pdop-threshold 3.0

# CSVs only — skip the cartopy and folium maps
python -m verify.track_outliers --no-map --no-html
```

## Outputs

Under `work/tables/`:

* `T_track_clean_downsampled.csv` — one fix per N minutes from the
  already-filtered cruise track. Drop this directly into QGIS / kepler.gl
  if you want a clean route map.
* `T_track_outliers.csv` — every flagged point, columns:
  `t_ns, lat, lon, fixType, numSV, hAcc_m, vAcc_m, category,
   prev_lat, prev_lon, dt_s, dist_km, implied_speed_kn,
   jamInd_L1, jamInd_L2L5, agcCnt_L1, agcCnt_L2L5,
   day, suspicion, region, class`
* `T_track_outliers_by_region.csv` — region × class pivot. Headline
  table for the paper appendix.

Under `work/figures/output/`:

* `track_outliers_map.pdf` — two-panel cartopy: clean track (left,
  time-graded), outliers coloured by class (right). Falls back to
  plain matplotlib if cartopy is not installed.
* `track_outliers_map.html` — folium interactive. Each class is a
  toggleable layer; clicking a marker shows the full context popup.

## Class taxonomy (precedence — top wins)

| class | meaning |
|---|---|
| `possible_spoof` | Speed outlier + tight `hAcc` + nominal `jamInd` — the canonical spoofing fingerprint already flagged by `analysis.anomalies`. |
| `spoofing_correlated` | `spoofing_check` suspicion ≥ 3 / 7. Worth manual review. |
| `rfi_correlated` | `jamInd > 64` at the same epoch on either RF block. |
| `scintillation_candidate` | Sub-Antarctic (lat ≤ −60°) anomaly AND that day's total cycle-slip count is in the upper quartile of the cruise. Keep for paper 2. |
| `cold_start` | Receiver clock not yet GPS-synced (`t_ns` outside 2017–2030 window). |
| `null_island` | `lat = lon = 0` (fix re-acquisition cache miss). |
| `low_geometry` | Passed the clean-track filter but pDOP > threshold (default 5.0). |
| `no_fix` | `fixType < 3`. |
| `speed_outlier` | Implied speed > 100 kn vs previous good fix, none of the above matched. |
| `unclassified` | Anomaly tag set but nothing matched. |

## What we already learned from the 4-day local sample

Across the 4 staged days (20250926, 20250930, 20260127, 20260415):

* 2,150 outliers total.
* `rfi_correlated` dominates: 1,279 in the Mediterranean (harbour
  RFI at Trieste pier) and 862 at the Antarctic station.
* The 862 Antarctic flags reflect the **elevated L2/L5 `jamInd`
  baseline** documented in §5.5 of the paper — they're a receiver-
  reporting artefact, not real interference. A future tightening
  would be: flag only excursions above the per-day median jamInd,
  not the absolute threshold of 64.
* 9 `no_fix` rows — all at Trieste pier on 20250926, the cold-start
  before the first GPS lock. Lat/lon are NOT null-island — they're
  the cached pier coordinates the receiver carries while seeking.

## Interpretation hints (for the manuscript)

* `rfi_correlated` in a sub-Antarctic / open-ocean context (away
  from harbours and the Antarctic-station baseline) is the
  interesting signal — likely environmental.
* `scintillation_candidate` is the candidate set for the polar
  ionospheric paper (paper 2). Cross-check against the planetary
  Kp index via `verify.correlate_kp`.
* `possible_spoof` and `spoofing_correlated` should currently be
  empty over the cruise (the cruise route avoids documented
  spoofing zones — see §5.6).

## Iteration

To tighten or relax the criteria, edit the constants in
`verify/track_outliers.py`:

* `HIGH_JAM_THRESHOLD` — change `rfi_correlated` cut.
* `pdop_threshold` CLI flag — tightens `low_geometry`.
* `_scintillation_slip_threshold` — change the slip-burst percentile.

After re-running, diff `T_track_outliers_by_region.csv` against the
previous version to see which class boundaries moved.
