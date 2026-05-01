# RUNBOOK — Full pipeline execution

This document describes the exact command sequence required to reproduce
the v5.1 manuscript headline numbers from raw input data.

The pipeline has 17 scripts plus the figure generator. Each script is
independent unless its "Depends on" line lists another. Running them in
the order below respects all dependencies.

Total wall time: approximately 8-12 minutes on a 2024-vintage laptop CPU
(single core, no GPU). The dominant cost is `07_chirps_rainfall.py` which
processes 264 GeoTIFFs.

---

## Prerequisites

1. Python 3.10, 3.11, or 3.12 in a virtual environment.
2. Dependencies installed: `pip install -r requirements.txt`.
3. Raw input data in place (see README "Data Acquisition").
4. `outputs/timeseries/` and `outputs/figures/` will be created by the scripts;
   no manual setup required.

---

## Execution sequence

All commands run from the `scripts/` directory.

### Stage 1: Upstream pipeline (independent, can be parallelised if desired)

```
python 01_grace_process.py            # ~30s  -> GRACE_TWS_IGP.csv
python 02_gldas_process.py            # <5s   -> GLDAS_processed.csv
python 03_modis_et_process.py         # <5s   -> MODIS_ET_monthly_2002_2023.csv,
                                      #          ET_bias_2002_2023.csv
python 04_well_process.py             # <10s  -> shallow_wells_annual.csv,
                                      #          deep_wells_annual.csv
python 07_chirps_rainfall.py          # ~2min -> CHIRPS_monthly_IGP.csv,
                                      #          CHIRPS_JJAS_annual.csv
```

`05_water_balance.py` and `06_ssa_trend.py` depend on Stage 1 outputs:

```
python 05_water_balance.py            # <5s   -> water_balance_standard.csv
                                      #   Depends on: 01, 02
python 06_ssa_trend.py                # ~5s   -> SSA_components.csv
                                      #   Depends on: 05
```

### Stage 2: Analysis (R-scripts)

```
python R1_well_filter_application.py  # <10s
                                      #   Depends on: (raw data only)
python R2_depth_cut_sensitivity.py    # ~30s
                                      #   Depends on: R1, 07
python R3_subseasonal_chirps.py       # ~30s-3min depending on whether
                                      #   daily CHIRPS TIFs are present
                                      #   Depends on: (raw data only)
python R4_change_point.py             # ~20s
                                      #   Depends on: 05, 06
python R6_ssa_quantitative.py         # ~10s
                                      #   Depends on: 05
python R7_source3_reframe.py          # <5s
                                      #   Depends on: 03
python R8_alluvial_mask.py            # ~30s
                                      #   Depends on: 01, raw GRACE NetCDF
python R9_monte_carlo_v3.py           # ~30s
                                      #   Depends on: 05, R7
```

### Stage 3: Figures

```
python F_make_v5_figures.py           # ~20s
                                      #   Depends on: 02, 05, 07, R2, R6, R7, R8, R9
```

---

## Dependency graph

```
                    Raw GRACE NetCDF
                          |
                       [01]------------+
                          |            |
   Raw GLDAS Giovanni     |            |
          |               |            |
       [02]----+          |            |
          |    |          |            |
   Raw MODIS   |          |            |
          |   [03]        |            |
          |    |          |            |
          |    +-->[R7]   |            |
          |    |    |     |            |
   Raw CGWB    |    |     |            |
          |    |    |     |            |
       [04]    |    |     |            |
       [R1]----+----+     |            |
          |    |    |     |            |
          |   [05]<-+-----+            |
          |    |    |                  |
          |   [06]<-+                  |
          |    |    |                  |
          |    |   [R4]                |
          |    |   [R6]                |
          |    +-->[R9]<---[R7 sigma]  |
          |                            |
          +------->[R2]                |
                                       |
   Raw CHIRPS                          |
          |                            |
       [07]                            |
       [R3]                            |
          |                            |
          +------->[R2]                |
                                       |
                  [R8]<-----------------+ (raw GRACE for re-clip)
                    |
                    v
                  [F] (consumes everything)
                    |
                    v
              12 manuscript figures
```

---

## Verification at each stage

After each script finishes, its final stdout block reports the key
computed values. The most important verification is at the end:

After running `R9_monte_carlo_v3.py`, you should see:

```
[R9] HEADLINE RESULT:
[R9]   Trend (Theil-Sen):    -34.86 mm/yr (expected -34.86)
[R9]   CI half-width:        +/-1.03 mm/yr (expected +/-1.03)
[R9]   Structural sigma:     +/-0.32 mm/yr (expected +/-0.32)
[R9]   Combined quadrature:  +/-1.08 mm/yr (expected +/-1.08)
[R9]   Cumulative loss:      ~413 km^3 (expected ~413)
```

If all five computed values match (or differ by less than 0.05 mm/yr for
the rates and 5 km^3 for the cumulative loss), the bundle reproduced the
manuscript correctly.

---

## Troubleshooting

### "GRACE NetCDF not found"

Place `GRCTellus.JPL.200204_202602.GLO.RL06.3M.MSCNv04CRI.nc` in
`data/grace/`. The exact filename matters; the script does not glob.

### "Missing GLDAS Giovanni CSV(s)"

Check that all six files are present in `data/gldas/` with the exact
filenames listed in `02_gldas_process.py`'s docstring.

### "ruptures not installed"

`pip install ruptures` (it's in requirements.txt; this error usually
means the environment was not activated).

### "No CHIRPS TIFs found"

Confirm `data/imd_rain/` contains 264 files matching the pattern
`chirps-v3.0.YYYY.MM.tif`. The script accepts any case in the filename
prefix (e.g. `CHIRPS-v3.0...` is also matched).

### "FATAL: proxy mask is empty" (R8 only)

This means no IGP grid cell has the threshold (default 5) CGWB wells.
Either:
- Provide an authoritative WHYMAP shapefile in `data/masks/whymap/`, or
- Lower the proxy threshold in `R8_alluvial_mask.py` (search for
  `build_proxy_mask(...)` and change the default `threshold=5`).

### Computed numbers differ from expected by more than ~0.5 mm/yr

The most common cause is using a different GRACE solution version. This
bundle is calibrated to `MSCNv04CRI`. If you use `MSCNv02CRI` or a CSR
mascon solution, expect a multi-mm/yr shift.

The second most common cause is a different CGWB snapshot. CGWB updates
its monitoring network periodically; a snapshot taken in 2025 will
include wells installed after the 2024 snapshot used for the manuscript.
This propagates into Table 2 numbers but not into the headline GRACE
trend, because the GRACE pipeline doesn't depend on well data.

---

## Re-running individual scripts

All scripts are idempotent: re-running overwrites previous outputs. There
is no clean-up step needed. The only state external to the scripts is the
`data/` directory (read-only).

---

## Estimated total run time

On a Lenovo ThinkPad P14s Gen 4 (2024) running Ubuntu 22.04 with Python 3.11:

```
Stage 1 (upstream):    3 min 40 s
Stage 2 (R-scripts):   2 min 50 s
Stage 3 (figures):     0 min 20 s
                       ----------
Total:                 6 min 50 s
```

If daily CHIRPS TIFs are present (~7,500 daily files), Stage 1 increases
by ~5-8 minutes due to the per-file rasterio open/clip overhead in
`R3_subseasonal_chirps.py`.
