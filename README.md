# IGP Groundwater Study — Reproducibility Bundle (v5.1)

This repository contains the complete analysis pipeline for the manuscript:

> Kumar, V. (2026). *Long-term groundwater depletion across the Upper
> Indo-Gangetic Plain: GRACE/GRACE-FO observations 2002-2023, depth-stratified
> well-network analysis, and Monte Carlo uncertainty quantification.*
> Submitted to *Water Resources Research*.

The bundle reproduces, from raw NASA / CGWB / CHIRPS data, the headline
results of the manuscript:

- **Trend**: −34.86 mm/yr groundwater storage decline
- **Non-parametric rank-based CI**: ±1.03 mm/yr (Theil-Sen 95%)
- **Within-JPL/GLDAS structural uncertainty**: ±0.32 mm/yr (5-source Monte Carlo, n=2000)
- **Combined quadrature**: ±1.08 mm/yr
- **Cumulative groundwater loss**: ~413 km³ over 22 years
- **Change-point**: August 2015 (binary segmentation, robust across three GWS variants)

---

## What this bundle is

A directly-runnable Python pipeline that takes raw input data as described
below and produces the CSV intermediates, summary text files, and the 12
manuscript figures (9 main + 3 supplementary).

The bundle is **scripts only**: raw input data is not redistributed because
the underlying datasets (GRACE, GLDAS, MODIS, CHIRPS, CGWB) have their own
licensing and acquisition pathways. Section [Data Acquisition](#data-acquisition)
documents how to obtain each input.

---

## Quick start

```bash
# 1. Clone and enter the bundle
cd IGP_GW_Study

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place raw data files into data/ (see "Data Acquisition" below)

# 5. Run the pipeline (see RUNBOOK.md for ordering)
cd scripts
python 01_grace_process.py
python 02_gldas_process.py
# ... etc, in numerical order, then R1, R2, ..., R9, then F
```

The full pipeline runs in about 8-12 minutes on a modern laptop
(2024-vintage CPU, no GPU required). Wall-time per script is documented
in `RUNBOOK.md`.

---

## Repository layout

```
IGP_GW_Study/
├── README.md                  This file.
├── RUNBOOK.md                 Exact command sequence and expected outputs.
├── LICENSE                    MIT.
├── requirements.txt           Pinned Python dependencies.
├── .zenodo.json               Zenodo deposit metadata.
├── .gitignore
│
├── scripts/                   18 Python files (see "Pipeline" below).
│   ├── config.py              Single source of truth for all constants.
│   ├── pipeline_utils.py      Shared SSA, Giovanni-CSV-loader, etc.
│   ├── 01_grace_process.py    Upstream pipeline (numbered 01-07).
│   ├── ...
│   ├── R1_well_filter_application.py  Analysis scripts (R1-R9).
│   ├── ...
│   └── F_make_v5_figures.py   Figure generator.
│
├── data/                      Place raw input data here (see below).
│   ├── grace/
│   ├── gldas/
│   ├── modis_et/
│   ├── cgwb_wells/
│   ├── imd_rain/
│   └── masks/
│
└── outputs/
    ├── timeseries/            Generated CSVs and summary text files.
    └── figures/               Generated PNG figures.
```

---

## Pipeline

The pipeline is organised in two stages.

### Upstream pipeline (scripts 01-07)

Reads raw input data and produces intermediate CSVs.

| Script | Output | Time |
|---|---|---|
| `01_grace_process.py` | GRACE_TWS_IGP.csv | ~30s |
| `02_gldas_process.py` | GLDAS_processed.csv | <5s |
| `03_modis_et_process.py` | MODIS_ET_monthly_2002_2023.csv, ET_bias_2002_2023.csv | <5s |
| `04_well_process.py` | shallow_wells_annual.csv, deep_wells_annual.csv | <10s |
| `05_water_balance.py` | water_balance_standard.csv | <5s |
| `06_ssa_trend.py` | SSA_components.csv | ~5s |
| `07_chirps_rainfall.py` | CHIRPS_monthly_IGP.csv, CHIRPS_JJAS_annual.csv | ~2 min |

### Analysis scripts (R1, R2, R3, R4, R6, R7, R8, R9)

Note: there is no R5; that script number was used for an iterative-SSA
sensitivity analysis whose logic is now embedded in the `fill_ssa_iterative`
function called by R6 and R9.

| Script | Manuscript anchor | Time |
|---|---|---|
| `R1_well_filter_application.py` | §3.4 (well selection) | <10s |
| `R2_depth_cut_sensitivity.py` | §5.4, Table 2 | ~30s |
| `R3_subseasonal_chirps.py` | §5.4 supplementary | depends on data |
| `R4_change_point.py` | §5.1 (Pettitt, Buishand, ruptures) | ~20s |
| `R6_ssa_quantitative.py` | §4.2, Fig S1 | ~10s |
| `R7_source3_reframe.py` | §3.5, Fig 5 | <5s |
| `R8_alluvial_mask.py` | §5.5, Fig S3 | ~30s |
| `R9_monte_carlo_v3.py` | §5.1 (headline), Fig 4 | ~30s |
| `F_make_v5_figures.py` | All 12 manuscript figures | ~20s |

See `RUNBOOK.md` for the exact command sequence and dependency graph.

---

## Data Acquisition

This bundle does not redistribute raw data because each source has its own
license and access conditions. The scripts expect the following files
to be present in `data/`.

### GRACE / GRACE-FO (1 file)

- **File**: `data/grace/GRCTellus.JPL.200204_202602.GLO.RL06.3M.MSCNv04CRI.nc`
- **Source**: NASA JPL PO.DAAC,
  https://podaac.jpl.nasa.gov/dataset/TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4
- **Access**: NASA Earthdata login required (free).

### GLDAS-2.1 Noah (6 Giovanni CSV exports)

The bundle uses time-averaged area-mean extracts from NASA Giovanni rather
than full grids, because the analysis only needs the IGP-domain mean. Use
Giovanni's Time-Averaged Area-Averaged Time Series tool:

- **Dataset**: GLDAS-2.1 Noah, monthly, 0.25° (`GLDAS_NOAH025_M.2.2`)
- **Bounding box**: 24-32.5° N, 73.5-80° E
- **Time range**: 2002-01 to 2023-12
- **Output**: 6 CSV files saved to `data/gldas/`:
  - `SM_0_10.csv` (variable: `SoilMoi0_10cm_inst`)
  - `SM_10_40.csv` (`SoilMoi10_40cm_inst`)
  - `SM_40_100.csv` (`SoilMoi40_100cm_inst`)
  - `SM_100_200.csv` (`SoilMoi100_200cm_inst`)
  - `SWE.csv` (`SWE_inst`)
  - `ET.csv` (`Evap_tavg`)
- **URL**: https://giovanni.gsfc.nasa.gov/giovanni/

### MODIS MOD16A2GF (1 AppEEARS Statistics CSV)

- **Product**: MOD16A2GF.061 (Terra Net Evapotranspiration Gap-Filled, 8-day)
- **Layer**: ET_500m
- **Region**: IGP bounding box
- **Period**: 2002-01-01 to 2023-12-31
- **File**: `data/modis_et/MOD16A2GF-061-Statistics.csv` (AppEEARS Area Statistics export)
- **URL**: https://appeears.earthdatacloud.nasa.gov/

### CGWB Wells (1 CSV)

- **File**: `data/cgwb_wells/cgwb_wells_raw.csv`
- **Source**: India-WRIS (Water Resources Information System),
  https://indiawris.gov.in/
- **Note**: The exact snapshot used in the manuscript is from CGWB's monthly
  release as of early 2024. Later snapshots may produce slightly different
  well counts (the bundle prints expected vs actual counts and tolerates
  modest differences).

### CHIRPS v3.0 monthly (264 GeoTIFFs)

- **Files**: `data/imd_rain/chirps-v3.0.YYYY.MM.tif` (264 files for 2002-01 through 2023-12)
- **Source**: Climate Hazards Center, https://www.chc.ucsb.edu/data/chirps
- **Note on directory name**: the directory is called `imd_rain/` for
  historical project reasons; CHIRPS is in fact a global product, not from
  the India Meteorological Department.

### Optional: Daily CHIRPS, IGP boundary, alluvial-extent masks

- `data/imd_rain/daily/` (daily CHIRPS TIFs, optional — enables R3's
  full sub-seasonal battery; without these R3 reports monthly-only metrics)
- `data/masks/IGP_boundary.geojson` (optional, for R3 spatial clipping;
  bbox fallback used if absent)
- `data/masks/whymap/` (optional, World-wide Hydrogeological Mapping)
- `data/masks/glim/` (optional, Global Lithological Map)

If WHYMAP and GLiM are both absent, R8 falls back to a well-density proxy
mask derived from the CGWB CSV, and prints a notice indicating which mask
was used.

---

## Verification

Each analysis script prints its computed headline numbers alongside the
manuscript expected values for manual comparison. A successful end-to-end
run should produce, in `R9`'s final report:

```
[R9] HEADLINE RESULT:
[R9]   Trend (Theil-Sen):    -34.86 mm/yr (expected -34.86)
[R9]   CI half-width:        +/-1.03 mm/yr (expected +/-1.03)
[R9]   Structural sigma:     +/-0.32 mm/yr (expected +/-0.32)
[R9]   Combined quadrature:  +/-1.08 mm/yr (expected +/-1.08)
[R9]   Cumulative loss:      ~413 km^3 (expected ~413)
```

If your computed numbers differ by more than ~0.05 mm/yr, the most likely
causes (in order of frequency) are:

1. **Different CGWB snapshot.** A newer CGWB CSV will have slightly different
   well counts, which propagates into different per-class composite means.
2. **Different GLDAS Giovanni extract.** Giovanni rounds time stamps to
   month-start; ensure your CSVs cover the full 2002-01 to 2023-12 range
   without gaps.
3. **Different GRACE solution version.** This bundle uses JPL RL06.3
   `MSCNv04CRI`. Earlier (RL06v02, RL06v03) or alternative (CSR, GSFC)
   solutions will produce different absolute trends.

---

## Citation

If you use this code or its derived results, please cite both the manuscript
and the deposited code:

> Kumar, V. (2026). Long-term groundwater depletion across the Upper
> Indo-Gangetic Plain. *Water Resources Research*. (DOI to be assigned)

> Kumar, V. (2026). IGP Groundwater Study — Reproducibility Bundle (v5.1).
> Zenodo. https://doi.org/10.5281/zenodo.19966722

---

## License

MIT License — see `LICENSE`. Free for academic and commercial use; please
cite the manuscript above for the scientific results.

---

## Contact

Vikash Kumar — vikasynced@gmail.com — Independent Researcher.

For bug reports specific to this code, please open an issue on the GitHub
repository (https://github.com/vikasynced/IGP-groundwater-bundle)
