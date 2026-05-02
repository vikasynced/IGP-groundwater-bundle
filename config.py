"""
config.py
=========
Centralised constants for the IGP groundwater reproducibility bundle.

All scripts in this bundle import their domain constants, depth bins,
hydroclimatic year classifications, and random seeds from this module.
This is the single source of truth: changing a value here propagates to
every downstream script automatically.

The numerical values here reproduce the v5.1 manuscript exactly. Editing
any value will cause the bundle to produce results that differ from the
manuscript; do not edit unless you intend that.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Project root (bundle-relative paths)
# -----------------------------------------------------------------------------
# All scripts derive their paths from this single anchor. The bundle is
# self-contained and runnable from any installation directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
TS_DIR       = OUTPUT_DIR / "timeseries"
FIG_DIR      = OUTPUT_DIR / "figures"

# -----------------------------------------------------------------------------
# Spatial domain
# -----------------------------------------------------------------------------
# Upper Indo-Gangetic Plain bounding box (manuscript §3.1)
LAT_MIN, LAT_MAX = 24.0, 32.5
LON_MIN, LON_MAX = 73.5, 80.0

# Total domain area for groundwater volume calculations (km^2)
# Used in cumulative loss arithmetic (manuscript §7)
DOMAIN_AREA_KM2 = 540_000

# -----------------------------------------------------------------------------
# Temporal domain
# -----------------------------------------------------------------------------
# Study period (inclusive)
STUDY_START_YEAR = 2002
STUDY_END_YEAR   = 2023
N_STUDY_YEARS    = STUDY_END_YEAR - STUDY_START_YEAR  # = 22 (intervals)

# GRACE/GRACE-FO baseline period for anomaly computation (manuscript §3.2)
BASELINE_START = "2004-01"
BASELINE_END   = "2009-12"

# GRACE-FO data gap (no observations available)
GRACE_GAP_START = "2017-06"
GRACE_GAP_END   = "2018-05"

# -----------------------------------------------------------------------------
# Depth stratification (manuscript §3.4, Table 2)
# -----------------------------------------------------------------------------
# Primary depth threshold separating shallow phreatic from deep semi-confined
# groundwater (m below ground level)
SHALLOW_DEPTH_MAX = 30.0    # <30 m  = shallow phreatic
DEEP_DEPTH_MIN    = 100.0   # >=100 m = deep semi-confined
# Sensitivity thresholds tested in R2 (Table 2)
DEEP_SENSITIVITY_THRESHOLDS = [100, 150, 200]

# -----------------------------------------------------------------------------
# Hydroclimatic year classification (manuscript §3.6)
# -----------------------------------------------------------------------------
# CHIRPS-derived JJAS rainfall classification:
#   wet  = JJAS total > climatological mean + 1 sigma
#   dry  = JJAS total < climatological mean - 1 sigma
# Years are produced by 07_chirps_rainfall.py and used downstream by R2/R3.
# Listed here for documentation; canonical values are in CHIRPS_JJAS_annual.csv.
WET_YEARS_CHIRPS = [2010, 2011, 2013, 2018, 2019]
DRY_YEARS_CHIRPS = [2002, 2004, 2009, 2014]

# Wet years usable for paired pre/post-monsoon analysis
# (excludes 2018 which falls in the GRACE-FO data gap)
WET_YEARS_TESTABLE = [y for y in WET_YEARS_CHIRPS if y != 2018]

# -----------------------------------------------------------------------------
# SSA parameters (manuscript §4.2)
# -----------------------------------------------------------------------------
# Window length L=60 chosen for trend-vs-annual separability (R6 diagnostics).
# Reproducer: changing this value will alter the gap-filled GWS series.
SSA_WINDOW_LENGTH      = 60      # months
SSA_GAPFILL_MAX_ITER   = 10      # iterative-SSA convergence cap
SSA_GAPFILL_TOLERANCE  = 0.5     # mm; convergence threshold

# Sensitivity values tested in R6
SSA_L_SENSITIVITY = [36, 60, 84, 120]

# -----------------------------------------------------------------------------
# Monte Carlo (manuscript §4.3, R9)
# -----------------------------------------------------------------------------
MC_N_ITERATIONS = 2000
MC_RANDOM_SEED  = 12345

# Error source magnitudes (within-JPL/GLDAS structural; manuscript Table 4)
SIGMA_GRACE_FORMAL_MM = 20.0   # Source 1: GRACE formal error
FRAC_SCALE_FACTOR     = 0.05   # Source 2: 5% of |TWS|
FRAC_GLDAS_SM         = 0.15   # Source 3: 15% of |SM_anom|
FRAC_SWE              = 0.20   # Source 5: 20% of |SWE_anom|
# Source 4 (inter-model ET) is loaded at runtime from R7's CSV output.

# Auxiliary random seeds for non-MC stochastic operations
SEED_BUISHAND_PERMUTATION = 12345  # R4 Buishand range test (was 42 in v4)

# -----------------------------------------------------------------------------
# Headline results for v5.1 (printed at end of each script for verification)
# -----------------------------------------------------------------------------
# These values come from the published manuscript. Each script's final block
# prints its own computed value alongside this expected value so the user can
# manually verify the bundle reproduced the correct number.
EXPECTED_TREND_MM_PER_YR              = -34.86
EXPECTED_THEILSEN_CI_HALFWIDTH        = 1.03
EXPECTED_STRUCTURAL_SIGMA_MM_PER_YR   = 0.32
EXPECTED_COMBINED_QUADRATURE          = 1.08
EXPECTED_CUMULATIVE_LOSS_KM3          = 413
EXPECTED_CHANGEPOINT_DATE             = "2015-08"
EXPECTED_DEEP_WET_MEAN_M              = 1.24
EXPECTED_SHALLOW_WET_MEAN_M           = 3.16
EXPECTED_N_WELLS_TOTAL                = 6604
EXPECTED_N_WELLS_SHALLOW              = 2855
EXPECTED_N_WELLS_TRANSITIONAL         = 2849
EXPECTED_N_WELLS_DEEP                 = 900
