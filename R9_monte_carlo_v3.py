"""
R9_monte_carlo_v3.py
====================
Monte Carlo uncertainty quantification for the IGP GWS trend.

PURPOSE
-------
This is the headline-numbers script of the bundle. It produces:

  1. The gap-filled GWS series via iterative SSA (manuscript §4.2 method M3,
     L = 60).
  2. Theil-Sen rank-based trend estimate with 95% CI (the "non-parametric
     rank-based" CI of half-width +/-1.03 mm/yr in §5.1).
  3. Monte Carlo ensemble (n = 2000) propagating five within-JPL/GLDAS
     structural uncertainty sources, yielding the structural sigma of
     +/-0.32 mm/yr and the combined-quadrature uncertainty of +/-1.08 mm/yr.
  4. Cumulative volumetric loss over the 22-year study period (~413 km^3
     across the 540,000 km^2 IGP domain).

The five Monte Carlo error sources (manuscript Table 4):

    1. GRACE formal error          sigma = 20 mm                (constant)
    2. Scale-factor uncertainty    sigma = 5% of |TWS|          (proportional)
    3. GLDAS SM structural         sigma = 15% of |SM_anom|     (proportional)
    4. Inter-model ET (R7)         sigma = month-of-year vector (per-month)
    5. SWE uncertainty             sigma = 20% of |SWE_anom|    (proportional)

INPUTS
------
  outputs/timeseries/water_balance_standard.csv   from 05_water_balance.py
  outputs/timeseries/R7_intermodel_sigma.csv      from R7_source3_reframe.py

OUTPUTS
-------
  outputs/timeseries/R9_gws_filled.csv
      Columns: time, GWS_observed, GWS_filled, fill_flag (1 if gap-filled)
  outputs/timeseries/R9_montecarlo_ensemble_summary.csv
      Per-month median and percentiles of the MC ensemble.
  outputs/timeseries/R9_trend_estimates.csv
      Theil-Sen point estimate, 95% CI, MC structural sigma, combined sigma.
  outputs/timeseries/R9_summary.txt

USAGE
-----
  cd path/to/IGP_GW_Study/scripts
  python R9_monte_carlo_v3.py    # requires 05 and R7 first
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    TS_DIR,
    SSA_WINDOW_LENGTH,
    MC_N_ITERATIONS, MC_RANDOM_SEED,
    SIGMA_GRACE_FORMAL_MM, FRAC_SCALE_FACTOR,
    FRAC_GLDAS_SM, FRAC_SWE,
    DOMAIN_AREA_KM2, N_STUDY_YEARS,
    EXPECTED_TREND_MM_PER_YR,
    EXPECTED_THEILSEN_CI_HALFWIDTH,
    EXPECTED_STRUCTURAL_SIGMA_MM_PER_YR,
    EXPECTED_COMBINED_QUADRATURE,
    EXPECTED_CUMULATIVE_LOSS_KM3,
)
from pipeline_utils import fill_ssa_iterative


WB_CSV    = TS_DIR / "water_balance_standard.csv"
SIGMA_CSV = TS_DIR / "R7_intermodel_sigma.csv"

OUT_FILLED   = TS_DIR / "R9_gws_filled.csv"
OUT_ENSEMBLE = TS_DIR / "R9_montecarlo_ensemble_summary.csv"
OUT_TRENDS   = TS_DIR / "R9_trend_estimates.csv"
OUT_SUMMARY  = TS_DIR / "R9_summary.txt"
OUT_FIG_BAND = Path(__file__).resolve().parent.parent / "outputs" / "figures" / "R9_MC_band.png"


def theil_sen(x_months, y_mm):
    """
    Theil-Sen (median-of-pairwise-slopes) regression, returning slope in
    mm/yr and the 95% confidence interval.

    Uses scipy.stats.theilslopes (Sen's modification: returns slope as
    median of pairwise slopes; CI from rank statistics, valid for n >= 10
    without distributional assumptions).
    """
    res = stats.theilslopes(y_mm, x_months, alpha=0.95)
    # res returns slope per unit-x. x is in months -> multiply by 12.
    slope = float(res.slope) * 12
    lo    = float(res.low_slope) * 12
    hi    = float(res.high_slope) * 12
    return slope, lo, hi


def main():
    TS_DIR.mkdir(parents=True, exist_ok=True)

    for path in [WB_CSV, SIGMA_CSV]:
        if not path.exists():
            raise FileNotFoundError(
                f"[R9] Required upstream output not found: {path}"
            )

    # -------------------------------------------------------------------------
    # Load and gap-fill GWS
    # -------------------------------------------------------------------------
    print(f"[R9] Loading water_balance_standard.csv ...")
    wb = pd.read_csv(WB_CSV, parse_dates=["time"])
    wb["time"] = wb["time"].dt.to_period("M").dt.to_timestamp()

    # Reindex onto a contiguous monthly axis so the gap appears as NaN.
    full_idx = pd.date_range(
        wb["time"].min(), wb["time"].max(), freq="MS",
    )
    wb = wb.set_index("time").reindex(full_idx)
    wb.index.name = "time"

    print(f"[R9] Gap-filling GWS via iterative SSA (M3, L = {SSA_WINDOW_LENGTH}) ...")
    gws_obs    = wb["GWS_standard"]
    gws_filled = fill_ssa_iterative(gws_obs, L=SSA_WINDOW_LENGTH)
    fill_flag  = gws_obs.isna().astype(int).values
    n_filled   = int(fill_flag.sum())
    print(f"[R9] Gap-filled months: {n_filled} / {len(gws_filled)}")

    pd.DataFrame({
        "time":         gws_filled.index,
        "GWS_observed": gws_obs.values,
        "GWS_filled":   gws_filled.values,
        "fill_flag":    fill_flag,
    }).to_csv(OUT_FILLED, index=False)

    # -------------------------------------------------------------------------
    # Theil-Sen point estimate + 95% CI
    # -------------------------------------------------------------------------
    months = np.arange(len(gws_filled))
    ts_slope, ts_lo, ts_hi = theil_sen(months, gws_filled.values)
    ci_half = (ts_hi - ts_lo) / 2
    print(f"[R9] Theil-Sen slope: {ts_slope:+.2f} mm/yr "
          f"(95% CI [{ts_lo:+.2f}, {ts_hi:+.2f}], half-width {ci_half:.2f})")

    # -------------------------------------------------------------------------
    # Monte Carlo: 5 error sources
    # -------------------------------------------------------------------------
    # Need TWS, SM_anom, SWE_anom on the filled axis. The water-balance CSV
    # has them but with NaNs in the gap; for sources 2/3/5 we need values
    # to scale, so use linear interpolation across the gap (the actual
    # uncertainty contribution during the gap doesn't affect the trend much
    # since the gap-fill already smooths these months).
    tws_filled      = wb["TWS"     ].interpolate(method="linear", limit_direction="both").values
    sm_anom_filled  = wb["SM_anom" ].interpolate(method="linear", limit_direction="both").values
    swe_anom_filled = wb["SWE_anom"].interpolate(method="linear", limit_direction="both").values

    # Source 4: per-month-of-year sigma vector (R7 output)
    sigma_per_month = pd.read_csv(SIGMA_CSV).set_index("month")["sigma_mm_month"]
    sigma_source4 = np.array([
        sigma_per_month.loc[t.month] for t in gws_filled.index
    ])

    print(f"[R9] Running Monte Carlo n = {MC_N_ITERATIONS}, "
          f"seed = {MC_RANDOM_SEED} ...")
    rng = np.random.default_rng(MC_RANDOM_SEED)
    n   = len(gws_filled)
    base_gws = gws_filled.values

    # Vectorised generation: sample all sources for all iterations at once.
    # Memory cost: 5 * n * MC_N_ITERATIONS * 8 bytes ~ 20 MB for n=265,
    # MC_N_ITERATIONS=2000. Comfortable.
    eps_grace  = rng.normal(0, SIGMA_GRACE_FORMAL_MM, size=(MC_N_ITERATIONS, n))
    eps_scale  = rng.normal(0, FRAC_SCALE_FACTOR * np.abs(tws_filled),
                            size=(MC_N_ITERATIONS, n))
    eps_sm     = rng.normal(0, FRAC_GLDAS_SM * np.abs(sm_anom_filled),
                            size=(MC_N_ITERATIONS, n))
    eps_et     = rng.normal(0, sigma_source4,
                            size=(MC_N_ITERATIONS, n))
    eps_swe    = rng.normal(0, FRAC_SWE * np.abs(swe_anom_filled),
                            size=(MC_N_ITERATIONS, n))

    # Each ensemble member is the deterministic GWS plus the perturbations.
    # Note: SM and SWE perturbations enter with a leading minus because
    # GWS = TWS - SM - SWE, so any positive SM error reduces GWS estimate.
    ensemble = (
        base_gws[None, :]
        + eps_grace
        + eps_scale
        - eps_sm
        - eps_et
        - eps_swe
    )

    # Per-month percentiles
    q = np.percentile(ensemble, [2.5, 16, 50, 84, 97.5], axis=0)
    ens_summary = pd.DataFrame({
        "time":      gws_filled.index,
        "p02_5":     q[0],
        "p16":       q[1],
        "median":    q[2],
        "p84":       q[3],
        "p97_5":     q[4],
        "std":       ensemble.std(axis=0, ddof=1),
    })
    ens_summary.to_csv(OUT_ENSEMBLE, index=False)

    # Per-iteration Theil-Sen slope
    ensemble_slopes = np.empty(MC_N_ITERATIONS)
    for k in range(MC_N_ITERATIONS):
        s = stats.theilslopes(ensemble[k], months, alpha=0.95).slope * 12
        ensemble_slopes[k] = s
    structural_sigma = float(ensemble_slopes.std(ddof=1))
    structural_mean  = float(ensemble_slopes.mean())

    combined_quadrature = float(np.sqrt(ci_half ** 2 + structural_sigma ** 2))

    # -------------------------------------------------------------------------
    # Cumulative volumetric loss
    # -------------------------------------------------------------------------
    # Cumulative depth loss = trend (mm/yr) * 22 years = ... mm
    # Volume = depth (m) * area (km^2) = ... km^3, with mm -> 1e-6 m
    cumulative_depth_mm = ts_slope * N_STUDY_YEARS
    cumulative_loss_km3 = abs(cumulative_depth_mm) * DOMAIN_AREA_KM2 * 1e-6

    # -------------------------------------------------------------------------
    # Save trend estimates
    # -------------------------------------------------------------------------
    trends_df = pd.DataFrame({
        "estimate":   [
            "theil_sen_slope",
            "theil_sen_ci_lo",
            "theil_sen_ci_hi",
            "theil_sen_ci_halfwidth",
            "mc_ensemble_mean_slope",
            "mc_structural_sigma",
            "combined_quadrature",
            "cumulative_loss_km3",
        ],
        "value_mm_yr": [
            ts_slope, ts_lo, ts_hi, ci_half,
            structural_mean, structural_sigma, combined_quadrature,
            np.nan,
        ],
        "value_km3":   [np.nan] * 7 + [cumulative_loss_km3],
    })
    trends_df.to_csv(OUT_TRENDS, index=False)

    # -------------------------------------------------------------------------
    # MC-band figure (Fig 4 of the manuscript)
    # -------------------------------------------------------------------------
    OUT_FIG_BAND.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(gws_filled.index, q[0], q[4], color="#5B9BD5", alpha=0.25,
                    label="95% MC band")
    ax.fill_between(gws_filled.index, q[1], q[3], color="#5B9BD5", alpha=0.45,
                    label="68% MC band")
    ax.plot(gws_filled.index, q[2], color="#1F4E79", lw=1.0, label="MC median")
    ax.plot(gws_filled.index[fill_flag == 1],
            base_gws[fill_flag == 1],
            "o", color="#E8A33D", markersize=3.5, alpha=0.85,
            markeredgecolor="black", markeredgewidth=0.3,
            label="Gap-filled (iterative SSA)")
    # Theil-Sen trend overlay
    trend_line = stats.theilslopes(base_gws, months, alpha=0.95)
    ax.plot(gws_filled.index,
            trend_line.intercept + trend_line.slope * months,
            color="#C00000", lw=1.5,
            label=f"Theil-Sen trend ({ts_slope:+.2f} mm/yr)")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("GWS anomaly (mm equivalent water height)")
    ax.set_title(
        f"GWS with {MC_N_ITERATIONS}-member Monte Carlo uncertainty bands "
        f"(seed = {MC_RANDOM_SEED})"
    )
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_FIG_BAND, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Human-readable summary
    # -------------------------------------------------------------------------
    lines = []
    lines.append("R9  MONTE CARLO TREND UNCERTAINTY  -  SUMMARY")
    lines.append("=" * 78)
    lines.append(f"Series length: {n} months ({n_filled} gap-filled)")
    lines.append(f"SSA window:    L = {SSA_WINDOW_LENGTH}")
    lines.append(f"MC iterations: {MC_N_ITERATIONS}, seed = {MC_RANDOM_SEED}")
    lines.append("")
    lines.append("HEADLINE RESULTS")
    lines.append("-" * 78)
    lines.append(f"  Theil-Sen slope:                   {ts_slope:+7.2f} mm/yr")
    lines.append(f"  Theil-Sen 95% CI:                  "
                 f"[{ts_lo:+.2f}, {ts_hi:+.2f}]")
    lines.append(f"  Non-parametric rank-based CI:      "
                 f"+/-{ci_half:.2f} mm/yr")
    lines.append(f"  MC structural sigma:               "
                 f"+/-{structural_sigma:.2f} mm/yr")
    lines.append(f"  Combined quadrature uncertainty:   "
                 f"+/-{combined_quadrature:.2f} mm/yr")
    lines.append(f"  Cumulative loss over {N_STUDY_YEARS} years: "
                 f"~{cumulative_loss_km3:.0f} km^3")
    lines.append("")
    lines.append("v5.1 MANUSCRIPT EXPECTED VALUES")
    lines.append("-" * 78)
    lines.append(f"  Theil-Sen slope:           "
                 f"{EXPECTED_TREND_MM_PER_YR:+.2f} mm/yr")
    lines.append(f"  CI half-width:             "
                 f"+/-{EXPECTED_THEILSEN_CI_HALFWIDTH:.2f} mm/yr")
    lines.append(f"  Structural sigma:          "
                 f"+/-{EXPECTED_STRUCTURAL_SIGMA_MM_PER_YR:.2f} mm/yr")
    lines.append(f"  Combined quadrature:       "
                 f"+/-{EXPECTED_COMBINED_QUADRATURE:.2f} mm/yr")
    lines.append(f"  Cumulative loss:           "
                 f"~{EXPECTED_CUMULATIVE_LOSS_KM3} km^3")
    lines.append("")
    lines.append("=" * 78)
    summary = "\n".join(lines)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------
    print(f"[R9]")
    print(f"[R9] HEADLINE RESULT:")
    print(f"[R9]   Trend (Theil-Sen):    {ts_slope:+.2f} mm/yr "
          f"(expected {EXPECTED_TREND_MM_PER_YR:+.2f})")
    print(f"[R9]   CI half-width:        +/-{ci_half:.2f} mm/yr "
          f"(expected +/-{EXPECTED_THEILSEN_CI_HALFWIDTH:.2f})")
    print(f"[R9]   Structural sigma:     +/-{structural_sigma:.2f} mm/yr "
          f"(expected +/-{EXPECTED_STRUCTURAL_SIGMA_MM_PER_YR:.2f})")
    print(f"[R9]   Combined quadrature:  +/-{combined_quadrature:.2f} mm/yr "
          f"(expected +/-{EXPECTED_COMBINED_QUADRATURE:.2f})")
    print(f"[R9]   Cumulative loss:      ~{cumulative_loss_km3:.0f} km^3 "
          f"(expected ~{EXPECTED_CUMULATIVE_LOSS_KM3})")
    print(f"[R9] Wrote: {OUT_FILLED.relative_to(OUT_FILLED.parents[2])}")
    print(f"[R9] Wrote: {OUT_ENSEMBLE.relative_to(OUT_ENSEMBLE.parents[2])}")
    print(f"[R9] Wrote: {OUT_TRENDS.relative_to(OUT_TRENDS.parents[2])}")
    print(f"[R9] Wrote: {OUT_SUMMARY.relative_to(OUT_SUMMARY.parents[2])}")
    print(f"[R9] Wrote: {OUT_FIG_BAND.relative_to(OUT_FIG_BAND.parents[2])}")


if __name__ == "__main__":
    main()
