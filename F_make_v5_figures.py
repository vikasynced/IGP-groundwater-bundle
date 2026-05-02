"""
F_make_v5_figures.py
====================
Generate the 12 manuscript figures for v5.1 in a consistent visual style.

PURPOSE
-------
Reads the CSV outputs from the upstream pipeline and the R-scripts, and
renders the publication figures. Figures with substantive data work
(Figs 1-3, 6-9, S2) are rendered here; figures that are just
relabelled diagnostics from R-scripts (Figs 4, 5, S1, S3) are copied
from those scripts' outputs with the manuscript-facing name.

OUTPUTS (manuscript figure numbering)
--------------------------------------
  Fig01_v5_study_area.png            wells map, IGP bbox, depth classes
  Fig02_v5_GRACE_TWS.png             TWS time series with gap markers
  Fig03_v5_gws_trend_changepoint.png gap-filled GWS + trend + change-point
  Fig04_v5_MC_band.png               Monte Carlo uncertainty band (from R9)
  Fig05_v5_ET_intercomparison.png    GLDAS-MODIS ET (from R7)
  Fig06_v5_well_groups.png           shallow vs deep secular trends
  Fig07_v5_depth_cut_response.png    depth-stratified wet-year response
  Fig08_v5_CHIRPS_rainfall.png       CHIRPS JJAS classification
  Fig09_v5_wet_year_per_year.png     per-wet-year deep response
  FigS1_v5_wcor_L60.png              w-correlation heatmap (from R6)
  FigS2_v5_GLDAS_components.png      GLDAS SM, SWE, ET separately
  FigS3_v5_alluvial_mask.png         alluvial-restricted TWS (from R8)

USAGE
-----
  cd path/to/IGP_GW_Study/scripts
  python F_make_v5_figures.py

  Each figure is wrapped in try/except so a single missing input file
  does not prevent the rest from rendering.
"""

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR, TS_DIR, FIG_DIR,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    SHALLOW_DEPTH_MAX, DEEP_DEPTH_MIN,
    SSA_WINDOW_LENGTH,
)
from pipeline_utils import fill_ssa_iterative


# Make sure outputs/figures exists
FIG_DIR.mkdir(parents=True, exist_ok=True)

# v5 visual style
def apply_v5_style():
    plt.rcParams.update({
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linewidth":    0.5,
        "legend.frameon":    False,
        "legend.fontsize":   9,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "lines.linewidth":   1.2,
    })


# Manuscript palette
COLOR_GWS     = "#1F4E79"   # dark blue
COLOR_TREND   = "#C00000"   # red
COLOR_BAND    = "#5B9BD5"   # light blue
COLOR_SHALLOW = "#2E7D32"   # green
COLOR_DEEP    = "#1F4E79"   # dark blue
COLOR_DRY     = "#A52A2A"   # brown
COLOR_WET     = "#1F4E79"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def safe_load(path):
    """Load a bundle CSV with automatic 'time' column detection."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    df = pd.read_csv(path)
    for cand in ["time", "date", "Time", "Date", "timestamp", "Unnamed: 0"]:
        if cand in df.columns:
            df["time"] = pd.to_datetime(df[cand])
            return df
    return df


def build_gap_filled_gws():
    """Build the gap-filled GWS series for figures that need it."""
    wb = safe_load(TS_DIR / "water_balance_standard.csv")
    n_dup = wb["time"].duplicated().sum()
    if n_dup:
        numeric_cols = [c for c in wb.columns if c != "time"
                        and pd.api.types.is_numeric_dtype(wb[c])]
        wb = wb[["time"] + numeric_cols].groupby("time", as_index=False).mean()
    wb = wb.sort_values("time").reset_index(drop=True)
    idx_full = pd.date_range(
        wb["time"].iloc[0].to_period("M").to_timestamp(),
        wb["time"].iloc[-1].to_period("M").to_timestamp(),
        freq="MS",
    )
    wb_full = (
        wb.set_index("time").reindex(idx_full)
        .rename_axis("time").reset_index()
    )
    gws = fill_ssa_iterative(wb_full["GWS_standard"],
                              L=SSA_WINDOW_LENGTH).values
    return wb_full["time"].values, gws, wb_full["GWS_standard"].isna().values


# =============================================================================
# Figure 1 — Study area + wells
# =============================================================================
def fig01_study_area():
    out_path = FIG_DIR / "Fig01_v5_study_area.png"
    wells_path = DATA_DIR / "cgwb_wells" / "cgwb_wells_raw.csv"
    if not wells_path.exists():
        print(f"  Fig 1: wells CSV not found at {wells_path}; SKIPPED")
        return
    wells = pd.read_csv(wells_path, low_memory=False, usecols=range(20))
    wells["Latitude"]   = pd.to_numeric(wells["Latitude"],   errors="coerce")
    wells["Longitude"]  = pd.to_numeric(wells["Longitude"],  errors="coerce")
    wells["Well Depth"] = pd.to_numeric(wells["Well Depth"], errors="coerce")
    m = (wells["Latitude"].between(LAT_MIN, LAT_MAX) &
         wells["Longitude"].between(LON_MIN, LON_MAX))
    wells = wells[m]
    shallow = wells[wells["Well Depth"] < SHALLOW_DEPTH_MAX]
    transit = wells[(wells["Well Depth"] >= SHALLOW_DEPTH_MAX) &
                    (wells["Well Depth"] <  DEEP_DEPTH_MIN)]
    deep    = wells[wells["Well Depth"] >= DEEP_DEPTH_MIN]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(shallow["Longitude"], shallow["Latitude"], s=2, alpha=0.25,
               c=COLOR_SHALLOW, label=f"Shallow <30 m (n={len(shallow)})")
    ax.scatter(transit["Longitude"], transit["Latitude"], s=2, alpha=0.25,
               c="#FFA500", label=f"Transitional 30-100 m (n={len(transit)})")
    ax.scatter(deep["Longitude"], deep["Latitude"], s=6, alpha=0.55,
               c=COLOR_DEEP, label=f"Deep >=100 m (n={len(deep)})")
    ax.plot([LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN],
            [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN],
            "k-", lw=1.5,
            label=f"Study domain ({LAT_MIN}-{LAT_MAX} N, {LON_MIN}-{LON_MAX} E)")
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.set_title("Upper Indo-Gangetic Plain study domain and CGWB monitoring wells")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9, frameon=True)
    ax.set_xlim(LON_MIN - 0.3, LON_MAX + 0.3)
    ax.set_ylim(LAT_MIN - 0.3, LAT_MAX + 0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Fig 1 rendered: {out_path.name}")


# =============================================================================
# Figure 2 — GRACE/GRACE-FO TWS time series with gap markers
# =============================================================================
def fig02_grace_tws():
    out_path = FIG_DIR / "Fig02_v5_GRACE_TWS.png"
    wb = safe_load(TS_DIR / "water_balance_standard.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(wb["time"], wb["TWS"], color=COLOR_GWS, lw=1.0)
    ax.axvspan(pd.Timestamp("2017-06-01"), pd.Timestamp("2018-05-31"),
               color="red", alpha=0.12,
               label="GRACE/GRACE-FO inter-mission gap")
    ax.set_xlabel("Year")
    ax.set_ylabel("TWS anomaly (mm equivalent water height)")
    ax.set_title("GRACE/GRACE-FO total water storage anomaly, IGP domain")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Fig 2 rendered: {out_path.name}")


# =============================================================================
# Figure 3 — Gap-filled GWS with trend + CIs + change-point marker
# =============================================================================
def fig03_gws_trend_changepoint():
    out_path = FIG_DIR / "Fig03_v5_gws_trend_changepoint.png"
    times, gws, was_filled = build_gap_filled_gws()
    times_pd = pd.to_datetime(times)
    t = np.arange(len(gws))
    ts_res = stats.theilslopes(gws, t, alpha=0.95)
    sl, ic = float(ts_res.slope), float(ts_res.intercept)
    lo, hi = float(ts_res.low_slope), float(ts_res.high_slope)
    trend    = ic + sl * t
    trend_lo = ic + lo * t
    trend_hi = ic + hi * t

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axvspan(pd.Timestamp("2017-06-01"), pd.Timestamp("2018-05-31"),
               color="red", alpha=0.15, lw=0,
               label="GRACE/GRACE-FO gap")
    ax.plot(times_pd[~was_filled], gws[~was_filled],
            color=COLOR_GWS, lw=0.9, label="Observed GWS")
    ax.plot(times_pd[was_filled], gws[was_filled],
            color="#E8A33D", lw=0.0, marker="o", markersize=3.5,
            alpha=0.85, markeredgecolor="black", markeredgewidth=0.3,
            label="Gap-filled (iterative SSA)")
    ax.fill_between(times_pd, trend_lo, trend_hi,
                    color=COLOR_TREND, alpha=0.18,
                    label="Theil-Sen 95% CI")
    ax.plot(times_pd, trend, color=COLOR_TREND, lw=1.5,
            label=f"Theil-Sen trend ({sl * 12:+.2f} mm/yr)")
    ax.axvline(pd.Timestamp("2015-08-01"), color="black", lw=1.0,
               ls="--", alpha=0.7,
               label="Change-point (Binseg, 2015-08)")
    ax.set_xlabel("Year")
    ax.set_ylabel("GWS anomaly (mm equivalent water height)")
    ax.set_title("Gap-filled GWS time series with Theil-Sen trend and change-point")
    ax.legend(loc="lower left", fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Fig 3 rendered: {out_path.name}")


# =============================================================================
# Figure 4 — MC band (rename from R9)
# =============================================================================
def fig04_mc_band():
    out_path = FIG_DIR / "Fig04_v5_MC_band.png"
    src = FIG_DIR / "R9_MC_band.png"
    if src.exists():
        shutil.copy(src, out_path)
        print(f"  Fig 4 copied from R9: {out_path.name}")
    else:
        print(f"  Fig 4: {src.name} not found; run R9_monte_carlo_v3.py first")


# =============================================================================
# Figure 5 — ET inter-comparison (rename from R7)
# =============================================================================
def fig05_et_intercomp():
    out_path = FIG_DIR / "Fig05_v5_ET_intercomparison.png"
    src = FIG_DIR / "R7_ET_intercomparison.png"
    if src.exists():
        shutil.copy(src, out_path)
        print(f"  Fig 5 copied from R7: {out_path.name}")
    else:
        print(f"  Fig 5: {src.name} not found; run R7_source3_reframe.py first")


# =============================================================================
# Figure 6 — Shallow vs deep well secular trends
# =============================================================================
def fig06_well_groups():
    out_path = FIG_DIR / "Fig06_v5_well_groups.png"
    obs = pd.read_csv(TS_DIR / "R2_per_well_year_changes.csv")
    shallow = obs[obs["depth_m"] < SHALLOW_DEPTH_MAX]
    deep    = obs[obs["depth_m"] >= DEEP_DEPTH_MIN]
    sh_yr = shallow.groupby("year")["may_bgl"].mean()
    dp_yr = deep.groupby("year")["may_bgl"].mean()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(sh_yr.index, sh_yr.values, marker="o", color=COLOR_SHALLOW,
            label=f"Shallow <30 m (n~{len(shallow.station_code.unique())} wells)",
            lw=1.4)
    ax.plot(dp_yr.index, dp_yr.values, marker="s", color=COLOR_DEEP,
            label=f"Deep >=100 m (n~{len(deep.station_code.unique())} wells)",
            lw=1.4)
    ax.invert_yaxis()
    ax.set_xlabel("Year")
    ax.set_ylabel("Pre-monsoon (May) depth to water (m bgl)\n"
                  "<- shallower water table        deeper water table ->")
    ax.set_title("Domain-mean May water level: shallow vs deep CGWB wells")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Fig 6 rendered: {out_path.name}")


# =============================================================================
# Figure 7 — Depth-stratified wet-year response
# =============================================================================
def fig07_depth_response():
    out_path = FIG_DIR / "Fig07_v5_depth_cut_response.png"
    res = pd.read_csv(TS_DIR / "R2_depth_cut_sensitivity.csv")
    # The bundle uses 'deep_ge100' (matches manuscript convention).
    order  = ["shallow_lt30", "transitional_30_100",
              "deep_ge100", "deep_100_150", "deep_ge150", "deep_ge200"]
    labels = ["<30 m", "30-100 m", ">=100 m",
              "100-150 m", ">=150 m", ">=200 m"]
    available = [(o, l) for o, l in zip(order, labels)
                 if o in res["depth_class"].values]
    order  = [a[0] for a in available]
    labels = [a[1] for a in available]
    res = res.set_index("depth_class").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(order))
    means = res["wet_mean_m"].values
    los   = res["wet_ci95_lo"].values
    his   = res["wet_ci95_hi"].values
    drys  = res["dry_mean_m"].values

    ax.bar(x - 0.2, means, 0.4, color=COLOR_WET,
           yerr=[means - los, his - means], capsize=5,
           label="Wet years (n=4): mean and 95% CI")
    ax.bar(x + 0.2, drys, 0.4, color=COLOR_DRY, alpha=0.7,
           label="Dry years (n=4): mean")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Well screen-depth class")
    ax.set_ylabel("Pre to post-monsoon water-level change\n"
                  "(m, positive = recharge)")
    ax.set_title("Depth-stratified wet-year and dry-year domain-mean response")
    cur_top = ax.get_ylim()[1]
    cur_bot = ax.get_ylim()[0]
    ax.set_ylim(cur_bot, cur_top * 1.20)
    ax.legend(loc="upper right", fontsize=9, bbox_to_anchor=(1.0, -0.15), ncol=2)
    n_y = cur_top * 1.10
    for xi, n in zip(x, res["n_wells"].values):
        ax.text(xi, n_y, f"n={int(n)}", ha="center", fontsize=8, color="gray")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Fig 7 rendered: {out_path.name}")


# =============================================================================
# Figure 8 — CHIRPS rainfall classification
# =============================================================================
def fig08_chirps_rainfall():
    out_path = FIG_DIR / "Fig08_v5_CHIRPS_rainfall.png"
    chirps = pd.read_csv(TS_DIR / "CHIRPS_JJAS_annual.csv")
    yr_col   = next((c for c in chirps.columns if "year" in c.lower()),
                    chirps.columns[0])
    rain_col = next((c for c in chirps.columns
                     if "rain" in c.lower() or "jjas" in c.lower()
                     or "total" in c.lower()),
                    chirps.columns[1])

    mu  = float(chirps[rain_col].mean())
    sig = float(chirps[rain_col].std(ddof=1))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = []
    for v in chirps[rain_col]:
        if v > mu + sig: colors.append(COLOR_WET)
        elif v < mu - sig: colors.append(COLOR_DRY)
        else: colors.append("#888888")
    ax.bar(chirps[yr_col], chirps[rain_col], color=colors)
    ax.axhline(mu, color="k", ls="-", lw=0.8,
               label=f"22-year mean ({mu:.0f} mm)")
    ax.axhline(mu + sig, color="k", ls="--", lw=0.6,
               label=f"Wet threshold (mean + sigma = {mu + sig:.0f} mm)")
    ax.axhline(mu - sig, color="k", ls="--", lw=0.6,
               label=f"Dry threshold (mean - sigma = {mu - sig:.0f} mm)")
    ax.set_xlabel("Year")
    ax.set_ylabel("JJAS total rainfall (mm)")
    ax.set_title("Annual JJAS rainfall, IGP domain (CHIRPS v3.0)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Fig 8 rendered: {out_path.name}")


# =============================================================================
# Figure 9 — Per-wet-year deep-well response
# =============================================================================
def fig09_per_wet_year():
    out_path = FIG_DIR / "Fig09_v5_wet_year_per_year.png"
    res = pd.read_csv(TS_DIR / "R2_depth_cut_sensitivity.csv")
    deep_row = res[res["depth_class"] == "deep_ge100"].iloc[0]
    yrs  = [2010, 2011, 2013, 2019]
    vals = [deep_row.get(f"y{y}_recharge_m", np.nan) for y in yrs]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar([str(y) for y in yrs], vals, color=COLOR_DEEP, alpha=0.85)
    ax.axhline(deep_row["wet_mean_m"], color=COLOR_TREND, lw=1.2,
               label=f"4-year mean ({deep_row['wet_mean_m']:+.2f} m)")
    ax.set_xlabel("Wet year")
    ax.set_ylabel("Domain-mean water-level rise May to November (m)")
    ax.set_title("Per-wet-year deep-well (>=100 m) response")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Fig 9 rendered: {out_path.name}")


# =============================================================================
# Supplementary
# =============================================================================
def figS1_wcor():
    out_path = FIG_DIR / "FigS1_v5_wcor_L60.png"
    src = FIG_DIR / "R6_wcorrelation_L60.png"
    if src.exists():
        shutil.copy(src, out_path)
        print(f"  Fig S1 copied from R6: {out_path.name}")
    else:
        print(f"  Fig S1: {src.name} not found; run R6_ssa_quantitative.py first")


def figS2_gldas_components():
    out_path = FIG_DIR / "FigS2_v5_GLDAS_components.png"
    gldas = safe_load(TS_DIR / "GLDAS_processed.csv")
    sm_col  = "SM_anom"     if "SM_anom"     in gldas.columns else None
    swe_col = "SWE_anom"    if "SWE_anom"    in gldas.columns else None
    et_col  = "ET_mm_month" if "ET_mm_month" in gldas.columns else None
    if not all([sm_col, swe_col, et_col]):
        print(f"  Fig S2: GLDAS_processed.csv missing required columns; SKIPPED")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, col, lbl in zip(
        axes,
        [sm_col, swe_col, et_col],
        ["Soil moisture anomaly (mm)",
         "SWE anomaly (mm)",
         "Evapotranspiration (mm/month)"],
    ):
        ax.plot(gldas["time"], gldas[col], color=COLOR_GWS, lw=0.9)
        ax.set_ylabel(lbl)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Year")
    axes[0].set_title("GLDAS-2.1 Noah land-surface variables, IGP domain")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Fig S2 rendered: {out_path.name}")


def figS3_alluvial():
    out_path = FIG_DIR / "FigS3_v5_alluvial_mask.png"
    src = FIG_DIR / "R8_alluvial_mask.png"
    if src.exists():
        shutil.copy(src, out_path)
        print(f"  Fig S3 copied from R8: {out_path.name}")
    else:
        print(f"  Fig S3: {src.name} not found; run R8_alluvial_mask.py first")


# =============================================================================
# Main
# =============================================================================
def main():
    apply_v5_style()
    figs = [
        ("Fig 1",  fig01_study_area),
        ("Fig 2",  fig02_grace_tws),
        ("Fig 3",  fig03_gws_trend_changepoint),
        ("Fig 4",  fig04_mc_band),
        ("Fig 5",  fig05_et_intercomp),
        ("Fig 6",  fig06_well_groups),
        ("Fig 7",  fig07_depth_response),
        ("Fig 8",  fig08_chirps_rainfall),
        ("Fig 9",  fig09_per_wet_year),
        ("Fig S1", figS1_wcor),
        ("Fig S2", figS2_gldas_components),
        ("Fig S3", figS3_alluvial),
    ]
    print(f"[F] Rendering {len(figs)} manuscript figures to {FIG_DIR}/")
    for label, fn in figs:
        try:
            fn()
        except Exception as e:
            print(f"  {label} FAILED: {e}")
    print(f"[F] Done.")


if __name__ == "__main__":
    main()
