"""
R2_depth_cut_sensitivity.py
============================
Depth-stratified pre/post-monsoon water-level analysis.

PURPOSE
-------
Computes per-well-per-year recharge proxies (May - November water level)
across six depth classes and reports wet-year vs dry-year composites with
95% confidence intervals. The output supports Table 2 of the manuscript
(§5.4) and demonstrates that the water-table response to monsoon rainfall
attenuates monotonically with screen depth, consistent with vadose-zone
pressure transmission rather than direct recharge to a confined aquifer.

The all-year correlation between JJAS rainfall and depth-stratified
recharge is reported as the primary inferential anchor (n=22), with the
small-n wet-year means (n=4 testable wet years) reported as descriptive
support. This avoids over-interpreting a 4-year composite as if it were
a robust hypothesis test.

INPUTS
------
  outputs/timeseries/wells_filtered_all.csv   from R1
  outputs/timeseries/CHIRPS_JJAS_annual.csv   from 07_chirps_rainfall.py

OUTPUTS
-------
  outputs/timeseries/R2_depth_cut_sensitivity.csv
      Per-depth-class summary: wet/dry means, CIs, all-year correlation.
  outputs/timeseries/R2_per_well_year_changes.csv
      Tidy long-form table of per-well-per-year recharge values.
  outputs/timeseries/R2_state_breakdown_deep.csv
      Per-state breakdown for the deep (>=100 m) class.
  outputs/timeseries/R2_summary.txt
      Human-readable summary.

USAGE
-----
  cd path/to/IGP_GW_Study/scripts
  python R2_depth_cut_sensitivity.py    # requires R1 and 07 first
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    TS_DIR,
    STUDY_START_YEAR, STUDY_END_YEAR,
    WET_YEARS_TESTABLE, DRY_YEARS_CHIRPS,
    EXPECTED_DEEP_WET_MEAN_M, EXPECTED_SHALLOW_WET_MEAN_M,
)
from pipeline_utils import pearson_with_p


WELLS_CSV  = TS_DIR / "wells_filtered_all.csv"
CHIRPS_CSV = TS_DIR / "CHIRPS_JJAS_annual.csv"

OUT_MAIN    = TS_DIR / "R2_depth_cut_sensitivity.csv"
OUT_PERWELL = TS_DIR / "R2_per_well_year_changes.csv"
OUT_STATES  = TS_DIR / "R2_state_breakdown_deep.csv"
OUT_SUMMARY = TS_DIR / "R2_summary.txt"

# Depth classes for the sensitivity matrix. Three primary cuts plus three
# sensitivity cuts at increasing depth, plus the shallow control.
DEPTH_TESTS = [
    ("shallow_lt30",        lambda d: d < 30,                  "Shallow control (<30 m)"),
    ("transitional_30_100", lambda d: (d >= 30) & (d < 100),   "Transitional (30-100 m)"),
    ("deep_ge100",          lambda d: d >= 100,                "Deep, primary cutoff (>=100 m)"),
    ("deep_100_150",        lambda d: (d >= 100) & (d < 150),  "Deep transitional (100-150 m)"),
    ("deep_ge150",          lambda d: d >= 150,                "Deep, sensitivity (>=150 m)"),
    ("deep_ge200",          lambda d: d >= 200,                "Deep, strict (>=200 m)"),
]

MONTH_PREFIXES = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}


def col_to_year(colname):
    yy = int(colname.split("-")[1])
    return 2000 + yy if yy < 50 else 1900 + yy


def find_seasonal_columns(columns):
    """Return dicts mapping year -> column name for May and November."""
    may_by_year, nov_by_year = {}, {}
    for c in columns:
        if len(c) != 6 or c[3] != "-" or c[:3] not in MONTH_PREFIXES:
            continue
        try:
            yy = int(c.split("-")[1])
        except ValueError:
            continue
        if yy < 0 or yy >= 50:
            continue
        year = col_to_year(c)
        if not (STUDY_START_YEAR <= year <= STUDY_END_YEAR):
            continue
        if c.startswith("May"):
            may_by_year[year] = c
        elif c.startswith("Nov"):
            nov_by_year[year] = c
    return may_by_year, nov_by_year


def main():
    TS_DIR.mkdir(parents=True, exist_ok=True)

    if not WELLS_CSV.exists():
        raise FileNotFoundError(
            f"{WELLS_CSV} not found. Run R1_well_filter_application.py first."
        )
    if not CHIRPS_CSV.exists():
        raise FileNotFoundError(
            f"{CHIRPS_CSV} not found. Run 07_chirps_rainfall.py first."
        )

    print(f"[R2] Loading filtered wells and CHIRPS data ...")
    wells  = pd.read_csv(WELLS_CSV, low_memory=False)
    chirps = pd.read_csv(CHIRPS_CSV)

    may_cols, nov_cols = find_seasonal_columns(wells.columns)
    common_years = sorted(set(may_cols) & set(nov_cols))
    print(f"[R2] Wells: {len(wells)}; "
          f"years with both May and November columns: {len(common_years)}")

    # -------------------------------------------------------------------------
    # Per-well, per-year recharge (May - November water level)
    # -------------------------------------------------------------------------
    # In CGWB convention, water level is depth-to-water (m below ground level):
    # larger value = deeper water table = LESS water in the aquifer.
    # Recharge over the monsoon manifests as the water table RISING,
    # i.e. depth-to-water DECREASING. So the recharge proxy is May - Nov:
    # positive values indicate recharge.
    records = []
    for _, w in wells.iterrows():
        for y in common_years:
            may_v = w.get(may_cols[y])
            nov_v = w.get(nov_cols[y])
            if pd.notna(may_v) and pd.notna(nov_v):
                records.append({
                    "station_code": w.get("Station Code", ""),
                    "state":        w.get("State", ""),
                    "lat":          w.get("Latitude", np.nan),
                    "lon":          w.get("Longitude", np.nan),
                    "depth_m":      w.get("depth_m", np.nan),
                    "year":         y,
                    "may_bgl":      float(may_v),    # pre-monsoon depth-to-water (m)
                    "nov_bgl":      float(nov_v),    # post-monsoon depth-to-water (m)
                    "recharge_m":   may_v - nov_v,   # may - nov; positive = recharge
                })
    obs = pd.DataFrame(records)
    obs.to_csv(OUT_PERWELL, index=False)
    print(f"[R2] Per-well-per-year observations: {len(obs)}")

    # -------------------------------------------------------------------------
    # Per-depth-class composites
    # -------------------------------------------------------------------------
    results = []
    for tag, depth_fn, description in DEPTH_TESTS:
        sub = obs[depth_fn(obs["depth_m"])]
        n_wells = sub["station_code"].nunique()
        if n_wells < 5:
            print(f"[R2]   [{tag}] {n_wells} wells - skipping")
            continue

        # Domain-mean recharge per year (mean across wells in that year)
        annual = sub.groupby("year").agg(
            domain_mean_recharge=("recharge_m", "mean"),
            n_wells_with_obs=("station_code", "nunique"),
            n_observations=("recharge_m", "count"),
        ).reset_index()

        # Wet-year composite (excluding 2018 due to GRACE-FO gap)
        wet_obs = annual[annual["year"].isin(WET_YEARS_TESTABLE)]
        dry_obs = annual[annual["year"].isin(DRY_YEARS_CHIRPS)]

        if len(wet_obs) >= 2:
            wet_mean = float(wet_obs["domain_mean_recharge"].mean())
            wet_std  = float(wet_obs["domain_mean_recharge"].std(ddof=1))
            n_wet    = len(wet_obs)
            ci_half  = float(stats.t.ppf(0.975, df=n_wet - 1) * wet_std / np.sqrt(n_wet))
            wet_lo, wet_hi = wet_mean - ci_half, wet_mean + ci_half
            n_pos = int((wet_obs["domain_mean_recharge"] > 0).sum())
        else:
            wet_mean = wet_std = wet_lo = wet_hi = np.nan
            n_wet = len(wet_obs); n_pos = 0

        if len(dry_obs) >= 2:
            dry_mean = float(dry_obs["domain_mean_recharge"].mean())
            n_dry    = len(dry_obs)
        else:
            dry_mean = np.nan
            n_dry = len(dry_obs)

        effect = (wet_mean - dry_mean) if (pd.notna(wet_mean) and pd.notna(dry_mean)) else np.nan

        # All-year correlation with CHIRPS JJAS rainfall (n = 22; this is the
        # primary inferential anchor in the manuscript).
        merged = annual.merge(chirps[["year", "JJAS_mm"]], on="year", how="inner")
        r_all, p_all, n_corr = pearson_with_p(
            merged["JJAS_mm"].values,
            merged["domain_mean_recharge"].values,
        )

        row = {
            "depth_class":    tag,
            "description":    description,
            "n_wells":        int(n_wells),
            "wet_mean_m":     wet_mean,
            "wet_ci95_lo":    wet_lo,
            "wet_ci95_hi":    wet_hi,
            "wet_n_years":    int(n_wet),
            "wet_n_positive": int(n_pos),
            "dry_mean_m":     dry_mean,
            "dry_n_years":    int(n_dry),
            "effect_size_m":  effect,
            "all_year_r":     r_all,
            "all_year_p":     p_all,
            "all_year_n":     int(n_corr),
        }
        # Per-wet-year detail (for the human-readable summary)
        for _, r in wet_obs.iterrows():
            row[f"y{int(r['year'])}_recharge_m"] = float(r["domain_mean_recharge"])
        results.append(row)

    res = pd.DataFrame(results)
    res.to_csv(OUT_MAIN, index=False)

    # -------------------------------------------------------------------------
    # State-level breakdown (deep wells)
    # -------------------------------------------------------------------------
    deep_obs = obs[obs["depth_m"] >= 100]
    state_breakdown = (
        deep_obs.groupby("state")
        .agg(n_wells=("station_code", "nunique"),
             n_observations=("recharge_m", "count"))
        .reset_index()
        .sort_values("n_wells", ascending=False)
    )
    state_breakdown.to_csv(OUT_STATES, index=False)

    # -------------------------------------------------------------------------
    # Human-readable summary
    # -------------------------------------------------------------------------
    lines = []
    lines.append("R2  DEPTH-CUT SENSITIVITY  -  SUMMARY")
    lines.append("=" * 78)
    lines.append(f"Wet years (testable, excluding 2018 in GRACE gap): {WET_YEARS_TESTABLE}")
    lines.append(f"Dry years:                                          {DRY_YEARS_CHIRPS}")
    lines.append(f"Total per-well-per-year observations: {len(obs)}")
    lines.append("")
    lines.append("DEPTH-CUT RESULTS")
    lines.append("-" * 78)
    for _, r in res.iterrows():
        lines.append("")
        lines.append(f"  [{r['depth_class']}]  {r['description']}")
        lines.append(f"    n_wells: {int(r['n_wells'])}")
        if pd.notna(r["wet_mean_m"]):
            lines.append(
                f"    Wet years (n={int(r['wet_n_years'])}): "
                f"mean = {r['wet_mean_m']:+.3f} m, "
                f"95% CI [{r['wet_ci95_lo']:+.3f}, {r['wet_ci95_hi']:+.3f}], "
                f"positive in {int(r['wet_n_positive'])}/{int(r['wet_n_years'])}"
            )
        if pd.notna(r["dry_mean_m"]):
            lines.append(
                f"    Dry years (n={int(r['dry_n_years'])}): "
                f"mean = {r['dry_mean_m']:+.3f} m"
            )
        if pd.notna(r["effect_size_m"]):
            lines.append(f"    Effect size (wet - dry): {r['effect_size_m']:+.3f} m")
        if pd.notna(r["all_year_r"]):
            lines.append(
                f"    All-year correlation with JJAS rainfall: "
                f"r = {r['all_year_r']:+.3f}, p = {r['all_year_p']:.4f}, "
                f"n = {int(r['all_year_n'])}"
            )
        # Per-wet-year detail
        yr_keys = [k for k in r.index if k.startswith("y2")]
        ystr = ", ".join(f"{k[1:]}: {r[k]:+.2f} m"
                         for k in yr_keys if pd.notna(r[k]))
        if ystr:
            lines.append(f"    Per-wet-year: {ystr}")
    lines.append("")
    lines.append("STATE-LEVEL BREAKDOWN (deep wells, >=100 m)")
    lines.append("-" * 78)
    lines.append(state_breakdown.to_string(index=False))
    lines.append("")
    lines.append("=" * 78)
    summary = "\n".join(lines)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    # -------------------------------------------------------------------------
    # Final report (compare with manuscript expected values)
    # -------------------------------------------------------------------------
    print(f"[R2] Wrote: {OUT_MAIN.relative_to(OUT_MAIN.parents[2])}")
    print(f"[R2] Wrote: {OUT_PERWELL.relative_to(OUT_PERWELL.parents[2])}")
    print(f"[R2] Wrote: {OUT_STATES.relative_to(OUT_STATES.parents[2])}")
    print(f"[R2] Wrote: {OUT_SUMMARY.relative_to(OUT_SUMMARY.parents[2])}")
    print()
    print(f"[R2] Manuscript verification (v5.1 §5.4, Table 2):")
    shl_row = res[res["depth_class"] == "shallow_lt30"]
    dep_row = res[res["depth_class"] == "deep_ge100"]
    if not shl_row.empty:
        v = float(shl_row["wet_mean_m"].iloc[0])
        print(f"[R2]   Shallow wet-year mean: {v:+.2f} m  "
              f"(expected {EXPECTED_SHALLOW_WET_MEAN_M:+.2f})")
    if not dep_row.empty:
        v = float(dep_row["wet_mean_m"].iloc[0])
        print(f"[R2]   Deep    wet-year mean: {v:+.2f} m  "
              f"(expected {EXPECTED_DEEP_WET_MEAN_M:+.2f})")


if __name__ == "__main__":
    main()
