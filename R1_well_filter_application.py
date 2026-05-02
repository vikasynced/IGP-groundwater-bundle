"""
R1_well_filter_application.py
==============================
Apply the v5.1 well-selection filter to the raw CGWB table and emit
depth-stratified subsets for downstream analysis.

PURPOSE
-------
The v5.1 manuscript (§3.4) defines a single, transparent rule for choosing
which CGWB wells enter the analysis: spatial filter to the IGP bounding
box, with no further filtering on aquifer type, well type, or record
length. This script is the canonical implementation of that rule.

Wells are then stratified into three classes by screen depth:

    Shallow:        depth < 30 m       (2,855 wells; phreatic aquifer)
    Transitional:   30 m <= depth < 100 m  (2,849 wells)
    Deep:           depth >= 100 m     (900 wells; semi-confined aquifer)

These three subsets, together with the merged "all wells" table, are
written as CSVs that R2 (depth-cut sensitivity) and R8 (alluvial mask)
read directly.

This script SUPERSEDES the v4 hypothesis-test workflow that selected
wells by an ad-hoc combination of geographic, aquifer, well-type, and
record-length filters. The audit history of how the v5.1 rule was chosen
is preserved in the project archive but is not part of the published
reproducibility bundle.

INPUT
-----
  data/cgwb_wells/cgwb_wells_raw.csv

OUTPUTS
-------
  outputs/timeseries/wells_filtered_all.csv
      All wells inside the IGP bounding box; one row per well.
      Includes Latitude, Longitude, depth_m, depth_class.

  outputs/timeseries/wells_filtered_shallow.csv      depth < 30 m
  outputs/timeseries/wells_filtered_transitional.csv 30 <= depth < 100 m
  outputs/timeseries/wells_filtered_deep.csv         depth >= 100 m

  outputs/timeseries/R1_filter_summary.txt
      Human-readable summary of well counts per state, depth class,
      and per-decade observation completeness.

USAGE
-----
  cd path/to/IGP_GW_Study/scripts
  python R1_well_filter_application.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR, TS_DIR,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    SHALLOW_DEPTH_MAX, DEEP_DEPTH_MIN,
    STUDY_START_YEAR, STUDY_END_YEAR,
    EXPECTED_N_WELLS_TOTAL, EXPECTED_N_WELLS_SHALLOW,
    EXPECTED_N_WELLS_TRANSITIONAL, EXPECTED_N_WELLS_DEEP,
)


WELLS_CSV = DATA_DIR / "cgwb_wells" / "cgwb_wells_raw.csv"

OUT_ALL    = TS_DIR / "wells_filtered_all.csv"
OUT_SHL    = TS_DIR / "wells_filtered_shallow.csv"
OUT_TRA    = TS_DIR / "wells_filtered_transitional.csv"
OUT_DEEP   = TS_DIR / "wells_filtered_deep.csv"
OUT_SUMMARY = TS_DIR / "R1_filter_summary.txt"

MONTH_PREFIXES = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}


def col_to_year(colname):
    """Convert a 'Mon-YY' column name to a full year (assumes 20YY)."""
    yy = int(colname.split("-")[1])
    return 2000 + yy if yy < 50 else 1900 + yy


def find_observation_columns(df):
    """
    Locate columns whose name matches the 'Mon-YY' monthly observation pattern
    AND whose year falls within the study window.
    """
    cols = []
    for c in df.columns:
        if len(c) != 6 or c[3] != "-" or c[:3] not in MONTH_PREFIXES:
            continue
        try:
            yy = int(c.split("-")[1])
        except ValueError:
            continue
        # Skip ill-formed suffixes (yy >= 50 would be 1950s and is implausible)
        if yy < 0 or yy >= 50:
            continue
        year = col_to_year(c)
        if STUDY_START_YEAR <= year <= STUDY_END_YEAR:
            cols.append(c)
    return cols


def main():
    TS_DIR.mkdir(parents=True, exist_ok=True)

    if not WELLS_CSV.exists():
        raise FileNotFoundError(
            f"CGWB wells CSV not found at {WELLS_CSV}.\n"
            f"Obtain cgwb_wells_raw.csv from the CGWB India-WRIS portal "
            f"and place it in {WELLS_CSV.parent}."
        )

    print(f"[R1] Loading raw CGWB wells table ...")
    df_raw = pd.read_csv(WELLS_CSV, low_memory=False)
    print(f"[R1] Total wells in raw table: {len(df_raw)}")

    # Coerce numeric columns
    df_raw["Latitude"]   = pd.to_numeric(df_raw["Latitude"],   errors="coerce")
    df_raw["Longitude"]  = pd.to_numeric(df_raw["Longitude"],  errors="coerce")
    df_raw["Well Depth"] = pd.to_numeric(df_raw["Well Depth"], errors="coerce")

    # -------------------------------------------------------------------------
    # v5.1 filter: spatial bounding box only
    # -------------------------------------------------------------------------
    in_bbox = (
        df_raw["Latitude"].between(LAT_MIN, LAT_MAX) &
        df_raw["Longitude"].between(LON_MIN, LON_MAX)
    )
    wells = df_raw.loc[in_bbox].copy().reset_index(drop=True)
    print(f"[R1] Wells inside IGP bbox: {len(wells)}")

    # -------------------------------------------------------------------------
    # Depth stratification
    # -------------------------------------------------------------------------
    depth = wells["Well Depth"]
    wells["depth_class"] = np.select(
        condlist=[
            depth < SHALLOW_DEPTH_MAX,
            (depth >= SHALLOW_DEPTH_MAX) & (depth < DEEP_DEPTH_MIN),
            depth >= DEEP_DEPTH_MIN,
        ],
        choicelist=["shallow", "transitional", "deep"],
        default="unknown",   # wells with NaN depth
    )

    # Standard renaming for downstream code
    wells["depth_m"] = wells["Well Depth"]

    # Order columns: identifying columns first, then class, then everything else
    front = ["Station Code", "State", "District", "Latitude", "Longitude",
             "depth_m", "depth_class"]
    front = [c for c in front if c in wells.columns]
    rest  = [c for c in wells.columns if c not in front]
    wells = wells[front + rest]

    n_shl = (wells["depth_class"] == "shallow").sum()
    n_tra = (wells["depth_class"] == "transitional").sum()
    n_dep = (wells["depth_class"] == "deep").sum()
    n_unk = (wells["depth_class"] == "unknown").sum()

    # -------------------------------------------------------------------------
    # Observation completeness diagnostic (per-decade)
    # -------------------------------------------------------------------------
    obs_cols = find_observation_columns(wells)
    n_obs_per_well = wells[obs_cols].notna().sum(axis=1)
    median_obs    = int(n_obs_per_well.median()) if len(n_obs_per_well) else 0
    pct_with_15yr = float((n_obs_per_well >= 15 * 12).mean() * 100) \
                    if len(n_obs_per_well) else 0.0

    # -------------------------------------------------------------------------
    # Save filtered CSVs
    # -------------------------------------------------------------------------
    wells.to_csv(OUT_ALL, index=False)
    wells[wells["depth_class"] == "shallow"     ].to_csv(OUT_SHL,  index=False)
    wells[wells["depth_class"] == "transitional"].to_csv(OUT_TRA,  index=False)
    wells[wells["depth_class"] == "deep"        ].to_csv(OUT_DEEP, index=False)

    # -------------------------------------------------------------------------
    # State-level breakdown
    # -------------------------------------------------------------------------
    state_breakdown = (
        wells.groupby("State")["depth_class"]
        .value_counts()
        .unstack(fill_value=0)
    )
    # Reorder columns by class order, dropping 'unknown' if zero
    cols_order = [c for c in ["shallow", "transitional", "deep", "unknown"]
                  if c in state_breakdown.columns]
    state_breakdown = state_breakdown[cols_order]
    state_breakdown["total"] = state_breakdown.sum(axis=1)
    state_breakdown = state_breakdown.sort_values("total", ascending=False)

    # -------------------------------------------------------------------------
    # Human-readable summary
    # -------------------------------------------------------------------------
    summary_lines = []
    summary_lines.append("R1  CGWB WELL FILTER APPLICATION  -  SUMMARY")
    summary_lines.append("=" * 78)
    summary_lines.append("")
    summary_lines.append("Filter rule (v5.1):")
    summary_lines.append(f"  Spatial: {LAT_MIN}-{LAT_MAX} N, {LON_MIN}-{LON_MAX} E")
    summary_lines.append("  No additional filters on aquifer type, well type, or record length.")
    summary_lines.append("")
    summary_lines.append("Depth classes:")
    summary_lines.append(f"  Shallow:      depth < {SHALLOW_DEPTH_MAX:.0f} m")
    summary_lines.append(f"  Transitional: {SHALLOW_DEPTH_MAX:.0f} <= depth < {DEEP_DEPTH_MIN:.0f} m")
    summary_lines.append(f"  Deep:         depth >= {DEEP_DEPTH_MIN:.0f} m")
    summary_lines.append("")
    summary_lines.append("WELL COUNTS")
    summary_lines.append("-" * 78)
    summary_lines.append(f"  Total raw input:                {len(df_raw):>6d}")
    summary_lines.append(f"  After bbox filter:              {len(wells):>6d}")
    summary_lines.append(f"    Shallow (<30 m):              {n_shl:>6d}")
    summary_lines.append(f"    Transitional (30-100 m):      {n_tra:>6d}")
    summary_lines.append(f"    Deep (>=100 m):               {n_dep:>6d}")
    if n_unk:
        summary_lines.append(f"    Unknown depth (excluded):     {n_unk:>6d}")
    summary_lines.append("")
    summary_lines.append("v5.1 manuscript expected counts (for verification):")
    summary_lines.append(f"  Total:        {EXPECTED_N_WELLS_TOTAL}")
    summary_lines.append(f"  Shallow:      {EXPECTED_N_WELLS_SHALLOW}")
    summary_lines.append(f"  Transitional: {EXPECTED_N_WELLS_TRANSITIONAL}")
    summary_lines.append(f"  Deep:         {EXPECTED_N_WELLS_DEEP}")
    summary_lines.append("")
    summary_lines.append("OBSERVATION COMPLETENESS")
    summary_lines.append("-" * 78)
    summary_lines.append(f"  Observation columns in study window "
                         f"({STUDY_START_YEAR}-{STUDY_END_YEAR}): {len(obs_cols)}")
    summary_lines.append(f"  Median non-null obs per well: {median_obs}")
    summary_lines.append(f"  Wells with >=15 years of observations: {pct_with_15yr:.1f}%")
    summary_lines.append("")
    summary_lines.append("STATE-LEVEL BREAKDOWN")
    summary_lines.append("-" * 78)
    summary_lines.append(state_breakdown.to_string())
    summary_lines.append("")
    summary_lines.append("=" * 78)
    summary = "\n".join(summary_lines)

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------
    print(f"[R1] Wells after filter: {len(wells)} "
          f"(expected {EXPECTED_N_WELLS_TOTAL})")
    print(f"[R1]   Shallow:      {n_shl:>5d} (expected {EXPECTED_N_WELLS_SHALLOW})")
    print(f"[R1]   Transitional: {n_tra:>5d} (expected {EXPECTED_N_WELLS_TRANSITIONAL})")
    print(f"[R1]   Deep:         {n_dep:>5d} (expected {EXPECTED_N_WELLS_DEEP})")
    if n_unk:
        print(f"[R1]   Unknown depth (excluded): {n_unk}")
    print(f"[R1] Wrote: {OUT_ALL.relative_to(OUT_ALL.parents[2])}")
    print(f"[R1] Wrote: {OUT_SHL.relative_to(OUT_SHL.parents[2])}")
    print(f"[R1] Wrote: {OUT_TRA.relative_to(OUT_TRA.parents[2])}")
    print(f"[R1] Wrote: {OUT_DEEP.relative_to(OUT_DEEP.parents[2])}")
    print(f"[R1] Wrote: {OUT_SUMMARY.relative_to(OUT_SUMMARY.parents[2])}")


if __name__ == "__main__":
    main()
