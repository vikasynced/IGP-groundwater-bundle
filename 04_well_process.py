"""
04_well_process.py
==================
Filter the CGWB wells table to the IGP domain and produce annual
pre/post-monsoon water-level aggregates per depth class.

PURPOSE
-------
The Central Ground Water Board (CGWB) of India publishes a national
table of monitoring wells with monthly observations of depth-to-water
(in m below ground level). This script:

  1. Filters to the IGP bounding box.
  2. Stratifies wells into shallow (<30 m) and deep (>=100 m) classes
     using the screen-depth column.
  3. Computes the domain-mean pre-monsoon (May) and post-monsoon (November)
     water level for each year 2002-2023.
  4. Computes the per-year recharge proxy as (May - Nov), where positive
     values indicate water-table rise (recharge).

Outputs are consumed by the legacy v4 hypothesis-test workflow that has
been superseded by R2 (depth-cut sensitivity); however, R2 itself uses
the raw CGWB CSV directly to access individual well records, so this
upstream script remains useful for the basic shallow/deep climatology
plot referenced in §5.4 of the manuscript.

INPUT
-----
  data/cgwb_wells/cgwb_wells_raw.csv

  Expected columns: Latitude, Longitude, Well Depth, plus monthly
  observation columns named like "May-02", "Nov-02", ..., "Dec-23".
  Units: water level in m below ground surface (positive downward).

OUTPUTS
-------
  outputs/timeseries/shallow_wells_annual.csv     <30 m wells
  outputs/timeseries/deep_wells_annual.csv        >=100 m wells
      Each: index=year, columns=may_wl, nov_wl, annual_change

  outputs/figures/04_CGWB_wells_climatology.png   diagnostic two-panel plot

USAGE
-----
  cd path/to/IGP_GW_Study/scripts
  python 04_well_process.py

NOTES
-----
The depth boundary at >=100 m (rather than the v4 ">100 m") was adopted
in v5 to make the cuts non-overlapping and to match the convention used
by R2's sensitivity analysis. Wells with exactly 100.0 m depth are
included in the deep class.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR, TS_DIR, FIG_DIR,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    SHALLOW_DEPTH_MAX, DEEP_DEPTH_MIN,
    STUDY_START_YEAR, STUDY_END_YEAR,
)


WELLS_CSV = DATA_DIR / "cgwb_wells" / "cgwb_wells_raw.csv"
OUT_SHALLOW = TS_DIR / "shallow_wells_annual.csv"
OUT_DEEP    = TS_DIR / "deep_wells_annual.csv"
OUT_FIG     = FIG_DIR / "04_CGWB_wells_climatology.png"

MONTH_PREFIXES = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}


def col_to_year(colname):
    """Convert a 'Mon-YY' column name to a full year (assumes 20YY)."""
    yy = int(colname.split("-")[1])
    return 2000 + yy if yy < 50 else 1900 + yy


def annual_mean(group, may_cols, nov_cols, may_years, nov_years):
    """
    Compute domain-mean May and November water level per year.

    Within each year the mean is taken across all wells in the depth
    class that have a non-null observation for that month. The recharge
    proxy is May - November: a positive value means water level rose
    over the monsoon (depth-to-water decreased).
    """
    rows = []
    for mc, nc, yr in zip(may_cols, nov_cols, may_years):
        if not (STUDY_START_YEAR <= yr <= STUDY_END_YEAR):
            continue
        may_val = pd.to_numeric(group[mc], errors="coerce").mean()
        nov_val = pd.to_numeric(group[nc], errors="coerce").mean()
        rows.append({
            "year":          yr,
            "may_wl":        may_val,
            "nov_wl":        nov_val,
            "annual_change": may_val - nov_val,
        })
    return pd.DataFrame(rows).set_index("year")


def main():
    TS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not WELLS_CSV.exists():
        raise FileNotFoundError(
            f"CGWB wells CSV not found at {WELLS_CSV}.\n"
            f"Obtain cgwb_wells_raw.csv from the CGWB India-WRIS portal "
            f"and place it in {WELLS_CSV.parent}."
        )

    print(f"[04] Loading CGWB wells table ...")
    df = pd.read_csv(WELLS_CSV, low_memory=False)
    print(f"[04] Total wells in raw table: {len(df)}")

    # Coerce numeric columns; CGWB occasionally has non-numeric placeholders.
    df["Latitude"]   = pd.to_numeric(df["Latitude"],   errors="coerce")
    df["Longitude"]  = pd.to_numeric(df["Longitude"],  errors="coerce")
    df["Well Depth"] = pd.to_numeric(df["Well Depth"], errors="coerce")

    # -------------------------------------------------------------------------
    # Spatial filter
    # -------------------------------------------------------------------------
    igp = df[
        df["Latitude"].between(LAT_MIN, LAT_MAX) &
        df["Longitude"].between(LON_MIN, LON_MAX)
    ].copy()
    print(f"[04] Wells in IGP bounding box: {len(igp)}")

    # -------------------------------------------------------------------------
    # Depth stratification (matches v5.1 manuscript, Table 2 cuts)
    # -------------------------------------------------------------------------
    shallow = igp[igp["Well Depth"] <  SHALLOW_DEPTH_MAX].copy()
    deep    = igp[igp["Well Depth"] >= DEEP_DEPTH_MIN  ].copy()
    print(f"[04] Shallow wells (<{SHALLOW_DEPTH_MAX:.0f} m): {len(shallow)}")
    print(f"[04] Deep wells    (>={DEEP_DEPTH_MIN:.0f} m): {len(deep)}")

    # -------------------------------------------------------------------------
    # Identify May/November observation columns
    # -------------------------------------------------------------------------
    # CGWB uses 'Mon-YY' format. Filter to columns whose suffix is a
    # plausible 2-digit year (>=02 to skip noise like 'Jan-1').
    may_cols, may_years = [], []
    nov_cols, nov_years = [], []
    for c in df.columns:
        if len(c) != 6 or c[3] != "-":
            continue
        if c[:3] not in MONTH_PREFIXES:
            continue
        try:
            yy = int(c.split("-")[1])
        except ValueError:
            continue
        if yy < 2:
            continue
        if c.startswith("May"):
            may_cols.append(c)
            may_years.append(col_to_year(c))
        elif c.startswith("Nov"):
            nov_cols.append(c)
            nov_years.append(col_to_year(c))

    if len(may_cols) != len(nov_cols):
        # Handle edge case where one season is missing for some year
        common = set(may_years) & set(nov_years)
        may_pairs = [(c, y) for c, y in zip(may_cols, may_years) if y in common]
        nov_pairs = [(c, y) for c, y in zip(nov_cols, nov_years) if y in common]
        may_cols  = [c for c, _ in sorted(may_pairs, key=lambda p: p[1])]
        may_years = [y for _, y in sorted(may_pairs, key=lambda p: p[1])]
        nov_cols  = [c for c, _ in sorted(nov_pairs, key=lambda p: p[1])]
        nov_years = [y for _, y in sorted(nov_pairs, key=lambda p: p[1])]

    # -------------------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------------------
    shallow_ts = annual_mean(shallow, may_cols, nov_cols, may_years, nov_years)
    deep_ts    = annual_mean(deep,    may_cols, nov_cols, may_years, nov_years)
    shallow_ts.to_csv(OUT_SHALLOW)
    deep_ts.to_csv(OUT_DEEP)

    # -------------------------------------------------------------------------
    # Diagnostic figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(shallow_ts.index, shallow_ts["may_wl"],
                 "o--", color="orange", label="Pre-monsoon (May)", linewidth=1.5)
    axes[0].plot(shallow_ts.index, shallow_ts["nov_wl"],
                 "s-",  color="brown",  label="Post-monsoon (November)",
                 linewidth=1.5)
    axes[0].set_ylabel("Water level (m bgl)")
    axes[0].set_title(
        f"Shallow wells (<{SHALLOW_DEPTH_MAX:.0f} m) — n = {len(shallow)}"
    )
    axes[0].legend()
    axes[0].invert_yaxis()  # deeper depth = lower on plot
    axes[0].grid(alpha=0.25)

    axes[1].plot(deep_ts.index, deep_ts["may_wl"],
                 "o--", color="steelblue", label="Pre-monsoon (May)", linewidth=1.5)
    axes[1].plot(deep_ts.index, deep_ts["nov_wl"],
                 "s-",  color="navy",      label="Post-monsoon (November)",
                 linewidth=1.5)
    axes[1].set_ylabel("Water level (m bgl)")
    axes[1].set_title(
        f"Deep wells (>={DEEP_DEPTH_MIN:.0f} m) — n = {len(deep)}"
    )
    axes[1].legend()
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Year")
    axes[1].grid(alpha=0.25)

    fig.suptitle(
        f"CGWB seasonal water-level climatology — IGP "
        f"({STUDY_START_YEAR}-{STUDY_END_YEAR})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------
    print(f"[04] Annual aggregates: {len(shallow_ts)} years")
    print(f"[04] Wrote: {OUT_SHALLOW.relative_to(OUT_SHALLOW.parents[2])}")
    print(f"[04] Wrote: {OUT_DEEP.relative_to(OUT_DEEP.parents[2])}")
    print(f"[04] Wrote: {OUT_FIG.relative_to(OUT_FIG.parents[2])}")


if __name__ == "__main__":
    main()
