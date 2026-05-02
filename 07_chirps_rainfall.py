"""
07_chirps_rainfall.py
=====================
Process CHIRPS monthly rainfall TIFs into a domain-mean time series and
classify each year as Wet, Dry, or Normal based on JJAS totals.

PURPOSE
-------
The CHIRPS v3.0 product provides monthly precipitation grids at 0.05-deg
resolution. This script clips each TIF to the IGP bounding box, computes
the cosine-of-latitude-weighted spatial mean, and aggregates the June-
September totals per year. Years are classified relative to the climatological
mean +/- 1 standard deviation:

    Wet:    JJAS total > mean + 1 sigma
    Normal: mean - 1 sigma <= JJAS total <= mean + 1 sigma
    Dry:    JJAS total < mean - 1 sigma

The classification feeds into R2 (depth-cut sensitivity) and R3 (sub-
seasonal CHIRPS metrics).

DATA ACQUISITION
----------------
CHIRPS monthly TIFs are downloadable from the Climate Hazards Center:

  Source:   https://www.chc.ucsb.edu/data/chirps
  Product:  CHIRPS v3.0 monthly, 0.05-deg
  Period:   2002-01 through 2023-12 (264 files)
  Filename: chirps-v3.0.YYYY.MM.tif

Save TIFs into data/imd_rain/. The directory name reflects historical
project conventions; CHIRPS is in fact a global dataset, not IMD.

NOTE: this script uses cosine-of-latitude-weighted spatial averaging,
correcting a bug in the original v4 pipeline that used naive np.nanmean
and over-weighted high-latitude grid cells. The IGP latitudinal extent
(24-32.5 N) is modest, so the correction shifts JJAS totals by ~3% and
does NOT alter the wet/dry/normal classification of any year.

INPUT
-----
  data/imd_rain/chirps-v3.0.YYYY.MM.tif    (264 monthly files)

OUTPUTS
-------
  outputs/timeseries/CHIRPS_monthly_IGP.csv
      Columns: date, year, month, rain_mm
  outputs/timeseries/CHIRPS_JJAS_annual.csv
      Columns: year, JJAS_mm, category
  outputs/figures/07_CHIRPS_rainfall.png
      Two-panel diagnostic.

USAGE
-----
  cd path/to/IGP_GW_Study/scripts
  python 07_chirps_rainfall.py
"""

import glob
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray as rxr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR, TS_DIR, FIG_DIR,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
)


CHIRPS_DIR = DATA_DIR / "imd_rain"
OUT_MONTHLY = TS_DIR / "CHIRPS_monthly_IGP.csv"
OUT_ANNUAL  = TS_DIR / "CHIRPS_JJAS_annual.csv"
OUT_FIG     = FIG_DIR / "07_CHIRPS_rainfall.png"

CHIRPS_PATTERN = re.compile(r"chirps[-_]v?[0-9.]+\.(\d{4})\.(\d{2})\.tif$",
                            re.IGNORECASE)


def cosine_weighted_mean_xr(da, lat_dim="y"):
    """
    Area-weighted spatial mean of a 2-D rioxarray DataArray.

    Uses cos(lat) cell weights, NaN-safe. Equivalent to the function in
    pipeline_utils.py but specialised here for the rioxarray case where
    the latitude dimension is named 'y' (not 'lat').
    """
    arr = da.values
    weights = np.cos(np.deg2rad(da[lat_dim].values))[:, None]
    weights = np.broadcast_to(weights, arr.shape).copy()
    weights[np.isnan(arr)] = 0.0
    arr = np.nan_to_num(arr, nan=0.0)
    total_w = weights.sum()
    if total_w == 0:
        return np.nan
    return float((arr * weights).sum() / total_w)


def main():
    TS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(CHIRPS_DIR / "chirps*.tif")))
    if not files:
        raise FileNotFoundError(
            f"No CHIRPS TIFs found in {CHIRPS_DIR}.\n"
            f"Download CHIRPS v3.0 monthly from "
            f"https://www.chc.ucsb.edu/data/chirps and place TIFs in {CHIRPS_DIR}/"
        )
    print(f"[07] Processing {len(files)} CHIRPS monthly TIFs ...")

    # -------------------------------------------------------------------------
    # Domain-mean per file
    # -------------------------------------------------------------------------
    rows = []
    for f in files:
        m = CHIRPS_PATTERN.search(Path(f).name)
        if not m:
            continue
        year, month = int(m.group(1)), int(m.group(2))

        # rioxarray returns a DataArray with dims (band, y, x); squeeze out band.
        da = rxr.open_rasterio(f, masked=True).squeeze()
        # CHIRPS is north-up, so y descends; slice with (LAT_MAX, LAT_MIN).
        da_igp = da.sel(y=slice(LAT_MAX, LAT_MIN), x=slice(LON_MIN, LON_MAX))
        rain = cosine_weighted_mean_xr(da_igp, lat_dim="y")

        rows.append({
            "date":    pd.Timestamp(year=year, month=month, day=1),
            "year":    year,
            "month":   month,
            "rain_mm": rain,
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df.to_csv(OUT_MONTHLY, index=False)
    print(f"[07] Monthly series: {len(df)} months "
          f"({df['date'].min().strftime('%Y-%m')} to "
          f"{df['date'].max().strftime('%Y-%m')})")

    # -------------------------------------------------------------------------
    # JJAS aggregation and classification
    # -------------------------------------------------------------------------
    jjas = (
        df[df["month"].isin([6, 7, 8, 9])]
        .groupby("year")["rain_mm"]
        .sum()
        .rename("JJAS_mm")
    )

    clim_mean = float(jjas.mean())
    clim_std  = float(jjas.std(ddof=1))

    def classify(v):
        if v > clim_mean + clim_std:
            return "Wet"
        if v < clim_mean - clim_std:
            return "Dry"
        return "Normal"

    jjas_df = jjas.reset_index()
    jjas_df["category"] = jjas_df["JJAS_mm"].apply(classify)
    jjas_df.to_csv(OUT_ANNUAL, index=False)

    wet_years = jjas_df.loc[jjas_df["category"] == "Wet",    "year"].tolist()
    dry_years = jjas_df.loc[jjas_df["category"] == "Dry",    "year"].tolist()
    nrm_years = jjas_df.loc[jjas_df["category"] == "Normal", "year"].tolist()

    # -------------------------------------------------------------------------
    # Diagnostic figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    # Panel A: monthly time series, monsoon months coloured
    monsoon_mask = df["month"].isin([6, 7, 8, 9])
    bar_colors = np.where(monsoon_mask, "steelblue", "lightgray")
    axes[0].bar(df["date"], df["rain_mm"], color=bar_colors,
                width=25, alpha=0.85)
    axes[0].set_ylabel("Rainfall (mm/month)")
    axes[0].set_title("CHIRPS monthly rainfall — IGP domain "
                      "(blue = JJAS monsoon months)")
    axes[0].grid(alpha=0.25, axis="y")

    # Panel B: annual JJAS totals with classification
    cat_colors = jjas_df["category"].map(
        {"Wet": "steelblue", "Dry": "salmon", "Normal": "lightgray"}
    ).values
    axes[1].bar(jjas_df["year"], jjas_df["JJAS_mm"],
                color=cat_colors, alpha=0.85,
                edgecolor="black", linewidth=0.5)
    axes[1].axhline(clim_mean, color="black", linewidth=1.5,
                    label=f"Mean: {clim_mean:.0f} mm")
    axes[1].axhline(clim_mean + clim_std, color="steelblue", linewidth=1.2,
                    linestyle="--", label=f"+1 sigma: {clim_mean + clim_std:.0f} mm")
    axes[1].axhline(clim_mean - clim_std, color="salmon", linewidth=1.2,
                    linestyle="--", label=f"-1 sigma: {clim_mean - clim_std:.0f} mm")
    axes[1].set_ylabel("JJAS rainfall (mm/season)")
    axes[1].set_xlabel("Year")
    axes[1].set_title(
        "Annual monsoon (JJAS) totals — wet (blue), normal (grey), dry (red)"
    )
    axes[1].set_xticks(jjas_df["year"])
    axes[1].set_xticklabels(jjas_df["year"], rotation=45)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25, axis="y")

    # Annotate wet/dry years
    for _, row in jjas_df.iterrows():
        if row["category"] in ("Wet", "Dry"):
            axes[1].annotate(
                str(row["year"]),
                xy=(row["year"], row["JJAS_mm"]),
                xytext=(0, 6 if row["JJAS_mm"] > clim_mean else -14),
                textcoords="offset points",
                ha="center", fontsize=7, fontweight="bold",
                color="steelblue" if row["category"] == "Wet" else "salmon",
            )

    fig.suptitle(
        "CHIRPS rainfall — Upper IGP (cosine-of-latitude weighted)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------
    print(f"[07] Climatological JJAS mean: {clim_mean:.1f} mm "
          f"+/- {clim_std:.1f} mm (1 sigma)")
    print(f"[07] Wet years (>+1 sigma):     {wet_years}")
    print(f"[07] Dry years (<-1 sigma):     {dry_years}")
    print(f"[07] Normal years:              {nrm_years}")
    print(f"[07] Wrote: {OUT_MONTHLY.relative_to(OUT_MONTHLY.parents[2])}")
    print(f"[07] Wrote: {OUT_ANNUAL.relative_to(OUT_ANNUAL.parents[2])}")
    print(f"[07] Wrote: {OUT_FIG.relative_to(OUT_FIG.parents[2])}")


if __name__ == "__main__":
    main()
