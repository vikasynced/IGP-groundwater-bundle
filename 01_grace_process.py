"""
01_grace_process.py
===================
Process GRACE/GRACE-FO mascon NetCDF into an IGP-domain TWS anomaly series.

PURPOSE
-------
Reads the JPL Release-06 mascon solution (CRI-corrected variant), applies
the per-pixel scale factor, restricts to the Indo-Gangetic Plain bounding
box, computes anomalies relative to the 2004-2009 baseline, and reduces
to a domain-mean monthly time series via cosine-of-latitude weighting.

The output CSV is consumed by 05_water_balance.py and the downstream
R-scripts (R4, R6, R8, R9). The original v4 pipeline duplicated the GRACE
processing across two scripts (02 and 06); this consolidated version
runs the processing once and saves the intermediate.

INPUT
-----
  data/grace/GRCTellus.JPL.200204_202602.GLO.RL06.3M.MSCNv04CRI.nc

OUTPUT
------
  outputs/timeseries/GRACE_TWS_IGP.csv
      Columns: time, TWS (mm equivalent water height, anomaly relative to
      2004-2009 baseline)

  outputs/figures/01_GRACE_TWS_anomaly.png
      Diagnostic time-series plot.

USAGE
-----
  cd path/to/IGP_GW_Study/scripts
  python 01_grace_process.py

REFERENCES
----------
  Watkins et al. (2015) JGR Solid Earth 120:2648-2671 — mascon solution.
  Wiese et al. (2016) WRR 52:7490-7502 — CRI scale-factor formulation.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Make sibling scripts importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR, TS_DIR, FIG_DIR,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    BASELINE_START, BASELINE_END,
    STUDY_START_YEAR, STUDY_END_YEAR,
)
from pipeline_utils import cosine_weighted_mean


GRACE_NC = DATA_DIR / "grace" / "GRCTellus.JPL.200204_202602.GLO.RL06.3M.MSCNv04CRI.nc"
OUT_CSV  = TS_DIR / "GRACE_TWS_IGP.csv"
OUT_FIG  = FIG_DIR / "01_GRACE_TWS_anomaly.png"


def main():
    TS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not GRACE_NC.exists():
        raise FileNotFoundError(
            f"GRACE NetCDF not found at {GRACE_NC}.\n"
            f"Download GRCTellus.JPL.200204_202602.GLO.RL06.3M.MSCNv04CRI.nc "
            f"from PO.DAAC and place it in {GRACE_NC.parent}."
        )

    # -------------------------------------------------------------------------
    # Load and apply scale-factor correction
    # -------------------------------------------------------------------------
    print(f"[01] Loading {GRACE_NC.name} ...")
    ds = xr.open_dataset(GRACE_NC)

    # Per-pixel CRI scale factor; lwe_thickness in cm of equivalent water.
    # Multiplying by 10 converts cm -> mm. The scale factor itself is
    # dimensionless and corrects for spatial leakage.
    lwe_scaled = ds["lwe_thickness"] * ds["scale_factor"] * 10.0  # mm EWH

    # Restrict to IGP bounding box. JPL mascons use 0-360 longitude
    # convention, but IGP lies entirely in the eastern hemisphere so the
    # numeric values 73.5-80 work in either convention.
    lwe_igp = lwe_scaled.sel(
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX),
    )

    # Apply land mask to drop ocean/lake pixels (matters for ambiguous
    # coastal grid cells but is a no-op for the inland IGP domain;
    # included for defensiveness).
    land = ds["land_mask"].sel(
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX),
    )
    lwe_igp = lwe_igp.where(land == 1)

    # -------------------------------------------------------------------------
    # Anomaly relative to 2004-2009 baseline
    # -------------------------------------------------------------------------
    baseline = lwe_igp.sel(time=slice(BASELINE_START, BASELINE_END)).mean("time")
    tws_anom = lwe_igp - baseline

    # -------------------------------------------------------------------------
    # Domain mean with cosine-of-latitude weighting
    # -------------------------------------------------------------------------
    # xarray's weighted().mean() does this internally; we use it here rather
    # than the pipeline_utils function because xarray retains the time index.
    weights = np.cos(np.deg2rad(tws_anom.lat))
    tws_ts  = tws_anom.weighted(weights).mean(("lat", "lon"))

    # -------------------------------------------------------------------------
    # Build a clean DataFrame, restrict to study window, save
    # -------------------------------------------------------------------------
    df = pd.DataFrame({
        "time": pd.to_datetime(tws_ts.time.values),
        "TWS":  tws_ts.values,
    })
    df = df[
        (df["time"].dt.year >= STUDY_START_YEAR) &
        (df["time"].dt.year <= STUDY_END_YEAR)
    ].reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)

    # -------------------------------------------------------------------------
    # Diagnostic figure
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["time"], df["TWS"], color="steelblue", linewidth=1.4)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("TWS anomaly (mm EWH)")
    ax.set_xlabel("Year")
    ax.set_title(
        f"GRACE/GRACE-FO TWS anomaly — Upper IGP "
        f"({STUDY_START_YEAR}-{STUDY_END_YEAR})\n"
        f"Baseline: {BASELINE_START} to {BASELINE_END}, JPL RL06.3 mascon"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------
    print(f"[01] Months in window: {len(df)} "
          f"({df['time'].iloc[0].strftime('%Y-%m')} to "
          f"{df['time'].iloc[-1].strftime('%Y-%m')})")
    print(f"[01] TWS range: {df['TWS'].min():+.1f} to {df['TWS'].max():+.1f} mm")
    print(f"[01] Wrote: {OUT_CSV.relative_to(OUT_CSV.parents[2])}")
    print(f"[01] Wrote: {OUT_FIG.relative_to(OUT_FIG.parents[2])}")


if __name__ == "__main__":
    main()
