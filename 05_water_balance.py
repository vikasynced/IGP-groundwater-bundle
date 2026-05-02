"""
05_water_balance.py
===================
Combine GRACE TWS and GLDAS soil-moisture / SWE anomalies into the
groundwater storage anomaly series via the standard water-balance approach.

PURPOSE
-------
The GRACE/GRACE-FO satellites measure total water storage (TWS) — the
vertically integrated mass of water in surface, soil, and aquifer
reservoirs. To isolate the groundwater component, we subtract the
soil-moisture and snow-water-equivalent anomalies from a land-surface
model (here GLDAS-2.1 Noah):

    GWS_anom = TWS_anom - SM_anom - SWE_anom

This is the "standard" approach used widely in GRACE groundwater studies
(Rodell et al. 2009; Tiwari et al. 2009). The output is the canonical
GWS series consumed by every downstream R-script.

The original v4 pipeline duplicated the GRACE NetCDF processing here;
that processing has been moved to 01_grace_process.py and this script
now reads only the intermediate CSVs.

INPUTS
------
  outputs/timeseries/GRACE_TWS_IGP.csv     from 01_grace_process.py
  outputs/timeseries/GLDAS_processed.csv   from 02_gldas_process.py

OUTPUT
------
  outputs/timeseries/water_balance_standard.csv
      Columns: time, TWS, SM_anom, SWE_anom, GWS_standard

  outputs/figures/05_water_balance.png
      Four-panel diagnostic.

USAGE
-----
  cd path/to/IGP_GW_Study/scripts
  python 05_water_balance.py    # requires 01 and 02 first

REFERENCES
----------
  Rodell, M., et al. (2009). Satellite-based estimates of groundwater
    depletion in India. Nature 460, 999-1002.
  Tiwari, V. M., et al. (2009). Dwindling groundwater resources in
    northern India. Geophys. Res. Lett. 36, L18401.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TS_DIR, FIG_DIR


GRACE_CSV = TS_DIR / "GRACE_TWS_IGP.csv"
GLDAS_CSV = TS_DIR / "GLDAS_processed.csv"
OUT_CSV   = TS_DIR / "water_balance_standard.csv"
OUT_FIG   = FIG_DIR / "05_water_balance.png"


def main():
    TS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for path in [GRACE_CSV, GLDAS_CSV]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required upstream output not found: {path}\n"
                f"Run the upstream scripts (01_grace_process.py, "
                f"02_gldas_process.py) before this one."
            )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------
    print(f"[05] Loading GRACE TWS and GLDAS components ...")
    grace = pd.read_csv(GRACE_CSV, parse_dates=["time"])
    gldas = pd.read_csv(GLDAS_CSV, parse_dates=["time"])

    # Align both series to month-start. GRACE timestamps are mid-month
    # (e.g. 2002-04-17, the centroid of the monthly mascon solution);
    # GLDAS is end-of-month-stamped. Normalising to month-start makes the
    # join unambiguous and the subsequent inner-join keeps only months
    # for which both datasets have an observation (this naturally drops
    # the GRACE-FO 2017-06 to 2018-05 gap from the merged series).
    grace["time"] = grace["time"].dt.to_period("M").dt.to_timestamp()
    gldas["time"] = gldas["time"].dt.to_period("M").dt.to_timestamp()

    # If GRACE has multiple solutions in the same calendar month (unusual
    # but possible at solution-version boundaries), average them.
    if grace["time"].duplicated().any():
        grace = grace.groupby("time", as_index=False).mean()

    merged = grace.merge(
        gldas[["time", "SM_anom", "SWE_anom"]],
        on="time",
        how="inner",
    ).sort_values("time").reset_index(drop=True)
    print(f"[05] Merged months: {len(merged)} "
          f"({merged['time'].iloc[0].strftime('%Y-%m')} to "
          f"{merged['time'].iloc[-1].strftime('%Y-%m')})")

    # -------------------------------------------------------------------------
    # Water-balance arithmetic
    # -------------------------------------------------------------------------
    merged["GWS_standard"] = (
        merged["TWS"] - merged["SM_anom"] - merged["SWE_anom"]
    )

    out = merged[["time", "TWS", "SM_anom", "SWE_anom", "GWS_standard"]]
    out.to_csv(OUT_CSV, index=False)

    # -------------------------------------------------------------------------
    # Diagnostic figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    axes[0].plot(out["time"], out["TWS"], color="steelblue", linewidth=1.3)
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_ylabel("TWS (mm)")
    axes[0].set_title("GRACE/GRACE-FO TWS anomaly")
    axes[0].grid(alpha=0.25)

    axes[1].plot(out["time"], out["SM_anom"], color="brown", linewidth=1.2)
    axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[1].set_ylabel("SM anomaly (mm)")
    axes[1].set_title("GLDAS soil-moisture anomaly (subtracted)")
    axes[1].grid(alpha=0.25)

    axes[2].plot(out["time"], out["SWE_anom"], color="cornflowerblue",
                 linewidth=1.2)
    axes[2].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[2].set_ylabel("SWE anomaly (mm)")
    axes[2].set_title("GLDAS snow-water-equivalent anomaly (subtracted)")
    axes[2].grid(alpha=0.25)

    # Linear trend overlay on GWS for visual sanity-check (the canonical
    # trend in the manuscript is computed by R9 with Theil-Sen on the
    # gap-filled series; this OLS slope is just for orientation).
    valid = out["GWS_standard"].notna()
    if valid.sum() > 0:
        x = np.arange(len(out))[valid.values]
        y = out.loc[valid, "GWS_standard"].values
        slope_per_month, intercept = np.polyfit(x, y, 1)
        slope_per_yr = slope_per_month * 12
        axes[3].plot(out["time"], out["GWS_standard"], color="darkred",
                     linewidth=1.3, label="GWS_standard")
        axes[3].plot(
            out["time"].iloc[valid.values],
            intercept + slope_per_month * x,
            color="black", linewidth=1.2, linestyle="-",
            label=f"OLS trend: {slope_per_yr:+.1f} mm/yr",
        )
        axes[3].legend()
    else:
        axes[3].plot(out["time"], out["GWS_standard"], color="darkred",
                     linewidth=1.3)
    axes[3].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[3].set_ylabel("GWS (mm)")
    axes[3].set_xlabel("Year")
    axes[3].set_title("Groundwater storage anomaly = TWS - SM_anom - SWE_anom")
    axes[3].grid(alpha=0.25)

    fig.suptitle(
        "Water-balance decomposition — Upper IGP",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------
    print(f"[05] Mean TWS:          {out['TWS'].mean():+.1f} mm")
    print(f"[05] Mean SM_anom:      {out['SM_anom'].mean():+.1f} mm")
    print(f"[05] Mean SWE_anom:     {out['SWE_anom'].mean():+.2f} mm")
    print(f"[05] Mean GWS_standard: {out['GWS_standard'].mean():+.1f} mm")
    print(f"[05] OLS GWS slope (orientation only): {slope_per_yr:+.2f} mm/yr")
    print(f"[05] Note: canonical Theil-Sen trend ({-34.86:+.2f} mm/yr) is")
    print(f"[05]       computed by R9 on the gap-filled series.")
    print(f"[05] Wrote: {OUT_CSV.relative_to(OUT_CSV.parents[2])}")
    print(f"[05] Wrote: {OUT_FIG.relative_to(OUT_FIG.parents[2])}")


if __name__ == "__main__":
    main()
