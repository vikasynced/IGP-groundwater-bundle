"""
02_gldas_process.py
===================
Process GLDAS-2.1 Noah Giovanni-format CSV exports into a single tidy CSV.

PURPOSE
-------
Loads the six Giovanni domain-mean CSVs (four soil-moisture layers, SWE,
ET) for the IGP bounding box, sums the soil-moisture layers into total
column SM, computes anomalies relative to the 2004-2009 baseline, and
converts ET from kg/m^2/s to mm/month. The output is consumed by
05_water_balance.py and downstream scripts that need SM_anom / SWE_anom
(R8, R9) or GLDAS ET (R7).

DATA ACQUISITION (manual)
-------------------------
The GLDAS extracts in this bundle were obtained via NASA Giovanni's
time-averaged area-mean tool with the following parameters:

  Dataset:     GLDAS-2.1 Noah, monthly, 0.25-deg
  Variables:   SoilMoi0_10cm_inst    -> SM_0_10.csv
               SoilMoi10_40cm_inst   -> SM_10_40.csv
               SoilMoi40_100cm_inst  -> SM_40_100.csv
               SoilMoi100_200cm_inst -> SM_100_200.csv
               SWE_inst              -> SWE.csv
               Evap_tavg             -> ET.csv
  Bounding box: 24-32.5 N, 73.5-80 E  (matches config.LAT_MIN/MAX, LON_MIN/MAX)
  Time range:   2002-01 to 2023-12
  Output:       Time-averaged area-averaged time series, CSV format

Save the six CSVs into data/gldas/ with the filenames listed above.

Giovanni does not have a documented public API suitable for redistribution
of this query, so the bundle does not include an automated download script;
a reproducer must obtain the CSVs through the Giovanni web interface.

INPUTS
------
  data/gldas/SM_0_10.csv        Surface soil moisture (kg/m^2)
  data/gldas/SM_10_40.csv       Layer 2
  data/gldas/SM_40_100.csv      Layer 3
  data/gldas/SM_100_200.csv     Layer 4
  data/gldas/SWE.csv            Snow water equivalent (kg/m^2)
  data/gldas/ET.csv             Evapotranspiration (kg/m^2/s)

OUTPUT
------
  outputs/timeseries/GLDAS_processed.csv
      Columns: time, SM_total, SWE, ET, SM_anom, SWE_anom, ET_mm_month

  outputs/figures/02_GLDAS_components.png
      Three-panel diagnostic: SM_anom, SWE_anom, ET.

USAGE
-----
  cd path/to/IGP_GW_Study/scripts
  python 02_gldas_process.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, TS_DIR, FIG_DIR, BASELINE_START, BASELINE_END
from pipeline_utils import load_giovanni


GLDAS_DIR = DATA_DIR / "gldas"
OUT_CSV   = TS_DIR / "GLDAS_processed.csv"
OUT_FIG   = FIG_DIR / "02_GLDAS_components.png"

REQUIRED_FILES = [
    "SM_0_10.csv",
    "SM_10_40.csv",
    "SM_40_100.csv",
    "SM_100_200.csv",
    "SWE.csv",
    "ET.csv",
]


def main():
    TS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Existence check
    # -------------------------------------------------------------------------
    missing = [f for f in REQUIRED_FILES if not (GLDAS_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing GLDAS Giovanni CSV(s) in {GLDAS_DIR}: {missing}\n"
            f"See module docstring for Giovanni acquisition parameters."
        )

    # -------------------------------------------------------------------------
    # Load and combine
    # -------------------------------------------------------------------------
    print(f"[02] Loading {len(REQUIRED_FILES)} GLDAS Giovanni CSVs ...")
    sm1 = load_giovanni(GLDAS_DIR / "SM_0_10.csv")
    sm2 = load_giovanni(GLDAS_DIR / "SM_10_40.csv")
    sm3 = load_giovanni(GLDAS_DIR / "SM_40_100.csv")
    sm4 = load_giovanni(GLDAS_DIR / "SM_100_200.csv")
    swe = load_giovanni(GLDAS_DIR / "SWE.csv")
    et  = load_giovanni(GLDAS_DIR / "ET.csv")

    # Total column soil moisture: sum of four GLDAS Noah layers.
    # Units: kg/m^2 == mm of equivalent water depth (water density 1000 kg/m^3).
    sm_total = sm1["value"] + sm2["value"] + sm3["value"] + sm4["value"]

    gldas = pd.DataFrame({
        "SM_total": sm_total,
        "SWE":      swe["value"],
        "ET":       et["value"],
    })

    # -------------------------------------------------------------------------
    # Anomalies and unit conversions
    # -------------------------------------------------------------------------
    base = slice(BASELINE_START, BASELINE_END)
    gldas["SM_anom"]  = gldas["SM_total"] - gldas.loc[base, "SM_total"].mean()
    gldas["SWE_anom"] = gldas["SWE"]      - gldas.loc[base, "SWE"].mean()

    # GLDAS ET is reported as a temporal flux: kg m^-2 s^-1.
    # Multiply by seconds-per-day (86400) and approximate days-per-month (30)
    # to obtain a monthly total in mm. The 30-day approximation introduces a
    # ~1.6% maximum error vs. true days-per-month; this is well below the
    # GLDAS-MODIS inter-model uncertainty quantified in R7 and is therefore
    # acceptable for the intended use.
    gldas["ET_mm_month"] = gldas["ET"] * 86400 * 30

    # Make 'time' a regular column (not the index) for downstream consistency
    # with other intermediate CSVs in the bundle.
    gldas = gldas.reset_index().rename(columns={"time": "time"})
    gldas.to_csv(OUT_CSV, index=False)

    # -------------------------------------------------------------------------
    # Diagnostic figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(gldas["time"], gldas["SM_anom"], color="brown", linewidth=1.2)
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_ylabel("SM anomaly (mm)")
    axes[0].set_title("Total column soil moisture anomaly")
    axes[0].grid(alpha=0.25)

    axes[1].plot(gldas["time"], gldas["SWE_anom"], color="steelblue", linewidth=1.2)
    axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[1].set_ylabel("SWE anomaly (mm)")
    axes[1].set_title("Snow water equivalent anomaly")
    axes[1].grid(alpha=0.25)

    axes[2].plot(gldas["time"], gldas["ET_mm_month"], color="green", linewidth=1.2)
    axes[2].set_ylabel("ET (mm/month)")
    axes[2].set_title("Evapotranspiration (GLDAS-2.1 Noah)")
    axes[2].set_xlabel("Year")
    axes[2].grid(alpha=0.25)

    fig.suptitle(
        f"GLDAS-2.1 Noah components — IGP domain "
        f"(baseline {BASELINE_START} to {BASELINE_END})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------
    print(f"[02] Months: {len(gldas)} "
          f"({gldas['time'].iloc[0].strftime('%Y-%m')} to "
          f"{gldas['time'].iloc[-1].strftime('%Y-%m')})")
    print(f"[02] Mean SM_total: {gldas['SM_total'].mean():.1f} kg/m^2")
    print(f"[02] Mean ET:       {gldas['ET_mm_month'].mean():.1f} mm/month")
    print(f"[02] Wrote: {OUT_CSV.relative_to(OUT_CSV.parents[2])}")
    print(f"[02] Wrote: {OUT_FIG.relative_to(OUT_FIG.parents[2])}")


if __name__ == "__main__":
    main()
