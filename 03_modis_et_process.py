"""
03_modis_et_process.py
======================
Aggregate MODIS MOD16A2GF 8-day ET to monthly totals and compute the
MODIS-vs-GLDAS difference series.

PURPOSE
-------
The MOD16A2GF gap-filled 8-day ET product is delivered as a Statistics CSV
from the AppEEARS interface. This script accumulates the 8-day means into
monthly totals, aligns them with the GLDAS ET series produced by
02_gldas_process.py, and computes the per-month difference

    diff_t = MODIS_t - GLDAS_t

which is consumed by R7 to derive the inter-model ET uncertainty
(Source 4 in the Monte Carlo).

Note on sign convention: in the v5.1 manuscript and in R7, the signed
difference is reported as GLDAS - MODIS (because GLDAS is the model
component being characterised). For the bias CSV produced by this
upstream script we retain MODIS - GLDAS to match the historical column
naming. R7 reverses the sign internally; both conventions yield identical
absolute uncertainty estimates.

DATA ACQUISITION (manual)
-------------------------
The MODIS ET extract was obtained from NASA AppEEARS:

  Product: MOD16A2GF.061 (Terra Net Evapotranspiration Gap-Filled, 8-day)
  Layer:   ET_500m
  Region:  IGP bounding box (24-32.5 N, 73.5-80 E)
  Period:  2002-01-01 to 2023-12-31
  Output:  Area Statistics CSV (MOD16A2GF-061-Statistics.csv)

Save the AppEEARS Statistics CSV into data/modis_et/.

INPUTS
------
  data/modis_et/MOD16A2GF-061-Statistics.csv     AppEEARS export (8-day)
  outputs/timeseries/GLDAS_processed.csv         from 02_gldas_process.py

OUTPUTS
-------
  outputs/timeseries/MODIS_ET_monthly_2002_2023.csv
      Columns: date, ET_mm_month
  outputs/timeseries/ET_bias_2002_2023.csv
      Columns: time, MODIS_ET, GLDAS_ET, ET_bias  (ET_bias = MODIS - GLDAS)
  outputs/figures/03_MODIS_vs_GLDAS_ET.png
      Three-panel diagnostic.

USAGE
-----
  cd path/to/IGP_GW_Study/scripts
  python 03_modis_et_process.py     # requires 02_gldas_process.py first
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, TS_DIR, FIG_DIR, STUDY_START_YEAR, STUDY_END_YEAR


MODIS_CSV = DATA_DIR / "modis_et" / "MOD16A2GF-061-Statistics.csv"
GLDAS_CSV = TS_DIR / "GLDAS_processed.csv"

OUT_MODIS = TS_DIR / "MODIS_ET_monthly_2002_2023.csv"
OUT_BIAS  = TS_DIR / "ET_bias_2002_2023.csv"
OUT_FIG   = FIG_DIR / "03_MODIS_vs_GLDAS_ET.png"


def load_modis_appeears(path):
    """
    Load the AppEEARS Statistics CSV and aggregate 8-day to monthly totals.

    AppEEARS reports the spatial mean of each 8-day composite. Summing the
    eight-day-period values within each calendar month gives a sensible
    monthly total because each 8-day composite represents an integrated
    flux over its 8-day window. Months where the 8-day windows straddle
    the calendar boundary are handled by attributing each window to the
    month containing its start date (AppEEARS convention).
    """
    df = pd.read_csv(path)
    et = df[df["Dataset"] == "ET_500m"].copy()
    et["Date"] = pd.to_datetime(et["Date"])
    et = et[
        (et["Date"] >= f"{STUDY_START_YEAR}-01-01") &
        (et["Date"] <= f"{STUDY_END_YEAR}-12-31")
    ]

    # AppEEARS Mean is the area-mean ET over the bbox in mm/8day.
    # Group 8-day composites by their start-month and sum.
    et["year_month"] = et["Date"].dt.to_period("M")
    monthly = et.groupby("year_month")["Mean"].sum().reset_index()
    monthly.columns = ["year_month", "ET_mm_month"]
    monthly["date"] = monthly["year_month"].dt.to_timestamp()
    return monthly[["date", "ET_mm_month"]].set_index("date")


def main():
    TS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not MODIS_CSV.exists():
        raise FileNotFoundError(
            f"MODIS AppEEARS CSV not found at {MODIS_CSV}.\n"
            f"See module docstring for AppEEARS acquisition parameters."
        )
    if not GLDAS_CSV.exists():
        raise FileNotFoundError(
            f"GLDAS processed CSV not found at {GLDAS_CSV}. "
            f"Run 02_gldas_process.py first."
        )

    # -------------------------------------------------------------------------
    # MODIS: 8-day -> monthly
    # -------------------------------------------------------------------------
    print(f"[03] Loading MODIS AppEEARS export ...")
    modis = load_modis_appeears(MODIS_CSV)
    modis.index = modis.index.to_period("M").to_timestamp()
    modis.to_csv(OUT_MODIS)

    # -------------------------------------------------------------------------
    # GLDAS ET (already in mm/month from 02_gldas_process.py)
    # -------------------------------------------------------------------------
    print(f"[03] Loading GLDAS ET from upstream output ...")
    gldas = pd.read_csv(GLDAS_CSV, parse_dates=["time"]).set_index("time")
    gldas.index = gldas.index.to_period("M").to_timestamp()
    gldas_et = gldas["ET_mm_month"]

    # -------------------------------------------------------------------------
    # Align and compute bias
    # -------------------------------------------------------------------------
    common = modis.index.intersection(gldas_et.index)
    modis_c = modis.loc[common, "ET_mm_month"]
    gldas_c = gldas_et.loc[common]
    bias    = modis_c - gldas_c

    bias_df = pd.DataFrame({
        "MODIS_ET": modis_c,
        "GLDAS_ET": gldas_c,
        "ET_bias":  bias,
    })
    bias_df.index.name = "time"
    bias_df.to_csv(OUT_BIAS)

    # -------------------------------------------------------------------------
    # Diagnostic figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Panel A: MODIS and GLDAS overlaid
    axes[0].plot(common, modis_c.values, color="darkgreen", linewidth=1.2,
                 label="MODIS MOD16A2GF", alpha=0.9)
    axes[0].plot(common, gldas_c.values, color="orange", linewidth=1.2,
                 linestyle="--", label="GLDAS-2.1 Noah", alpha=0.9)
    axes[0].set_ylabel("ET (mm/month)")
    axes[0].set_title("MODIS vs GLDAS evapotranspiration — IGP domain")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # Panel B: bias time series, bar coloured by sign
    bar_colors = ["steelblue" if v >= 0 else "salmon" for v in bias.values]
    axes[1].bar(common, bias.values, color=bar_colors, alpha=0.8, width=25)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axhline(bias.mean(), color="red", linewidth=1.5, linestyle="--",
                    label=f"Mean: {bias.mean():+.1f} mm/month")
    axes[1].set_ylabel("MODIS - GLDAS (mm/month)")
    axes[1].set_title("ET bias — blue = MODIS higher, red = GLDAS higher")
    axes[1].legend()
    axes[1].grid(alpha=0.25, axis="y")

    # Panel C: monthly climatology of the bias
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    bias_by_month = [bias[bias.index.month == m].values for m in range(1, 13)]
    bp = axes[2].boxplot(
        bias_by_month, labels=month_names,
        patch_artist=True,
        medianprops={"color": "red", "linewidth": 2},
        flierprops={"marker": ".", "markersize": 4},
    )
    rabi_months = {10, 11, 12, 1, 2, 3}
    for i, patch in enumerate(bp["boxes"]):
        m = i + 1
        patch.set_facecolor("steelblue" if m in rabi_months else "salmon")
        patch.set_alpha(0.7)
    axes[2].axhline(0, color="black", linewidth=1.5)
    axes[2].set_ylabel("ET bias (mm/month)")
    axes[2].set_title("Seasonal climatology — blue = Rabi (Oct-Mar), red = Kharif (Jun-Sep)")
    axes[2].set_xlabel("Month")
    axes[2].grid(alpha=0.25, axis="y")

    fig.suptitle(
        "MODIS MOD16A2GF vs GLDAS-2.1 Noah — IGP domain (2002-2023)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------
    print(f"[03] MODIS months: {len(modis)} "
          f"({modis.index.min().strftime('%Y-%m')} to "
          f"{modis.index.max().strftime('%Y-%m')})")
    print(f"[03] Common months (MODIS & GLDAS): {len(common)}")
    print(f"[03] Mean MODIS ET: {modis_c.mean():.1f} mm/month")
    print(f"[03] Mean GLDAS ET: {gldas_c.mean():.1f} mm/month")
    print(f"[03] Mean bias (MODIS - GLDAS): {bias.mean():+.1f} mm/month")
    print(f"[03] Wrote: {OUT_MODIS.relative_to(OUT_MODIS.parents[2])}")
    print(f"[03] Wrote: {OUT_BIAS.relative_to(OUT_BIAS.parents[2])}")
    print(f"[03] Wrote: {OUT_FIG.relative_to(OUT_FIG.parents[2])}")


if __name__ == "__main__":
    main()
