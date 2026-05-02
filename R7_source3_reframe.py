"""
R7_source3_reframe.py
=====================
Inter-model evapotranspiration uncertainty for the Monte Carlo error budget.

PURPOSE
-------
The v4 Monte Carlo treated GLDAS soil-moisture uncertainty via a fractional
multiplier on |SM_anom| (Source 3). The v5.1 reframe (manuscript §3.5)
replaces this with an inter-model uncertainty derived from the GLDAS-MODIS
ET difference: the per-month standard deviation of the ET difference is
used as a structural-uncertainty estimate for the GLDAS soil-moisture
budget closure.

This script:
  1. Loads the per-month ET bias series (MODIS - GLDAS) produced by
     03_modis_et_process.py.
  2. Decomposes the difference into mean + seasonal cycle + residual
     components.
  3. Computes the month-of-year standard deviation of the difference,
     which becomes Source 4 in R9's Monte Carlo.

The signed convention used here (GLDAS - MODIS) matches the v5.1
manuscript text. The bias CSV from 03_modis_et_process.py has the
opposite sign (MODIS - GLDAS, retained for historical column naming);
the conversion is a unary minus.

INPUT
-----
  outputs/timeseries/ET_bias_2002_2023.csv     from 03_modis_et_process.py

OUTPUTS
-------
  outputs/timeseries/R7_intermodel_sigma.csv
      Columns: month, sigma_mm_month
      Per-calendar-month standard deviation of (GLDAS - MODIS).
      Read by R9 as the Source 4 uncertainty.
  outputs/timeseries/R7_bias_decomposition.csv
      Per-month decomposition into mean, seasonal cycle, residual.
  outputs/figures/R7_ET_intercomparison.png
      Three-panel diagnostic for §5.2.
  outputs/timeseries/R7_summary.txt

USAGE
-----
  cd path/to/IGP_GW_Study/scripts
  python R7_source3_reframe.py    # requires 03 first
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


BIAS_CSV   = TS_DIR / "ET_bias_2002_2023.csv"
OUT_SIGMA  = TS_DIR / "R7_intermodel_sigma.csv"
OUT_DECOMP = TS_DIR / "R7_bias_decomposition.csv"
OUT_FIG    = FIG_DIR / "R7_ET_intercomparison.png"
OUT_SUMM   = TS_DIR / "R7_summary.txt"


def main():
    TS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not BIAS_CSV.exists():
        raise FileNotFoundError(
            f"[R7] {BIAS_CSV} not found. Run 03_modis_et_process.py first."
        )

    print(f"[R7] Loading ET_bias_2002_2023.csv ...")
    df = pd.read_csv(BIAS_CSV, parse_dates=["time"])

    # Convert to v5.1 convention: signed difference is GLDAS - MODIS
    # (so that POSITIVE values mean GLDAS overestimates relative to MODIS,
    # matching the manuscript text in §5.2).
    df["diff_GLDAS_minus_MODIS"] = df["GLDAS_ET"] - df["MODIS_ET"]

    # -------------------------------------------------------------------------
    # Decomposition: mean + seasonal cycle + residual
    # -------------------------------------------------------------------------
    diff = df["diff_GLDAS_minus_MODIS"]
    grand_mean = float(diff.mean())

    df["month"] = df["time"].dt.month
    seasonal = df.groupby("month")["diff_GLDAS_minus_MODIS"].mean()
    df["seasonal_cycle"] = df["month"].map(seasonal) - grand_mean
    df["residual"]       = diff - grand_mean - df["seasonal_cycle"]

    decomp = df[["time", "month", "GLDAS_ET", "MODIS_ET",
                 "diff_GLDAS_minus_MODIS", "seasonal_cycle", "residual"]]
    decomp.to_csv(OUT_DECOMP, index=False)

    # -------------------------------------------------------------------------
    # Per-month-of-year standard deviation (the Source 4 vector for R9)
    # -------------------------------------------------------------------------
    # Use ddof=1 sample standard deviation (matches the n=22 sample size
    # interpretation; would be ~3% smaller with ddof=0).
    sigma_per_month = (
        df.groupby("month")["diff_GLDAS_minus_MODIS"]
        .std(ddof=1)
        .reset_index()
        .rename(columns={"diff_GLDAS_minus_MODIS": "sigma_mm_month"})
    )
    sigma_per_month.to_csv(OUT_SIGMA, index=False)

    # Annualised structural sigma estimate: quadrature sum across months,
    # divided by 12 (treating each month as an independent contribution).
    sigma_annual_proxy = float(np.sqrt((sigma_per_month["sigma_mm_month"] ** 2).sum())) / 12

    # -------------------------------------------------------------------------
    # Diagnostic figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    # Panel A: GLDAS and MODIS overlaid
    axes[0].plot(df["time"], df["GLDAS_ET"], color="orange", linewidth=1.2,
                 linestyle="--", label="GLDAS-2.1 Noah")
    axes[0].plot(df["time"], df["MODIS_ET"], color="darkgreen", linewidth=1.2,
                 label="MODIS MOD16A2GF")
    axes[0].set_ylabel("ET (mm/month)")
    axes[0].set_title("Evapotranspiration: GLDAS vs MODIS — IGP domain")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # Panel B: signed difference (GLDAS - MODIS)
    bar_colors = ["steelblue" if v >= 0 else "salmon"
                  for v in df["diff_GLDAS_minus_MODIS"].values]
    axes[1].bar(df["time"], df["diff_GLDAS_minus_MODIS"].values,
                color=bar_colors, alpha=0.8, width=25)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axhline(grand_mean, color="red", linewidth=1.5, linestyle="--",
                    label=f"Grand mean: {grand_mean:+.1f} mm/month")
    axes[1].set_ylabel("GLDAS - MODIS (mm/month)")
    axes[1].set_title("Signed inter-model difference — blue = GLDAS higher, "
                      "red = MODIS higher")
    axes[1].legend()
    axes[1].grid(alpha=0.25, axis="y")

    # Panel C: per-month sigma (the Source 4 vector)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    axes[2].bar(np.arange(1, 13), sigma_per_month["sigma_mm_month"].values,
                color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.5)
    axes[2].set_xticks(np.arange(1, 13))
    axes[2].set_xticklabels(month_names)
    axes[2].set_ylabel("Std dev of GLDAS - MODIS (mm/month)")
    axes[2].set_title("Per-month standard deviation (used as Source 4 in R9 Monte Carlo)")
    axes[2].grid(alpha=0.25, axis="y")

    fig.suptitle(
        "Inter-model ET uncertainty — Source 4 of R9 Monte Carlo",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Human-readable summary
    # -------------------------------------------------------------------------
    lines = []
    lines.append("R7  INTER-MODEL ET UNCERTAINTY  -  SUMMARY")
    lines.append("=" * 78)
    lines.append(f"Months in series: {len(df)}")
    lines.append(f"Grand mean of (GLDAS - MODIS): {grand_mean:+.2f} mm/month")
    lines.append(f"Annualised structural sigma proxy: {sigma_annual_proxy:.2f} mm/yr")
    lines.append("")
    lines.append("PER-MONTH STANDARD DEVIATION (Source 4 vector for R9)")
    lines.append("-" * 78)
    lines.append(sigma_per_month.round(2).to_string(index=False))
    lines.append("")
    lines.append("=" * 78)
    summary = "\n".join(lines)
    with open(OUT_SUMM, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------
    print(f"[R7] Grand mean (GLDAS - MODIS): {grand_mean:+.2f} mm/month")
    print(f"[R7] Annualised sigma proxy:     {sigma_annual_proxy:.2f} mm/yr")
    print(f"[R7] Wrote: {OUT_SIGMA.relative_to(OUT_SIGMA.parents[2])}")
    print(f"[R7] Wrote: {OUT_DECOMP.relative_to(OUT_DECOMP.parents[2])}")
    print(f"[R7] Wrote: {OUT_FIG.relative_to(OUT_FIG.parents[2])}")
    print(f"[R7] Wrote: {OUT_SUMM.relative_to(OUT_SUMM.parents[2])}")


if __name__ == "__main__":
    main()
