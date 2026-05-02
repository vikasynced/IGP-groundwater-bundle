"""
06_ssa_trend.py
===============
Singular Spectrum Analysis decomposition of the GWS_standard series.

PURPOSE
-------
Decomposes the groundwater-storage anomaly series into trend, annual-cycle,
and residual components via Singular Spectrum Analysis with window length
L = 60 (the value chosen in §4.2 of the manuscript and validated by R6).
The output is consumed by R4 (change-point analysis on the SSA trend) and
by F_make_v5_figures.py (decomposition figure for the manuscript).

The trend slope reported here is computed on the OBSERVED (no gap-fill)
trend component, using the trend-free pre-whitened Mann-Kendall test
(Yue-Wang modification). The headline manuscript trend (-34.86 mm/yr) is
computed downstream by R9 on the gap-filled series with Theil-Sen
regression; this script's purpose is the decomposition itself, not the
canonical trend.

INPUT
-----
  outputs/timeseries/water_balance_standard.csv   from 05_water_balance.py

OUTPUT
------
  outputs/timeseries/SSA_components.csv
      Columns: time, GWS_original, SSA_trend, SSA_annual, SSA_residual

  outputs/figures/06_SSA_decomposition.png
      Four-panel diagnostic.

USAGE
-----
  cd path/to/IGP_GW_Study/scripts
  python 06_ssa_trend.py    # requires 05 first

NOTES
-----
The trend grouping (RC1+RC2 -> trend, RC3-RC6 -> annual cycle) is the
standard SSA convention for series with a single dominant secular trend
plus annual seasonality (Vautard et al. 1992; Golyandina & Zhigljavsky
2013, ch. 2). R6 verifies this grouping quantitatively via w-correlation
analysis at multiple L values; the verification is reported in §4.2 and
Fig. S1 of the manuscript.

REFERENCES
----------
  Vautard, R., Yiou, P., & Ghil, M. (1992). Singular-spectrum analysis:
    A toolkit for short, noisy chaotic signals. Physica D 58, 95-126.
  Yue, S., & Wang, C. Y. (2002). Applicability of pre-whitening to
    eliminate the influence of serial correlation on the Mann-Kendall
    test. Water Resour. Res. 38, 1068.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TS_DIR, FIG_DIR, SSA_WINDOW_LENGTH
from pipeline_utils import ssa_decompose


WB_CSV  = TS_DIR / "water_balance_standard.csv"
OUT_CSV = TS_DIR / "SSA_components.csv"
OUT_FIG = FIG_DIR / "06_SSA_decomposition.png"


def main():
    TS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not WB_CSV.exists():
        raise FileNotFoundError(
            f"water_balance_standard.csv not found at {WB_CSV}. "
            f"Run 05_water_balance.py first."
        )

    # -------------------------------------------------------------------------
    # Load and drop NaN rows (the gap)
    # -------------------------------------------------------------------------
    print(f"[06] Loading water_balance_standard.csv ...")
    df = pd.read_csv(WB_CSV, parse_dates=["time"])
    valid = df["GWS_standard"].notna()
    times = df.loc[valid, "time"].reset_index(drop=True)
    gws   = df.loc[valid, "GWS_standard"].values
    print(f"[06] Series length after gap removal: {len(gws)} months")

    # -------------------------------------------------------------------------
    # SSA decomposition
    # -------------------------------------------------------------------------
    L = SSA_WINDOW_LENGTH
    print(f"[06] Running SSA with L = {L} ...")
    rcs, singular_values = ssa_decompose(gws, L=L, n_components=12)

    # Standard grouping for a series with secular trend + annual cycle.
    # R6 verifies separability quantitatively for L = 36, 60, 84, 120.
    trend  = rcs[0] + rcs[1]
    annual = rcs[2] + rcs[3] + rcs[4] + rcs[5]
    residual = gws - trend - annual

    # Variance shares (eigenvalue percentage of total spectrum)
    total_eig = float(np.sum(singular_values ** 2))
    trend_eig_pct  = 100 * (singular_values[0] ** 2 + singular_values[1] ** 2) / total_eig
    annual_eig_pct = 100 * sum(singular_values[i] ** 2 for i in range(2, 6)) / total_eig

    # -------------------------------------------------------------------------
    # Mann-Kendall trend tests (orientation only)
    # -------------------------------------------------------------------------
    mk_full  = mk.original_test(gws)
    mk_trend = mk.yue_wang_modification_test(trend)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    out = pd.DataFrame({
        "time":         times,
        "GWS_original": gws,
        "SSA_trend":    trend,
        "SSA_annual":   annual,
        "SSA_residual": residual,
    })
    out.to_csv(OUT_CSV, index=False)

    # -------------------------------------------------------------------------
    # Diagnostic figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    axes[0].plot(times, gws, color="darkred", linewidth=1.2)
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_ylabel("mm EWH")
    axes[0].set_title("Original GWS (water-balance method)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(times, trend, color="navy", linewidth=2.0)
    axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[1].set_ylabel("mm EWH")
    axes[1].set_title(
        f"SSA trend (RC1+RC2) — TFPW-MK slope: {mk_trend.slope * 12:+.1f} mm/yr, "
        f"p = {mk_trend.p:.4f}, eig share: {trend_eig_pct:.1f}%"
    )
    axes[1].grid(alpha=0.25)

    axes[2].plot(times, annual, color="seagreen", linewidth=1.2)
    axes[2].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[2].set_ylabel("mm EWH")
    axes[2].set_title(
        f"SSA annual cycle (RC3-RC6) — eig share: {annual_eig_pct:.1f}%"
    )
    axes[2].grid(alpha=0.25)

    axes[3].plot(times, residual, color="gray", linewidth=1.0)
    axes[3].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[3].set_ylabel("mm EWH")
    axes[3].set_xlabel("Year")
    axes[3].set_title(f"SSA residual — std: {residual.std(ddof=1):.1f} mm")
    axes[3].grid(alpha=0.25)

    fig.suptitle(
        f"SSA decomposition of GWS — Upper IGP (L = {L})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------
    print(f"[06] SSA window L = {L}")
    print(f"[06] Trend variance share (eigenvalue): {trend_eig_pct:.1f}%")
    print(f"[06] Annual cycle share (eigenvalue):   {annual_eig_pct:.1f}%")
    print(f"[06] TFPW-MK on SSA trend: {mk_trend.slope * 12:+.2f} mm/yr, "
          f"p = {mk_trend.p:.4f}")
    print(f"[06] Note: canonical headline trend ({-34.86:+.2f} mm/yr) is in R9.")
    print(f"[06] Wrote: {OUT_CSV.relative_to(OUT_CSV.parents[2])}")
    print(f"[06] Wrote: {OUT_FIG.relative_to(OUT_FIG.parents[2])}")


if __name__ == "__main__":
    main()
