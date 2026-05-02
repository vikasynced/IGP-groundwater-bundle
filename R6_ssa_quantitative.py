"""
R6_ssa_quantitative.py
======================
Quantitative SSA diagnostics: L-sensitivity, eigenvalue percentages,
weighted correlation matrix, separability metric.

PURPOSE
-------
The choice of SSA window length L is the key methodological decision in
the trend extraction (manuscript §4.2). This script provides the
quantitative justification for choosing L = 60:

  1. Decomposes GWS at four candidate window lengths: L = 36, 60, 84, 120.
  2. For each L, reports the eigenvalue percentage of the leading
     components and the variance explained by each reconstructed component
     (the latter accounts for the diagonal-averaging step's effect on
     component variances).
  3. Computes the weighted correlation matrix between RCs (Vautard et al.
     1992 Section 3) at each L. Off-diagonal w-correlations near zero
     indicate good component separability.
  4. Auto-detects the trend group (RC1+RC2) and the leading annual pair
     (typically RC3,RC4 with periods near 12 months) at each L, and
     reports their period and amplitude.
  5. Computes a separability metric: the maximum off-diagonal
     w-correlation among the first 6 RCs. Lower is better.

INPUT
-----
  outputs/timeseries/water_balance_standard.csv   from 05_water_balance.py

OUTPUTS
-------
  outputs/timeseries/R6_ssa_L_sensitivity.csv
      One row per (L, component_index) with eigenvalue %, RC variance %.
  outputs/timeseries/R6_ssa_separability.csv
      One row per L with max off-diagonal w-correlation and the chosen
      trend/annual groupings.
  outputs/figures/R6_wcorrelation_L60.png
      W-correlation matrix heatmap at L = 60 (the chosen value).
  outputs/timeseries/R6_summary.txt

USAGE
-----
  cd path/to/IGP_GW_Study/scripts
  python R6_ssa_quantitative.py    # requires 05 first

REFERENCES
----------
  Vautard, R., Yiou, P., & Ghil, M. (1992). Singular-spectrum analysis:
    A toolkit for short, noisy chaotic signals. Physica D 58, 95-126.
  Golyandina, N. & Zhigljavsky, A. (2013). Singular Spectrum Analysis
    for Time Series. Springer Briefs in Statistics.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TS_DIR, FIG_DIR, SSA_L_SENSITIVITY, SSA_WINDOW_LENGTH
from pipeline_utils import ssa_decompose, fill_ssa_iterative


WB_CSV    = TS_DIR / "water_balance_standard.csv"
OUT_LSENS = TS_DIR / "R6_ssa_L_sensitivity.csv"
OUT_SEP   = TS_DIR / "R6_ssa_separability.csv"
OUT_FIG   = FIG_DIR / "R6_wcorrelation_L60.png"
OUT_SUMM  = TS_DIR / "R6_summary.txt"


def w_correlation_matrix(rcs, L):
    """
    Weighted correlation matrix among reconstructed components.

    For a series of length N decomposed via SSA with window L, the
    diagonal-averaging weights are:
      w_i = i + 1            for 0 <= i < L*
      w_i = L*               for L* <= i < N - L*
      w_i = N - i            for N - L* <= i < N
    where L* = min(L, N - L + 1) - 1.

    The w-correlation between RCs i and j is then
      <RC_i, RC_j>_w / (||RC_i||_w * ||RC_j||_w)
    where <a, b>_w = sum w_k a_k b_k.

    Values near 0 indicate good separability of the two components.
    """
    n = rcs.shape[1]
    Lstar = min(L, n - L + 1) - 1
    w = np.empty(n)
    for k in range(n):
        if k < Lstar:
            w[k] = k + 1
        elif k < n - Lstar:
            w[k] = Lstar
        else:
            w[k] = n - k
    w = w / w.sum()  # normalise (only relative magnitudes matter)

    K = rcs.shape[0]
    norms = np.sqrt(np.einsum("k,ik,ik->i", w, rcs, rcs))
    W = np.einsum("k,ik,jk->ij", w, rcs, rcs) / np.outer(norms, norms)
    return W


def detect_period(component, dt_months=1):
    """
    Estimate the dominant period (in months) of a component via FFT.

    Returns np.inf for components dominated by the zero-frequency bin
    (i.e., trend-like components with no clear oscillation).
    """
    n = len(component)
    x = component - component.mean()
    spec = np.abs(np.fft.rfft(x)) ** 2
    if len(spec) <= 1:
        return np.inf
    spec[0] = 0  # ignore DC
    k = int(np.argmax(spec))
    if k == 0:
        return np.inf
    return n * dt_months / k


def detect_groupings(rcs, n_check=8):
    """
    Auto-detect the SSA trend group and the leading annual pair.

    Trend group: the leading components whose dominant period exceeds 60
    months (5 years) -- typically RC1, sometimes RC1+RC2.
    Annual pair: the first pair of consecutive RCs whose periods are
    both within (10, 14) months and whose individual variances differ
    by less than a factor of 3 (Vautard's separability heuristic for
    quasi-degenerate eigenvalue pairs).
    """
    K = min(n_check, rcs.shape[0])
    periods = [detect_period(rcs[i]) for i in range(K)]
    variances = np.array([np.var(rcs[i], ddof=1) for i in range(K)])

    trend_idx = []
    for i in range(K):
        if periods[i] > 60 or np.isinf(periods[i]):
            trend_idx.append(i)
        else:
            break  # trend is always the leading components

    annual_pair = None
    for i in range(len(trend_idx), K - 1):
        if (10 <= periods[i] <= 14 and 10 <= periods[i + 1] <= 14):
            ratio = max(variances[i], variances[i + 1]) / min(variances[i], variances[i + 1])
            if ratio < 3.0:
                annual_pair = (i, i + 1)
                break

    return trend_idx, annual_pair, periods, variances


def main():
    TS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not WB_CSV.exists():
        raise FileNotFoundError(
            f"[R6] {WB_CSV} not found. Run 05_water_balance.py first."
        )

    print(f"[R6] Loading water_balance_standard.csv and gap-filling via "
          f"iterative SSA (M3) at L = {SSA_WINDOW_LENGTH} ...")
    df = pd.read_csv(WB_CSV, parse_dates=["time"])
    raw = df.set_index("time")["GWS_standard"]
    filled = fill_ssa_iterative(raw, L=SSA_WINDOW_LENGTH)

    print(f"[R6] Series length (gap-filled): {len(filled)} months")

    # -------------------------------------------------------------------------
    # L sensitivity sweep
    # -------------------------------------------------------------------------
    n_check = 12
    rcs_by_L = {}
    sv_by_L  = {}
    grouping_by_L = {}

    lsens_rows = []
    sep_rows   = []
    print(f"[R6] Decomposing at L = {SSA_L_SENSITIVITY} ...")
    for L in SSA_L_SENSITIVITY:
        rcs, sv = ssa_decompose(filled.values, L=L, n_components=n_check)
        rcs_by_L[L] = rcs
        sv_by_L[L]  = sv

        # Eigenvalue percentages (singular value squared / total spectrum)
        total_eig = float(np.sum(sv ** 2))
        eig_pct = 100 * (sv ** 2) / total_eig

        # RC variance percentages (variance share AFTER diagonal averaging)
        total_var = np.var(filled.values, ddof=1) * len(filled)  # sum-of-squares
        rc_var_pct = []
        for i in range(min(n_check, len(rcs))):
            ssq = float(np.sum(rcs[i] ** 2))  # rough proxy; not exact share
            rc_var_pct.append(100 * ssq / total_var)

        for i in range(min(n_check, len(sv))):
            lsens_rows.append({
                "L":            L,
                "component":    i + 1,
                "singular_val": float(sv[i]),
                "eig_pct":      float(eig_pct[i]),
                "rc_var_pct":   float(rc_var_pct[i]),
            })

        # Detect groupings and compute separability
        trend_idx, annual_pair, periods, variances = detect_groupings(rcs, n_check=n_check)
        W = w_correlation_matrix(rcs[: min(6, rcs.shape[0])], L=L)
        # max off-diagonal in absolute value
        off_diag = np.abs(W - np.eye(W.shape[0]))
        max_off_diag = float(off_diag.max())
        grouping_by_L[L] = (trend_idx, annual_pair, periods, variances, W)

        sep_rows.append({
            "L": L,
            "trend_components":  ",".join(str(i + 1) for i in trend_idx) or "-",
            "annual_pair":       (f"{annual_pair[0]+1},{annual_pair[1]+1}"
                                  if annual_pair else "-"),
            "annual_period_mo":  (np.mean([periods[annual_pair[0]],
                                            periods[annual_pair[1]]])
                                  if annual_pair else np.nan),
            "max_off_diag_wcor": max_off_diag,
        })

    pd.DataFrame(lsens_rows).to_csv(OUT_LSENS, index=False)
    pd.DataFrame(sep_rows).to_csv(OUT_SEP, index=False)

    # -------------------------------------------------------------------------
    # W-correlation heatmap at the chosen L
    # -------------------------------------------------------------------------
    L_choice = SSA_WINDOW_LENGTH
    if L_choice in grouping_by_L:
        _, _, _, _, W = grouping_by_L[L_choice]
        labels = [f"RC{i+1}" for i in range(W.shape[0])]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(np.abs(W), cmap="viridis", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(W.shape[0])); ax.set_yticks(range(W.shape[0]))
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        for i in range(W.shape[0]):
            for j in range(W.shape[0]):
                ax.text(j, i, f"{abs(W[i, j]):.2f}",
                        ha="center", va="center",
                        color="white" if abs(W[i, j]) < 0.5 else "black",
                        fontsize=9)
        ax.set_title(f"|w-correlation| matrix, L = {L_choice}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(OUT_FIG, dpi=200)
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Human-readable summary
    # -------------------------------------------------------------------------
    lines = []
    lines.append("R6  SSA QUANTITATIVE DIAGNOSTICS  -  SUMMARY")
    lines.append("=" * 78)
    lines.append(f"Series length (gap-filled): {len(filled)} months")
    lines.append("")
    lines.append("L SENSITIVITY -- LEADING SINGULAR VALUES")
    lines.append("-" * 78)
    lsens = pd.DataFrame(lsens_rows)
    pivot = lsens.pivot(index="component", columns="L", values="eig_pct").round(2)
    lines.append("Eigenvalue percentages (% of total spectrum):")
    lines.append(pivot.to_string())
    lines.append("")
    lines.append("SEPARABILITY AND GROUPING")
    lines.append("-" * 78)
    lines.append(pd.DataFrame(sep_rows).round(3).to_string(index=False))
    lines.append("")
    lines.append(f"CHOSEN L = {L_choice}")
    lines.append("-" * 78)
    if L_choice in grouping_by_L:
        trend_idx, annual_pair, periods, variances, _ = grouping_by_L[L_choice]
        trend_str  = ", ".join(f"RC{i+1}" for i in trend_idx) if trend_idx else "(none)"
        annual_str = (f"RC{annual_pair[0]+1}+RC{annual_pair[1]+1} "
                      f"(period ~{np.mean([periods[annual_pair[0]], periods[annual_pair[1]]]):.1f} mo)"
                      if annual_pair else "(not detected)")
        lines.append(f"  Trend components:    {trend_str}")
        lines.append(f"  Annual pair:         {annual_str}")
    lines.append("")
    lines.append("=" * 78)
    summary = "\n".join(lines)
    with open(OUT_SUMM, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------
    print(f"[R6] Wrote: {OUT_LSENS.relative_to(OUT_LSENS.parents[2])}")
    print(f"[R6] Wrote: {OUT_SEP.relative_to(OUT_SEP.parents[2])}")
    print(f"[R6] Wrote: {OUT_FIG.relative_to(OUT_FIG.parents[2])}")
    print(f"[R6] Wrote: {OUT_SUMM.relative_to(OUT_SUMM.parents[2])}")
    print(f"[R6]")
    print(f"[R6] Separability summary:")
    for row in sep_rows:
        print(f"[R6]   L = {row['L']:>3d}: trend = RC{row['trend_components']:<3s}, "
              f"annual = {row['annual_pair']:<5s}, "
              f"max |w-corr| = {row['max_off_diag_wcor']:.3f}")


if __name__ == "__main__":
    main()
