"""
R4_change_point.py
==================
Change-point detection on the GWS series via four complementary methods.

PURPOSE
-------
Tests whether the IGP GWS series contains a single statistically detectable
change in the mean/trend, using:

  1. Pettitt test          (non-parametric, sensitive to mean-shift)
  2. Buishand range test   (non-parametric, sensitive to mean-shift; runs
                            with Monte Carlo permutations for p-value)
  3. ruptures Binseg       (binary segmentation with l2 cost; runs at
                            n_breakpoints = 1, 2, and 3 for sensitivity)
  4. ruptures PELT         (penalised exact partition; sweeps penalty
                            parameter and reports the most stable result)

Each method is applied to three variants of the GWS series:

  (a) raw GWS_standard (with the 2017-06 to 2018-05 GRACE-FO gap left as NaN)
  (b) raw GWS_standard with linear interpolation across the gap
  (c) SSA trend component (smooth, gap-free reconstruction)

This triple-variant design checks that the detected change-point is not
an artefact of the GRACE-FO gap.

INPUTS
------
  outputs/timeseries/water_balance_standard.csv   from 05_water_balance.py
  outputs/timeseries/SSA_components.csv           from 06_ssa_trend.py

OUTPUTS
-------
  outputs/timeseries/R4_changepoints.csv
      Per-method, per-variant detected change-points and p-values.
  outputs/timeseries/R4_pre_post_slopes.csv
      Local pre- and post-gap OLS slopes (window-based diagnostic).
  outputs/timeseries/R4_summary.txt
      Human-readable summary.

USAGE
-----
  cd path/to/IGP_GW_Study/scripts
  python R4_change_point.py        # requires 05 and 06 first

DEPENDENCIES
------------
  ruptures (pip install ruptures); the Pettitt and Buishand tests are
  implemented in this file because there is no widely-maintained Python
  package for them as of v5.1 development.

REFERENCES
----------
  Pettitt, A. N. (1979). A non-parametric approach to the change-point
    problem. Appl. Stat. 28, 126-135.
  Buishand, T. A. (1982). Some methods for testing the homogeneity of
    rainfall records. J. Hydrol. 58, 11-27.
  Villarini, G., et al. (2009). On the stationarity of annual flood peaks
    in the continental United States. WRR 45, W08417.
  Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of
    offline change point detection methods. Signal Process. 167, 107299.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    TS_DIR,
    GRACE_GAP_START, GRACE_GAP_END,
    SEED_BUISHAND_PERMUTATION,
    EXPECTED_CHANGEPOINT_DATE,
)


WB_CSV  = TS_DIR / "water_balance_standard.csv"
SSA_CSV = TS_DIR / "SSA_components.csv"

OUT_CSV     = TS_DIR / "R4_changepoints.csv"
OUT_SLOPES  = TS_DIR / "R4_pre_post_slopes.csv"
OUT_SUMMARY = TS_DIR / "R4_summary.txt"

LOCAL_SLOPE_WINDOW_MONTHS = 30   # Pre/post-gap diagnostic window length


# -----------------------------------------------------------------------------
# Test implementations
# -----------------------------------------------------------------------------
def pettitt_test(x):
    """
    Pettitt non-parametric change-point test.

    Returns
    -------
    idx : int
        0-based index of the most likely change-point (the t* maximising
        |U_t|).
    p   : float
        Approximate two-sided p-value (Pettitt 1979 formula, valid for
        n > ~30).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return 0, np.nan
    # Pairwise sign accumulator U_t = sum_{i<=t} sum_{j>t} sgn(x_i - x_j)
    # O(n^2) but small n in this study (~250 months) so direct is fine.
    U = np.zeros(n)
    for t in range(n - 1):
        # sign matrix for (i <= t, j > t)
        diff = x[: t + 1, None] - x[t + 1:][None, :]
        U[t] = np.sign(diff).sum()
    idx = int(np.argmax(np.abs(U)))
    K   = float(np.abs(U[idx]))
    # Two-sided approximate p-value (Pettitt 1979 eq. 2.6)
    p = 2.0 * np.exp(-6.0 * K ** 2 / (n ** 3 + n ** 2))
    return idx, min(p, 1.0)


def buishand_range_test(x, n_perm=2000, seed=SEED_BUISHAND_PERMUTATION):
    """
    Buishand range test (R/sqrt(n)) for a single change-point in the mean.

    The R/sqrt(n) statistic is referenced to the empirical permutation
    distribution rather than the asymptotic table; this is the practice
    recommended by Villarini et al. (2009) for moderate sample sizes
    (n in 50-500). Permutation seed is fixed at SEED_BUISHAND_PERMUTATION
    in config for reproducibility.

    Returns
    -------
    idx : int
        Index of the most likely change-point.
    p   : float
        Permutation p-value.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return 0, np.nan
    cumsum  = np.cumsum(x - x.mean())
    obs_R   = cumsum.max() - cumsum.min()
    obs_idx = int(np.argmax(np.abs(cumsum)))
    rng     = np.random.default_rng(seed)
    null_R  = np.empty(n_perm)
    for k in range(n_perm):
        sh = rng.permutation(x)
        cs = np.cumsum(sh - sh.mean())
        null_R[k] = cs.max() - cs.min()
    p = float((null_R >= obs_R).sum() + 1) / (n_perm + 1)
    return obs_idx, p


def ruptures_binseg(x, n_bkps=1):
    """ruptures Binary Segmentation with l2 cost."""
    if not HAS_RUPTURES:
        return None
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    algo = rpt.Binseg(model="l2").fit(x)
    bkps = algo.predict(n_bkps=n_bkps)
    return [b - 1 for b in bkps[:-1]]   # exclude trailing n; convert to 0-based


def ruptures_pelt(x, pen=10.0):
    """ruptures PELT with l2 cost."""
    if not HAS_RUPTURES:
        return None
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    algo = rpt.Pelt(model="l2").fit(x)
    bkps = algo.predict(pen=pen)
    return [b - 1 for b in bkps[:-1]]


# -----------------------------------------------------------------------------
# Local slopes (pre/post-gap diagnostic)
# -----------------------------------------------------------------------------
def local_ols_slope(times, values, window_end, window_months):
    """
    OLS slope (mm/yr) over the window of size `window_months` ending at
    or just before `window_end` (a pandas Timestamp).
    """
    end_idx = times.searchsorted(window_end, side="right")
    start_idx = max(0, end_idx - window_months)
    sub_t = times[start_idx:end_idx]
    sub_v = values[start_idx:end_idx]
    valid = ~np.isnan(sub_v)
    if valid.sum() < 6:
        return np.nan
    x = np.arange(valid.sum())
    slope_per_month, _ = np.polyfit(x, sub_v[valid], 1)
    return float(slope_per_month * 12)


def main():
    TS_DIR.mkdir(parents=True, exist_ok=True)

    if not HAS_RUPTURES:
        raise SystemExit(
            "[R4] FATAL: ruptures not installed. `pip install ruptures`."
        )
    for path in [WB_CSV, SSA_CSV]:
        if not path.exists():
            raise FileNotFoundError(
                f"[R4] Required upstream output not found: {path}"
            )

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print(f"[R4] Loading water_balance_standard.csv and SSA_components.csv ...")
    wb  = pd.read_csv(WB_CSV,  parse_dates=["time"])
    ssa = pd.read_csv(SSA_CSV, parse_dates=["time"])

    # Build the three variants.
    # Variant (a): raw GWS_standard, with NaN for gap months.
    # We need a continuous monthly time axis here so that change-point indices
    # map cleanly to dates. Reindex onto full monthly range from min to max.
    full_idx = pd.date_range(
        wb["time"].min().to_period("M").to_timestamp(),
        wb["time"].max().to_period("M").to_timestamp(),
        freq="MS",
    )
    raw = wb.set_index("time")["GWS_standard"]
    raw.index = raw.index.to_period("M").to_timestamp()
    raw = raw.reindex(full_idx)
    raw_filled = raw.interpolate(method="linear", limit_direction="both")

    # Variant (c): SSA trend (no gaps; SSA was computed on observed-only series,
    # so its time index excludes the gap; reindex with linear interpolation
    # across the gap).
    trend = ssa.set_index("time")["SSA_trend"]
    trend.index = trend.index.to_period("M").to_timestamp()
    trend = trend.reindex(full_idx).interpolate(method="linear", limit_direction="both")

    variants = {
        "raw_with_gap":     raw,
        "raw_interpolated": raw_filled,
        "ssa_trend":        trend,
    }

    # -------------------------------------------------------------------------
    # Run all method x variant combinations
    # -------------------------------------------------------------------------
    rows = []
    print(f"[R4] Running change-point tests across 3 variants x 4 methods ...")
    for vname, series in variants.items():
        # For the raw_with_gap variant, drop NaN for tests that can't handle them.
        clean = series.dropna()
        clean_times = clean.index
        clean_vals  = clean.values

        # Pettitt
        idx, p = pettitt_test(clean_vals)
        rows.append({
            "variant": vname, "method": "Pettitt",
            "n_breakpoints": 1, "n_samples": len(clean_vals),
            "cp_index_within_clean": int(idx),
            "cp_date": str(clean_times[idx].date()),
            "p_value": float(p),
        })

        # Buishand range, permutation p-value
        idx, p = buishand_range_test(clean_vals)
        rows.append({
            "variant": vname, "method": "Buishand",
            "n_breakpoints": 1, "n_samples": len(clean_vals),
            "cp_index_within_clean": int(idx),
            "cp_date": str(clean_times[idx].date()),
            "p_value": float(p),
        })

        # ruptures Binseg, n=1, n=2, n=3
        for nb in [1, 2, 3]:
            bkps = ruptures_binseg(clean_vals, n_bkps=nb)
            if bkps is None or not bkps:
                continue
            for k, b in enumerate(bkps, start=1):
                rows.append({
                    "variant": vname,
                    "method": f"Binseg(n={nb})",
                    "n_breakpoints": nb,
                    "n_samples": len(clean_vals),
                    "cp_index_within_clean": int(b),
                    "cp_date": str(clean_times[b].date()),
                    "p_value": np.nan,
                    "bkpt_rank": k,
                })

        # ruptures PELT, sweep penalty
        for pen in [5.0, 10.0, 20.0, 40.0]:
            bkps = ruptures_pelt(clean_vals, pen=pen)
            if bkps is None:
                continue
            for k, b in enumerate(bkps, start=1):
                rows.append({
                    "variant": vname,
                    "method": f"PELT(pen={pen})",
                    "n_breakpoints": len(bkps),
                    "n_samples": len(clean_vals),
                    "cp_index_within_clean": int(b),
                    "cp_date": str(clean_times[b].date()),
                    "p_value": np.nan,
                    "bkpt_rank": k,
                })

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)

    # -------------------------------------------------------------------------
    # Pre/post-gap local slopes (diagnostic for whether the
    # change-point really is a gap artefact)
    # -------------------------------------------------------------------------
    pre_end  = pd.Timestamp(GRACE_GAP_START)
    post_end = pd.Timestamp(GRACE_GAP_END)

    pre_slopes  = {}
    post_slopes = {}
    for vname, series in variants.items():
        v = series.values
        t = np.asarray(series.index, dtype="datetime64[ns]")
        pre_slopes[vname]  = local_ols_slope(t, v, pre_end,  LOCAL_SLOPE_WINDOW_MONTHS)
        # For "post" we want the window starting AFTER the gap end
        post_window_end_idx = series.index.searchsorted(post_end, side="right") + LOCAL_SLOPE_WINDOW_MONTHS
        if post_window_end_idx > len(series):
            post_window_end_idx = len(series)
        post_window_end = series.index[post_window_end_idx - 1]
        post_slopes[vname] = local_ols_slope(t, v, post_window_end, LOCAL_SLOPE_WINDOW_MONTHS)

    slopes_df = pd.DataFrame({
        "variant":     list(variants.keys()),
        "pre_gap_slope_mm_yr":  [pre_slopes[v]  for v in variants],
        "post_gap_slope_mm_yr": [post_slopes[v] for v in variants],
        "window_months": LOCAL_SLOPE_WINDOW_MONTHS,
    })
    slopes_df.to_csv(OUT_SLOPES, index=False)

    # -------------------------------------------------------------------------
    # Human-readable summary
    # -------------------------------------------------------------------------
    lines = []
    lines.append("R4  CHANGE-POINT DETECTION  -  SUMMARY")
    lines.append("=" * 78)
    lines.append("")
    lines.append("DETECTED CHANGE-POINTS (Pettitt, Buishand, Binseg n=1)")
    lines.append("-" * 78)
    primary = res[res["method"].isin(["Pettitt", "Buishand", "Binseg(n=1)"])]
    for _, r in primary.iterrows():
        p_str = f"p = {r['p_value']:.4f}" if pd.notna(r["p_value"]) else "(no p)"
        lines.append(f"  [{r['variant']:18s}] {r['method']:13s} -> "
                     f"{r['cp_date']}  {p_str}")
    lines.append("")
    lines.append("BINSEG n=2 AND n=3 (for sensitivity)")
    lines.append("-" * 78)
    multi = res[res["method"].isin(["Binseg(n=2)", "Binseg(n=3)"])]
    if len(multi):
        lines.append(multi[["variant", "method", "bkpt_rank",
                            "cp_date"]].to_string(index=False))
    lines.append("")
    lines.append("PELT PENALTY SWEEP")
    lines.append("-" * 78)
    pelt = res[res["method"].str.startswith("PELT", na=False)]
    if len(pelt):
        lines.append(pelt[["variant", "method", "n_breakpoints",
                           "bkpt_rank", "cp_date"]].to_string(index=False))
    lines.append("")
    lines.append(f"PRE/POST-GAP LOCAL SLOPES "
                 f"(window = {LOCAL_SLOPE_WINDOW_MONTHS} months)")
    lines.append("-" * 78)
    lines.append(slopes_df.round(2).to_string(index=False))
    lines.append("")
    lines.append("=" * 78)
    summary = "\n".join(lines)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------
    print(f"[R4] Wrote: {OUT_CSV.relative_to(OUT_CSV.parents[2])}")
    print(f"[R4] Wrote: {OUT_SLOPES.relative_to(OUT_SLOPES.parents[2])}")
    print(f"[R4] Wrote: {OUT_SUMMARY.relative_to(OUT_SUMMARY.parents[2])}")
    print(f"[R4]")
    print(f"[R4] Manuscript verification (v5.1 §5.1):")
    print(f"[R4]   Expected primary change-point: {EXPECTED_CHANGEPOINT_DATE}")
    bs1 = primary[(primary["method"] == "Binseg(n=1)") &
                  (primary["variant"] == "raw_interpolated")]
    if not bs1.empty:
        print(f"[R4]   Computed (Binseg n=1, interpolated): "
              f"{bs1['cp_date'].iloc[0]}")


if __name__ == "__main__":
    main()
