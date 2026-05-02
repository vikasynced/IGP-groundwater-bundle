"""
R3_subseasonal_chirps.py
========================
Sub-seasonal CHIRPS rainfall metrics for wet-year characterisation.

PURPOSE
-------
Computes monthly and (optionally) daily sub-seasonal monsoon metrics for
each year 2002-2023, then compares wet/dry/2019 composites. The output
quantitatively replaces post-hoc narratives about why certain wet years
produced atypical groundwater responses (manuscript §5.4 supplementary
discussion).

If daily CHIRPS TIFs are present in data/imd_rain/daily/, the full
sub-seasonal battery is computed (onset DOY, 7/10-day max, dry-spell
length, early/late ratio). If only monthly TIFs are available, the
reduced monthly-only metrics are computed instead.

INPUTS
------
  data/imd_rain/chirps-v3.0.YYYY.MM.tif         (monthly; required)
  data/imd_rain/daily/*.tif                     (daily; optional)
  data/masks/IGP_boundary.geojson               (optional, for clipping)

OUTPUTS
-------
  outputs/timeseries/R3_subseasonal_metrics.csv
  outputs/timeseries/R3_wet_vs_dry_comparison.csv
  outputs/timeseries/R3_summary.txt

USAGE
-----
  cd path/to/IGP_GW_Study/scripts
  python R3_subseasonal_chirps.py

NOTES
-----
This script uses cosine-of-latitude-weighted spatial averaging. The
original v4 implementation used np.nanmean which over-weights high-
latitude grid cells; at the IGP latitude range the correction is ~3%
on absolute totals but does not affect wet/dry classification or
relative ranking among years.
"""

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import rioxarray
    import geopandas as gpd
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR, TS_DIR,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    WET_YEARS_CHIRPS, DRY_YEARS_CHIRPS,
    STUDY_START_YEAR, STUDY_END_YEAR,
)


MONTHLY_DIR = DATA_DIR / "imd_rain"
DAILY_DIR   = DATA_DIR / "imd_rain" / "daily"
MASK_PATH   = DATA_DIR / "masks" / "IGP_boundary.geojson"

OUT_METRICS = TS_DIR / "R3_subseasonal_metrics.csv"
OUT_COMPARE = TS_DIR / "R3_wet_vs_dry_comparison.csv"
OUT_SUMMARY = TS_DIR / "R3_summary.txt"

ALL_YEARS    = list(range(STUDY_START_YEAR, STUDY_END_YEAR + 1))
WET_EXCL_2019 = [y for y in WET_YEARS_CHIRPS if y != 2019]


def _load_mask():
    """Try to load the IGP boundary mask; return None if unavailable."""
    if not HAS_GEO or not MASK_PATH.exists():
        return None
    try:
        return gpd.read_file(MASK_PATH)
    except (OSError, ValueError) as e:
        print(f"[R3] Mask file present but unreadable ({e}); using bbox fallback.")
        return None


def _domain_mean_chirps(tif_path, mask_gdf=None):
    """
    Cosine-of-latitude-weighted spatial mean of a CHIRPS monthly or
    daily TIF, restricted to the IGP domain (mask if available, else bbox).
    """
    da = rioxarray.open_rasterio(tif_path).squeeze()

    # Filter the dataset's nodata flag if set
    nodata = da.rio.nodata
    if nodata is not None:
        da = da.where(da != nodata)

    # Restrict to mask (preferred) or bbox (fallback)
    if mask_gdf is not None:
        try:
            da = da.rio.clip(mask_gdf.geometry, mask_gdf.crs, drop=True)
        except (rioxarray.exceptions.NoDataInBounds, ValueError):
            da = da.sel(y=slice(LAT_MAX, LAT_MIN), x=slice(LON_MIN, LON_MAX))
    else:
        da = da.sel(y=slice(LAT_MAX, LAT_MIN), x=slice(LON_MIN, LON_MAX))

    arr     = da.values.astype(float)
    weights = np.cos(np.deg2rad(da["y"].values))[:, None]
    weights = np.broadcast_to(weights, arr.shape).copy()
    weights[np.isnan(arr)] = 0.0
    arr = np.nan_to_num(arr, nan=0.0)
    total_w = weights.sum()
    if total_w == 0:
        return np.nan
    return float((arr * weights).sum() / total_w)


def _parse_monthly_filename(fn):
    """Match chirps-v3.0.YYYY.MM.tif (case insensitive)."""
    m = re.search(r"chirps[-_]v?[0-9.]+\.(\d{4})\.(\d{2})\.tif$",
                  Path(fn).name, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _parse_daily_filename(fn):
    """Match chirps-v?.YYYY.MM.DD.tif (case insensitive)."""
    m = re.search(r"chirps[-_]v?[0-9.]+\.(\d{4})\.(\d{2})\.(\d{2})\.tif$",
                  Path(fn).name, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None, None, None


def compute_monthly_metrics(mask):
    """Build a (year x month) monthly precipitation table for the IGP domain."""
    files = sorted(glob.glob(str(MONTHLY_DIR / "chirps*.tif")))
    rows  = []
    for f in files:
        y, m = _parse_monthly_filename(f)
        if y is None or not (STUDY_START_YEAR <= y <= STUDY_END_YEAR):
            continue
        v = _domain_mean_chirps(f, mask)
        rows.append({"year": y, "month": m, "rainfall_mm": v})
    if not rows:
        return None
    monthly = (
        pd.DataFrame(rows)
        .pivot_table(index="year", columns="month", values="rainfall_mm",
                     aggfunc="mean")
        .reindex(columns=range(1, 13))
    )
    monthly.columns = [f"m{i:02d}" for i in monthly.columns]
    return monthly


def compute_daily_metrics_for_year(year, mask):
    """Sub-seasonal metrics for one year if daily TIFs exist; else None."""
    files = sorted(glob.glob(str(DAILY_DIR / f"*chirps*{year}*.tif")))
    if not files:
        return None
    daily_records = []
    for f in files:
        y, m, d = _parse_daily_filename(f)
        if y != year:
            continue
        v = _domain_mean_chirps(f, mask)
        if not np.isnan(v):
            daily_records.append({
                "date":      pd.Timestamp(year=y, month=m, day=d),
                "precip_mm": v,
            })
    if len(daily_records) < 30:
        return None
    daily = pd.DataFrame(daily_records).sort_values("date").reset_index(drop=True)
    jjas = daily[
        (daily["date"] >= pd.Timestamp(year, 6, 1)) &
        (daily["date"] <= pd.Timestamp(year, 9, 30))
    ].copy()
    if len(jjas) < 30:
        return None

    metrics = {}
    metrics["daily_jjas_mean"]  = float(jjas["precip_mm"].mean())
    metrics["daily_jjas_total"] = float(jjas["precip_mm"].sum())

    # Onset = first 5-day window with rolling-mean > 10 mm/day
    roll5 = jjas["precip_mm"].rolling(5, min_periods=5).mean()
    onset_mask = roll5 > 10
    if onset_mask.any():
        onset_idx  = onset_mask.idxmax()
        onset_date = jjas.loc[onset_idx, "date"]
        metrics["onset_doy"] = int(onset_date.dayofyear)
    else:
        metrics["onset_doy"] = np.nan

    metrics["max_7day_mm"]            = float(jjas["precip_mm"].rolling(7,  min_periods=7).sum().max())
    metrics["max_10day_mm"]           = float(jjas["precip_mm"].rolling(10, min_periods=10).sum().max())
    metrics["n_heavy_days_gt10mm"]    = int((jjas["precip_mm"] > 10).sum())

    # Longest run of consecutive days with precip < 1 mm
    is_dry  = (jjas["precip_mm"] < 1).astype(int).values
    longest = 0; current = 0
    for v in is_dry:
        if v == 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    metrics["longest_dry_spell_days"] = int(longest)

    early = float(jjas[jjas["date"].dt.month.isin([6, 7])]["precip_mm"].sum())
    late  = float(jjas[jjas["date"].dt.month.isin([8, 9])]["precip_mm"].sum())
    metrics["early_late_ratio"] = (early / late) if late > 0 else np.nan
    return metrics


def compute_daily_metrics_all_years(mask):
    if not DAILY_DIR.exists():
        print(f"[R3] Daily CHIRPS directory not found: {DAILY_DIR}; "
              f"monthly-only metrics will be reported.")
        return None
    rows = []
    for y in ALL_YEARS:
        m = compute_daily_metrics_for_year(y, mask)
        if m is not None:
            m["year"] = y
            rows.append(m)
    return pd.DataFrame(rows).set_index("year") if rows else None


def wet_vs_dry_table(metrics_df):
    """Group means for Wet, Wet-excluding-2019, Dry, and 2019 alone."""
    rows = []
    for col in metrics_df.columns:
        if not pd.api.types.is_numeric_dtype(metrics_df[col]):
            continue
        wet_vals      = metrics_df.loc[metrics_df.index.intersection(WET_YEARS_CHIRPS), col]
        wet_excl_2019 = metrics_df.loc[metrics_df.index.intersection(WET_EXCL_2019),    col]
        dry_vals      = metrics_df.loc[metrics_df.index.intersection(DRY_YEARS_CHIRPS), col]
        v2019         = metrics_df.loc[2019, col] if 2019 in metrics_df.index else np.nan
        rows.append({
            "metric":              col,
            "wet_mean":            float(wet_vals.mean()) if len(wet_vals) else np.nan,
            "wet_excl_2019_mean":  float(wet_excl_2019.mean()) if len(wet_excl_2019) else np.nan,
            "dry_mean":            float(dry_vals.mean()) if len(dry_vals) else np.nan,
            "val_2019":            float(v2019) if pd.notna(v2019) else np.nan,
            "_2019_vs_other_wet":  (float(v2019) - float(wet_excl_2019.mean()))
                                   if (pd.notna(v2019) and len(wet_excl_2019)) else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    TS_DIR.mkdir(parents=True, exist_ok=True)

    if not HAS_GEO:
        raise SystemExit(
            "[R3] FATAL: rioxarray and geopandas are required. Install with "
            "`pip install rioxarray geopandas`."
        )
    if not MONTHLY_DIR.exists():
        raise FileNotFoundError(
            f"[R3] CHIRPS monthly directory not found: {MONTHLY_DIR}"
        )

    print(f"[R3] Loading IGP boundary mask (if available) ...")
    mask = _load_mask()
    print(f"[R3]   Mask source: "
          f"{'shapefile' if mask is not None else 'bbox fallback'}")

    print(f"[R3] Computing monthly CHIRPS metrics ...")
    monthly = compute_monthly_metrics(mask)
    if monthly is None:
        raise SystemExit("[R3] FATAL: no readable CHIRPS monthly TIFs found.")

    print(f"[R3] Computing daily CHIRPS metrics (if available) ...")
    daily = compute_daily_metrics_all_years(mask)
    has_daily = daily is not None

    # Build per-year combined table
    per_year = monthly.copy()
    per_year["jjas_total"] = per_year[["m06", "m07", "m08", "m09"]].sum(axis=1, min_count=1)
    per_year["peak_month"] = per_year[["m06", "m07", "m08", "m09"]].idxmax(axis=1)
    per_year["jun_share"]  = per_year["m06"] / per_year["jjas_total"]
    per_year["sep_share"]  = per_year["m09"] / per_year["jjas_total"]
    per_year["monthly_onset_proxy"] = (per_year["m06"] >= 50).astype(int)

    if has_daily:
        per_year = per_year.join(daily, how="left")

    per_year.to_csv(OUT_METRICS)

    comp = wet_vs_dry_table(per_year)
    comp.to_csv(OUT_COMPARE, index=False)

    # -------------------------------------------------------------------------
    # Human-readable summary
    # -------------------------------------------------------------------------
    lines = []
    lines.append("R3  SUB-SEASONAL CHIRPS METRICS  -  SUMMARY")
    lines.append("=" * 78)
    lines.append(f"Daily CHIRPS available: {has_daily}")
    lines.append(f"  ({'Full sub-seasonal battery' if has_daily else 'Monthly-only metrics'})")
    lines.append(f"Mask source: {'shapefile' if mask is not None else 'bbox fallback'}")
    lines.append("")
    lines.append("PER-YEAR METRICS (key columns)")
    lines.append("-" * 78)
    key_cols = ["m06", "m07", "m08", "m09", "jjas_total", "peak_month", "jun_share"]
    if has_daily:
        key_cols += ["onset_doy", "max_7day_mm", "max_10day_mm",
                     "n_heavy_days_gt10mm", "longest_dry_spell_days",
                     "early_late_ratio"]
    present = [c for c in key_cols if c in per_year.columns]
    lines.append(per_year[present].round(2).to_string())
    lines.append("")
    lines.append("WET vs DRY vs 2019")
    lines.append("-" * 78)
    lines.append(comp.round(2).to_string(index=False))
    lines.append("")
    lines.append("=" * 78)
    summary = "\n".join(lines)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------
    print(f"[R3] Years processed: {len(per_year)}")
    if 2019 in per_year.index and "jjas_total" in per_year.columns:
        v = float(per_year.loc[2019, "jjas_total"])
        print(f"[R3] 2019 JJAS total: {v:.0f} mm")
    print(f"[R3] Wrote: {OUT_METRICS.relative_to(OUT_METRICS.parents[2])}")
    print(f"[R3] Wrote: {OUT_COMPARE.relative_to(OUT_COMPARE.parents[2])}")
    print(f"[R3] Wrote: {OUT_SUMMARY.relative_to(OUT_SUMMARY.parents[2])}")


if __name__ == "__main__":
    main()
