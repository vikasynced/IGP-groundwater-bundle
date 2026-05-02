"""
pipeline_utils.py
=================
Shared utilities used across the IGP groundwater reproducibility bundle.

Functions here are imported by both upstream pipeline scripts (01-07) and
downstream R-scripts. Centralising them avoids the ~150 lines of code
duplication in the original scripts and ensures, for example, that the
SSA decomposition is identical across R6 and R9.

Contents
--------
  load_giovanni              Parse NASA Giovanni-format CSV exports.
  cosine_weighted_mean       Latitude-weighted spatial mean of an xr.DataArray.
  find_time_column           Detect the time column in a heterogeneous CSV.
  ssa_decompose              Singular Spectrum Analysis (vectorised).
  fill_ssa_iterative         Gap-fill via iterative SSA (manuscript §4.2 M3).
  pearson_with_p             Pearson correlation with two-sided p-value.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Input loaders
# -----------------------------------------------------------------------------
def load_giovanni(filepath):
    """
    Load a NASA Giovanni time-series CSV export.

    Giovanni CSVs prepend a variable-length comment header before the data
    rows. This routine locates the first data line by scanning for the first
    line whose first non-whitespace character is a digit, then reads from
    there as a two-column (time, value) table.

    Parameters
    ----------
    filepath : str or Path
        Path to a Giovanni-format CSV file.

    Returns
    -------
    pd.DataFrame
        Indexed on a datetime 'time' column, with a single 'value' column.

    Notes
    -----
    Used for GLDAS soil moisture, SWE, and ET extracts.
    """
    filepath = Path(filepath)
    with open(filepath, "r") as f:
        lines = f.readlines()
    start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s and s[0].isdigit():
            start = i
            break
    df = pd.read_csv(filepath, skiprows=start, header=None)
    df.columns = ["time", "value"]
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def find_time_column(df, candidates=None):
    """
    Locate the time column in a CSV with heterogeneous naming conventions.

    Several files in the pipeline are written with different time-column
    names: 'time', 'date', 'Unnamed: 0' (a common pandas artefact when an
    index column was written without a name). This helper centralises the
    detection so individual scripts don't each carry the same search logic.

    Parameters
    ----------
    df : pd.DataFrame
    candidates : list of str, optional
        Names to try in order. Defaults to a sensible standard list.

    Returns
    -------
    str
        Name of the column that holds the time information.

    Raises
    ------
    KeyError
        If no candidate name appears in df's columns.
    """
    if candidates is None:
        candidates = ["time", "date", "Time", "Date", "timestamp", "Unnamed: 0"]
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(
        f"No time column found. Tried {candidates}; "
        f"available columns are {list(df.columns)}."
    )


# -----------------------------------------------------------------------------
# Spatial weighting
# -----------------------------------------------------------------------------
def cosine_weighted_mean(da, lat_dim="lat", mask=None):
    """
    Compute area-weighted spatial mean of a DataArray using cos(lat) weights.

    For a regular lat/lon grid, raw arithmetic means over-weight high
    latitudes because cells become geometrically narrower as |lat| increases.
    The cosine-of-latitude weighting corrects this. Across the IGP latitude
    range (24-32.5 N), the correction is ~3% relative to a naive np.nanmean.

    Parameters
    ----------
    da : xr.DataArray
        Must have a latitude dimension and at least one other dimension to
        average over (typically lon, with optional time).
    lat_dim : str, default 'lat'
        Name of the latitude dimension.
    mask : np.ndarray of bool, optional
        Per-pixel mask broadcastable to da's spatial dimensions. Pixels
        where mask is False are excluded from the weighted mean (e.g.,
        ocean pixels, or alluvial-extent restriction in R8).

    Returns
    -------
    np.ndarray or float
        The weighted mean along all non-time dimensions.
    """
    arr = da.values
    weights = np.cos(np.deg2rad(da[lat_dim].values))[:, None]
    weights = np.broadcast_to(weights, arr.shape[1:])
    if mask is not None:
        weights = weights * mask
    weights = weights * (~np.isnan(arr))
    return np.nansum(arr * weights, axis=(1, 2)) / np.nansum(weights, axis=(1, 2))


# -----------------------------------------------------------------------------
# SSA (Singular Spectrum Analysis)
# -----------------------------------------------------------------------------
def _hankelise(matrix):
    """
    Diagonal-average a rectangular matrix to produce a 1-D Hankelised series.

    This is the SSA reconstruction step: each anti-diagonal of the trajectory
    matrix is replaced by its mean. Vectorised version, equivalent to (and
    significantly faster than) a triple-nested loop accumulator.

    Parameters
    ----------
    matrix : np.ndarray, shape (L, K)
        A rank-1 elementary matrix from SSA SVD: s_k * u_k * v_k^T.

    Returns
    -------
    np.ndarray, shape (L + K - 1,)
        The Hankelised reconstructed component.
    """
    L, K = matrix.shape
    n    = L + K - 1
    # Each anti-diagonal d (where d = i + j, with 0 <= i < L, 0 <= j < K)
    # contributes to position d in the output. Use np.add.at on the indices.
    out = np.zeros(n, dtype=matrix.dtype)
    cnt = np.zeros(n, dtype=matrix.dtype)
    i_idx, j_idx = np.indices((L, K))
    diag_idx = i_idx + j_idx
    np.add.at(out, diag_idx.ravel(), matrix.ravel())
    np.add.at(cnt, diag_idx.ravel(), 1)
    return out / cnt


def ssa_decompose(x, L, n_components=None):
    """
    Perform Singular Spectrum Analysis on a 1-D series.

    Builds the trajectory matrix of embedding dimension L, performs SVD,
    and returns the reconstructed components (RCs) along with the singular
    values. Implementation follows Vautard et al. (1992) and Golyandina &
    Zhigljavsky (2013).

    Parameters
    ----------
    x : array-like
        Input series. Must contain no NaNs (use fill_ssa_iterative if it does).
    L : int
        Window length (embedding dimension). Will be capped at len(x) // 2.
    n_components : int, optional
        Number of leading components to return. Defaults to all.

    Returns
    -------
    rcs : np.ndarray, shape (n_components, len(x))
        Reconstructed components, ordered by descending singular value.
    singular_values : np.ndarray
        Full vector of singular values (length min(L, K)). Useful for
        eigenvalue percentage diagnostics even when n_components is small.
    """
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        raise ValueError(
            "ssa_decompose: input contains NaN. "
            "Use fill_ssa_iterative() to gap-fill first."
        )
    n = len(x)
    L = min(L, n // 2)
    K = n - L + 1

    # Trajectory matrix: columns are length-L sliding windows of x.
    X = np.column_stack([x[i:i + L] for i in range(K)])
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    if n_components is None:
        n_components = len(s)
    n_components = min(n_components, len(s))

    rcs = np.empty((n_components, n))
    for k in range(n_components):
        Xk = s[k] * np.outer(U[:, k], Vt[k, :])
        rcs[k] = _hankelise(Xk)
    return rcs, s


def fill_ssa_iterative(series, L=60, max_iter=10, tol=0.5, n_components=3):
    """
    Gap-fill a series via iterative SSA (manuscript §4.2 method M3).

    The series is first linearly interpolated to provide a complete starting
    estimate, then SSA is run on it; the leading n_components reconstructed
    components are summed and used to overwrite the originally-missing
    values. This is iterated until convergence (or max_iter reached).

    Parameters
    ----------
    series : pd.Series
        Input series with NaN at gap locations.
    L : int, default 60
        SSA window length.
    max_iter : int, default 10
        Maximum number of iterative passes.
    tol : float, default 0.5
        Convergence threshold (in input units): iteration stops when the
        maximum change between successive iterations falls below this value.
    n_components : int, default 3
        Number of leading SSA components used to define the smooth
        reconstruction that fills gaps.

    Returns
    -------
    pd.Series
        Same index as input, with no NaN values. Originally observed values
        are preserved exactly; only originally missing values are filled.
    """
    s = series.copy().reset_index(drop=True)
    observed_mask = ~s.isna()
    cur = s.interpolate(method="linear", limit_direction="both").values

    for _ in range(max_iter):
        try:
            rcs, _ = ssa_decompose(cur, L=L, n_components=n_components)
        except (np.linalg.LinAlgError, ValueError):
            # SVD failed to converge or the trajectory matrix is degenerate.
            # Return the current best estimate; this is exceptionally rare on
            # real data but possible if the series is too short relative to L.
            break
        recon = rcs.sum(axis=0)
        new = cur.copy()
        new[~observed_mask.values] = recon[~observed_mask.values]
        if np.max(np.abs(new - cur)) < tol:
            cur = new
            break
        cur = new

    return pd.Series(cur, index=series.index)


# -----------------------------------------------------------------------------
# Statistical helpers
# -----------------------------------------------------------------------------
def pearson_with_p(x, y):
    """
    Pearson correlation with two-sided p-value, NaN-safe.

    Wraps scipy.stats.pearsonr but drops paired NaN rows first so the
    user does not have to do this explicitly in every script.

    Parameters
    ----------
    x, y : array-like
        Equal-length sequences. Pairs where either is NaN are dropped.

    Returns
    -------
    r : float
    p : float
    n : int
        Number of finite pairs used in the computation.
    """
    from scipy import stats
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan, np.nan, int(valid.sum())
    r, p = stats.pearsonr(x[valid], y[valid])
    return float(r), float(p), int(valid.sum())
