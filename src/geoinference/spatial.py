"""Spatial and temporal dependence diagnostics for geoinference.

These experimental functions measure within-cluster correlation and how fast
it decays along geographic distance or elapsed time. They describe dependence;
they do not replace the standard-error estimators in :mod:`geoinference.inference`.
other pairwise-distance matrix. They are *diagnostics*: they describe the
dependence structure (and how much information itineraries cost), they do not
replace the standard-error estimators in ``inference.py``.

The core is axis-agnostic. ``empirical_variogram``, ``morans_i``, and
``effective_n`` all take a precomputed n×n distance matrix, so the same code
serves the spatial axis (``haversine_matrix``), the temporal axis
(``time_gap_matrix``), and any future axis (e.g. same/different enumerator).

References:
    Griffith, D.A. (2005). Effective geographic sample size in the presence
        of spatial autocorrelation.
    Watson, P.A. (2021). A note on the variogram-based effective sample size.
        J. Applied Statistics.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

# Mean Earth radius (meters), for great-circle distances.
_EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True, slots=True)
class DependenceDiagnostics:
    """Experimental spatial and temporal dependence diagnostics."""

    within_between_ratio: float = float("nan")
    morans_i_space: float = float("nan")
    morans_i_space_p: float = float("nan")
    variogram_range_m: float = float("nan")
    spatial_corr_ratio: float = float("nan")
    n_eff_space: float = float("nan")
    morans_i_time: float = float("nan")
    morans_i_time_p: float = float("nan")
    variogram_range_s: float = float("nan")
    temporal_corr_ratio: float = float("nan")
    n_eff_time: float = float("nan")


def _as_finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Return a finite one-dimensional float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_distance_matrix(dist: np.ndarray, n: int) -> np.ndarray:
    """Validate and return an n by n pairwise-distance matrix."""
    matrix = np.asarray(dist, dtype=float)
    if matrix.shape != (n, n):
        raise ValueError(f"dist must have shape ({n}, {n}), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("dist must contain finite, nonnegative distances")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("dist must be symmetric")
    if not np.allclose(np.diag(matrix), 0):
        raise ValueError("dist must have a zero diagonal")
    return matrix


# ─── Pairwise distance matrices (the "axes") ─────────────────────────


def haversine_matrix(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances in meters between lon/lat points.

    Args:
        lon: Longitudes in decimal degrees.
        lat: Latitudes in decimal degrees.

    Returns:
        n×n symmetric matrix of distances in meters (zero diagonal).

    Raises:
        ValueError: If coordinates are misaligned, nonfinite, or outside valid
            longitude and latitude bounds.
    """
    lon_values = _as_finite_vector(lon, "lon")
    lat_values = _as_finite_vector(lat, "lat")
    if lon_values.shape != lat_values.shape:
        raise ValueError("lon and lat must have the same length")
    if np.any((lon_values < -180) | (lon_values > 180)):
        raise ValueError("longitude must lie between -180 and 180 degrees")
    if np.any((lat_values < -90) | (lat_values > 90)):
        raise ValueError("latitude must lie between -90 and 90 degrees")
    lon_r = np.radians(lon_values)
    lat_r = np.radians(lat_values)
    dlon = lon_r[:, None] - lon_r[None, :]
    dlat = lat_r[:, None] - lat_r[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_r)[:, None] * np.cos(lat_r)[None, :] * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    out: np.ndarray = 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
    return out


def _to_epoch_seconds(timestamps: np.ndarray) -> np.ndarray:
    """Convert timestamps to float epoch seconds.

    Accepts numeric epoch seconds (returned as-is) or anything pandas can
    parse to datetimes (datetime64, Timestamps, ISO strings).
    """
    ts = np.asarray(timestamps)
    if ts.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if np.issubdtype(ts.dtype, np.number):
        result = ts.astype(float)
        if not np.all(np.isfinite(result)):
            raise ValueError("timestamps must contain only finite values")
        return result
    # Force nanosecond resolution before the integer cast: to_numpy() can return
    # datetime64 at us/ms/s depending on the source, which would mis-scale gaps.
    try:
        parsed = pd.to_datetime(ts, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("timestamps could not be parsed as datetimes") from error
    if np.asarray(pd.isna(parsed)).any():
        raise ValueError("timestamps cannot contain missing values")
    ns = parsed.to_numpy().astype("datetime64[ns]").astype("int64")
    out: np.ndarray = ns.astype(float) / 1e9
    return out


def time_gap_matrix(timestamps: np.ndarray) -> np.ndarray:
    """Pairwise absolute time differences in seconds.

    Args:
        timestamps: datetime64 / Timestamps / ISO strings, or numeric epoch
            seconds.

    Returns:
        n×n symmetric matrix of absolute time gaps in seconds (zero diagonal).
    """
    secs = _to_epoch_seconds(timestamps)
    out: np.ndarray = np.abs(secs[:, None] - secs[None, :])
    return out


# ─── Empirical variogram and exponential fit ─────────────────────────


def empirical_variogram(
    values: np.ndarray,
    dist: np.ndarray,
    n_bins: int = 15,
    max_dist: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classical (Matheron) empirical semivariogram over distance bins.

    Semivariance per bin is the mean of ``0.5 * (z_i - z_j)**2`` over unique
    pairs whose distance falls in the bin.

    Args:
        values: Length-n outcome array.
        dist: n×n pairwise-distance matrix (any axis).
        n_bins: Number of distance bins.
        max_dist: Maximum distance to include. Defaults to half the maximum
            pairwise distance (the conventional cutoff).

    Returns:
        (lag_centers, semivariance, pair_counts) for non-empty bins only.

    Raises:
        ValueError: If inputs are malformed or outside their domains.
    """
    values = _as_finite_vector(values, "values")
    n = len(values)
    matrix = _validate_distance_matrix(dist, n)
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if max_dist is not None and (not np.isfinite(max_dist) or max_dist <= 0):
        raise ValueError("max_dist must be finite and positive")
    iu, ju = np.triu_indices(n, k=1)
    d = matrix[iu, ju]
    sq = 0.5 * (values[iu] - values[ju]) ** 2

    if max_dist is None:
        max_dist = float(d.max()) / 2.0 if d.size else 0.0

    keep = (d > 0) & (d <= max_dist)
    d = d[keep]
    sq = sq[keep]
    if d.size == 0:
        empty = np.array([])
        return empty, empty, empty

    edges = np.linspace(0.0, max_dist, n_bins + 1)
    idx = np.clip(np.digitize(d, edges, right=True) - 1, 0, n_bins - 1)

    lags = np.full(n_bins, np.nan)
    gamma = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins)
    for b in range(n_bins):
        sel = idx == b
        c = int(sel.sum())
        counts[b] = c
        if c > 0:
            lags[b] = float(d[sel].mean())
            gamma[b] = float(sq[sel].mean())

    nonempty = counts > 0
    return lags[nonempty], gamma[nonempty], counts[nonempty]


def _exponential_model(
    h: np.ndarray, nugget: float, partial_sill: float, rng_: float
) -> np.ndarray:
    """Exponential variogram: nugget + partial_sill * (1 - exp(-h/range))."""
    return nugget + partial_sill * (1.0 - np.exp(-h / rng_))


def fit_variogram(
    lags: np.ndarray,
    gamma: np.ndarray,
    counts: np.ndarray,
    model: str = "exponential",
) -> tuple[float, float, float]:
    """Weighted least-squares fit of an exponential variogram.

    The fitted correlation function is ``rho(h) = ((C1-C0)/C1) * exp(-h/r)``,
    so ``r`` is the e-folding scale (correlation falls to 1/e at h = r); the
    conventional "effective range" is ~3r.

    Args:
        lags: Bin-center distances from ``empirical_variogram``.
        gamma: Semivariances from ``empirical_variogram``.
        counts: Pair counts per bin (used as fit weights).
        model: Only ``"exponential"`` is supported.

    Returns:
        (nugget C0, sill C1, range r). NaNs if the fit cannot be performed.

    Raises:
        ValueError: If ``model`` names anything but ``"exponential"``.
    """
    if model != "exponential":
        raise ValueError(f"Unsupported variogram model: {model!r}")

    lags = _as_finite_vector(lags, "lags")
    gamma = _as_finite_vector(gamma, "gamma")
    counts = _as_finite_vector(counts, "counts")
    if not (lags.shape == gamma.shape == counts.shape):
        raise ValueError("lags, gamma, and counts must have the same length")
    if np.any(lags < 0) or np.any(gamma < 0) or np.any(counts <= 0):
        raise ValueError("lags and gamma must be nonnegative and counts positive")
    nan = (float("nan"), float("nan"), float("nan"))
    if lags.size < 3:
        return nan

    gmax = float(np.nanmax(gamma))
    lmax = float(np.nanmax(lags))
    if not np.isfinite(gmax) or gmax <= 0 or lmax <= 0:
        return nan

    p0 = [0.0, gmax, lmax / 3.0]
    bounds = ([0.0, 0.0, 1e-9], [gmax + 1e-12, 5.0 * gmax + 1e-9, 10.0 * lmax])
    sigma = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    try:
        popt, _ = optimize.curve_fit(
            _exponential_model,
            lags,
            gamma,
            p0=p0,
            sigma=sigma,
            bounds=bounds,
            maxfev=10_000,
        )
    except (RuntimeError, ValueError):
        return nan

    nugget = float(popt[0])
    sill = float(popt[0] + popt[1])  # C1 = nugget + partial sill
    rng_ = float(popt[2])
    return nugget, sill, rng_


# ─── Moran's I and effective sample size ──────────────────────────────


def morans_i(
    values: np.ndarray,
    dist: np.ndarray,
    cutoff: float,
    n_perm: int = 999,
    seed: int = 0,
) -> tuple[float, float]:
    """Global Moran's I with binary distance-cutoff weights.

    Weight ``w_ij = 1`` if ``0 < dist_ij <= cutoff`` else 0. The p-value is a
    two-sided permutation test on the observed statistic.

    Args:
        values: Length-n outcome array.
        dist: n×n pairwise-distance matrix.
        cutoff: Neighbor distance threshold (same units as ``dist``).
        n_perm: Number of permutations for the p-value.
        seed: RNG seed for the permutation test.

    Returns:
        (I, p_value). NaNs if undefined (e.g. no neighbor pairs).

    Raises:
        TypeError: If ``seed`` is not an integer.
        ValueError: If inputs are malformed or outside their domains.
    """
    values = _as_finite_vector(values, "values")
    n = len(values)
    matrix = _validate_distance_matrix(dist, n)
    if n < 3:
        raise ValueError("Moran's I requires at least three values")
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")
    if isinstance(n_perm, bool) or not isinstance(n_perm, int) or n_perm < 1:
        raise ValueError("n_perm must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    w = ((matrix > 0) & (matrix <= cutoff)).astype(float)
    np.fill_diagonal(w, 0.0)
    s0 = w.sum()
    if s0 == 0:
        return float("nan"), float("nan")

    z = values - values.mean()
    denom = float(np.sum(z**2))
    if denom == 0:
        return float("nan"), float("nan")

    def _stat(zv: np.ndarray) -> float:
        return float((n / s0) * float(zv @ w @ zv) / denom)

    obs = _stat(z)

    rng = np.random.default_rng(seed)
    simulated = np.array([_stat(rng.permutation(z)) for _ in range(n_perm)])
    null_mean = float(np.mean(simulated))
    extreme = np.count_nonzero(np.abs(simulated - null_mean) >= abs(obs - null_mean))
    return float(obs), float((extreme + 1) / (n_perm + 1))


def effective_n(
    values: np.ndarray,
    dist: np.ndarray,
    nugget: float,
    sill: float,
    range_: float,
) -> float:
    """Variogram-based effective sample size under autocorrelation.

    ``n_eff = n / (1 + (1/n) * sum_{i!=j} rho(d_ij))`` with the exponential
    correlation ``rho(d) = ((sill - nugget)/sill) * exp(-d/range_)`` (Griffith
    2005; Watson 2021). Reduces to ``n`` when there is no autocorrelation.

    Args:
        values: The observations, used only for their count.
        dist: Square pairwise distance matrix over those observations.
        nugget: Variogram nugget C0.
        sill: Variogram sill C1.
        range_: Variogram e-folding range r.

    Returns:
        Effective sample size (<= n), or n if the fit was degenerate.

    Raises:
        ValueError: If the distance matrix or variogram parameters are outside
            their supported domains.
    """
    values = _as_finite_vector(values, "values")
    n = len(values)
    matrix = _validate_distance_matrix(dist, n)
    if n < 2:
        return float(n)
    if not all(np.isfinite(value) for value in (nugget, sill, range_)):
        raise ValueError("nugget, sill, and range_ must be finite")
    if nugget < 0 or sill <= 0 or nugget > sill or range_ <= 0:
        raise ValueError("require 0 <= nugget <= sill, sill > 0, and range_ > 0")

    corr_ratio = (sill - nugget) / sill
    if not np.isfinite(corr_ratio) or corr_ratio <= 0:
        return float(n)

    rho = corr_ratio * np.exp(-matrix / range_)
    np.fill_diagonal(rho, 0.0)
    deff = 1.0 + float(rho.sum()) / n
    if deff <= 0:
        return float(n)
    return float(n / deff)


def within_between_contrast(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Mean semivariance of same-cluster vs different-cluster pairs.

    This is the "does the prior point predict the next one, versus a point
    elsewhere?" contrast. ``ratio = within / between`` near 1 means
    itineraries look like representative subsamples (dispersed); ``ratio``
    well below 1 means within-itinerary pairs are much more alike (compact
    routes), which inflates the design effect.

    Args:
        values: One value per observation.
        labels: Cluster label per observation, aligned to ``values``.

    Returns:
        ``{"within", "between", "ratio"}`` (semivariances; NaN where undefined).

    Raises:
        ValueError: If values and labels are malformed or misaligned.
    """
    values = _as_finite_vector(values, "values")
    labels = np.asarray(labels)
    n = len(values)
    if labels.ndim != 1 or len(labels) != n:
        raise ValueError("labels must be one-dimensional and aligned with values")
    if pd.isna(labels).any():
        raise ValueError("labels cannot contain missing values")
    nan = {"within": float("nan"), "between": float("nan"), "ratio": float("nan")}
    if n < 2:
        return nan

    iu, ju = np.triu_indices(n, k=1)
    sq = 0.5 * (values[iu] - values[ju]) ** 2
    same = labels[iu] == labels[ju]

    within = float(sq[same].mean()) if np.any(same) else float("nan")
    between = float(sq[~same].mean()) if np.any(~same) else float("nan")
    ratio = (
        within / between
        if np.isfinite(between) and between > 0 and np.isfinite(within)
        else float("nan")
    )
    return {"within": within, "between": between, "ratio": ratio}


def _axis_diagnostics(
    values: np.ndarray, dist: np.ndarray, seed: int
) -> tuple[float, float, float, float, float]:
    """Compute variogram, Moran, and effective-size diagnostics for one axis."""
    lags, semivariance, counts = empirical_variogram(values, dist)
    nugget, sill, range_value = fit_variogram(lags, semivariance, counts)
    corr_ratio = (
        (sill - nugget) / sill if np.isfinite(sill) and sill > 0 else float("nan")
    )
    effective = (
        effective_n(values, dist, nugget, sill, range_value)
        if all(np.isfinite(value) for value in (nugget, sill, range_value))
        and 0 <= nugget <= sill
        and sill > 0
        and range_value > 0
        else float(len(values))
    )
    positive = dist[dist > 0]
    if positive.size == 0:
        return float("nan"), float("nan"), range_value, corr_ratio, effective
    cutoff = (
        range_value
        if np.isfinite(range_value) and range_value > 0
        else float(np.median(positive))
    )
    moran, p_value = morans_i(values, dist, cutoff, seed=seed)
    return moran, p_value, range_value, corr_ratio, effective


def dependence_diagnostics(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    lon: np.ndarray | None = None,
    lat: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    max_points: int = 2500,
    seed: int = 0,
) -> DependenceDiagnostics:
    """Compute experimental within-cluster dependence diagnostics.

    Args:
        values: One finite outcome per observation.
        labels: Cluster labels aligned with ``values``.
        lon: Optional longitudes in decimal degrees.
        lat: Optional latitudes in decimal degrees.
        timestamps: Optional datetimes or epoch seconds.
        max_points: Maximum observations used for pairwise diagnostics.
        seed: Random seed used for subsampling and permutations.

    Returns:
        Spatial and temporal descriptive diagnostics.

    Raises:
        TypeError: If ``seed`` is not an integer.
        ValueError: If arrays are misaligned, incomplete, or outside their
            documented domains.
    """
    outcomes = _as_finite_vector(values, "values")
    cluster_labels = np.asarray(labels)
    if cluster_labels.ndim != 1 or len(cluster_labels) != len(outcomes):
        raise ValueError("labels must be one-dimensional and aligned with values")
    if len(outcomes) < 3:
        raise ValueError("dependence diagnostics require at least three observations")
    if pd.isna(cluster_labels).any():
        raise ValueError("labels cannot contain missing values")
    if (lon is None) != (lat is None):
        raise ValueError("lon and lat must be supplied together")
    if (
        isinstance(max_points, bool)
        or not isinstance(max_points, int)
        or max_points < 3
    ):
        raise ValueError("max_points must be an integer of at least 3")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    coordinate_lon = None if lon is None else _as_finite_vector(lon, "lon")
    coordinate_lat = None if lat is None else _as_finite_vector(lat, "lat")
    time_values = None if timestamps is None else np.asarray(timestamps)
    for name, array in (
        ("lon", coordinate_lon),
        ("lat", coordinate_lat),
        ("timestamps", time_values),
    ):
        if array is not None and len(array) != len(outcomes):
            raise ValueError(f"{name} must be aligned with values")

    selected = np.arange(len(outcomes))
    if len(outcomes) > max_points:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(outcomes), size=max_points, replace=False))
        warnings.warn(
            f"dependence diagnostics used {max_points} of {len(outcomes)} observations",
            UserWarning,
            stacklevel=2,
        )
    outcomes = outcomes[selected]
    cluster_labels = cluster_labels[selected]
    contrast = within_between_contrast(outcomes, cluster_labels)
    result: dict[str, float] = {"within_between_ratio": contrast["ratio"]}

    if coordinate_lon is not None and coordinate_lat is not None:
        space = _axis_diagnostics(
            outcomes,
            haversine_matrix(coordinate_lon[selected], coordinate_lat[selected]),
            seed,
        )
        result.update(
            morans_i_space=space[0],
            morans_i_space_p=space[1],
            variogram_range_m=space[2],
            spatial_corr_ratio=space[3],
            n_eff_space=space[4],
        )
    if time_values is not None:
        time = _axis_diagnostics(outcomes, time_gap_matrix(time_values[selected]), seed)
        result.update(
            morans_i_time=time[0],
            morans_i_time_p=time[1],
            variogram_range_s=time[2],
            temporal_corr_ratio=time[3],
            n_eff_time=time[4],
        )
    return DependenceDiagnostics(**result)
