"""Shared point-estimation and uncertainty machinery."""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy import stats as sp_stats

from .types import Diagnostics, Estimate, InferenceMethod, ResolvedInferenceMethod


@dataclass(frozen=True, slots=True)
class _PreparedEstimand:
    """Point estimate and linearized contributions for one estimand."""

    name: str
    variables: tuple[str, ...]
    estimate: float
    contributions: np.ndarray
    evaluate: Callable[[np.ndarray], float]
    population_size: int | None = None


def _prepare_mean(values: np.ndarray, variable: str) -> _PreparedEstimand:
    estimate = float(np.mean(values))
    n = len(values)
    return _PreparedEstimand(
        name="mean",
        variables=(variable,),
        estimate=estimate,
        contributions=(values - estimate) / n,
        evaluate=lambda rows: float(np.mean(values[rows])),
    )


def _prepare_total(
    values: np.ndarray, variable: str, population_size: int
) -> _PreparedEstimand:
    mean = float(np.mean(values))
    n = len(values)
    return _PreparedEstimand(
        name="total",
        variables=(variable,),
        estimate=population_size * mean,
        contributions=population_size * (values - mean) / n,
        evaluate=lambda rows: float(population_size * np.mean(values[rows])),
        population_size=population_size,
    )


def _prepare_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    numerator_name: str,
    denominator_name: str,
) -> _PreparedEstimand:
    denominator_total = float(np.sum(denominator))
    estimate = float(np.sum(numerator) / denominator_total)

    def evaluate(rows: np.ndarray) -> float:
        draw_denominator = float(np.sum(denominator[rows]))
        if draw_denominator <= 0:
            return float("nan")
        return float(np.sum(numerator[rows]) / draw_denominator)

    return _PreparedEstimand(
        name="ratio",
        variables=(numerator_name, denominator_name),
        estimate=estimate,
        contributions=(numerator - estimate * denominator) / denominator_total,
        evaluate=evaluate,
    )


def _prepare_mean_of_ratios(
    numerator: np.ndarray,
    denominator: np.ndarray,
    numerator_name: str,
    denominator_name: str,
) -> _PreparedEstimand:
    ratios = numerator / denominator
    estimate = float(np.mean(ratios))
    n = len(ratios)
    return _PreparedEstimand(
        name="mean_of_ratios",
        variables=(numerator_name, denominator_name),
        estimate=estimate,
        contributions=(ratios - estimate) / n,
        evaluate=lambda rows: float(np.mean(ratios[rows])),
    )


def _iid_standard_error(contributions: np.ndarray) -> float:
    """Return the finite-sample iid standard error."""
    n = len(contributions)
    return float(np.sqrt(n / (n - 1) * np.sum(contributions**2)))


def _cluster_standard_error(
    contributions: np.ndarray, labels: np.ndarray, n_clusters: int
) -> float:
    """Return the cluster-sandwich standard error."""
    cluster_contributions = np.bincount(
        labels, weights=contributions, minlength=n_clusters
    )
    return float(
        np.sqrt(n_clusters / (n_clusters - 1) * np.sum(cluster_contributions**2))
    )


def _bootstrap_distribution(
    estimand: _PreparedEstimand,
    labels: np.ndarray,
    n_units: int,
    reps: int,
    seed: int | None,
) -> np.ndarray:
    """Return exactly ``reps`` valid pairs-bootstrap estimates."""
    rng = np.random.default_rng(seed)
    unit_rows = [np.flatnonzero(labels == unit) for unit in range(n_units)]
    estimates: list[float] = []
    attempts = 0
    max_attempts = 2 * reps
    while len(estimates) < reps and attempts < max_attempts:
        draw = rng.integers(0, n_units, size=n_units)
        rows = np.concatenate([unit_rows[unit] for unit in draw])
        value = estimand.evaluate(rows)
        if np.isfinite(value):
            estimates.append(value)
        attempts += 1
    if len(estimates) < reps:
        raise ValueError(
            "fewer than half the bootstrap draws were defined; the sample has "
            "too little positive denominator information"
        )
    return np.asarray(estimates, dtype=float)


def _validate_options(
    inference: InferenceMethod,
    confidence_level: float,
    bootstrap_reps: int,
    seed: int | None,
    has_clusters: bool,
) -> ResolvedInferenceMethod:
    methods = {"design", "iid", "cluster", "bootstrap"}
    if not isinstance(inference, str) or inference not in methods:
        raise ValueError(
            f"inference must be one of {sorted(methods)}, got {inference!r}"
        )
    if isinstance(confidence_level, bool) or not isinstance(confidence_level, Real):
        raise TypeError("confidence_level must be a real number")
    if not np.isfinite(confidence_level):
        raise ValueError("confidence_level must be finite and strictly between 0 and 1")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be finite and strictly between 0 and 1")
    if isinstance(bootstrap_reps, bool) or not isinstance(bootstrap_reps, int):
        raise TypeError("bootstrap_reps must be an integer")
    if bootstrap_reps < 2:
        raise ValueError("bootstrap_reps must be at least 2")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
    if inference == "cluster" and not has_clusters:
        raise ValueError("cluster inference requires a declared cluster column")
    if inference == "design":
        return "cluster" if has_clusters else "iid"
    return inference


def _diagnostics(
    contributions: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    has_clusters: bool,
) -> Diagnostics:
    iid_se = _iid_standard_error(contributions)
    if not has_clusters:
        return Diagnostics(
            iid_standard_error=iid_se,
            cluster_standard_error=None,
            design_effect=None,
            effective_sample_size=float(len(contributions)),
            cluster_sizes=None,
            effective_cluster_count=None,
        )

    cluster_se = _cluster_standard_error(contributions, labels, n_clusters)
    design_effect = None
    effective_sample_size = float(len(contributions))
    if iid_se > 0:
        design_effect = (cluster_se / iid_se) ** 2
        if design_effect > 0:
            effective_sample_size /= design_effect
    sizes_array = np.bincount(labels, minlength=n_clusters)
    effective_clusters = float(sizes_array.sum() ** 2 / np.sum(sizes_array**2))
    return Diagnostics(
        iid_standard_error=iid_se,
        cluster_standard_error=cluster_se,
        design_effect=design_effect,
        effective_sample_size=effective_sample_size,
        cluster_sizes=tuple(int(size) for size in sizes_array),
        effective_cluster_count=effective_clusters,
    )


def _run_inference(
    estimand: _PreparedEstimand,
    labels: np.ndarray,
    n_clusters: int,
    has_clusters: bool,
    *,
    inference: InferenceMethod,
    confidence_level: float,
    bootstrap_reps: int,
    seed: int | None,
) -> Estimate:
    """Compute uncertainty and construct a public result."""
    resolved = _validate_options(
        inference, confidence_level, bootstrap_reps, seed, has_clusters
    )
    diagnostics = _diagnostics(estimand.contributions, labels, n_clusters, has_clusters)
    alpha = 1 - confidence_level

    if resolved == "bootstrap":
        draws = _bootstrap_distribution(
            estimand, labels, n_clusters, bootstrap_reps, seed
        )
        standard_error = float(np.std(draws, ddof=1))
        quantiles = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
        interval = (float(quantiles[0]), float(quantiles[1]))
    else:
        if resolved == "cluster":
            cluster_se = diagnostics.cluster_standard_error
            if cluster_se is None:
                raise RuntimeError("cluster standard error was not computed")
            standard_error = cluster_se
            critical_value = float(sp_stats.t.ppf(1 - alpha / 2, df=n_clusters - 1))
        else:
            standard_error = diagnostics.iid_standard_error
            critical_value = float(sp_stats.norm.ppf(1 - alpha / 2))
        interval = (
            estimand.estimate - critical_value * standard_error,
            estimand.estimate + critical_value * standard_error,
        )

    return Estimate(
        estimand=estimand.name,
        variables=estimand.variables,
        estimate=estimand.estimate,
        standard_error=standard_error,
        confidence_interval=interval,
        confidence_level=confidence_level,
        inference_method=resolved,
        n_observations=len(labels),
        n_clusters=n_clusters if has_clusters else None,
        population_size=estimand.population_size,
        diagnostics=diagnostics,
    )
