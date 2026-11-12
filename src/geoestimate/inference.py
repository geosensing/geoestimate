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


@dataclass(frozen=True, slots=True)
class _Design:
    """Row-level design information shared by every estimand.

    ``units`` labels each row's primary sampling unit: its cluster when one is
    declared, otherwise the row itself. ``unit_strata`` gives the stratum of each
    unit. ``calibration_cells`` labels post-strata when the weights were
    calibrated to known shares.
    """

    weights: np.ndarray
    units: np.ndarray
    n_units: int
    unit_strata: np.ndarray
    row_strata: np.ndarray
    n_strata: int
    has_clusters: bool
    calibration_cells: np.ndarray | None = None


def _prepare_mean(
    values: np.ndarray, variable: str, weights: np.ndarray, name: str = "mean"
) -> _PreparedEstimand:
    weight_total = float(np.sum(weights))
    estimate = float(np.sum(weights * values) / weight_total)

    def evaluate(rows: np.ndarray) -> float:
        draw_weights = weights[rows]
        return float(np.sum(draw_weights * values[rows]) / np.sum(draw_weights))

    return _PreparedEstimand(
        name=name,
        variables=(variable,),
        estimate=estimate,
        contributions=weights * (values - estimate) / weight_total,
        evaluate=evaluate,
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


def _prepare_weighted_total(
    values: np.ndarray, variable: str, weights: np.ndarray
) -> _PreparedEstimand:
    """Horvitz-Thompson total: weights are inverse inclusion probabilities."""
    return _PreparedEstimand(
        name="total",
        variables=(variable,),
        estimate=float(np.sum(weights * values)),
        contributions=weights * values,
        evaluate=lambda rows: float(np.sum(weights[rows] * values[rows])),
    )


def _prepare_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    numerator_name: str,
    denominator_name: str,
    weights: np.ndarray,
) -> _PreparedEstimand:
    denominator_total = float(np.sum(weights * denominator))
    estimate = float(np.sum(weights * numerator) / denominator_total)

    def evaluate(rows: np.ndarray) -> float:
        draw_weights = weights[rows]
        draw_denominator = float(np.sum(draw_weights * denominator[rows]))
        if draw_denominator <= 0:
            return float("nan")
        return float(np.sum(draw_weights * numerator[rows]) / draw_denominator)

    return _PreparedEstimand(
        name="ratio",
        variables=(numerator_name, denominator_name),
        estimate=estimate,
        contributions=weights
        * (numerator - estimate * denominator)
        / denominator_total,
        evaluate=evaluate,
    )


def _prepare_mean_of_ratios(
    numerator: np.ndarray,
    denominator: np.ndarray,
    numerator_name: str,
    denominator_name: str,
    weights: np.ndarray,
) -> _PreparedEstimand:
    ratios = numerator / denominator
    weight_total = float(np.sum(weights))
    estimate = float(np.sum(weights * ratios) / weight_total)

    def evaluate(rows: np.ndarray) -> float:
        draw_weights = weights[rows]
        return float(np.sum(draw_weights * ratios[rows]) / np.sum(draw_weights))

    return _PreparedEstimand(
        name="mean_of_ratios",
        variables=(numerator_name, denominator_name),
        estimate=estimate,
        contributions=weights * (ratios - estimate) / weight_total,
        evaluate=evaluate,
    )


def _calibrated_contributions(contributions: np.ndarray, design: _Design) -> np.ndarray:
    """Remove what post-stratification already fixes.

    Calibrating weights to known cell shares removes between-cell sampling
    variation, so only each row's departure from its cell's weighted mean
    contributes to the variance.
    """
    cells = design.calibration_cells
    if cells is None:
        return contributions
    cell_contribution = np.bincount(cells, weights=contributions)
    cell_weight = np.bincount(cells, weights=design.weights)
    return contributions - design.weights * (cell_contribution / cell_weight)[cells]


def _standard_error(
    contributions: np.ndarray,
    units: np.ndarray,
    n_units: int,
    unit_strata: np.ndarray,
    n_strata: int,
) -> float:
    """Return the with-replacement linearization standard error.

    Unit totals are centered within strata, so strata contribute no
    between-stratum variation.
    """
    totals = np.bincount(units, weights=contributions, minlength=n_units)
    counts = np.bincount(unit_strata, minlength=n_strata)
    means = np.bincount(unit_strata, weights=totals, minlength=n_strata) / counts
    centered = totals - means[unit_strata]
    scale = counts / (counts - 1)
    return float(np.sqrt(np.sum(scale[unit_strata] * centered**2)))


def _bootstrap_distribution(
    estimand: _PreparedEstimand,
    design: _Design,
    reps: int,
    seed: int | None,
) -> np.ndarray:
    """Return exactly ``reps`` valid pairs-bootstrap estimates.

    Units are resampled with replacement within each stratum.
    """
    rng = np.random.default_rng(seed)
    unit_rows = [np.flatnonzero(design.units == unit) for unit in range(design.n_units)]
    stratum_units = [
        np.flatnonzero(design.unit_strata == stratum)
        for stratum in range(design.n_strata)
    ]
    estimates: list[float] = []
    attempts = 0
    max_attempts = 2 * reps
    while len(estimates) < reps and attempts < max_attempts:
        draw = np.concatenate(
            [rng.choice(units, size=len(units)) for units in stratum_units]
        )
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


def _diagnostics(contributions: np.ndarray, design: _Design) -> Diagnostics:
    rows = np.arange(len(contributions))
    iid_se = _standard_error(
        contributions, rows, len(rows), design.row_strata, design.n_strata
    )
    if not design.has_clusters:
        return Diagnostics(
            iid_standard_error=iid_se,
            cluster_standard_error=None,
            design_effect=None,
            effective_sample_size=float(len(contributions)),
            cluster_sizes=None,
            effective_cluster_count=None,
        )

    cluster_se = _standard_error(
        contributions,
        design.units,
        design.n_units,
        design.unit_strata,
        design.n_strata,
    )
    design_effect = None
    effective_sample_size = float(len(contributions))
    if iid_se > 0:
        design_effect = (cluster_se / iid_se) ** 2
        if design_effect > 0:
            effective_sample_size /= design_effect
    sizes_array = np.bincount(design.units, minlength=design.n_units)
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
    design: _Design,
    *,
    inference: InferenceMethod,
    confidence_level: float,
    bootstrap_reps: int,
    seed: int | None,
) -> Estimate:
    """Compute uncertainty and construct a public result."""
    resolved = _validate_options(
        inference, confidence_level, bootstrap_reps, seed, design.has_clusters
    )
    contributions = _calibrated_contributions(estimand.contributions, design)
    diagnostics = _diagnostics(contributions, design)
    alpha = 1 - confidence_level

    if resolved == "bootstrap":
        if design.calibration_cells is not None:
            raise ValueError(
                "bootstrap inference is not available for post-stratified samples"
            )
        draws = _bootstrap_distribution(estimand, design, bootstrap_reps, seed)
        standard_error = float(np.std(draws, ddof=1))
        quantiles = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
        interval = (float(quantiles[0]), float(quantiles[1]))
    else:
        if resolved == "cluster":
            cluster_se = diagnostics.cluster_standard_error
            if cluster_se is None:
                raise RuntimeError("cluster standard error was not computed")
            standard_error = cluster_se
            degrees = design.n_units - design.n_strata
            critical_value = float(sp_stats.t.ppf(1 - alpha / 2, df=degrees))
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
        n_observations=len(design.units),
        n_clusters=design.n_units if design.has_clusters else None,
        population_size=estimand.population_size,
        diagnostics=diagnostics,
    )
