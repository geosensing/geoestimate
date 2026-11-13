"""Point estimates and uncertainty, with variance estimation delegated to ``svy``."""

from dataclasses import dataclass, replace
from numbers import Real
from typing import Literal

import numpy as np
import polars as pl
import svy
from scipy import stats as sp_stats

from .types import Diagnostics, Estimate, InferenceMethod, ResolvedInferenceMethod


@dataclass(frozen=True, slots=True)
class _Estimand:
    """What to estimate: a weighted mean, a ratio of totals, or a total.

    ``scale`` multiplies a mean into a total when the population size is known.
    """

    name: str
    variables: tuple[str, ...]
    kind: Literal["mean", "ratio", "total"]
    values: np.ndarray
    denominator: np.ndarray | None = None
    scale: float = 1.0
    population_size: int | None = None


@dataclass(frozen=True, slots=True)
class _Design:
    """Row-level design information shared by every estimand.

    ``units`` labels each row's primary sampling unit: its cluster when one is
    declared, otherwise the row itself. ``cells`` and ``cell_shares`` are set
    when the sample is post-stratified: each row's cell and each cell's known
    population share.
    """

    weights: np.ndarray
    units: np.ndarray
    n_units: int
    row_strata: np.ndarray
    n_strata: int
    has_clusters: bool
    cells: np.ndarray | None = None
    cell_shares: np.ndarray | None = None


def _post_stratified_total(estimand: _Estimand, design: _Design) -> _Estimand:
    """Rewrite a post-stratified total as a plain total of its linearized values.

    Shares fix the mix of cells, not the population size, so the total is the
    estimated size times the post-stratified mean. Its exact first derivative
    is the value below; ``svy`` substitutes regression residuals, which agree
    only when the sample's mix is already close to the shares.
    """
    weights, cells, shares = design.weights, design.cells, design.cell_shares
    if cells is None or shares is None:
        raise RuntimeError("the design is not post-stratified")
    cell_weight = np.bincount(cells, weights=weights)
    cell_mean = np.bincount(cells, weights=weights * estimand.values) / cell_weight
    factor = shares * weights.sum() / cell_weight
    mean = float(np.sum(shares * cell_mean))
    values = mean + factor[cells] * (estimand.values - cell_mean[cells])
    return replace(estimand, values=values)


def _linearized(
    estimand: _Estimand, design: _Design, *, clustered: bool
) -> tuple[float, float]:
    """Return the estimate and its Taylor-linearization standard error."""
    if estimand.kind == "total" and design.cells is not None:
        estimand = _post_stratified_total(estimand, design)
        design = replace(design, cells=None, cell_shares=None)
    columns = {
        "weight": design.weights,
        "unit": design.units,
        "stratum": design.row_strata,
        "values": estimand.values,
    }
    if estimand.denominator is not None:
        columns["denominator"] = estimand.denominator
    if design.cells is not None:
        columns["cell"] = design.cells
    sample = svy.Sample(
        pl.DataFrame(columns),
        svy.Design(
            wgt="weight",
            psu="unit" if clustered else None,
            stratum="stratum" if design.n_strata > 1 else None,
        ),
    )
    if design.cells is not None and design.cell_shares is not None:
        sample = sample.weighting.poststratify(
            shares=dict(enumerate(design.cell_shares.tolist())), cells="cell"
        )
    estimation = sample.estimation
    if estimand.kind == "ratio":
        result = estimation.ratio("values", "denominator")
    elif estimand.kind == "total":
        result = estimation.total("values")
    else:
        result = estimation.mean("values")
    if isinstance(result, list):
        raise TypeError("svy returned several estimates for one variable")
    row = result.to_polars().row(0, named=True)
    return estimand.scale * float(row["est"]), estimand.scale * float(row["se"])


def _evaluate(estimand: _Estimand, weights: np.ndarray, rows: np.ndarray) -> float:
    """Point estimate on a resampled set of rows."""
    draw_weights = weights[rows]
    numerator = float(np.sum(draw_weights * estimand.values[rows]))
    if estimand.kind == "total":
        return estimand.scale * numerator
    if estimand.denominator is None:
        denominator = float(np.sum(draw_weights))
    else:
        denominator = float(np.sum(draw_weights * estimand.denominator[rows]))
    if denominator <= 0:
        return float("nan")
    return estimand.scale * numerator / denominator


def _bootstrap_distribution(
    estimand: _Estimand,
    design: _Design,
    reps: int,
    seed: int | None,
) -> np.ndarray:
    """Return exactly ``reps`` valid pairs-bootstrap estimates."""
    rng = np.random.default_rng(seed)
    unit_rows = [np.flatnonzero(design.units == unit) for unit in range(design.n_units)]
    estimates: list[float] = []
    attempts = 0
    max_attempts = 2 * reps
    while len(estimates) < reps and attempts < max_attempts:
        draw = rng.integers(0, design.n_units, size=design.n_units)
        rows = np.concatenate([unit_rows[unit] for unit in draw])
        value = _evaluate(estimand, design.weights, rows)
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


def _diagnostics(estimand: _Estimand, design: _Design) -> tuple[float, Diagnostics]:
    """Return the point estimate with the analytic diagnostics."""
    n_rows = len(design.units)
    estimate, iid_se = _linearized(estimand, design, clustered=False)
    if not design.has_clusters:
        return estimate, Diagnostics(
            iid_standard_error=iid_se,
            cluster_standard_error=None,
            design_effect=None,
            effective_sample_size=float(n_rows),
            cluster_sizes=None,
            effective_cluster_count=None,
        )

    _, cluster_se = _linearized(estimand, design, clustered=True)
    design_effect = None
    effective_sample_size = float(n_rows)
    if iid_se > 0:
        design_effect = (cluster_se / iid_se) ** 2
        if design_effect > 0:
            effective_sample_size /= design_effect
    sizes_array = np.bincount(design.units, minlength=design.n_units)
    effective_clusters = float(sizes_array.sum() ** 2 / np.sum(sizes_array**2))
    return estimate, Diagnostics(
        iid_standard_error=iid_se,
        cluster_standard_error=cluster_se,
        design_effect=design_effect,
        effective_sample_size=effective_sample_size,
        cluster_sizes=tuple(int(size) for size in sizes_array),
        effective_cluster_count=effective_clusters,
    )


def _run_inference(
    estimand: _Estimand,
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
    estimate, diagnostics = _diagnostics(estimand, design)
    alpha = 1 - confidence_level

    if resolved == "bootstrap":
        if design.cells is not None:
            raise ValueError(
                "bootstrap inference is not available for post-stratified samples"
            )
        if design.n_strata > 1:
            # Resampling within strata understates variance when strata hold
            # few units, unless the weights are rescaled draw by draw.
            raise ValueError(
                "bootstrap inference is not available for stratified samples"
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
            estimate - critical_value * standard_error,
            estimate + critical_value * standard_error,
        )

    return Estimate(
        estimand=estimand.name,
        variables=estimand.variables,
        estimate=estimate,
        standard_error=standard_error,
        confidence_interval=interval,
        confidence_level=confidence_level,
        inference_method=resolved,
        n_observations=len(design.units),
        n_clusters=design.n_units if design.has_clusters else None,
        population_size=estimand.population_size,
        diagnostics=diagnostics,
    )
