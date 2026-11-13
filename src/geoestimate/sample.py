"""Sample declaration and estimation methods."""

from collections.abc import Mapping
from copy import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .inference import _Design, _Estimand, _run_inference
from .types import Estimate, InferenceMethod


class Sample:
    """Bind observations to a sampling design.

    Notes:
        ``Sample`` supports design weights, strata, clusters, and
        post-stratification. Linearized standard errors come from ``svy``;
        the bootstrap resamples primary units. Variance
        treats primary units as drawn with replacement, so it applies no finite
        population correction. It does not implement PPS or GRTS variance
        estimation.
    """

    __slots__ = ("_cluster", "_data", "_design", "_strata", "_weights")

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        cluster: str | None = None,
        weights: str | None = None,
        strata: str | None = None,
    ) -> None:
        """Create a validated snapshot of the sample.

        Args:
            data: One row per observation.
            cluster: Optional column identifying primary sampling or collection
                units, such as itineraries, walks, or fieldwork days.
            weights: Optional column of positive weights. For a probability
                sample these are inverse inclusion probabilities.
            strata: Optional column identifying design strata. Clusters are
                treated as nested within strata.

        Raises:
            TypeError: If data is not a pandas DataFrame.
            ValueError: If the sample, cluster, weights, or strata declaration
                is invalid.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if len(data) < 2:
            raise ValueError("data must contain at least two rows")

        copied = data.copy(deep=True).reset_index(drop=True)
        for role, column in (
            ("cluster", cluster),
            ("weights", weights),
            ("strata", strata),
        ):
            self._check_column(copied, role, column)

        if weights is None:
            weight_values = np.ones(len(copied), dtype=float)
        else:
            try:
                weight_values = copied[weights].to_numpy(dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(f"column {weights!r} must be numeric") from error
            if not np.all(np.isfinite(weight_values)) or np.any(weight_values <= 0):
                raise ValueError("weights must be finite and positive")

        if strata is None:
            row_strata = np.zeros(len(copied), dtype=int)
            n_strata = 1
        else:
            row_strata, unique_strata = pd.factorize(copied[strata], sort=False)
            n_strata = len(unique_strata)

        if cluster is None:
            units = np.arange(len(copied), dtype=int)
            n_units = len(copied)
            unit_strata = row_strata
        else:
            # The same cluster label in two strata names two different units.
            keys = pd.MultiIndex.from_arrays([row_strata, copied[cluster]])
            units, unique_units = pd.factorize(keys, sort=False)
            n_units = len(unique_units)
            unit_strata = np.zeros(n_units, dtype=int)
            unit_strata[units] = row_strata

        if np.any(np.bincount(unit_strata, minlength=n_strata) < 2):
            unit_name = "clusters" if cluster is not None else "rows"
            scope = "every stratum" if strata is not None else "the sample"
            raise ValueError(f"{scope} must contain at least two {unit_name}")

        self._data = copied
        self._cluster = cluster
        self._weights = weights
        self._strata = strata
        self._design = _Design(
            weights=weight_values,
            units=units.astype(int),
            n_units=n_units,
            row_strata=row_strata.astype(int),
            n_strata=n_strata,
            has_clusters=cluster is not None,
        )

    @staticmethod
    def _check_column(data: pd.DataFrame, role: str, column: str | None) -> None:
        if column is None:
            return
        if not isinstance(column, str) or not column.strip():
            raise ValueError(f"{role} must be a non-empty column name or None")
        if column not in data:
            raise ValueError(
                f"{role} column {column!r} not found; "
                f"available columns: {list(data.columns)}"
            )
        if data[column].isna().any():
            raise ValueError(f"{role} column {column!r} cannot contain missing values")

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        cluster: str | None = None,
        weights: str | None = None,
        strata: str | None = None,
    ) -> "Sample":
        """Read a Parquet, CSV, or TSV file and construct a sample."""
        from .io import _read_table

        return cls(_read_table(path), cluster=cluster, weights=weights, strata=strata)

    def post_stratify(self, by: str, shares: Mapping[object, float]) -> "Sample":
        """Rescale weights so each cell of ``by`` has a known population share.

        Use this when the sample's mix differs from the population's for a
        reason the design did not intend: uneven nonresponse, or a collection
        route that over-represents some kinds of road. Variance then counts
        only variation within cells. The shares fix the mix, not the population
        size, so a weighted total keeps the sampling error of the weight total.

        Args:
            by: Column whose values define the cells.
            shares: Population share of each cell. Shares must be positive, sum
                to one, and name exactly the cells present in the sample.

        Returns:
            A new sample with calibrated weights.

        Raises:
            ValueError: If the column or the shares are invalid, or the sample
                is already post-stratified.
        """
        if self._design.cells is not None:
            raise ValueError(
                "the sample is already post-stratified; calibrating to a second "
                "set of shares is not supported"
            )
        self._check_column(self._data, "by", by)
        cells, unique_cells = pd.factorize(self._data[by], sort=False)
        present = set(unique_cells)
        if set(shares) != present:
            raise ValueError(
                "shares must name exactly the cells present in the sample; "
                f"sample cells: {sorted(map(repr, present))}, "
                f"shares cells: {sorted(map(repr, shares))}"
            )
        targets = np.array([float(shares[cell]) for cell in unique_cells])
        if np.any(targets <= 0) or not np.isclose(targets.sum(), 1.0):
            raise ValueError("shares must be positive and sum to one")

        return self._with_design(
            replace(self._design, cells=cells.astype(int), cell_shares=targets)
        )

    def _with_design(self, design: _Design) -> "Sample":
        """Return a sample over the same rows under another design."""
        self_copy = copy(self)
        self_copy._design = design  # noqa: SLF001
        return self_copy

    @property
    def cluster(self) -> str | None:
        """Return the declared cluster-column name."""
        return self._cluster

    @property
    def weights(self) -> str | None:
        """Return the declared weight-column name."""
        return self._weights

    @property
    def strata(self) -> str | None:
        """Return the declared stratum-column name."""
        return self._strata

    @property
    def n_observations(self) -> int:
        """Return the number of sampled observations."""
        return len(self._data)

    @property
    def n_clusters(self) -> int | None:
        """Return the declared cluster count, or ``None`` for unclustered samples."""
        return self._design.n_units if self._cluster is not None else None

    def _numeric(self, variable: str, *, role: str = "variable") -> np.ndarray:
        if not isinstance(variable, str) or not variable.strip():
            raise ValueError(f"{role} must be a non-empty column name")
        if variable not in self._data:
            raise ValueError(
                f"column {variable!r} not found; available columns: "
                f"{list(self._data.columns)}"
            )
        try:
            values = self._data[variable].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"column {variable!r} must be numeric") from error
        if not np.all(np.isfinite(values)):
            raise ValueError(f"column {variable!r} must contain only finite values")
        return values

    def _run(
        self,
        estimand: _Estimand,
        *,
        inference: InferenceMethod,
        confidence_level: float,
        bootstrap_reps: int,
        seed: int | None,
    ) -> Estimate:
        return _run_inference(
            estimand,
            self._design,
            inference=inference,
            confidence_level=confidence_level,
            bootstrap_reps=bootstrap_reps,
            seed=seed,
        )

    def mean(
        self,
        variable: str,
        *,
        inference: InferenceMethod = "design",
        confidence_level: float = 0.95,
        bootstrap_reps: int = 2000,
        seed: int | None = None,
    ) -> Estimate:
        """Estimate the population mean of a numeric variable."""
        return self._run(
            _Estimand("mean", (variable,), "mean", self._numeric(variable)),
            inference=inference,
            confidence_level=confidence_level,
            bootstrap_reps=bootstrap_reps,
            seed=seed,
        )

    def share_at_least(
        self,
        variable: str,
        threshold: float,
        *,
        inference: InferenceMethod = "design",
        confidence_level: float = 0.95,
        bootstrap_reps: int = 2000,
        seed: int | None = None,
    ) -> Estimate:
        """Estimate the population share of units at or above a threshold."""
        at_least = (self._numeric(variable) >= threshold).astype(float)
        return self._run(
            _Estimand("share_at_least", (variable,), "mean", at_least),
            inference=inference,
            confidence_level=confidence_level,
            bootstrap_reps=bootstrap_reps,
            seed=seed,
        )

    def total(
        self,
        variable: str,
        *,
        population_size: int | None = None,
        inference: InferenceMethod = "design",
        confidence_level: float = 0.95,
        bootstrap_reps: int = 2000,
        seed: int | None = None,
    ) -> Estimate:
        """Estimate a population total.

        With declared weights this is the Horvitz-Thompson total, the weighted
        sum. Without weights it is ``population_size`` times the sample mean.

        Args:
            variable: Numeric column to total.
            population_size: Number of units in the population. Required for
                unweighted samples and not allowed for weighted ones.
            inference: Uncertainty method.
            confidence_level: Coverage of the confidence interval.
            bootstrap_reps: Number of bootstrap draws.
            seed: Seed for bootstrap draws.

        Returns:
            The estimated total.

        Raises:
            TypeError: If population_size is not an integer.
            ValueError: If population_size is missing, redundant, or too small.
        """
        values = self._numeric(variable)
        if self._weights is not None:
            if population_size is not None:
                raise ValueError(
                    "population_size is not used with declared weights; the "
                    "weights already scale the sample to the population"
                )
            estimand = _Estimand("total", (variable,), "total", values)
        else:
            if population_size is None:
                raise ValueError("population_size is required without weights")
            if isinstance(population_size, bool) or not isinstance(
                population_size, int
            ):
                raise TypeError("population_size must be an integer")
            if population_size < len(self._data):
                raise ValueError("population_size must be at least the sample size")
            estimand = _Estimand(
                "total",
                (variable,),
                "mean",
                values,
                scale=float(population_size),
                population_size=population_size,
            )
        return self._run(
            estimand,
            inference=inference,
            confidence_level=confidence_level,
            bootstrap_reps=bootstrap_reps,
            seed=seed,
        )

    def ratio(
        self,
        numerator: str,
        denominator: str,
        *,
        inference: InferenceMethod = "design",
        confidence_level: float = 0.95,
        bootstrap_reps: int = 2000,
        seed: int | None = None,
    ) -> Estimate:
        """Estimate a ratio of population totals."""
        numerator_values = self._numeric(numerator, role="numerator")
        denominator_values = self._numeric(denominator, role="denominator")
        if np.any(denominator_values < 0):
            raise ValueError("denominator values must be nonnegative")
        if np.sum(denominator_values) <= 0:
            raise ValueError("denominator must have a positive sample total")
        return self._run(
            _Estimand(
                "ratio",
                (numerator, denominator),
                "ratio",
                numerator_values,
                denominator_values,
            ),
            inference=inference,
            confidence_level=confidence_level,
            bootstrap_reps=bootstrap_reps,
            seed=seed,
        )

    def mean_of_ratios(
        self,
        numerator: str,
        denominator: str,
        *,
        inference: InferenceMethod = "design",
        confidence_level: float = 0.95,
        bootstrap_reps: int = 2000,
        seed: int | None = None,
    ) -> Estimate:
        """Estimate the population mean of row-level numerator/denominator ratios."""
        numerator_values = self._numeric(numerator, role="numerator")
        denominator_values = self._numeric(denominator, role="denominator")
        if np.any(denominator_values <= 0):
            raise ValueError("mean_of_ratios requires every denominator to be positive")
        return self._run(
            _Estimand(
                "mean_of_ratios",
                (numerator, denominator),
                "mean",
                numerator_values / denominator_values,
            ),
            inference=inference,
            confidence_level=confidence_level,
            bootstrap_reps=bootstrap_reps,
            seed=seed,
        )

    def __repr__(self) -> str:
        """Return a concise representation."""
        return (
            f"Sample(n_observations={len(self._data)}, "
            f"cluster={self._cluster!r}, n_clusters={self.n_clusters!r}, "
            f"weights={self._weights!r}, strata={self._strata!r})"
        )
