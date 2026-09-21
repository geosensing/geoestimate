"""Equal-probability sample declaration and estimation methods."""

from pathlib import Path

import numpy as np
import pandas as pd

from .inference import (
    _prepare_mean,
    _prepare_mean_of_ratios,
    _prepare_ratio,
    _prepare_total,
    _PreparedEstimand,
    _run_inference,
)
from .types import Estimate, InferenceMethod


class Sample:
    """Bind observations to an equal-probability sampling design.

    Notes:
        ``Sample`` supports equal-probability observations with iid or
        cluster-sandwich inference. It does not implement weights, strata,
        finite population corrections, PPS, or GRTS variance estimation.
    """

    __slots__ = ("_cluster", "_data", "_labels", "_n_clusters")

    def __init__(self, data: pd.DataFrame, *, cluster: str | None = None) -> None:
        """Create a validated snapshot of the sample.

        Args:
            data: One row per sampled population unit.
            cluster: Optional column identifying independent sampling or collection
                clusters.

        Raises:
            TypeError: If data is not a pandas DataFrame.
            ValueError: If the sample or cluster declaration is invalid.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if len(data) < 2:
            raise ValueError("data must contain at least two rows")
        if cluster is not None and (
            not isinstance(cluster, str) or not cluster.strip()
        ):
            raise ValueError("cluster must be a non-empty column name or None")

        copied = data.copy(deep=True).reset_index(drop=True)
        if cluster is None:
            labels = np.arange(len(copied), dtype=int)
            n_clusters = len(copied)
        else:
            if cluster not in copied:
                raise ValueError(
                    f"cluster column {cluster!r} not found; "
                    f"available columns: {list(copied.columns)}"
                )
            if copied[cluster].isna().any():
                raise ValueError(
                    f"cluster column {cluster!r} cannot contain missing values"
                )
            labels, unique = pd.factorize(copied[cluster], sort=False)
            n_clusters = len(unique)
            if n_clusters < 2:
                raise ValueError("clustered inference requires at least two clusters")

        self._data = copied
        self._cluster = cluster
        self._labels = labels.astype(int)
        self._n_clusters = n_clusters

    @classmethod
    def from_file(cls, path: str | Path, *, cluster: str | None = None) -> "Sample":
        """Read a Parquet, CSV, or TSV file and construct a sample."""
        from .io import _read_table

        return cls(_read_table(path), cluster=cluster)

    @property
    def cluster(self) -> str | None:
        """Return the declared cluster-column name."""
        return self._cluster

    @property
    def n_observations(self) -> int:
        """Return the number of sampled observations."""
        return len(self._data)

    @property
    def n_clusters(self) -> int | None:
        """Return the declared cluster count, or ``None`` for iid samples."""
        return self._n_clusters if self._cluster is not None else None

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
        estimand: _PreparedEstimand,
        *,
        inference: InferenceMethod,
        confidence_level: float,
        bootstrap_reps: int,
        seed: int | None,
    ) -> Estimate:
        return _run_inference(
            estimand,
            self._labels,
            self._n_clusters,
            self._cluster is not None,
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
            _prepare_mean(self._numeric(variable), variable),
            inference=inference,
            confidence_level=confidence_level,
            bootstrap_reps=bootstrap_reps,
            seed=seed,
        )

    def total(
        self,
        variable: str,
        *,
        population_size: int,
        inference: InferenceMethod = "design",
        confidence_level: float = 0.95,
        bootstrap_reps: int = 2000,
        seed: int | None = None,
    ) -> Estimate:
        """Estimate a population total as population size times the sample mean."""
        if isinstance(population_size, bool) or not isinstance(population_size, int):
            raise TypeError("population_size must be an integer")
        if population_size < len(self._data):
            raise ValueError("population_size must be at least the sample size")
        return self._run(
            _prepare_total(self._numeric(variable), variable, population_size),
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
            _prepare_ratio(
                numerator_values, denominator_values, numerator, denominator
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
            _prepare_mean_of_ratios(
                numerator_values, denominator_values, numerator, denominator
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
            f"cluster={self._cluster!r}, n_clusters={self.n_clusters!r})"
        )
