"""Public result types for :mod:`geoestimate`."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd

InferenceMethod = Literal["design", "iid", "cluster", "bootstrap"]
ResolvedInferenceMethod = Literal["iid", "cluster", "bootstrap"]


@dataclass(frozen=True, slots=True)
class Diagnostics:
    """Analytic diagnostics for one estimate.

    Attributes:
        iid_standard_error: Standard error that treats rows as independent.
        cluster_standard_error: Cluster-sandwich standard error, when declared.
        design_effect: Cluster variance divided by iid variance, when defined.
        effective_sample_size: Observation count divided by the design effect.
        cluster_sizes: Number of observations in each declared cluster.
        effective_cluster_count: Kish effective count of declared clusters.
    """

    iid_standard_error: float
    cluster_standard_error: float | None
    design_effect: float | None
    effective_sample_size: float
    cluster_sizes: tuple[int, ...] | None
    effective_cluster_count: float | None


@dataclass(frozen=True, slots=True)
class Estimate:
    """A scalar estimate with uncertainty and design diagnostics."""

    estimand: str
    variables: tuple[str, ...]
    estimate: float
    standard_error: float
    confidence_interval: tuple[float, float]
    confidence_level: float
    inference_method: ResolvedInferenceMethod
    n_observations: int
    n_clusters: int | None
    population_size: int | None
    diagnostics: Diagnostics

    def summary(self) -> str:
        """Return a compact, human-readable summary."""
        level = f"{100 * self.confidence_level:g}%"
        variable_text = ", ".join(self.variables)
        lines = [
            "=" * 60,
            "geoestimate: Estimate",
            "=" * 60,
            f"Estimand: {self.estimand}",
            f"Variables: {variable_text}",
            f"Observations: {self.n_observations}",
        ]
        if self.n_clusters is not None:
            lines.append(f"Clusters: {self.n_clusters}")
        if self.population_size is not None:
            lines.append(f"Population size: {self.population_size}")
        lines.extend(
            [
                f"Estimate: {self.estimate:.6g}",
                f"Standard error: {self.standard_error:.6g}",
                (
                    f"{level} confidence interval: "
                    f"[{self.confidence_interval[0]:.6g}, "
                    f"{self.confidence_interval[1]:.6g}]"
                ),
                f"Inference: {self.inference_method}",
                "=" * 60,
            ]
        )
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Return the scalar result as a one-row tidy table."""
        is_univariate = self.estimand in {"mean", "total"}
        return pd.DataFrame(
            [
                {
                    "estimand": self.estimand,
                    "variable": self.variables[0] if is_univariate else None,
                    "numerator": None if is_univariate else self.variables[0],
                    "denominator": None if is_univariate else self.variables[1],
                    "estimate": self.estimate,
                    "standard_error": self.standard_error,
                    "conf_low": self.confidence_interval[0],
                    "conf_high": self.confidence_interval[1],
                    "confidence_level": self.confidence_level,
                    "inference_method": self.inference_method,
                    "n_observations": self.n_observations,
                    "n_clusters": self.n_clusters,
                    "population_size": self.population_size,
                }
            ]
        )

    def __repr__(self) -> str:
        """Return a concise representation."""
        return (
            f"Estimate(estimand={self.estimand!r}, estimate={self.estimate:.6g}, "
            f"standard_error={self.standard_error:.6g}, "
            f"inference_method={self.inference_method!r})"
        )
