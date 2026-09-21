"""Structured results returned by :func:`geoinference.estimate`."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SEResult:
    """Standard-error estimates and the selected method."""

    naive: float
    cluster: float | None
    bootstrap: float | None
    recommended: float
    method_used: str

    @property
    def cluster_to_naive(self) -> float:
        """Return the cluster standard error divided by the naive one."""
        if self.cluster is None or self.naive <= 0:
            return float("nan")
        return self.cluster / self.naive


@dataclass(frozen=True, slots=True)
class CIResult:
    """Confidence intervals and the selected method."""

    normal: tuple[float, float]
    t: tuple[float, float] | None
    bootstrap: tuple[float, float] | None
    recommended: tuple[float, float]
    method_used: str
    level: float


@dataclass(frozen=True, slots=True)
class Diagnostics:
    """Data and clustering diagnostics for an inference result."""

    n_obs: int
    n_positive_frames: int
    n_empty_frames: int
    empty_frame_rate: float
    n_clusters: int
    cluster_sizes: np.ndarray
    cluster_size_mean: float
    cluster_size_cv: float
    n_clusters_eff: float
    icc: float
    deff: float
    n_eff: float
    se_photo_mean_cluster_to_naive: float
    ratio_bias_approx: float


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Point estimates, uncertainty estimates, and diagnostics."""

    ratio: float
    photo_mean: float
    ratio_se: SEResult
    photo_mean_se: SEResult
    ratio_ci: CIResult
    photo_mean_ci: CIResult
    diagnostics: Diagnostics
    design_name: str
    n_obs: int
    n_clusters: int

    def summary(self) -> str:
        """Return a human-readable summary."""
        level = f"{100 * self.ratio_ci.level:g}%"
        lines = [
            "=" * 60,
            "geoinference: Inference Result",
            "=" * 60,
            f"Design: {self.design_name}",
            (
                f"Observations: {self.n_obs} "
                f"({self.diagnostics.n_positive_frames} with people, "
                f"{self.diagnostics.n_empty_frames} empty)"
            ),
            f"Clusters: {self.n_clusters}",
            "",
            "Ratio estimand (people-weighted)",
            f"  Estimate:  {self.ratio:.4f}",
            f"  SE:        {self.ratio_se.recommended:.4f} "
            f"({self.ratio_se.method_used})",
            f"  {level} CI:    [{self.ratio_ci.recommended[0]:.4f}, "
            f"{self.ratio_ci.recommended[1]:.4f}] ({self.ratio_ci.method_used})",
            "",
            "Photo-level mean (location-weighted)",
            f"  Estimate:  {self.photo_mean:.4f}",
            f"  SE:        {self.photo_mean_se.recommended:.4f} "
            f"({self.photo_mean_se.method_used})",
            f"  {level} CI:    [{self.photo_mean_ci.recommended[0]:.4f}, "
            f"{self.photo_mean_ci.recommended[1]:.4f}] "
            f"({self.photo_mean_ci.method_used})",
            "",
            "Diagnostics",
            f"  ICC:                  {self.diagnostics.icc:.4f}",
            f"  Design effect:        {self.diagnostics.deff:.2f}",
            f"  Effective N:          {self.diagnostics.n_eff:.1f}",
            f"  Effective clusters:   {self.diagnostics.n_clusters_eff:.1f}",
            f"  Ratio bias O(1/N):    {self.diagnostics.ratio_bias_approx:.6f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a compact representation."""
        return (
            f"InferenceResult(ratio={self.ratio:.4f}, "
            f"photo_mean={self.photo_mean:.4f}, n={self.n_obs}, "
            f"G={self.n_clusters}, design={self.design_name!r})"
        )
