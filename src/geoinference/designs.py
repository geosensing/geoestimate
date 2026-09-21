"""Sampling-design declarations for the stable inference API."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PointDesign:
    """Describe an equal-probability sample of observation locations.

    Args:
        cluster_var: Column identifying independent sampling or collection
            clusters. Leave unset only when rows can be treated as independent.

    Notes:
        This design does not implement unequal-probability weighting, GRTS
        variance estimation, annotation-subsampling corrections, or finite
        population corrections.
    """

    cluster_var: str | None = None

    def __post_init__(self) -> None:
        """Validate the cluster-column name."""
        if self.cluster_var is not None and not self.cluster_var.strip():
            raise ValueError("cluster_var must be a non-empty column name")

    @property
    def name(self) -> str:
        """Return a short design name."""
        if self.has_clusters:
            return "point_equal_probability_clustered"
        return "point_equal_probability"

    @property
    def has_clusters(self) -> bool:
        """Report whether the design declares clusters."""
        return self.cluster_var is not None

    @property
    def recommended_se_method(self) -> str:
        """Return the standard-error method implied by the design."""
        return "cluster" if self.has_clusters else "naive"


@dataclass(frozen=True, slots=True)
class WalkDesign:
    """Describe observations grouped into independent walks.

    The class selects walk-clustered inference. It does not make a random walk
    self-weighting and does not correct unequal location-selection probabilities.
    Use it only when the target is the encountered-frame population or selection
    has already been corrected upstream.

    Args:
        walk_var: Column identifying independent walks.
    """

    walk_var: str = "walk_id"

    def __post_init__(self) -> None:
        """Validate the walk-column name."""
        if not self.walk_var.strip():
            raise ValueError("walk_var must be a non-empty column name")

    @property
    def name(self) -> str:
        """Return a short design name."""
        return "walk_clustered"

    @property
    def has_clusters(self) -> bool:
        """Report that walks define clusters."""
        return True

    @property
    def cluster_var(self) -> str:
        """Return the column identifying walks."""
        return self.walk_var

    @property
    def recommended_se_method(self) -> str:
        """Return the standard-error method implied by the design."""
        return "cluster"


Design = PointDesign | WalkDesign
