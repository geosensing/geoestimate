"""Inference for equal-probability spatial observation surveys.

The stable API estimates people-weighted and frame-weighted proportions from
annotated frames. A design declaration selects independent or cluster-sandwich
uncertainty estimates.

Quick start:
    >>> import pandas as pd
    >>> from geoinference import PointDesign, estimate
    >>> df = pd.DataFrame({
    ...     "n_women": [3, 4, 2, 5],
    ...     "n_people": [10, 10, 10, 10],
    ...     "itinerary_id": [0, 0, 1, 1],
    ... })
    >>> design = PointDesign(cluster_var="itinerary_id")
    >>> result = estimate(df, "n_women", "n_people", design=design)
    >>> round(result.ratio, 3)
    0.35
"""

from importlib.metadata import version

from .designs import Design, PointDesign, WalkDesign
from .inference import estimate
from .io import estimate_from_file, read_frames
from .types import CIResult, Diagnostics, InferenceResult, SEResult

__version__ = version("geoinference")

__all__ = [
    "CIResult",
    "Design",
    "Diagnostics",
    "InferenceResult",
    "PointDesign",
    "SEResult",
    "WalkDesign",
    "estimate",
    "estimate_from_file",
    "read_frames",
]
