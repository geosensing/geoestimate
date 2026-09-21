"""Design-aware estimation for equal-probability observation samples.

Quick start:
    >>> import pandas as pd
    >>> from geoestimate import Sample
    >>> frames = pd.DataFrame({
    ...     "n_women": [3, 4, 2, 5],
    ...     "n_people": [10, 10, 10, 10],
    ...     "itinerary_id": [0, 0, 1, 1],
    ... })
    >>> sample = Sample(frames, cluster="itinerary_id")
    >>> round(sample.ratio("n_women", "n_people").estimate, 3)
    0.35
"""

from importlib.metadata import version

from .sample import Sample
from .types import Diagnostics, Estimate, InferenceMethod

__version__ = version("geoestimate")

__all__ = ["Diagnostics", "Estimate", "InferenceMethod", "Sample"]
