# geoinference

Inference for equal-probability spatial observation surveys.

[![PyPI](https://img.shields.io/pypi/v/geoinference.svg)](https://pypi.org/project/geoinference/)
[![CI](https://github.com/geosensing/geoinference/actions/workflows/ci.yml/badge.svg)](https://github.com/geosensing/geoinference/actions/workflows/ci.yml)
[![Docs](https://github.com/geosensing/geoinference/actions/workflows/docs.yml/badge.svg)](https://geosensing.github.io/geoinference/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

`geoinference` estimates two proportions from annotated frames and reports iid
or cluster-sandwich uncertainty. The stable API assumes that observation
locations have equal selection probabilities. It does not implement PPS, GRTS,
nonresponse weighting, annotation-subsampling corrections, or finite population
corrections.

## Install

```bash
pip install geoinference
```

## Estimate proportions

```python
import pandas as pd

from geoinference import PointDesign, estimate

frames = pd.DataFrame(
    {
        "n_women": [3, 4, 2, 5],
        "n_people": [10, 10, 10, 10],
        "itinerary_id": [0, 0, 1, 1],
        "longitude": [77.20, 77.21, 77.22, 77.23],
        "latitude": [28.60, 28.61, 28.62, 28.63],
    }
)

result = estimate(
    frames,
    design=PointDesign(cluster_var="itinerary_id"),
)
print(result.summary())
```

Declare a cluster column only when clusters are independent sampling or
collection units. Clustered designs use a cluster-sandwich standard error and a
Student t interval with cluster degrees of freedom. Omitting the column treats
rows as independent and uses an iid standard error with a normal interval.
Clustering is a design decision, not a test selected after inspecting the
outcomes ([Abadie et al., 2023](https://doi.org/10.1093/qje/qjad005)).

## Estimands

The people-weighted ratio is:

```text
sum(n_women) / sum(n_people)
```

It estimates the fraction of observed people who are women. A finite-sample
ratio estimator is generally biased, so the result includes a first-order bias
diagnostic rather than claiming exact unbiasedness.

The location-weighted result is the mean of `n_women / n_people` over frames
that contain at least one person. It estimates the mean observed proportion at
a sampled frame. Empty frames contribute to neither estimand.

Counts must be finite and nonnegative, and `n_women` cannot exceed
`n_people`. The estimator rejects undefined samples, incomplete cluster labels,
and designs with too few clusters for their requested uncertainty method.

## Bootstrap intervals

The analytic interval is the default. Request a pairs bootstrap explicitly:

```python
result = estimate(
    frames,
    design=PointDesign(cluster_var="itinerary_id"),
    bootstrap=True,
    bootstrap_reps=2000,
    se_method="bootstrap",
    ci_method="bootstrap",
    seed=42,
)
```

The bootstrap resamples the declared clusters, or individual rows for an
independent design. It is not the default small-cluster correction.

## Read a frame table

```python
from geoinference import estimate_from_file

result = estimate_from_file(
    "frames.parquet",
    cluster_var="itinerary_id",
)
```

`read_frames` and `estimate_from_file` accept Parquet, CSV, and TSV. CSV and TSV
may be compressed. Parquet preserves count and timestamp types and is the
preferred interchange format.

The same operation is available from the shell:

```bash
geoinference estimate frames.parquet --cluster-var itinerary_id
```

## Walk data

`WalkDesign(walk_var="walk_id")` groups observations by independent walks. It
does not make a random walk self-weighting. Use it only when the target is the
encountered-frame population or when selection probabilities have already been
corrected upstream.

## Experimental validation tools

`geoinference.spatial`, `geoinference.simulate`, and `geoinference.pipeline`
are experimental. They are useful for diagnosing dependence and validating a
collection design, but their interfaces may change before the stable inference
API does.

Spatial and temporal dependence diagnostics are separate from estimation:

```python
from geoinference.spatial import dependence_diagnostics

diagnostics = dependence_diagnostics(
    frames.loc[frames["n_people"] > 0, "n_women"].to_numpy()
    / frames.loc[frames["n_people"] > 0, "n_people"].to_numpy(),
    frames.loc[frames["n_people"] > 0, "itinerary_id"].to_numpy(),
    lon=frames.loc[frames["n_people"] > 0, "longitude"].to_numpy(),
    lat=frames.loc[frames["n_people"] > 0, "latitude"].to_numpy(),
)
```

Install the optional pipeline dependencies to validate a real
`geo-sampling` to `allocator` geometry:

```bash
pip install "geoinference[pipeline]"
```

From a source checkout, run the complete validation example:

```bash
python examples/validate_with_allocator.py
```

## License

MIT
