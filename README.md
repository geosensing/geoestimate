# geoestimate

Design-aware estimates for equal-probability observation samples.

[![PyPI](https://img.shields.io/pypi/v/geoestimate.svg)](https://pypi.org/project/geoestimate/)
[![Downloads](https://static.pepy.tech/badge/geoestimate)](https://pepy.tech/projects/geoestimate)
[![CI](https://github.com/geosensing/geoestimate/actions/workflows/ci.yml/badge.svg)](https://github.com/geosensing/geoestimate/actions/workflows/ci.yml)
[![Docs](https://github.com/geosensing/geoestimate/actions/workflows/docs.yml/badge.svg)](https://geosensing.github.io/geoestimate/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

`geoestimate` estimates means, population totals, ratios of totals, and means
of row-level ratios. It reports iid, cluster-sandwich, or pairs-bootstrap
uncertainty. The stable API assumes that rows have equal inclusion
probabilities.

The package does not implement unequal-probability weights, PPS, GRTS,
stratification, finite population corrections, nonresponse adjustments,
annotation-subsampling corrections, or walk-spacing corrections.

## Install

```bash
pip install geoestimate
```

## Estimate from a sample

```python
import pandas as pd

from geoestimate import Sample

frames = pd.DataFrame(
    {
        "n_women": [3, 4, 2, 5],
        "n_people": [10, 10, 10, 10],
        "itinerary_id": [0, 0, 1, 1],
    }
)

sample = Sample(frames, cluster="itinerary_id")

mean_people = sample.mean("n_people")
total_people = sample.total("n_people", population_size=50_000)
people_share = sample.ratio("n_women", "n_people")
location_share = sample.mean_of_ratios("n_women", "n_people")

print(people_share.summary())
```

Declare a cluster only when it identifies independent sampling or collection
units. A declared cluster makes cluster-sandwich inference with a Student-t
interval the default. Without a cluster, the default is an iid standard error
with a normal interval.

## Choose the estimand

`mean("n_people")` estimates the average count per population unit. For a
binary variable, the mean is a population proportion.

`total("n_people", population_size=N)` estimates `N * mean(n_people)`. The
population size counts the same row-level units represented by the sample. The
method does not apply a finite population correction.

`ratio("n_women", "n_people")` estimates:

```text
sum(n_women) / sum(n_people)
```

This ratio weights rows by their denominator. Individual denominators may be
zero, but they must be nonnegative and their sample total must be positive.

`mean_of_ratios("n_women", "n_people")` estimates:

```text
mean(n_women / n_people)
```

This estimand gives every row equal weight. Every denominator must be positive.
Filter the DataFrame before constructing `Sample` when the target population
excludes rows with zero denominators.

## Select inference

The default `inference="design"` follows the declared sample design. You can
request a method explicitly:

```python
bootstrap = sample.ratio(
    "n_women",
    "n_people",
    inference="bootstrap",
    bootstrap_reps=2_000,
    seed=42,
)
```

The bootstrap resamples declared clusters, or individual rows when no cluster
is declared. It reports the standard deviation of the bootstrap estimates and
a percentile interval. Explicit `inference="iid"` is available as a sensitivity
comparison for clustered samples.

## Read files and use the command line

`Sample.from_file()` accepts Parquet, CSV, compressed CSV, and TSV files:

```python
sample = Sample.from_file("frames.parquet", cluster="itinerary_id")
result = sample.ratio("n_women", "n_people")
```

The command line exposes the same four estimands:

```bash
geoestimate mean frames.parquet --variable n_people --cluster itinerary_id
geoestimate total frames.parquet --variable n_people --population-size 50000
geoestimate ratio frames.parquet --numerator n_women --denominator n_people
geoestimate mean-of-ratios frames.parquet --numerator n_women --denominator n_people
```

## Experimental validation tools

`geoestimate.spatial`, `geoestimate.simulate`, and `geoestimate.pipeline` help
diagnose dependence and validate collection designs. Their interfaces may
change before the stable inference API does.

Install the optional pipeline dependencies to validate a real `geo-sampling`
to `allocator` geometry:

```bash
pip install "geoestimate[pipeline]"
python examples/validate_with_allocator.py
```

## License

MIT
