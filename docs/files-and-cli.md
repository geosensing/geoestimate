# Files and command line

## Read a sample

`Sample.from_file()` accepts Parquet, CSV, compressed CSV, and TSV files.
Parquet is the best choice when column types must survive a round trip.

```python
from geoestimate import Sample

sample = Sample.from_file("frames.parquet", cluster="itinerary_id")
result = sample.mean("n_people")
```

Externally compressed Parquet files are rejected because Parquet already
handles compression and an extra suffix hides the file format.

## Command line

The command line exposes the same estimands and option names as `Sample`:

```bash
geoestimate mean frames.parquet --variable n_people --cluster itinerary_id
geoestimate total frames.parquet --variable n_people --population-size 50000
geoestimate ratio frames.parquet --numerator n_women --denominator n_people
geoestimate mean-of-ratios frames.parquet --numerator n_women --denominator n_people
```

Use `--inference bootstrap`, `--bootstrap-reps`, and `--seed` for bootstrap
inference. Use `--confidence-level 0.90` to request a 90 percent interval.
