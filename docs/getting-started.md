# Getting started

Install the package from PyPI:

```bash
pip install geoestimate
```

Create one `Sample` for a DataFrame and its sampling design. Then request the
estimand you need.

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
result = sample.ratio("n_women", "n_people")

print(result.estimate)
print(result.standard_error)
print(result.confidence_interval)
```

`Sample` takes a snapshot of the DataFrame. Later changes to `frames` do not
change the sample or its estimates.

The default `inference="design"` uses cluster-sandwich inference when you
declare a cluster. It uses iid inference otherwise. Read {doc}`estimands` before
choosing between ratios and means of ratios.

## Work with results

Every method returns an immutable `Estimate` with the same fields. Use
`summary()` for display or `to_frame()` to combine results with pandas.

```python
mean_result = sample.mean("n_people")
ratio_result = sample.ratio("n_women", "n_people")

table = pd.concat(
    [mean_result.to_frame(), ratio_result.to_frame()],
    ignore_index=True,
)
```
