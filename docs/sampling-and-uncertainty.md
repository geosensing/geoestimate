# Sampling and uncertainty

`geoestimate` assumes that rows have equal inclusion probabilities. The
declaration `Sample(data, cluster="itinerary_id")` changes the uncertainty
calculation, not the point estimate.

## Design-based default

`inference="design"` follows the declared design:

- Without a cluster, it reports an iid standard error and normal interval.
- With a cluster, it sums linearized contributions within clusters, applies the
  finite-cluster correction, and reports a Student-t interval with `G - 1`
  degrees of freedom.

Clusters must be independent sampling or collection units. Spatial proximity
or outcome correlation alone does not decide whether clustering is appropriate.

## Bootstrap

`inference="bootstrap"` resamples the design units. It resamples individual
rows for an iid sample and whole clusters for a clustered sample.

```python
result = sample.ratio(
    "n_women",
    "n_people",
    inference="bootstrap",
    bootstrap_reps=2_000,
    seed=42,
)
```

The result contains the standard deviation of the bootstrap estimates and a
percentile confidence interval. Ratio resamples with a zero denominator total
are undefined. The implementation redraws them up to a fixed attempt limit and
fails if the sample contains too little positive denominator information.

## Diagnostics

Every result contains iid and, when declared, clustered analytic standard
errors. The design effect is the clustered variance divided by the iid
variance. The effective sample size divides the observation count by that
design effect. These are descriptive diagnostics, not corrections for an
unsupported sampling design.
