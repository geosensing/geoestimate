# Sampling and uncertainty

`geoestimate` gives every row equal weight unless you declare weights. The
declaration `Sample(data, cluster="itinerary_id")` changes the uncertainty
calculation, not the point estimate.

## Weights, strata, and post-stratification

`Sample(data, weights="w", strata="road_class", cluster="itinerary_id")`
declares the design:

- `weights` are inverse inclusion probabilities. Means and ratios become
  weighted means and ratios of weighted totals.
- `strata` are groups the sample was drawn within. Variance counts only
  variation among units inside each stratum, and clusters are treated as nested
  in strata. Every stratum needs at least two units.
- Degrees of freedom for clustered intervals are clusters minus strata.

`sample.post_stratify("road_class", shares)` returns a sample whose weights
are rescaled so each cell has its known population share. Use it when the
sample's mix differs from the population's for a reason the design did not
intend, such as nonresponse that varies by road class. Variance then counts
only each row's departure from its cell's weighted mean. A sample can be
post-stratified once; calibrating to a second set of shares raises.

Standard errors treat units as drawn with replacement and apply no finite
population correction, which overstates variance slightly when the sample is a
large share of the population.

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
rows for an iid sample and whole clusters for a clustered sample. It is not
available for stratified or post-stratified samples.

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
