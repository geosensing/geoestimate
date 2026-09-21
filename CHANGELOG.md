# Changelog

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## 0.1.0 - 2026-09-20

### Added

- People-weighted ratio and location-weighted frame estimators.
- Iid and cluster-sandwich standard errors with normal and cluster-t intervals.
- Optional pairs-bootstrap standard errors and percentile intervals.
- Parquet, CSV, and TSV input through Python and command-line interfaces.
- Experimental spatial diagnostics and collection-design simulation tools.
- Integration helpers for `geo-sampling` and `allocator` geometries.

### Correctness

- Reject invalid counts, undefined samples, incomplete clusters, invalid
  confidence levels, and unavailable uncertainty methods at the public boundary.
- Keep simulated visits inside their assigned shifts and reject infeasible routes.
- Match convenience-scheduler capacity to its requested itinerary count.
- Use common population draws when comparing simulation pipelines and methods.
- Use the requested standard error when measuring simulation coverage.
- Retain every bootstrap resample for which the estimand is defined.
- Center two-sided Moran permutation tests on their permutation null distribution.

### Removed before first release

- PPS, GRTS, annotation-subsampling, finite-population, and walk-spacing options
  that did not affect the estimator.
- The unverified public wild-cluster-bootstrap interval.
- Spatial diagnostics from the stable `estimate` result.
