# Changelog

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## 0.1.1 - 2026-09-20

### Changed

- Renamed the distribution, import package, command, documentation, and
  repository from `geoinference` to `geoestimate`.
- Replaced the proportion-specific entry point with an immutable `Sample`
  object and scalar `Estimate` results.
- Added inference for means, population totals with known population size,
  ratios of totals, and means of row-level ratios.
- Simplified uncertainty selection to design-based, iid, cluster-sandwich, or
  pairs-bootstrap inference with matching confidence intervals.
- Reorganized the documentation around tasks, estimands, sampling assumptions,
  file workflows, and stable versus experimental APIs.

### Correctness

- Require users to define the analysis domain before estimating row-level
  ratios with zero denominators.
- Require a known row-level population size for population totals and state
  that finite population corrections are not implemented.
- Validate every estimand's variables and option domains at the public boundary.
- Cross-check equal-probability analytic standard errors against the R `survey`
  package under equivalent designs.

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
