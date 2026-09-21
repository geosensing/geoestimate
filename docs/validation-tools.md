# Validation tools

The stable estimator is small enough to validate against hand calculations and
reference software. The package also includes experimental tools for checking
whether a proposed field design behaves well under a known data-generating
process.

`geoestimate.spatial` measures spatial and temporal dependence in observed
values. These diagnostics do not replace a sampling design or alter an
estimate's standard error.

`geoestimate.simulate` generates known spatial and temporal populations and
measures bias, standard-error calibration, and interval coverage.

`geoestimate.pipeline` connects simulations to the optional `geo-sampling` and
`allocator` packages. Install those dependencies with:

```bash
pip install "geoestimate[pipeline]"
python examples/validate_with_allocator.py
```

These modules remain experimental because their reference checks cover fewer
configurations than the stable inference API.
