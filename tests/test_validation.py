import numpy as np
import pandas as pd

from geoestimate import Sample


def _calibration_run(*, clustered: bool) -> dict[str, tuple[float, float, float]]:
    rng = np.random.default_rng(20260920 + clustered)
    n_sims = 300
    n_clusters = 24
    cluster_size = 5
    population_size = 1_000
    records: dict[str, list[tuple[float, float, bool]]] = {
        "mean": [],
        "total": [],
        "ratio": [],
        "mean_of_ratios": [],
    }
    truths = {
        "mean": 2.0,
        "total": 2_000.0,
        "ratio": 2.0,
        "mean_of_ratios": 2.0,
    }

    for _ in range(n_sims):
        cluster = np.repeat(np.arange(n_clusters), cluster_size)
        if clustered:
            shared = np.repeat(rng.normal(0, 0.4, n_clusters), cluster_size)
        else:
            shared = rng.normal(0, 0.4, len(cluster))
        value = 2 + shared + rng.normal(0, 0.8, len(cluster))
        denominator = rng.lognormal(0, 0.35, len(cluster))
        data = pd.DataFrame(
            {
                "value": value,
                "denominator": denominator,
                "numerator": value * denominator,
                "cluster": cluster,
            }
        )
        sample = Sample(data, cluster="cluster" if clustered else None)
        estimates = {
            "mean": sample.mean("value"),
            "total": sample.total("value", population_size=population_size),
            "ratio": sample.ratio("numerator", "denominator"),
            "mean_of_ratios": sample.mean_of_ratios("numerator", "denominator"),
        }
        for name, result in estimates.items():
            truth = truths[name]
            low, high = result.confidence_interval
            records[name].append(
                (result.estimate, result.standard_error, low <= truth <= high)
            )

    summary: dict[str, tuple[float, float, float]] = {}
    for name, values in records.items():
        array = np.asarray(values, dtype=float)
        empirical_sd = float(np.std(array[:, 0], ddof=1))
        mean_se = float(np.mean(array[:, 1]))
        coverage = float(np.mean(array[:, 2]))
        summary[name] = (float(np.mean(array[:, 0])), mean_se / empirical_sd, coverage)
    return summary


def test_iid_inference_calibrates_all_estimands() -> None:
    summary = _calibration_run(clustered=False)
    truths = {"mean": 2.0, "total": 2_000.0, "ratio": 2.0, "mean_of_ratios": 2.0}
    for name, (mean_estimate, se_ratio, coverage) in summary.items():
        tolerance = 40 if name == "total" else 0.04
        assert abs(mean_estimate - truths[name]) < tolerance
        assert 0.85 < se_ratio < 1.15
        assert 0.90 < coverage < 0.99


def test_cluster_inference_calibrates_all_estimands() -> None:
    summary = _calibration_run(clustered=True)
    truths = {"mean": 2.0, "total": 2_000.0, "ratio": 2.0, "mean_of_ratios": 2.0}
    for name, (mean_estimate, se_ratio, coverage) in summary.items():
        tolerance = 40 if name == "total" else 0.04
        assert abs(mean_estimate - truths[name]) < tolerance
        assert 0.85 < se_ratio < 1.15
        assert 0.90 < coverage < 0.99
