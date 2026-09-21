import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from geoinference.spatial import (
    dependence_diagnostics,
    effective_n,
    empirical_variogram,
    fit_variogram,
    haversine_matrix,
    morans_i,
    time_gap_matrix,
    within_between_contrast,
)


def test_haversine_known_distance_and_symmetry() -> None:
    distances = haversine_matrix(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    assert distances[0, 1] == pytest.approx(111_195, rel=1e-3)
    assert_allclose(distances, distances.T)
    assert_allclose(np.diag(distances), 0)


def test_time_gap_parses_datetimes_at_second_scale() -> None:
    times = np.array(["2026-01-01T00:00:00", "2026-01-01T00:01:30"])
    gaps = time_gap_matrix(times)
    assert gaps[0, 1] == pytest.approx(90)


def test_empirical_variogram_known_pairs() -> None:
    values = np.array([0.0, 2.0, 4.0])
    distances = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    lags, gamma, counts = empirical_variogram(values, distances, n_bins=2, max_dist=2)
    assert counts.sum() == 3
    assert np.average(gamma, weights=counts) == pytest.approx(4.0)
    assert len(lags) == 2


def test_variogram_fit_recovers_exponential_shape() -> None:
    lags = np.linspace(1, 10, 20)
    gamma = 0.2 + 0.8 * (1 - np.exp(-lags / 3))
    counts = np.full(20, 100)
    nugget, sill, range_value = fit_variogram(lags, gamma, counts)
    assert nugget == pytest.approx(0.2, abs=0.02)
    assert sill == pytest.approx(1.0, abs=0.02)
    assert range_value == pytest.approx(3.0, abs=0.2)


def test_morans_two_sided_p_value_is_centered_on_null_distribution() -> None:
    values = np.array([0.0, 0.1, 0.9, 1.0])
    distances = np.abs(np.arange(4)[:, None] - np.arange(4)[None, :]).astype(float)
    observed, p_value = morans_i(values, distances, cutoff=1.0, n_perm=99, seed=12)

    weights = ((distances > 0) & (distances <= 1)).astype(float)
    centered = values - values.mean()
    denominator = np.sum(centered**2)

    def statistic(vector: np.ndarray) -> float:
        return float(4 / weights.sum() * (vector @ weights @ vector) / denominator)

    rng = np.random.default_rng(12)
    simulated = np.array([statistic(rng.permutation(centered)) for _ in range(99)])
    null_mean = simulated.mean()
    expected_p = (
        np.count_nonzero(np.abs(simulated - null_mean) >= abs(observed - null_mean)) + 1
    ) / 100
    assert p_value == pytest.approx(expected_p)


def test_effective_n_limiting_cases() -> None:
    values = np.arange(5, dtype=float)
    distances = np.ones((5, 5)) - np.eye(5)
    assert effective_n(values, distances, 1.0, 1.0, 2.0) == 5
    correlated = effective_n(values, distances, 0.0, 1.0, 1e9)
    assert correlated == pytest.approx(1.0, rel=1e-6)
    with pytest.raises(ValueError, match="nugget"):
        effective_n(values, distances, -1.0, 1.0, 2.0)


def test_within_between_known_contrast() -> None:
    result = within_between_contrast(
        np.array([0.0, 0.0, 1.0, 1.0]), np.array([0, 0, 1, 1])
    )
    assert result["within"] == 0
    assert result["between"] == pytest.approx(0.5)
    assert result["ratio"] == 0


def test_dependence_diagnostics_returns_separate_result() -> None:
    values = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.5])
    labels = np.array([0, 0, 1, 1, 2, 2])
    lon = np.linspace(-1, 1, 6)
    lat = np.linspace(50, 51, 6)
    times = pd.date_range("2026-01-01", periods=6, freq="h").to_numpy()
    result = dependence_diagnostics(
        values, labels, lon=lon, lat=lat, timestamps=times, seed=3
    )
    assert np.isfinite(result.within_between_ratio)
    assert 1 <= result.n_eff_space <= len(values)
    assert 1 <= result.n_eff_time <= len(values)


@pytest.mark.parametrize(
    "matrix",
    [
        np.ones((2, 3)),
        np.array([[0.0, 1.0], [2.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.0, -1.0], [-1.0, 0.0]]),
    ],
)
def test_bad_distance_matrices_fail(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError, match="dist"):
        effective_n(np.array([1.0, 2.0]), matrix, 0, 1, 1)


def test_spatial_argument_domains_fail() -> None:
    with pytest.raises(ValueError, match="same length"):
        haversine_matrix(np.array([0.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="latitude"):
        haversine_matrix(np.array([0.0]), np.array([100.0]))
    with pytest.raises(ValueError, match="n_bins"):
        empirical_variogram(np.arange(3), np.zeros((3, 3)), n_bins=0)
    with pytest.raises(ValueError, match="same length"):
        fit_variogram(np.arange(3), np.arange(2), np.arange(3) + 1)
    with pytest.raises(ValueError, match="n_perm"):
        morans_i(np.arange(3), np.zeros((3, 3)), cutoff=1, n_perm=0)
    with pytest.raises(ValueError, match="together"):
        dependence_diagnostics(np.arange(3), np.arange(3), lon=np.arange(3))
