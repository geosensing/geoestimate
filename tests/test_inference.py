import dataclasses

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats as sp_stats

from geoestimate import Estimate, Sample


@pytest.fixture
def frames() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_women": [1.0, 2.0, 4.0, 1.0, 0.0, 3.0],
            "n_people": [2.0, 4.0, 8.0, 5.0, 0.0, 6.0],
            "itinerary_id": ["a", "a", "b", "b", "c", "c"],
        }
    )


def test_sample_contract_and_snapshot(frames: pd.DataFrame) -> None:
    sample = Sample(frames, cluster="itinerary_id")
    assert sample.n_observations == 6
    assert sample.n_clusters == 3
    assert sample.cluster == "itinerary_id"
    assert "n_observations=6" in repr(sample)

    frames["n_people"] = 100
    assert sample.mean("n_people").estimate == pytest.approx(25 / 6)


def test_four_estimands_match_definitions(frames: pd.DataFrame) -> None:
    sample = Sample(frames)
    assert sample.mean("n_people").estimate == pytest.approx(25 / 6)
    assert sample.total("n_people", population_size=60).estimate == pytest.approx(250)
    assert sample.ratio("n_women", "n_people").estimate == pytest.approx(11 / 25)

    positive = frames.loc[frames["n_people"] > 0]
    expected = np.mean(positive["n_women"] / positive["n_people"])
    assert Sample(positive).mean_of_ratios("n_women", "n_people").estimate == (
        pytest.approx(expected)
    )


def test_iid_mean_and_total_standard_errors_scale(frames: pd.DataFrame) -> None:
    sample = Sample(frames)
    mean = sample.mean("n_people")
    total = sample.total("n_people", population_size=60)
    expected = float(frames["n_people"].std(ddof=1) / np.sqrt(len(frames)))
    assert mean.standard_error == pytest.approx(expected)
    assert total.standard_error == pytest.approx(60 * expected)
    assert total.confidence_interval == pytest.approx(
        (
            60 * mean.confidence_interval[0],
            60 * mean.confidence_interval[1],
        )
    )


def test_ratio_iid_standard_error_matches_delta_method(
    frames: pd.DataFrame,
) -> None:
    result = Sample(frames).ratio("n_women", "n_people")
    women = frames["n_women"].to_numpy()
    people = frames["n_people"].to_numpy()
    scores = women - result.estimate * people
    expected = np.sqrt(
        len(frames) * np.sum(scores**2) / ((len(frames) - 1) * people.sum() ** 2)
    )
    assert result.standard_error == pytest.approx(expected)


def test_cluster_standard_error_and_interval_match_hand_calculation(
    frames: pd.DataFrame,
) -> None:
    result = Sample(frames, cluster="itinerary_id").ratio("n_women", "n_people")
    scores = (
        frames["n_women"].to_numpy() - result.estimate * frames["n_people"].to_numpy()
    )
    cluster_scores = np.array(
        [
            scores[frames["itinerary_id"].to_numpy() == label].sum()
            for label in ("a", "b", "c")
        ]
    )
    expected_se = np.sqrt(
        3 / 2 * np.sum(cluster_scores**2) / frames["n_people"].sum() ** 2
    )
    critical = sp_stats.t.ppf(0.975, df=2)
    assert result.inference_method == "cluster"
    assert result.standard_error == pytest.approx(expected_se)
    assert result.confidence_interval == pytest.approx(
        (
            result.estimate - critical * expected_se,
            result.estimate + critical * expected_se,
        )
    )


def test_r_survey_reference_values(frames: pd.DataFrame) -> None:
    clustered = Sample(frames, cluster="itinerary_id")
    assert clustered.mean("n_people").estimate == pytest.approx(4.1666666667)
    assert clustered.mean("n_people").standard_error == pytest.approx(1.1666666667)
    assert clustered.total("n_people", population_size=10).estimate == pytest.approx(
        41.6666666667
    )
    assert clustered.total(
        "n_people", population_size=10
    ).standard_error == pytest.approx(11.6666666667)
    assert clustered.ratio("n_women", "n_people").estimate == pytest.approx(0.44)
    assert clustered.ratio("n_women", "n_people").standard_error == pytest.approx(
        0.0432
    )

    positive = frames.loc[frames["n_people"] > 0]
    mean_ratio = Sample(positive, cluster="itinerary_id").mean_of_ratios(
        "n_women", "n_people"
    )
    assert mean_ratio.estimate == pytest.approx(0.44)
    assert mean_ratio.standard_error == pytest.approx(0.05499091)


def test_rows_and_cluster_names_do_not_change_results(frames: pd.DataFrame) -> None:
    original = Sample(frames, cluster="itinerary_id").ratio("n_women", "n_people")
    changed = frames.sample(frac=1, random_state=7).reset_index(drop=True)
    changed["itinerary_id"] = changed["itinerary_id"].map({"a": 12, "b": -4, "c": 99})
    permuted = Sample(changed, cluster="itinerary_id").ratio("n_women", "n_people")
    assert_allclose(
        [original.estimate, original.standard_error],
        [permuted.estimate, permuted.standard_error],
    )


def test_explicit_iid_inference_on_clustered_sample(frames: pd.DataFrame) -> None:
    sample = Sample(frames, cluster="itinerary_id")
    result = sample.mean("n_people", inference="iid")
    assert result.inference_method == "iid"
    assert result.standard_error == result.diagnostics.iid_standard_error
    assert result.diagnostics.cluster_standard_error is not None


def test_numpy_real_confidence_level_is_supported(frames: pd.DataFrame) -> None:
    result = Sample(frames).mean(
        "n_people",
        confidence_level=np.float32(0.9),  # type: ignore[arg-type]
    )
    assert result.confidence_level == pytest.approx(0.9)


def test_bootstrap_is_reproducible_and_honors_level(frames: pd.DataFrame) -> None:
    sample = Sample(frames, cluster="itinerary_id")
    first = sample.ratio(
        "n_women",
        "n_people",
        inference="bootstrap",
        bootstrap_reps=499,
        seed=8,
    )
    second = sample.ratio(
        "n_women",
        "n_people",
        inference="bootstrap",
        bootstrap_reps=499,
        seed=8,
    )
    narrow = sample.ratio(
        "n_women",
        "n_people",
        inference="bootstrap",
        bootstrap_reps=499,
        seed=8,
        confidence_level=0.8,
    )
    assert first.standard_error == second.standard_error
    assert first.confidence_interval == second.confidence_interval
    span_95 = first.confidence_interval[1] - first.confidence_interval[0]
    span_80 = narrow.confidence_interval[1] - narrow.confidence_interval[0]
    assert span_80 < span_95


def test_bootstrap_redraws_undefined_ratio_samples() -> None:
    data = pd.DataFrame(
        {
            "numerator": [1.0, 0.0, 0.0],
            "denominator": [1.0, 0.0, 0.0],
        }
    )
    result = Sample(data).ratio(
        "numerator",
        "denominator",
        inference="bootstrap",
        bootstrap_reps=20,
        seed=4,
    )
    assert result.estimate == 1
    assert result.standard_error == 0


def test_result_is_immutable_tidy_and_uses_requested_level(
    frames: pd.DataFrame,
) -> None:
    result = Sample(frames).mean("n_people", confidence_level=0.8)
    assert isinstance(result, Estimate)
    assert "80% confidence interval" in result.summary()
    assert "95%" not in result.summary()
    assert result.to_frame().loc[0, "estimate"] == result.estimate
    assert result.to_frame().loc[0, "variable"] == "n_people"
    assert result.to_frame().loc[0, "numerator"] is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.estimate = 0  # type: ignore[misc]


@pytest.mark.parametrize("level", [0, 1, -0.1, 1.1, np.nan, np.inf])
def test_invalid_confidence_level_fails(frames: pd.DataFrame, level: float) -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        Sample(frames).mean("n_people", confidence_level=level)


@pytest.mark.parametrize("reps", [0, 1, -1])
def test_invalid_bootstrap_reps_fail(frames: pd.DataFrame, reps: int) -> None:
    with pytest.raises(ValueError, match="bootstrap_reps"):
        Sample(frames).mean("n_people", bootstrap_reps=reps)


def test_option_types_are_not_coerced(frames: pd.DataFrame) -> None:
    sample = Sample(frames)
    with pytest.raises(TypeError, match="bootstrap_reps"):
        sample.mean("n_people", bootstrap_reps=20.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="seed"):
        sample.mean("n_people", seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inference"):
        sample.mean("n_people", inference="guess")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inference"):
        sample.mean("n_people", inference=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real number"):
        sample.mean("n_people", confidence_level="0.95")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real number"):
        sample.mean("n_people", confidence_level=True)
    with pytest.raises(ValueError, match="requires a declared cluster"):
        sample.mean("n_people", inference="cluster")


def test_sample_rejects_invalid_data_and_clusters(frames: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        Sample([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="two rows"):
        Sample(pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="non-empty"):
        Sample(frames, cluster=" ")
    with pytest.raises(ValueError, match="not found"):
        Sample(frames, cluster="missing")
    missing = frames.copy()
    missing.loc[0, "itinerary_id"] = None
    with pytest.raises(ValueError, match="missing"):
        Sample(missing, cluster="itinerary_id")
    with pytest.raises(ValueError, match="two clusters"):
        Sample(frames.assign(itinerary_id="same"), cluster="itinerary_id")


def test_variables_must_exist_and_be_finite_numeric(frames: pd.DataFrame) -> None:
    sample = Sample(frames)
    with pytest.raises(ValueError, match="non-empty"):
        sample.mean("")
    with pytest.raises(ValueError, match="not found"):
        sample.mean("missing")
    with pytest.raises(ValueError, match="numeric"):
        Sample(frames.assign(label="bad")).mean("label")
    with pytest.raises(ValueError, match="finite"):
        Sample(frames.assign(value=np.nan)).mean("value")
    with pytest.raises(ValueError, match="finite"):
        Sample(frames.assign(value=np.inf)).mean("value")


@pytest.mark.parametrize("population_size", [True, 20.5, "20"])
def test_total_requires_integer_population_size(
    frames: pd.DataFrame, population_size: object
) -> None:
    with pytest.raises(TypeError, match="integer"):
        Sample(frames).total(
            "n_people",
            population_size=population_size,  # type: ignore[arg-type]
        )


def test_total_rejects_population_smaller_than_sample(frames: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="at least the sample size"):
        Sample(frames).total("n_people", population_size=5)


def test_ratio_denominator_domains(frames: pd.DataFrame) -> None:
    sample = Sample(frames)
    with pytest.raises(ValueError, match="nonnegative"):
        Sample(frames.assign(denominator=[1, 1, 1, 1, 1, -1])).ratio(
            "n_women", "denominator"
        )
    with pytest.raises(ValueError, match="positive sample total"):
        Sample(frames.assign(denominator=0)).ratio("n_women", "denominator")
    with pytest.raises(ValueError, match="every denominator to be positive"):
        sample.mean_of_ratios("n_women", "n_people")


def test_diagnostics_match_direct_definitions(frames: pd.DataFrame) -> None:
    result = Sample(frames, cluster="itinerary_id").mean("n_people")
    diagnostics = result.diagnostics
    assert diagnostics.cluster_sizes == (2, 2, 2)
    assert diagnostics.effective_cluster_count == 3
    assert diagnostics.cluster_standard_error == result.standard_error
    assert diagnostics.design_effect is not None
    assert diagnostics.design_effect == pytest.approx(
        (result.standard_error / diagnostics.iid_standard_error) ** 2
    )
    assert diagnostics.effective_sample_size == pytest.approx(
        len(frames) / diagnostics.design_effect
    )
