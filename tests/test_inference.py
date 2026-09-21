import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from geoinference import PointDesign, WalkDesign, estimate


@pytest.fixture
def frames() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_women": [1.0, 2.0, 4.0, 1.0, 0.0, 3.0],
            "n_people": [2.0, 4.0, 8.0, 5.0, 0.0, 6.0],
            "itinerary_id": ["a", "a", "b", "b", "c", "c"],
        }
    )


def test_point_design_contract() -> None:
    independent = PointDesign()
    clustered = PointDesign(cluster_var="itinerary_id")
    assert independent.name == "point_equal_probability"
    assert independent.recommended_se_method == "naive"
    assert clustered.name == "point_equal_probability_clustered"
    assert clustered.recommended_se_method == "cluster"
    with pytest.raises(ValueError, match="non-empty"):
        PointDesign(cluster_var=" ")
    with pytest.raises(TypeError):
        PointDesign(sampling="pps")  # type: ignore[call-arg]


def test_walk_design_contract() -> None:
    design = WalkDesign()
    assert design.cluster_var == "walk_id"
    assert design.recommended_se_method == "cluster"
    with pytest.raises(ValueError, match="non-empty"):
        WalkDesign(walk_var="")
    with pytest.raises(TypeError):
        WalkDesign(spacing_m=10)  # type: ignore[call-arg]


def test_estimands_match_definitions(frames: pd.DataFrame) -> None:
    result = estimate(frames)
    assert result.ratio == pytest.approx(11 / 25)
    expected_mean = np.mean([0.5, 0.5, 0.5, 0.2, 0.5])
    assert result.photo_mean == pytest.approx(expected_mean)
    assert result.ratio_se.method_used == "naive"
    assert result.ratio_ci.method_used == "normal"


def test_naive_ratio_se_matches_delta_method(frames: pd.DataFrame) -> None:
    result = estimate(frames)
    women = frames["n_women"].to_numpy()
    people = frames["n_people"].to_numpy()
    scores = women - result.ratio * people
    expected = np.sqrt(
        len(frames) * np.sum(scores**2) / ((len(frames) - 1) * people.sum() ** 2)
    )
    assert result.ratio_se.naive == pytest.approx(expected)


def test_cluster_se_matches_hand_calculation(frames: pd.DataFrame) -> None:
    design = PointDesign(cluster_var="itinerary_id")
    result = estimate(frames, design=design)
    scores = frames["n_women"].to_numpy() - result.ratio * frames["n_people"].to_numpy()
    cluster_scores = np.array(
        [
            scores[frames["itinerary_id"].to_numpy() == label].sum()
            for label in ("a", "b", "c")
        ]
    )
    expected = np.sqrt(
        3 / 2 * np.sum(cluster_scores**2) / frames["n_people"].sum() ** 2
    )
    assert result.ratio_se.cluster == pytest.approx(expected)
    assert result.ratio_se.recommended == pytest.approx(expected)
    assert result.ratio_ci.method_used == "t"


def test_rows_and_cluster_names_do_not_change_results(frames: pd.DataFrame) -> None:
    design = PointDesign(cluster_var="itinerary_id")
    original = estimate(frames, design=design)
    changed = frames.sample(frac=1, random_state=7).reset_index(drop=True)
    changed["itinerary_id"] = changed["itinerary_id"].map({"a": 12, "b": -4, "c": 99})
    permuted = estimate(changed, design=design)
    assert_allclose(
        [
            original.ratio,
            original.photo_mean,
            original.ratio_se.recommended,
            original.photo_mean_se.recommended,
        ],
        [
            permuted.ratio,
            permuted.photo_mean,
            permuted.ratio_se.recommended,
            permuted.photo_mean_se.recommended,
        ],
    )


def test_bootstrap_is_reproducible_and_honors_level(frames: pd.DataFrame) -> None:
    design = PointDesign(cluster_var="itinerary_id")
    first = estimate(frames, design=design, bootstrap=True, bootstrap_reps=499, seed=8)
    second = estimate(frames, design=design, bootstrap=True, bootstrap_reps=499, seed=8)
    narrow = estimate(
        frames,
        design=design,
        bootstrap=True,
        bootstrap_reps=499,
        seed=8,
        ci_level=0.8,
    )
    assert first.ratio_ci.bootstrap == second.ratio_ci.bootstrap
    assert first.ratio_se.bootstrap == second.ratio_se.bootstrap
    assert first.ratio_ci.bootstrap is not None
    assert narrow.ratio_ci.bootstrap is not None
    span_95 = first.ratio_ci.bootstrap[1] - first.ratio_ci.bootstrap[0]
    span_80 = narrow.ratio_ci.bootstrap[1] - narrow.ratio_ci.bootstrap[0]
    assert span_80 < span_95


def test_explicit_bootstrap_selection(frames: pd.DataFrame) -> None:
    design = PointDesign(cluster_var="itinerary_id")
    result = estimate(
        frames,
        design=design,
        bootstrap=True,
        bootstrap_reps=199,
        se_method="bootstrap",
    )
    assert result.ratio_se.method_used == "bootstrap"
    assert result.ratio_ci.method_used == "bootstrap"
    with pytest.raises(ValueError, match="unavailable"):
        estimate(
            frames,
            design=design,
            se_method="bootstrap",
            ci_method="normal",
        )
    with pytest.raises(ValueError, match="unavailable"):
        estimate(frames, design=design, ci_method="bootstrap")


def test_cluster_method_requires_cluster_design(frames: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unavailable"):
        estimate(frames, se_method="cluster")
    with pytest.raises(ValueError, match="unavailable"):
        estimate(frames, ci_method="t")


def test_summary_uses_requested_level(frames: pd.DataFrame) -> None:
    result = estimate(frames, ci_level=0.8)
    summary = result.summary()
    assert "80% CI" in summary
    assert "95% CI" not in summary


@pytest.mark.parametrize("level", [0, 1, -0.1, 1.1, np.nan, np.inf, True])
def test_invalid_ci_level_fails(frames: pd.DataFrame, level: float) -> None:
    with pytest.raises(ValueError, match="ci_level"):
        estimate(frames, ci_level=level)


@pytest.mark.parametrize("reps", [0, 1, -1])
def test_invalid_bootstrap_reps_fail(frames: pd.DataFrame, reps: int) -> None:
    with pytest.raises(ValueError, match="bootstrap_reps"):
        estimate(frames, bootstrap=True, bootstrap_reps=reps)


def test_bootstrap_reps_type_is_not_coerced(frames: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match="bootstrap_reps"):
        estimate(frames, bootstrap=True, bootstrap_reps=20.5)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("women", "people", "message"),
    [
        ([-1, 0], [1, 1], "nonnegative"),
        ([2, 0], [1, 1], "cannot exceed"),
        ([0, 0], [0, 0], "observed person"),
        ([np.nan, 0], [1, 1], "finite"),
        ([np.inf, 0], [1, 1], "finite"),
        ([1, 0], [1, 0], "two frames"),
    ],
)
def test_invalid_count_domains_fail(
    women: list[float], people: list[float], message: str
) -> None:
    data = pd.DataFrame({"n_women": women, "n_people": people})
    with pytest.raises(ValueError, match=message):
        estimate(data)


def test_empty_and_single_row_data_fail() -> None:
    with pytest.raises(ValueError, match="two rows"):
        estimate(pd.DataFrame(columns=["n_women", "n_people"]))
    with pytest.raises(ValueError, match="two rows"):
        estimate(pd.DataFrame({"n_women": [1], "n_people": [2]}))


def test_missing_and_nonnumeric_columns_fail(frames: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="columns not found"):
        estimate(frames.drop(columns="n_women"))
    bad = frames.copy()
    bad["n_women"] = "bad"
    with pytest.raises(ValueError, match="numeric"):
        estimate(bad)


def test_cluster_labels_must_exist_and_be_complete(frames: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="cluster column"):
        estimate(frames, design=PointDesign(cluster_var="missing"))
    missing = frames.copy()
    missing.loc[0, "itinerary_id"] = None
    with pytest.raises(ValueError, match="missing"):
        estimate(missing, design=PointDesign(cluster_var="itinerary_id"))
    one = frames.assign(itinerary_id="same")
    with pytest.raises(ValueError, match="at least two clusters"):
        estimate(one, design=PointDesign(cluster_var="itinerary_id"))


def test_photo_mean_requires_positive_frames_in_two_clusters(
    frames: pd.DataFrame,
) -> None:
    data = frames.copy()
    data.loc[data["itinerary_id"] != "a", ["n_women", "n_people"]] = 0
    with pytest.raises(ValueError, match="at least two clusters"):
        estimate(data, design=PointDesign(cluster_var="itinerary_id"))


def test_method_names_fail_at_boundary(frames: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="se_method"):
        estimate(frames, se_method="guess")
    with pytest.raises(ValueError, match="ci_method"):
        estimate(frames, ci_method="guess")
