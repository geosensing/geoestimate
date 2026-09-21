from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import geoinference.pipeline as pipeline
from geoinference.io import estimate_from_file, main, read_frames
from geoinference.pipeline import (
    Scene,
    assign_visit_times,
    make_scene,
    points_from_roads,
)


def test_points_from_segments_interpolates_interior_points() -> None:
    roads = pd.DataFrame(
        {
            "start_long": [0.0],
            "start_lat": [0.0],
            "end_long": [4.0],
            "end_lat": [2.0],
        }
    )
    points = points_from_roads(roads, per_segment=2)
    assert points["longitude"].tolist() == pytest.approx([1.0, 3.0])
    assert points["latitude"].tolist() == pytest.approx([0.5, 1.5])


def test_assign_visit_times_keeps_visits_inside_shifts() -> None:
    points = pd.DataFrame(
        {"longitude": [0.0, 0.001, 0.002, 0.003], "latitude": [0.0] * 4}
    )
    scene = assign_visit_times(
        points,
        [[0, 1], [2, 3]],
        days=1,
        shifts_per_day=2,
        speed_m_per_min=100,
        dwell_min=1,
        day_minutes=60,
        seed=4,
    )
    assert len(scene) == 4
    assert np.all((scene.time_of_day_min >= 0) & (scene.time_of_day_min <= 60))
    assert np.all(scene.time_of_day_min[scene.itinerary_id == 0] <= 30)
    assert np.all(scene.time_of_day_min[scene.itinerary_id == 1] >= 30)
    assert scene.timestamp_s.max() <= 60 * 60


def test_assign_visit_times_rejects_infeasible_route() -> None:
    points = pd.DataFrame({"longitude": [0.0, 1.0], "latitude": [0.0, 0.0]})
    with pytest.raises(ValueError, match="route 0 needs"):
        assign_visit_times(
            points,
            [[0, 1]],
            days=1,
            speed_m_per_min=1,
            dwell_min=1,
            day_minutes=10,
        )


def test_assign_visit_times_rejects_more_routes_than_slots() -> None:
    points = pd.DataFrame({"longitude": [0.0, 0.1], "latitude": [0.0, 0.1]})
    with pytest.raises(ValueError, match="available day/shift slots"):
        assign_visit_times(points, [[0], [1]], days=1, shifts_per_day=1)


@pytest.mark.parametrize("routes", [[[0, 0]], [[0]], [[0, 2]], [[]]])
def test_assign_visit_times_rejects_bad_routes(routes: list[list[int]]) -> None:
    points = pd.DataFrame({"longitude": [0.0, 0.1], "latitude": [0.0, 0.1]})
    with pytest.raises((TypeError, ValueError)):
        assign_visit_times(points, routes)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"days": 0},
        {"shifts_per_day": 0},
        {"speed_m_per_min": 0},
        {"dwell_min": -1},
        {"day_minutes": np.nan},
    ],
)
def test_assign_visit_times_rejects_bad_operation_values(
    kwargs: dict[str, float],
) -> None:
    points = pd.DataFrame({"longitude": [0.0], "latitude": [0.0]})
    with pytest.raises(ValueError, match="must be"):
        assign_visit_times(points, [[0]], **kwargs)  # pyright: ignore[reportArgumentType]


def test_scene_rejects_misaligned_arrays() -> None:
    with pytest.raises(ValueError, match="aligned"):
        Scene(
            longitude=np.array([0.0, 1.0]),
            latitude=np.array([0.0]),
            itinerary_id=np.array([0, 1]),
            timestamp_s=np.array([0.0, 1.0]),
            time_of_day_min=np.array([0.0, 1.0]),
        )


def test_make_scene_defaults_have_capacity_for_default_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = pd.DataFrame({"longitude": np.zeros(200), "latitude": np.zeros(200)})
    routes = [[index] for index in range(200)]
    monkeypatch.setattr(
        pipeline,
        "build_itineraries",
        lambda *args, **kwargs: (points, routes),
    )
    scene = make_scene(points)
    assert len(scene) == 200
    assert scene.n_itineraries == 200


def test_file_readers_and_estimator(tmp_path: Path) -> None:
    frames = pd.DataFrame(
        {
            "n_women": [1, 2, 2, 3],
            "n_people": [2, 4, 5, 6],
            "itinerary_id": [0, 0, 1, 1],
        }
    )
    csv_path = tmp_path / "frames.csv.gz"
    parquet_path = tmp_path / "frames.parquet"
    frames.to_csv(csv_path, index=False)
    frames.to_parquet(parquet_path, index=False)
    pd.testing.assert_frame_equal(read_frames(csv_path), frames)
    pd.testing.assert_frame_equal(read_frames(parquet_path), frames)
    result = estimate_from_file(parquet_path, cluster_var="itinerary_id")
    assert result.n_clusters == 2


def test_external_parquet_compression_and_unknown_suffix_fail() -> None:
    with pytest.raises(ValueError, match="compressed Parquet"):
        read_frames("frames.parquet.gz")
    with pytest.raises(ValueError, match="infer file format"):
        read_frames("frames.json")


def test_cli_prints_requested_level(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "frames.csv"
    pd.DataFrame({"n_women": [1, 2, 2, 3], "n_people": [2, 4, 5, 6]}).to_csv(
        path, index=False
    )
    main(["estimate", str(path), "--ci-level", "0.8"])
    assert "80% CI" in capsys.readouterr().out
