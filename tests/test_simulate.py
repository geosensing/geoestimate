import numpy as np
import pytest

from geoinference.simulate import (
    Pipeline,
    PopulationFactory,
    SimConfig,
    collect,
    evaluate_scene,
    run_pipeline,
)


def small_config(**kwargs: object) -> SimConfig:
    values: dict[str, object] = {
        "grid_n": 5,
        "time_grid_n": 8,
        "n_itineraries": 3,
        "n_sims": 4,
        "day_min": 120.0,
        "shift_min": 30.0,
        "dwell_min": 1.0,
        "speed_m_per_min": 100.0,
        "seed": 27,
    }
    values.update(kwargs)
    return SimConfig(**values)  # type: ignore[arg-type]


def test_population_rejects_times_outside_the_day() -> None:
    cfg = small_config()
    population = PopulationFactory(cfg).draw(np.random.default_rng(1))
    with pytest.raises(ValueError, match="simulated day"):
        population.p_at(np.array([0]), np.array([cfg.day_min + 1]))


def test_collection_times_stay_inside_day() -> None:
    cfg = small_config()
    population = PopulationFactory(cfg).draw(np.random.default_rng(1))
    frames = collect(population, Pipeline("compact"), cfg, np.random.default_rng(2))
    assert len(frames) >= cfg.n_itineraries
    assert frames["timestamp"].between(0, cfg.day_min * 60).all()


def test_pipeline_display_name_does_not_change_rng_stream() -> None:
    cfg = small_config()
    factory = PopulationFactory(cfg)
    first = run_pipeline(factory, Pipeline("first", routing="compact"), cfg)
    second = run_pipeline(factory, Pipeline("second", routing="compact"), cfg)
    assert first.bias == second.bias
    assert first.true_sd == second.true_sd
    assert first.coverage == second.coverage


def test_scene_label_does_not_change_rng_stream() -> None:
    cfg = small_config()
    factory = PopulationFactory(cfg)
    sample = np.arange(12)
    itinerary = np.repeat(np.arange(3), 4)
    time_of_day = np.linspace(5, 100, 12)
    timestamps = time_of_day * 60
    first = evaluate_scene(
        factory,
        sample,
        itinerary,
        time_of_day,
        timestamps,
        cfg,
        spatial_diag=False,
        label="one",
    )
    second = evaluate_scene(
        factory,
        sample,
        itinerary,
        time_of_day,
        timestamps,
        cfg,
        spatial_diag=False,
        label="two",
    )
    assert first.bias == second.bias
    assert first.true_sd == second.true_sd
    assert first.coverage == second.coverage


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grid_n": 1},
        {"time_grid_n": 1},
        {"n_itineraries": 0},
        {"n_sims": 0},
        {"extent_deg": 0},
        {"day_min": 0},
        {"shift_min": -1},
        {"shift_min": 121, "day_min": 120},
        {"range_s_m": -1},
        {"sd_t": -1},
    ],
)
def test_bad_simulation_config_fails(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=r"must|cannot"):
        small_config(**kwargs)


def test_population_factory_requires_coordinate_pair() -> None:
    with pytest.raises(ValueError, match="together"):
        PopulationFactory(small_config(), lon=np.array([0.0]))


def test_pipeline_and_method_domains_fail() -> None:
    with pytest.raises(ValueError, match="routing"):
        Pipeline("bad", routing="unknown")
    cfg = small_config()
    with pytest.raises(ValueError, match="se_method"):
        run_pipeline(PopulationFactory(cfg), Pipeline("p"), cfg, se_method="wcb")


def test_scene_arrays_must_align() -> None:
    cfg = small_config()
    with pytest.raises(ValueError, match="aligned"):
        evaluate_scene(
            PopulationFactory(cfg),
            np.array([0, 1]),
            np.array([0]),
            np.array([1.0, 2.0]),
            np.array([60.0, 120.0]),
            cfg,
        )
