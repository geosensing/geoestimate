"""Bridge from the real geosensing pipeline to a geoinference simulation scene.

This turns the actual ``geo_sampling`` to ``allocator`` output into a fixed
scene with point coordinates, itinerary membership, and visit times. The scene
``geoinference.simulate.evaluate_scene`` can validate a DGP against, and that
mirrors what the annotated frames look like in production.

``allocator`` and ``geo_sampling`` are optional; install them with
``pip install geoinference[pipeline]`` (or ``uv pip install -e ../allocator
../geo_sampling`` for local checkouts). They are imported lazily so core
geoinference keeps no heavy dependencies.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .spatial import haversine_matrix

_PIPELINE_HINT = "install the pipeline extra:  pip install geoinference[pipeline]"


@dataclass(frozen=True, slots=True)
class Scene:
    """A realized field operation: where, in which itinerary, and when.

    All arrays are per annotated frame and aligned by position.
    """

    longitude: np.ndarray
    latitude: np.ndarray
    itinerary_id: np.ndarray
    timestamp_s: np.ndarray  # absolute seconds across the whole operation
    time_of_day_min: np.ndarray  # minutes within the field "day" (drives diurnal DGP)

    def __post_init__(self) -> None:
        """Validate aligned, finite scene arrays."""
        arrays = {
            "longitude": np.asarray(self.longitude),
            "latitude": np.asarray(self.latitude),
            "itinerary_id": np.asarray(self.itinerary_id),
            "timestamp_s": np.asarray(self.timestamp_s),
            "time_of_day_min": np.asarray(self.time_of_day_min),
        }
        lengths = {len(array) for array in arrays.values() if array.ndim == 1}
        if any(array.ndim != 1 for array in arrays.values()) or len(lengths) != 1:
            raise ValueError("scene arrays must be one-dimensional and aligned")
        for name in ("longitude", "latitude", "timestamp_s", "time_of_day_min"):
            if not np.all(np.isfinite(arrays[name].astype(float))):
                raise ValueError(f"{name} must contain only finite values")
        if pd.isna(arrays["itinerary_id"]).any():
            raise ValueError("itinerary_id cannot contain missing values")

    def __len__(self) -> int:
        """Return the number of frames.

        Returns:
            One per longitude entry.
        """
        return len(self.longitude)

    def to_frame(self) -> pd.DataFrame:
        """Annotated-frame layout (minus the outcome columns)."""
        return pd.DataFrame(
            {
                "itinerary_id": self.itinerary_id,
                "longitude": self.longitude,
                "latitude": self.latitude,
                "timestamp": self.timestamp_s,
            }
        )

    @property
    def n_itineraries(self) -> int:
        """Count the distinct itineraries the frames belong to.

        Returns:
            The number of unique itinerary ids.
        """
        return int(np.unique(self.itinerary_id).size)

    @property
    def day_span(self) -> float:
        """Number of days the operation spans (from absolute timestamps)."""
        if len(self) == 0:
            return 0.0
        return float((self.timestamp_s.max() - self.timestamp_s.min()) / 86_400.0)


def points_from_roads(roads: pd.DataFrame | str, per_segment: int = 1) -> pd.DataFrame:
    """Point locations along road segments, from the road-segment schema.

    Accepts a DataFrame or a CSV path with the ``geo_sampling`` / ``allocator``
    columns ``start_lat, start_long, end_lat, end_long`` (and returns
    ``longitude``/``latitude``). If the frame already has ``longitude`` /
    ``latitude`` it is returned unchanged.

    ``per_segment`` interpolates that many evenly-spaced points along each
    segment (1 = midpoint). Densifying gives a larger candidate universe and
    hence a small sampling fraction for realistic validation.
    """
    if (
        isinstance(per_segment, bool)
        or not isinstance(per_segment, int)
        or per_segment < 1
    ):
        raise ValueError("per_segment must be a positive integer")
    df = pd.read_csv(roads) if isinstance(roads, str) else roads
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("roads must contain at least one row")
    if {"longitude", "latitude"}.issubset(df.columns):
        points = pd.DataFrame(
            {
                "longitude": df["longitude"].to_numpy(dtype=float),
                "latitude": df["latitude"].to_numpy(dtype=float),
            }
        )
        haversine_matrix(points["longitude"].to_numpy(), points["latitude"].to_numpy())
        return points
    needed = {"start_lat", "start_long", "end_lat", "end_long"}
    if not needed.issubset(df.columns):
        raise ValueError(
            f"roads must have {sorted(needed)} or longitude/latitude; "
            f"got {list(df.columns)}"
        )
    s_lon = df["start_long"].to_numpy(dtype=float)
    s_lat = df["start_lat"].to_numpy(dtype=float)
    e_lon = df["end_long"].to_numpy(dtype=float)
    e_lat = df["end_lat"].to_numpy(dtype=float)
    # Fractions at segment-interior points (midpoint for per_segment == 1).
    fracs = (np.arange(per_segment) + 0.5) / per_segment
    lon = (s_lon[:, None] + fracs[None, :] * (e_lon - s_lon)[:, None]).ravel()
    lat = (s_lat[:, None] + fracs[None, :] * (e_lat - s_lat)[:, None]).ravel()
    points = pd.DataFrame({"longitude": lon, "latitude": lat})
    haversine_matrix(points["longitude"].to_numpy(), points["latitude"].to_numpy())
    return points


def build_itineraries(
    points: pd.DataFrame,
    method: str = "random_partition",
    n_itineraries: int | None = None,
    max_distance: float | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, list[list[int]]]:
    """Partition points into itineraries with ``allocator`` (offline haversine).

    Args:
        points: Table containing longitude and latitude columns.
        method: Allocator partitioning method.
        n_itineraries: Requested itinerary count, when supported by the method.
        max_distance: Optional maximum route distance in meters.
        seed: Optional allocator random seed.

    Returns:
        The allocator data table and routes as point-index lists in visit order.

    Raises:
        ImportError: If the optional allocator package is unavailable.
        TypeError: If ``seed`` is not an integer or None.
        ValueError: If points or options are outside their supported domains.
        RuntimeError: If allocator does not return a complete partition.
    """
    if not isinstance(points, pd.DataFrame) or points.empty:
        raise ValueError("points must be a non-empty DataFrame")
    if not {"longitude", "latitude"}.issubset(points.columns):
        raise ValueError("points must contain longitude and latitude")
    haversine_matrix(
        points["longitude"].to_numpy(dtype=float),
        points["latitude"].to_numpy(dtype=float),
    )
    if not method.strip():
        raise ValueError("method must be non-empty")
    if n_itineraries is not None and (
        isinstance(n_itineraries, bool)
        or not isinstance(n_itineraries, int)
        or n_itineraries < 1
        or n_itineraries > len(points)
    ):
        raise ValueError("n_itineraries must lie between 1 and the point count")
    if max_distance is not None and (
        not np.isfinite(max_distance) or max_distance <= 0
    ):
        raise ValueError("max_distance must be finite and positive")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
    try:
        from allocator import (  # pyright: ignore[reportMissingImports]
            create_itineraries,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(f"allocator not available; {_PIPELINE_HINT}") from exc

    result = create_itineraries(
        points,
        method=method,
        n_itineraries=n_itineraries,
        max_distance=max_distance,
        distance="haversine",
        seed=seed,
    )
    routes = [[int(i) for i in route] for route in result.itineraries]
    routed = [point for route in routes for point in route]
    if len(routed) != len(points) or len(set(routed)) != len(points):
        raise RuntimeError("allocator must return every point exactly once")
    return result.data.reset_index(drop=True), routes


def assign_visit_times(
    points: pd.DataFrame,
    routes: list[list[int]],
    *,
    days: int = 14,
    shifts_per_day: int = 1,
    speed_m_per_min: float = 80.0,
    dwell_min: float = 2.0,
    day_minutes: float = 600.0,
    stagger_starts: bool = True,
    seed: int | None = None,
) -> Scene:
    """Spread itineraries over a multi-day operation and time every frame.

    Each itinerary is a single shift assigned to one (day, slot). Within a
    shift, visit times accumulate along the route as travel (haversine metres /
    ``speed_m_per_min``) plus a per-point ``dwell_min``. Shifts are spread
    round-robin across ``days × shifts_per_day`` so timestamps span the whole
    operation while each shift's frames stay close in time-of-day.

    Returns a ``Scene`` whose ``time_of_day_min`` drives the diurnal field and
    whose ``timestamp_s`` is the absolute time fed to ``estimate``'s temporal
    diagnostic.
    """
    if not isinstance(points, pd.DataFrame) or points.empty:
        raise ValueError("points must be a non-empty DataFrame")
    if not {"longitude", "latitude"}.issubset(points.columns):
        raise ValueError("points must contain longitude and latitude")
    for name, value in (("days", days), ("shifts_per_day", shifts_per_day)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (
        ("speed_m_per_min", speed_m_per_min),
        ("dwell_min", dwell_min),
        ("day_minutes", day_minutes),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")

    lon = points["longitude"].to_numpy(dtype=float)
    lat = points["latitude"].to_numpy(dtype=float)
    n = len(lon)
    haversine_matrix(lon, lat)
    if not routes:
        raise ValueError("routes must contain at least one itinerary")
    if any(not route for route in routes):
        raise ValueError("routes cannot contain empty itineraries")
    flat_routes = [point for route in routes for point in route]
    if any(
        isinstance(point, bool) or not isinstance(point, (int, np.integer))
        for point in flat_routes
    ):
        raise TypeError("route indices must be integers")
    if any(point < 0 or point >= n for point in flat_routes):
        raise ValueError("route index is outside the points table")
    if len(flat_routes) != n or len(set(flat_routes)) != n:
        raise ValueError("routes must contain every point exactly once")
    rng = np.random.default_rng(seed)

    itinerary_id = np.full(n, -1, dtype=int)
    time_of_day = np.full(n, np.nan)
    timestamp_s = np.full(n, np.nan)
    n_slots = days * shifts_per_day
    if len(routes) > n_slots:
        raise ValueError(
            f"{len(routes)} routes exceed the {n_slots} available day/shift slots"
        )
    slot_len_min = day_minutes / shifts_per_day

    for k, route in enumerate(routes):
        slot = k
        day = slot // shifts_per_day
        within_day_slot = slot % shifts_per_day
        legs = np.zeros(len(route), dtype=float)
        for position in range(1, len(route)):
            previous, current = route[position - 1], route[position]
            legs[position] = haversine_matrix(
                np.array([lon[previous], lon[current]]),
                np.array([lat[previous], lat[current]]),
            )[0, 1]
        route_duration = float(legs.sum() / speed_m_per_min + len(route) * dwell_min)
        if route_duration > slot_len_min:
            raise ValueError(
                f"route {k} needs {route_duration:.1f} minutes but its shift "
                f"allows {slot_len_min:.1f}"
            )
        slot_start = within_day_slot * slot_len_min
        latest_start = slot_start + slot_len_min - route_duration
        if stagger_starts:
            start_tod = float(rng.uniform(slot_start, latest_start))
        else:
            start_tod = slot_start

        t = start_tod
        for pos, pt in enumerate(route):
            t += legs[pos] / speed_m_per_min
            t += dwell_min
            itinerary_id[pt] = k
            time_of_day[pt] = t
            timestamp_s[pt] = (day * 86_400.0) + t * 60.0
    return Scene(
        longitude=lon,
        latitude=lat,
        itinerary_id=itinerary_id,
        timestamp_s=timestamp_s,
        time_of_day_min=time_of_day,
    )


def make_scene(
    roads: pd.DataFrame | str,
    method: str = "random_partition",
    n_itineraries: int = 200,
    seed: int | None = 0,
    **time_kwargs: object,
) -> Scene:
    """Convenience: roads CSV/frame → points → itineraries → timed ``Scene``."""
    points = points_from_roads(roads)
    data, routes = build_itineraries(
        points, method=method, n_itineraries=n_itineraries, seed=seed
    )
    if time_kwargs.get("shifts_per_day") is None:
        days = time_kwargs.get("days", 14)
        if isinstance(days, bool) or not isinstance(days, int) or days < 1:
            raise ValueError("days must be a positive integer")
        time_kwargs["shifts_per_day"] = int(np.ceil(len(routes) / days))
    return assign_visit_times(data, routes, seed=seed, **time_kwargs)  # type: ignore[arg-type]


def subsample_scene(
    universe: pd.DataFrame | str,
    n_sample: int,
    method: str = "kmeans_tsp",
    n_itineraries: int = 80,
    seed: int = 0,
    stagger_starts: bool = True,
    days: int = 14,
    shifts_per_day: int | None = None,
) -> tuple[np.ndarray, Scene]:
    """Sample a survey out of a city universe and route it into itineraries.

    Treats ``universe`` (all candidate road segments for a city) as the
    population, draws ``n_sample`` of them (SRS as a stand-in for live
    ``geo_sampling``), and routes the sample with the allocator. Returns
    ``(sample_idx, scene)`` where ``sample_idx`` indexes the universe (so the
    field can be drawn on the whole city and the sample scored against the city
    mean in ``geoinference.simulate.evaluate_scene``).
    """
    if isinstance(n_sample, bool) or not isinstance(n_sample, int) or n_sample < 1:
        raise ValueError("n_sample must be a positive integer")
    if (
        isinstance(n_itineraries, bool)
        or not isinstance(n_itineraries, int)
        or n_itineraries < 1
    ):
        raise ValueError("n_itineraries must be a positive integer")
    if n_itineraries > n_sample:
        raise ValueError("n_itineraries cannot exceed n_sample")
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise ValueError("days must be a positive integer")
    if shifts_per_day is not None and (
        isinstance(shifts_per_day, bool)
        or not isinstance(shifts_per_day, int)
        or shifts_per_day < 1
    ):
        raise ValueError("shifts_per_day must be a positive integer or None")
    points = points_from_roads(universe)
    n_uni = len(points)
    if n_sample > n_uni:
        raise ValueError("n_sample cannot exceed the universe size")
    rng = np.random.default_rng(seed)
    sample_idx = np.sort(rng.choice(n_uni, size=n_sample, replace=False))
    pts = points.iloc[sample_idx].reset_index(drop=True)
    data, routes = build_itineraries(
        pts, method=method, n_itineraries=n_itineraries, seed=seed
    )
    if shifts_per_day is None:
        shifts_per_day = int(np.ceil(len(routes) / days))
    scene = assign_visit_times(
        data,
        routes,
        days=days,
        shifts_per_day=shifts_per_day,
        stagger_starts=stagger_starts,
        seed=seed,
    )
    if len(scene) != len(sample_idx):
        raise RuntimeError("allocator dropped points; sample/scene misaligned")
    return sample_idx, scene
