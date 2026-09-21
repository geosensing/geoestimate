"""Space-time Monte Carlo harness for stress-testing itinerary-based collection.

The point of this module is *limiting-case reasoning*. We simulate a known
space-time data-generating process (DGP) with no measurement noise, run real
collection pipelines (sampling + routing under a fixed per-route time budget)
on top, push the assumptions to their extremes, and check what happens to the
bias, standard-error calibration, and confidence-interval coverage that
``geoestimate.Sample`` produces.

The analytically-predictable limits (the "truth table"):

Spatial correlation length ``range_s_m`` vs. the route's inter-point spacing
    - 0  (white noise): information ∝ point count → COMPACT routing is BLUE;
      ICC≈0, deff≈1, naive SE correct.
    - ∞  (field constant in space): n_eff_space → 1; extra spatial points carry
      no information; only the time axis can buy precision.

Temporal structure (``diurnal_amp``, ``range_t_min``, start-time policy)
    - none: start-time policy is irrelevant, no temporal bias.
    - strong diurnal + synchronized starts: β̂ is biased toward the sampled
      time-of-day by a fixed amount; more points do NOT fix it; staggering
      start times across the day does.

Key separations the extremes make obvious:
    - Correlation moves VARIANCE / SE / coverage, never the bias of the mean.
    - Bias comes from SELECTION: spatial coverage gaps and non-representative
      time windows.
    - The coverage cliff (naive SE under compact routing) is an INTERIOR
      phenomenon: benign at both correlation extremes, worst in the middle.

Everything works in lon/lat with great-circle distances so it exercises the
same code path as real data.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .sample import Sample
from .spatial import dependence_diagnostics, haversine_matrix
from .types import InferenceMethod


def _expit(x: np.ndarray) -> np.ndarray:
    """Logistic transform from the logit scale to a probability/ratio in (0, 1)."""
    out: np.ndarray = 1.0 / (1.0 + np.exp(-x))
    return out


# ─── Configuration ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Parameters of the space-time DGP and the field operation.

    Defaults describe a ~5.5 km square region surveyed in shifts of a few
    hours. Push any parameter to an extreme to read off a limiting case.
    """

    # Spatial population (a grid of candidate locations).
    grid_n: int = 18  # grid is grid_n × grid_n candidate points
    extent_deg: float = 0.05  # box side in degrees (~5.5 km)
    range_s_m: float = 800.0  # spatial correlation length (meters)
    sd_s: float = 1.0  # spatial latent SD (logit scale)
    base_logit: float = -0.85  # baseline (≈0.30 ratio at the mean)

    # Temporal population (over a daily window).
    day_min: float = 600.0  # length of the survey day (minutes)
    range_t_min: float = 60.0  # temporal correlation length (minutes)
    diurnal_amp: float = 0.0  # amplitude of the time-of-day trend (logit)
    sd_t: float = 0.0  # temporal stochastic SD (logit scale)
    time_grid_n: int = 40  # resolution of the temporal population grid

    # Field operation (one shift = one itinerary).
    n_itineraries: int = 8  # routes per sample (= number of clusters)
    shift_min: float = 120.0  # time budget per route (minutes)
    speed_m_per_min: float = 80.0  # walking speed
    dwell_min: float = 2.0  # annotation dwell per point

    # Monte Carlo.
    n_sims: int = 200
    seed: int = 12345

    def grid_side(self) -> int:
        """Number of candidate points per grid side (grid is this squared)."""
        return self.grid_n

    def __post_init__(self) -> None:
        """Validate the simulation domain."""
        for name in ("grid_n", "time_grid_n", "n_itineraries", "n_sims"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.grid_n < 2 or self.time_grid_n < 2:
            raise ValueError("grid_n and time_grid_n must be at least 2")
        positive = (
            "extent_deg",
            "day_min",
            "shift_min",
            "speed_m_per_min",
            "dwell_min",
        )
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        finite = (
            "range_s_m",
            "sd_s",
            "base_logit",
            "range_t_min",
            "diurnal_amp",
            "sd_t",
        )
        for name in finite:
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.range_s_m < 0 or self.sd_s < 0 or self.range_t_min < 0 or self.sd_t < 0:
            raise ValueError("ranges and standard deviations must be nonnegative")
        if self.shift_min > self.day_min:
            raise ValueError("shift_min cannot exceed day_min")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")


# ─── DGP: draw a space-time population ────────────────────────────────


@dataclass(frozen=True, slots=True)
class Population:
    """One realized space-time population (no measurement noise)."""

    lon: np.ndarray  # (N,) candidate longitudes
    lat: np.ndarray  # (N,) candidate latitudes
    dist_m: np.ndarray  # (N,N) haversine distances
    g_s: np.ndarray  # (N,) spatial latent
    t_grid: np.ndarray  # (T,) population time grid (minutes)
    temporal: np.ndarray  # (T,) temporal latent on t_grid
    base_logit: float
    beta_true: float  # space-time average ratio (the estimand)

    def __post_init__(self) -> None:
        """Validate aligned geometry and process arrays."""
        n = len(self.lon)
        if n < 2 or self.lon.shape != (n,) or self.lat.shape != (n,):
            raise ValueError(
                "lon and lat must be aligned vectors with at least two points"
            )
        if self.g_s.shape != (n,) or self.dist_m.shape != (n, n):
            raise ValueError("spatial arrays must align with lon and lat")
        if self.t_grid.ndim != 1 or self.temporal.shape != self.t_grid.shape:
            raise ValueError("temporal arrays must be aligned vectors")
        arrays = (self.lon, self.lat, self.dist_m, self.g_s, self.t_grid, self.temporal)
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("population arrays must contain only finite values")
        if not np.isfinite(self.base_logit) or not 0 <= self.beta_true <= 1:
            raise ValueError("population scalars are outside their domains")

    def p_at(self, idx: np.ndarray, times_min: np.ndarray) -> np.ndarray:
        """True ratio p(s_i, t_i) at points ``idx`` visited at ``times_min``."""
        idx = np.asarray(idx, dtype=int)
        times_min = np.asarray(times_min, dtype=float)
        if idx.ndim != 1 or times_min.ndim != 1 or len(idx) != len(times_min):
            raise ValueError("idx and times_min must be aligned one-dimensional arrays")
        if np.any(idx < 0) or np.any(idx >= len(self.lon)):
            raise ValueError("idx contains a location outside the population")
        if not np.all(np.isfinite(times_min)):
            raise ValueError("times_min must contain only finite values")
        if np.any(times_min < self.t_grid[0]) or np.any(times_min > self.t_grid[-1]):
            raise ValueError("times_min must lie inside the simulated day")
        tval = np.interp(times_min, self.t_grid, self.temporal)
        return _expit(self.base_logit + self.g_s[idx] + tval)


def _build_grid(cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """Regular lon/lat grid of candidate locations covering the region."""
    side = cfg.grid_side()
    axis = np.linspace(0.0, cfg.extent_deg, side)
    lon_g, lat_g = np.meshgrid(axis, axis)
    return lon_g.ravel(), lat_g.ravel()


def _spatial_chol(dist_m: np.ndarray, range_s_m: float, sd_s: float) -> np.ndarray:
    """Cholesky factor of the exponential spatial covariance.

    ``range_s_m <= 0`` ⇒ white noise (identity); very large ⇒ ~constant field.
    """
    n = dist_m.shape[0]
    if range_s_m <= 0:
        return np.eye(n) * sd_s
    cov = (sd_s**2) * np.exp(-dist_m / range_s_m)
    cov += 1e-9 * np.eye(n)
    return np.linalg.cholesky(cov)


def _temporal_chol(t_grid: np.ndarray, range_t_min: float, sd_t: float) -> np.ndarray:
    """Cholesky factor of the exponential temporal covariance on the time grid.

    Returns a zero matrix when there is no stochastic temporal component
    (``sd_t <= 0`` or ``range_t_min <= 0``).
    """
    n = len(t_grid)
    if sd_t <= 0 or range_t_min <= 0:
        return np.zeros((n, n))
    d = np.abs(t_grid[:, None] - t_grid[None, :])
    cov = (sd_t**2) * np.exp(-d / range_t_min) + 1e-9 * np.eye(n)
    return np.linalg.cholesky(cov)


class PopulationFactory:
    """Precomputes the geometry + covariance factors, then draws populations.

    Factoring once per configuration keeps the Monte Carlo loop cheap: each
    draw is just two matrix-vector products.
    """

    def __init__(
        self,
        cfg: SimConfig,
        lon: np.ndarray | None = None,
        lat: np.ndarray | None = None,
    ) -> None:
        """Build geometry + covariance factors.

        With ``lon``/``lat`` given, the field is defined on those real point
        locations (e.g. allocator/geo_sampling output); otherwise a regular
        ``grid_n × grid_n`` grid is built from ``cfg``.
        """
        self.cfg = cfg
        if (lon is None) != (lat is None):
            raise ValueError("lon and lat must be supplied together")
        if lon is not None and lat is not None:
            self.lon = np.asarray(lon, dtype=float)
            self.lat = np.asarray(lat, dtype=float)
            if (
                self.lon.ndim != 1
                or self.lat.shape != self.lon.shape
                or len(self.lon) < 2
            ):
                raise ValueError(
                    "lon and lat must be aligned vectors with at least two points"
                )
        else:
            self.lon, self.lat = _build_grid(cfg)
        self.dist_m = haversine_matrix(self.lon, self.lat)
        self.l_s = _spatial_chol(self.dist_m, cfg.range_s_m, cfg.sd_s)
        self.t_grid = np.linspace(0.0, cfg.day_min, cfg.time_grid_n)
        self.l_t = _temporal_chol(self.t_grid, cfg.range_t_min, cfg.sd_t)
        # Deterministic diurnal trend (one full cycle over the day).
        self.diurnal = cfg.diurnal_amp * np.sin(2.0 * np.pi * self.t_grid / cfg.day_min)

    def draw(self, rng: np.random.Generator) -> Population:
        """Draw one space-time population and its true space-time-average ratio."""
        cfg = self.cfg
        n = len(self.lon)
        g_s = self.l_s @ rng.standard_normal(n)
        temporal = self.diurnal + self.l_t @ rng.standard_normal(len(self.t_grid))

        # Estimand: the space-time average ratio over all locations × the day.
        # E_{s,t}[ expit(base + g_s(s) + temporal(t)) ].
        tval = temporal[None, :]  # (1,T)
        latent = cfg.base_logit + g_s[:, None] + tval  # (N,T)
        beta_true = float(_expit(latent).mean())

        return Population(
            lon=self.lon,
            lat=self.lat,
            dist_m=self.dist_m,
            g_s=g_s,
            t_grid=self.t_grid,
            temporal=temporal,
            base_logit=cfg.base_logit,
            beta_true=beta_true,
        )


# ─── Collection pipelines (routing under a time budget) ──────────────


@dataclass(frozen=True, slots=True)
class Pipeline:
    """A named collection strategy.

    Args:
        routing: How the next point is chosen. "compact" selects the nearest unvisited,
            "dispersed" (farthest from already-visited, i.e. maximin),
            "systematic" (a fixed space-filling order from a random offset),
            or "srs" (uniformly random next).
        staggered_starts: If True, route start times are spread across the day;
            if False, all routes start at the same time-of-day (synchronized).
    """

    name: str
    routing: str = "compact"
    staggered_starts: bool = True

    def __post_init__(self) -> None:
        """Validate the routing strategy and display name."""
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        allowed = {"compact", "dispersed", "systematic", "srs"}
        if self.routing not in allowed:
            raise ValueError(f"routing must be one of {sorted(allowed)}")


def _systematic_order(pop: Population) -> np.ndarray:
    """A boustrophedon ("snake") ordering of grid points for even coverage."""
    order: list[int] = []
    for band, latitude in enumerate(np.unique(pop.lat)):
        row = np.flatnonzero(pop.lat == latitude)
        row = row[np.argsort(pop.lon[row])]
        if band % 2:
            row = row[::-1]
        order.extend(row.tolist())
    return np.asarray(order, dtype=int)


def _route(
    pop: Population,
    pipe: Pipeline,
    start_time_min: float,
    cfg: SimConfig,
    rng: np.random.Generator,
    systematic_order: np.ndarray,
) -> tuple[list[int], list[float], float]:
    """Walk one route under the time budget; return (idx, visit_times, dist_m)."""
    n = len(pop.lon)
    visited: list[int] = []
    times: list[float] = []
    total_dist = 0.0
    t = start_time_min
    budget = cfg.shift_min

    start = int(rng.integers(n))
    cur = start
    available = np.ones(n, dtype=bool)

    # Position within the systematic order (start at a random offset).
    sys_pos = int(rng.integers(n)) if pipe.routing == "systematic" else 0

    while True:
        if available[cur]:
            # Cost to "collect" the current point is the dwell time.
            if t - start_time_min + cfg.dwell_min > budget:
                break
            t += cfg.dwell_min
            visited.append(cur)
            times.append(t)
            available[cur] = False

        if not available.any():
            break

        # Choose the next point per the routing policy.
        cand = np.where(available)[0]
        if pipe.routing == "compact":
            nxt = int(cand[np.argmin(pop.dist_m[cur, cand])])
        elif pipe.routing == "dispersed":
            if visited:
                dmin = pop.dist_m[np.ix_(cand, np.array(visited, dtype=int))].min(
                    axis=1
                )
                nxt = int(cand[np.argmax(dmin)])
            else:
                nxt = int(cand[np.argmax(pop.dist_m[cur, cand])])
        elif pipe.routing == "srs":
            nxt = int(rng.choice(cand))
        elif pipe.routing == "systematic":
            nxt = cur
            for _ in range(n):
                sys_pos = (sys_pos + 1) % n
                c = int(systematic_order[sys_pos])
                if available[c]:
                    nxt = c
                    break
        else:
            raise ValueError(f"Unknown routing: {pipe.routing!r}")

        travel = float(pop.dist_m[cur, nxt])
        travel_time = travel / cfg.speed_m_per_min
        if t - start_time_min + travel_time + cfg.dwell_min > budget:
            break
        t += travel_time
        total_dist += travel
        cur = nxt

    return visited, times, total_dist


def collect(
    pop: Population, pipe: Pipeline, cfg: SimConfig, rng: np.random.Generator
) -> pd.DataFrame:
    """Run all K routes of a pipeline and return an annotated-frame DataFrame."""
    systematic_order = _systematic_order(pop)
    rows_idx: list[int] = []
    rows_time: list[float] = []
    rows_itin: list[int] = []
    total_dist = 0.0

    for k in range(cfg.n_itineraries):
        if pipe.staggered_starts:
            feasible_window = cfg.day_min - cfg.shift_min
            start_t = feasible_window * (k + 0.5) / cfg.n_itineraries
        else:
            start_t = 0.0
        idx, times, dist = _route(pop, pipe, start_t, cfg, rng, systematic_order)
        total_dist += dist
        rows_idx.extend(idx)
        rows_time.extend(times)
        rows_itin.extend([k] * len(idx))

    idx_arr = np.array(rows_idx, dtype=int)
    time_arr = np.array(rows_time, dtype=float)
    p = pop.p_at(idx_arr, time_arr)

    df = pd.DataFrame(
        {
            "n_women": p,  # no measurement noise: w = p
            "n_people": np.ones(len(p)),  # h = 1, so ratio == mean(p)
            "itinerary_id": rows_itin,
            "longitude": pop.lon[idx_arr],
            "latitude": pop.lat[idx_arr],
            "timestamp": time_arr * 60.0,  # minutes → epoch-like seconds
        }
    )
    df.attrs["total_dist_m"] = total_dist
    return df


# ─── Monte Carlo runner and metrics ──────────────────────────────────


@dataclass
class PipelineResult:
    """Monte Carlo metrics for one (pipeline, SE method) combination.

    ``bias`` and ``true_sd`` describe the sampling distribution of beta-hat
    across sims; ``se_sd_ratio`` (mean SE / true SD) and ``coverage`` describe
    whether the reported uncertainty is honest.
    """

    name: str
    se_method: str
    n_sims: int
    mean_n: float
    mean_dist_km: float
    bias: float
    true_sd: float
    mean_se: float
    se_sd_ratio: float  # mean SE / true SD; ~1 is well-calibrated
    coverage: float  # CI coverage of beta_true (nominal = ci_level)
    mean_n_eff_space: float
    mean_within_between: float

    def as_row(self) -> dict[str, float | str | int]:
        """Return a flat, rounded dict of metrics for tabular display."""
        return {
            "pipeline": self.name,
            "SE": self.se_method,
            "n": round(self.mean_n, 1),
            "dist_km": round(self.mean_dist_km, 2),
            "bias": round(self.bias, 4),
            "true_SD": round(self.true_sd, 4),
            "mean_SE": round(self.mean_se, 4),
            "SE/SD": round(self.se_sd_ratio, 2),
            "coverage": round(self.coverage, 3),
            "n_eff_sp": round(self.mean_n_eff_space, 1),
            "win/btwn": round(self.mean_within_between, 2),
        }


def _inference_method(se_method: str) -> InferenceMethod:
    """Translate simulation vocabulary to the stable inference vocabulary."""
    methods: dict[str, InferenceMethod] = {
        "auto": "design",
        "naive": "iid",
        "cluster": "cluster",
        "boot": "bootstrap",
    }
    return methods[se_method]


def _aggregate(
    name: str,
    se_method: str,
    diffs: list[float],
    ses: list[float],
    covers: int,
    ns: list[float],
    dists: list[float],
    neff: list[float],
    wb: list[float],
    valid: int,
) -> PipelineResult:
    """Collapse per-sim records into a ``PipelineResult``."""
    diffs_a = np.array(diffs)
    finite_neff = np.asarray(neff, dtype=float)
    finite_neff = finite_neff[np.isfinite(finite_neff)]
    finite_wb = np.asarray(wb, dtype=float)
    finite_wb = finite_wb[np.isfinite(finite_wb)]
    sd = float(np.std(diffs_a, ddof=1)) if valid > 1 else float("nan")
    return PipelineResult(
        name=name,
        se_method=se_method,
        n_sims=valid,
        mean_n=float(np.mean(ns)) if ns else float("nan"),
        mean_dist_km=float(np.mean(dists)) if dists else float("nan"),
        bias=float(np.mean(diffs_a)) if valid else float("nan"),
        true_sd=sd,
        mean_se=float(np.mean(ses)) if ses else float("nan"),
        se_sd_ratio=float(np.mean(ses) / sd) if valid > 1 and sd > 0 else float("nan"),
        coverage=covers / valid if valid else float("nan"),
        mean_n_eff_space=(
            float(np.mean(finite_neff)) if finite_neff.size else float("nan")
        ),
        mean_within_between=(
            float(np.mean(finite_wb)) if finite_wb.size else float("nan")
        ),
    )


def run_pipeline(
    factory: PopulationFactory,
    pipe: Pipeline,
    cfg: SimConfig,
    se_method: str = "auto",
    ci_level: float = 0.95,
    spatial_diag: bool = False,
) -> PipelineResult:
    """Monte Carlo a single synthetic pipeline; return aggregated metrics.

    Args:
        factory: Precomputed population factory (geometry + covariance).
        pipe: The collection strategy to simulate (synthetic routing).
        cfg: Simulation configuration (DGP + field operation + Monte Carlo).
        se_method: Which standard error drives the CI: "auto", "naive",
            "cluster" (analytic robust with t_{G-1}), or "boot" (pairs bootstrap).
        ci_level: Nominal confidence level.
        spatial_diag: If True, also collect the spatial dependence diagnostics.

    Returns:
        A ``PipelineResult`` over ``cfg.n_sims`` simulations.

    Raises:
        ValueError: If the method or confidence level is unsupported.
    """
    if se_method not in {"auto", "naive", "cluster", "boot"}:
        raise ValueError("se_method must be 'auto', 'naive', 'cluster', or 'boot'")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be strictly between 0 and 1")
    diffs: list[float] = []
    ses: list[float] = []
    covers = 0
    ns: list[float] = []
    dists: list[float] = []
    neff: list[float] = []
    wb: list[float] = []
    valid = 0

    for simulation in range(cfg.n_sims):
        pop_rng = np.random.default_rng(
            np.random.SeedSequence([cfg.seed, simulation, 0])
        )
        route_rng = np.random.default_rng(
            np.random.SeedSequence([cfg.seed, simulation, 1])
        )
        pop = factory.draw(pop_rng)
        df = collect(pop, pipe, cfg, route_rng)
        if len(df) < cfg.n_itineraries:
            continue
        res = Sample(df, cluster="itinerary_id").ratio(
            "n_women",
            "n_people",
            inference=_inference_method(se_method),
            confidence_level=ci_level,
            bootstrap_reps=599,
            seed=simulation,
        )
        se = res.standard_error
        lo, hi = res.confidence_interval
        if not (np.isfinite(se) and np.isfinite(lo) and np.isfinite(hi)):
            continue
        valid += 1
        diffs.append(res.estimate - pop.beta_true)
        ses.append(se)
        covers += int(lo <= pop.beta_true <= hi)
        ns.append(float(len(df)))
        dists.append(float(df.attrs["total_dist_m"]) / 1000.0)
        if spatial_diag:
            diag = dependence_diagnostics(
                df["n_women"].to_numpy(),
                df["itinerary_id"].to_numpy(),
                lon=df["longitude"].to_numpy(),
                lat=df["latitude"].to_numpy(),
                timestamps=df["timestamp"].to_numpy(),
                seed=simulation,
            )
            neff.append(diag.n_eff_space)
            wb.append(diag.within_between_ratio)

    return _aggregate(
        pipe.name, se_method, diffs, ses, covers, ns, dists, neff, wb, valid
    )


def evaluate_scene(
    factory: PopulationFactory,
    sample_idx: np.ndarray,
    itinerary_id: np.ndarray,
    time_of_day_min: np.ndarray,
    timestamp_s: np.ndarray,
    cfg: SimConfig,
    se_method: str = "auto",
    ci_level: float = 0.95,
    spatial_diag: bool = True,
    label: str = "scene",
) -> PipelineResult:
    """Design-conditional Monte Carlo of a realized survey against the city mean.

    ``factory`` carries the field on the **whole city universe**; the survey is
    a fixed **subsample** of it (``sample_idx``) with a fixed itinerary
    partition and visit times (you ran geo_sampling + the allocator once). Each
    simulation redraws the outcome field, estimates the ratio from the observed
    sample, and checks whether the CI covers the *universe* space-time mean.
    Because the sample is a strict subset of the universe, β̂ carries genuine
    spatial sampling error. The cluster SE must capture that error, so this
    is a real test of whether ``estimate``'s SE/CI is honest for that design.

    Args:
        factory: ``PopulationFactory(cfg, lon, lat)`` built on ALL city points.
        sample_idx: Indices (into the factory universe) of the observed frames.
        itinerary_id: Per-frame cluster labels, aligned to ``sample_idx``.
        time_of_day_min: Per-frame time-of-day in ``[0, day_min]`` minutes. This
            drives the diurnal/temporal field component.
        timestamp_s: Per-frame absolute timestamp in seconds across the whole
            operation, used by the optional temporal diagnostics.
        cfg: Simulation configuration (DGP + Monte Carlo).
        se_method: SE estimator to use, or "auto" to take the design's.
        ci_level: Coverage the interval claims.
        spatial_diag: Whether to accumulate the spatial dependence diagnostics.
        label: Display name for the result row.

    Returns:
        A ``PipelineResult`` over ``cfg.n_sims`` field redraws.

    Raises:
        ValueError: If the method, confidence level, or scene arrays are invalid.
    """
    if se_method not in {"auto", "naive", "cluster", "boot"}:
        raise ValueError("se_method must be 'auto', 'naive', 'cluster', or 'boot'")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be strictly between 0 and 1")
    idx = np.asarray(sample_idx, dtype=int)
    m = len(idx)
    tod = np.asarray(time_of_day_min, dtype=float)
    itin = np.asarray(itinerary_id)
    ts = np.asarray(timestamp_s, dtype=float)
    if any(array.ndim != 1 for array in (idx, tod, itin, ts)):
        raise ValueError("scene arrays must be one-dimensional")
    if not (len(idx) == len(tod) == len(itin) == len(ts)) or m < 2:
        raise ValueError("scene arrays must be aligned and contain at least two rows")
    if np.any(idx < 0) or np.any(idx >= len(factory.lon)):
        raise ValueError("sample_idx contains an index outside the population")
    if not np.all(np.isfinite(tod)) or not np.all(np.isfinite(ts)):
        raise ValueError("scene times must be finite")
    if pd.isna(itin).any():
        raise ValueError("itinerary_id cannot contain missing values")
    lon_s = factory.lon[idx]
    lat_s = factory.lat[idx]

    diffs: list[float] = []
    ses: list[float] = []
    covers = 0
    neff: list[float] = []
    wb: list[float] = []
    valid = 0

    for simulation in range(cfg.n_sims):
        pop_rng = np.random.default_rng(
            np.random.SeedSequence([cfg.seed, simulation, 0])
        )
        pop = factory.draw(pop_rng)
        p = pop.p_at(idx, tod)
        frames = pd.DataFrame(
            {
                "n_women": p,
                "n_people": np.ones(m),
                "itinerary_id": itin,
                "longitude": lon_s,
                "latitude": lat_s,
                "timestamp": ts,
            }
        )
        res = Sample(frames, cluster="itinerary_id").ratio(
            "n_women",
            "n_people",
            inference=_inference_method(se_method),
            confidence_level=ci_level,
            bootstrap_reps=599,
            seed=simulation,
        )
        se = res.standard_error
        lo, hi = res.confidence_interval
        if not (np.isfinite(se) and np.isfinite(lo) and np.isfinite(hi)):
            continue
        valid += 1
        diffs.append(res.estimate - pop.beta_true)
        ses.append(se)
        covers += int(lo <= pop.beta_true <= hi)
        if spatial_diag:
            diag = dependence_diagnostics(
                frames["n_women"].to_numpy(),
                frames["itinerary_id"].to_numpy(),
                lon=lon_s,
                lat=lat_s,
                timestamps=ts,
                seed=simulation,
            )
            neff.append(diag.n_eff_space)
            wb.append(diag.within_between_ratio)

    return _aggregate(
        label, se_method, diffs, ses, covers, [float(m)], [], neff, wb, valid
    )


def run_experiment(
    cfg: SimConfig,
    pipelines: list[Pipeline],
    se_method: str = "auto",
    ci_level: float = 0.95,
    spatial_diag: bool = False,
) -> list[PipelineResult]:
    """Run every pipeline against one shared population factory (one DGP)."""
    factory = PopulationFactory(cfg)
    return [
        run_pipeline(factory, p, cfg, se_method, ci_level, spatial_diag)
        for p in pipelines
    ]


def results_table(results: list[PipelineResult]) -> str:
    """Render results as a fixed-width table."""
    rows = [r.as_row() for r in results]
    if not rows:
        return "(no results)"
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    head = "  ".join(c.rjust(widths[c]) for c in cols)
    lines = [head, "-" * len(head)]
    lines.extend("  ".join(str(r[c]).rjust(widths[c]) for c in cols) for r in rows)
    return "\n".join(lines)


def default_pipelines() -> list[Pipeline]:
    """The four reference routing strategies, all with staggered start times."""
    return [
        Pipeline("compact", routing="compact", staggered_starts=True),
        Pipeline("dispersed", routing="dispersed", staggered_starts=True),
        Pipeline("systematic", routing="systematic", staggered_starts=True),
        Pipeline("srs", routing="srs", staggered_starts=True),
    ]


def main() -> None:
    """Reproduce the headline experiments and print them as tables.

    Run with ``python -m geoestimate.simulate``.
    """
    import warnings

    warnings.simplefilter("ignore")

    print("\n[1] Spatial extremes (no time effect): corr=0 vs corr->inf")
    for label, rs in [
        ("corr = 0  (white noise)", 0.0),
        ("corr -> inf (constant)", 1e7),
    ]:
        cfg = SimConfig(range_s_m=rs, diurnal_amp=0.0, sd_t=0.0, n_sims=150, grid_n=16)
        print(f"\n  {label}:")
        print(
            results_table(run_experiment(cfg, default_pipelines(), spatial_diag=True))
        )

    print("\n[2] SE coverage vs spatial correlation (compact routing, K=8)")
    compact = Pipeline("compact", routing="compact")
    print(f"  {'range_s':>8} | {'naive':>6} | {'cluster-t':>9}")
    for rs in [0, 800, 2000, 6000]:
        cfg = SimConfig(
            range_s_m=float(rs), diurnal_amp=0.0, sd_t=0.0, n_sims=200, grid_n=16
        )
        fac = PopulationFactory(cfg)
        cov = {
            m: run_pipeline(fac, compact, cfg, se_method=m).coverage
            for m in ("naive", "cluster")
        }
        print(f"  {rs:>8} | {cov['naive']:>6.2f} | {cov['cluster']:>9.2f}")

    print("\n[3] Temporal bias (strong diurnal): synchronized vs staggered start times")
    cfg = SimConfig(range_s_m=600.0, diurnal_amp=1.5, sd_t=0.0, n_sims=150, grid_n=16)
    pipes = [
        Pipeline("compact-synced", routing="compact", staggered_starts=False),
        Pipeline("compact-stagger", routing="compact", staggered_starts=True),
    ]
    print(results_table(run_experiment(cfg, pipes)))


if __name__ == "__main__":
    main()
