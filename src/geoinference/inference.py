"""Point estimation and uncertainty for equal-probability frame samples."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .designs import Design, PointDesign
from .types import CIResult, Diagnostics, InferenceResult, SEResult


def _ratio_estimator(women: np.ndarray, people: np.ndarray) -> float:
    """Return the ratio of total women to total people."""
    return float(women.sum() / people.sum())


def _photo_mean_estimator(women: np.ndarray, people: np.ndarray) -> float:
    """Return the mean frame proportion among frames containing people."""
    positive = people > 0
    return float(np.mean(women[positive] / people[positive]))


def _ratio_bias_approx(women: np.ndarray, people: np.ndarray) -> float:
    """Return the first-order iid bias approximation for a ratio estimator."""
    n = len(women)
    ratio = _ratio_estimator(women, people)
    mean_people = float(np.mean(people))
    var_people = float(np.var(people, ddof=1))
    cov_counts = float(np.cov(women, people, ddof=1)[0, 1])
    return (ratio * var_people - cov_counts) / (n * mean_people**2)


def _naive_se_ratio(women: np.ndarray, people: np.ndarray) -> float:
    """Return the iid delta-method standard error for the ratio."""
    n = len(women)
    ratio = _ratio_estimator(women, people)
    scores = women - ratio * people
    variance = n * np.sum(scores**2) / ((n - 1) * people.sum() ** 2)
    return float(np.sqrt(variance))


def _naive_se_mean(women: np.ndarray, people: np.ndarray) -> float:
    """Return the iid standard error for the mean frame proportion."""
    positive = people > 0
    proportions = women[positive] / people[positive]
    return float(np.std(proportions, ddof=1) / np.sqrt(len(proportions)))


def _cluster_scores(
    values: np.ndarray, labels: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Sum observation-level scores within integer-coded clusters."""
    return np.bincount(labels, weights=values, minlength=n_clusters)


def _cluster_se_ratio(
    women: np.ndarray, people: np.ndarray, labels: np.ndarray, n_clusters: int
) -> float:
    """Return the cluster-sandwich standard error for the ratio."""
    ratio = _ratio_estimator(women, people)
    scores = _cluster_scores(women - ratio * people, labels, n_clusters)
    variance = n_clusters / (n_clusters - 1) * np.sum(scores**2) / people.sum() ** 2
    return float(np.sqrt(variance))


def _cluster_se_mean(
    women: np.ndarray, people: np.ndarray, labels: np.ndarray
) -> tuple[float, int]:
    """Return the cluster-sandwich SE and contributing cluster count."""
    positive = people > 0
    proportions = women[positive] / people[positive]
    labels_positive = labels[positive]
    _, compact_labels = np.unique(labels_positive, return_inverse=True)
    n_clusters = int(compact_labels.max()) + 1
    if n_clusters < 2:
        return float("nan"), n_clusters
    mean = float(np.mean(proportions))
    scores = _cluster_scores(proportions - mean, compact_labels, n_clusters)
    variance = n_clusters / (n_clusters - 1) * np.sum(scores**2) / len(proportions) ** 2
    return float(np.sqrt(variance)), n_clusters


def _bootstrap_distribution(
    women: np.ndarray,
    people: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    estimator: Callable[[np.ndarray, np.ndarray], float],
    sampled_clusters: np.ndarray,
) -> np.ndarray:
    """Evaluate an estimator on shared pairs-bootstrap draws."""
    cluster_rows = [np.flatnonzero(labels == cluster) for cluster in range(n_clusters)]
    estimates = np.empty(len(sampled_clusters), dtype=float)
    for index, draw in enumerate(sampled_clusters):
        rows = np.concatenate([cluster_rows[cluster] for cluster in draw])
        if people[rows].sum() <= 0:
            estimates[index] = np.nan
        else:
            estimates[index] = estimator(women[rows], people[rows])
    return estimates[np.isfinite(estimates)]


def _bootstrap_summary(
    estimates: np.ndarray, reps: int, ci_level: float
) -> tuple[float, tuple[float, float]]:
    """Return a bootstrap standard error and percentile interval."""
    if len(estimates) < reps / 2:
        raise ValueError(
            "fewer than half the bootstrap draws produced defined estimates; "
            "the sample has too little positive-count information"
        )
    alpha = 1 - ci_level
    quantiles = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    interval = (float(quantiles[0]), float(quantiles[1]))
    return float(np.std(estimates, ddof=1)), interval


def _compute_icc(values: np.ndarray, labels: np.ndarray) -> float:
    """Return a one-way random-effects intraclass correlation estimate."""
    unique, compact = np.unique(labels, return_inverse=True)
    n_clusters = len(unique)
    n = len(values)
    if n_clusters < 2 or n <= n_clusters:
        return float("nan")

    counts = np.bincount(compact).astype(float)
    means = np.bincount(compact, weights=values) / counts
    grand_mean = float(np.mean(values))
    ss_between = float(np.sum(counts * (means - grand_mean) ** 2))
    ss_within = float(np.sum((values - means[compact]) ** 2))
    ms_between = ss_between / (n_clusters - 1)
    ms_within = ss_within / (n - n_clusters)
    mean_size = (n - np.sum(counts**2) / n) / (n_clusters - 1)
    denominator = ms_between + (mean_size - 1) * ms_within
    if denominator <= 0:
        return 0.0
    return float(max((ms_between - ms_within) / denominator, 0.0))


def _validate_inputs(
    data: pd.DataFrame,
    women_var: str,
    people_var: str,
    design: Design,
    ci_level: float,
    bootstrap: bool,
    bootstrap_reps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Validate public inputs and return numeric arrays plus cluster codes."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data must contain at least two rows")
    missing = [column for column in (women_var, people_var) if column not in data]
    if missing:
        raise ValueError(
            f"columns not found: {missing}; available columns: {list(data.columns)}"
        )

    try:
        women = data[women_var].to_numpy(dtype=float)
        people = data[people_var].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("women and people columns must be numeric") from error

    if len(women) < 2:
        raise ValueError("data must contain at least two rows")
    if not np.all(np.isfinite(women)) or not np.all(np.isfinite(people)):
        raise ValueError("women and people counts must be finite")
    if np.any(women < 0) or np.any(people < 0):
        raise ValueError("women and people counts must be nonnegative")
    if np.any(women > people):
        raise ValueError("women counts cannot exceed people counts")
    if people.sum() <= 0:
        raise ValueError("at least one observed person is required")
    if np.count_nonzero(people > 0) < 2:
        raise ValueError("at least two frames containing people are required")
    if isinstance(ci_level, bool) or not np.isfinite(ci_level) or not 0 < ci_level < 1:
        raise ValueError("ci_level must be finite and strictly between 0 and 1")
    if bootstrap:
        if isinstance(bootstrap_reps, bool) or not isinstance(bootstrap_reps, int):
            raise TypeError("bootstrap_reps must be an integer")
        if bootstrap_reps < 2:
            raise ValueError("bootstrap_reps must be at least 2")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    if design.has_clusters:
        cluster_var = design.cluster_var
        if cluster_var not in data:
            raise ValueError(
                f"cluster column {cluster_var!r} not found; "
                f"available columns: {list(data.columns)}"
            )
        cluster_values = data[cluster_var]
        if cluster_values.isna().any():
            raise ValueError(
                f"cluster column {cluster_var!r} cannot contain missing values"
            )
        labels, unique = pd.factorize(cluster_values, sort=False)
        n_clusters = len(unique)
        if n_clusters < 2:
            raise ValueError("clustered inference requires at least two clusters")
    else:
        labels = np.arange(len(data), dtype=int)
        n_clusters = len(data)

    return women, people, labels.astype(int), n_clusters


def _select_se(
    naive: float,
    cluster: float | None,
    bootstrap: float | None,
    method: str,
) -> SEResult:
    """Select a requested standard error without falling back silently."""
    choices = {"naive": naive, "cluster": cluster, "bootstrap": bootstrap}
    if method not in choices:
        raise ValueError(f"se_method must be one of {sorted(choices)}, got {method!r}")
    selected = choices[method]
    if selected is None:
        raise ValueError(f"se_method={method!r} is unavailable for this call")
    return SEResult(
        naive=naive,
        cluster=cluster,
        bootstrap=bootstrap,
        recommended=selected,
        method_used=method,
    )


def _build_ci(
    estimate_value: float,
    standard_error: float,
    n_clusters: int | None,
    ci_level: float,
    bootstrap_interval: tuple[float, float] | None,
    recommended_se_method: str,
    ci_method: str | None,
) -> CIResult:
    """Construct supported intervals and select one explicitly."""
    alpha = 1 - ci_level
    z_value = float(sp_stats.norm.ppf(1 - alpha / 2))
    normal = (
        estimate_value - z_value * standard_error,
        estimate_value + z_value * standard_error,
    )
    t_interval = None
    if n_clusters is not None:
        t_value = float(sp_stats.t.ppf(1 - alpha / 2, df=n_clusters - 1))
        t_interval = (
            estimate_value - t_value * standard_error,
            estimate_value + t_value * standard_error,
        )

    intervals = {"normal": normal, "t": t_interval, "bootstrap": bootstrap_interval}
    selected_method = ci_method
    if selected_method is None:
        if recommended_se_method == "cluster":
            selected_method = "t"
        elif recommended_se_method == "bootstrap":
            selected_method = "bootstrap"
        else:
            selected_method = "normal"
    if selected_method not in intervals:
        raise ValueError(
            f"ci_method must be one of {sorted(intervals)}, got {selected_method!r}"
        )
    selected = intervals[selected_method]
    if selected is None:
        raise ValueError(f"ci_method={selected_method!r} is unavailable for this call")
    return CIResult(
        normal=normal,
        t=t_interval,
        bootstrap=bootstrap_interval,
        recommended=selected,
        method_used=selected_method,
        level=ci_level,
    )


def estimate(
    data: pd.DataFrame,
    women_var: str = "n_women",
    people_var: str = "n_people",
    design: Design | None = None,
    ci_level: float = 0.95,
    bootstrap: bool = False,
    bootstrap_reps: int = 2000,
    seed: int = 42,
    se_method: str | None = None,
    ci_method: str | None = None,
) -> InferenceResult:
    """Estimate two proportions from an equal-probability frame sample.

    The ratio estimand is ``sum(women) / sum(people)``. It weights frames by
    their observed number of people. The photo-level mean is the unweighted mean
    of ``women / people`` over frames containing at least one person.

    Args:
        data: One row per annotated frame.
        women_var: Column containing observed women counts.
        people_var: Column containing observed people counts.
        design: Equal-probability point or walk design. The default treats rows
            as independent.
        ci_level: Confidence level strictly between zero and one.
        bootstrap: Compute pairs-bootstrap results over design units.
        bootstrap_reps: Number of bootstrap draws when ``bootstrap`` is true.
        seed: Random seed for bootstrap draws.
        se_method: Selected standard error: ``"naive"``, ``"cluster"``, or
            ``"bootstrap"``. The design selects a default when omitted.
        ci_method: Selected interval: ``"normal"``, ``"t"``, or
            ``"bootstrap"``. The design and selected standard error determine
            the default.

    Returns:
        Estimates, uncertainty measures, and descriptive diagnostics.

    Raises:
        TypeError: If an argument has the wrong structural type.
        ValueError: If data or options fall outside the supported domain.

    Notes:
        The ratio estimator is generally biased in finite samples. The reported
        bias value is a first-order iid approximation, not a correction.
        Clustered inference assumes the declared clusters are independent.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if bootstrap and (
        isinstance(bootstrap_reps, bool) or not isinstance(bootstrap_reps, int)
    ):
        raise TypeError("bootstrap_reps must be an integer")
    resolved_design = design if design is not None else PointDesign()
    women, people, labels, n_clusters = _validate_inputs(
        data,
        women_var,
        people_var,
        resolved_design,
        ci_level,
        bootstrap,
        bootstrap_reps,
        seed,
    )
    has_clusters = resolved_design.has_clusters

    ratio = _ratio_estimator(women, people)
    photo_mean = _photo_mean_estimator(women, people)
    ratio_naive = _naive_se_ratio(women, people)
    mean_naive = _naive_se_mean(women, people)

    ratio_cluster = None
    mean_cluster = None
    mean_cluster_count = None
    if has_clusters:
        ratio_cluster = _cluster_se_ratio(women, people, labels, n_clusters)
        mean_cluster, mean_cluster_count = _cluster_se_mean(women, people, labels)
        if mean_cluster_count < 2:
            raise ValueError(
                "photo-level clustered inference requires people in at least "
                "two clusters"
            )

    ratio_bootstrap = None
    mean_bootstrap = None
    ratio_bootstrap_ci = None
    mean_bootstrap_ci = None
    if bootstrap:
        rng = np.random.default_rng(seed)
        sampled_clusters = rng.integers(
            0, n_clusters, size=(bootstrap_reps, n_clusters), endpoint=False
        )
        ratio_draws = _bootstrap_distribution(
            women,
            people,
            labels,
            n_clusters,
            _ratio_estimator,
            sampled_clusters,
        )
        mean_draws = _bootstrap_distribution(
            women,
            people,
            labels,
            n_clusters,
            _photo_mean_estimator,
            sampled_clusters,
        )
        ratio_bootstrap, ratio_bootstrap_ci = _bootstrap_summary(
            ratio_draws, bootstrap_reps, ci_level
        )
        mean_bootstrap, mean_bootstrap_ci = _bootstrap_summary(
            mean_draws, bootstrap_reps, ci_level
        )

    selected_se_method = se_method or resolved_design.recommended_se_method
    ratio_se = _select_se(
        ratio_naive, ratio_cluster, ratio_bootstrap, selected_se_method
    )
    mean_se = _select_se(mean_naive, mean_cluster, mean_bootstrap, selected_se_method)
    ratio_ci = _build_ci(
        ratio,
        ratio_se.recommended,
        n_clusters if has_clusters else None,
        ci_level,
        ratio_bootstrap_ci,
        selected_se_method,
        ci_method,
    )
    mean_ci = _build_ci(
        photo_mean,
        mean_se.recommended,
        mean_cluster_count,
        ci_level,
        mean_bootstrap_ci,
        selected_se_method,
        ci_method,
    )

    positive = people > 0
    positive_labels = labels[positive]
    proportions = women[positive] / people[positive]
    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    cluster_size_mean = float(np.mean(cluster_sizes))
    cluster_size_cv = float(np.std(cluster_sizes) / cluster_size_mean)
    n_clusters_eff = float(cluster_sizes.sum() ** 2 / np.sum(cluster_sizes**2))
    icc = _compute_icc(proportions, positive_labels) if has_clusters else float("nan")
    design_effect = (
        1 + (cluster_size_mean - 1) * icc if np.isfinite(icc) else float("nan")
    )
    effective_n = (
        float(len(proportions) / design_effect)
        if np.isfinite(design_effect) and design_effect > 0
        else float(len(proportions))
    )
    diagnostics = Diagnostics(
        n_obs=len(data),
        n_positive_frames=int(np.count_nonzero(positive)),
        n_empty_frames=int(np.count_nonzero(~positive)),
        empty_frame_rate=float(np.mean(~positive)),
        n_clusters=n_clusters,
        cluster_sizes=cluster_sizes,
        cluster_size_mean=cluster_size_mean,
        cluster_size_cv=cluster_size_cv,
        n_clusters_eff=n_clusters_eff,
        icc=icc,
        deff=design_effect,
        n_eff=effective_n,
        se_photo_mean_cluster_to_naive=mean_se.cluster_to_naive,
        ratio_bias_approx=_ratio_bias_approx(women, people),
    )
    return InferenceResult(
        ratio=ratio,
        photo_mean=photo_mean,
        ratio_se=ratio_se,
        photo_mean_se=mean_se,
        ratio_ci=ratio_ci,
        photo_mean_ci=mean_ci,
        diagnostics=diagnostics,
        design_name=resolved_design.name,
        n_obs=len(data),
        n_clusters=n_clusters,
    )
