"""Weighted, stratified, clustered and post-stratified estimates match R's survey.

The reference values come from an independent implementation; see
``tests/fixtures/make_survey_reference.R``.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from geoestimate import Sample

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE = json.loads((FIXTURES / "survey_reference.json").read_text())
DESIGNS = {
    "weights": {"weights": "w"},
    "weights_cluster": {"weights": "w", "cluster": "psu"},
    "weights_strata": {"weights": "w", "strata": "stratum"},
    "weights_strata_cluster": {"weights": "w", "strata": "stratum", "cluster": "psu"},
    "post_stratified": {"weights": "w", "cluster": "psu"},
}


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "survey_reference.csv")


@pytest.mark.parametrize("design", sorted(DESIGNS))
@pytest.mark.parametrize("estimand", ["mean", "ratio", "total", "share_at_least"])
def test_matches_r_survey(data: pd.DataFrame, design: str, estimand: str) -> None:
    sample = Sample(data, **DESIGNS[design])
    if design == "post_stratified":
        sample = sample.post_stratify("cell", REFERENCE["shares"])
    estimate = {
        "mean": lambda: sample.mean("level"),
        "ratio": lambda: sample.ratio("y", "z"),
        "total": lambda: sample.total("y"),
        "share_at_least": lambda: sample.share_at_least("level", 62),
    }[estimand]()
    expected = REFERENCE[design][estimand]
    # R's reference for the post-stratified total is the exact delta method;
    # svy uses regression residuals, which agree only to first order.
    tolerance = 1e-3 if (design, estimand) == ("post_stratified", "total") else 1e-8
    assert estimate.estimate == pytest.approx(expected["estimate"], rel=1e-8)
    assert estimate.standard_error == pytest.approx(expected["se"], rel=tolerance)


def test_unit_weights_reproduce_unweighted_estimates(data: pd.DataFrame) -> None:
    plain = Sample(data, cluster="psu").ratio("y", "z")
    weighted = Sample(data.assign(one=1.0), cluster="psu", weights="one").ratio(
        "y", "z"
    )
    assert weighted.estimate == pytest.approx(plain.estimate)
    assert weighted.standard_error == pytest.approx(plain.standard_error)


def test_post_stratification_hits_the_target_shares(data: pd.DataFrame) -> None:
    sample = Sample(data, weights="w").post_stratify("cell", REFERENCE["shares"])
    indicator = data.assign(residential=(data["cell"] == "residential").astype(float))
    share = (
        Sample(indicator, weights="w")
        .post_stratify("cell", REFERENCE["shares"])
        .mean("residential")
    )
    assert share.estimate == pytest.approx(0.6)
    assert share.standard_error == pytest.approx(0.0, abs=1e-12)
    assert sample.n_observations == len(data)


def test_invalid_declarations_are_rejected(data: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        Sample(data.assign(w=0.0), weights="w")
    with pytest.raises(ValueError, match="at least two clusters"):
        Sample(
            data.assign(one="x", lone=data["psu"] == 1), cluster="one", strata="lone"
        )
    with pytest.raises(ValueError, match="exactly the cells"):
        Sample(data).post_stratify("cell", {"residential": 1.0})
    with pytest.raises(ValueError, match="sum to one"):
        Sample(data).post_stratify(
            "cell", {"residential": 0.5, "tertiary": 0.5, "primary": 0.5}
        )
    with pytest.raises(ValueError, match="not used with declared weights"):
        Sample(data, weights="w").total("y", population_size=10_000)
    with pytest.raises(ValueError, match="required without weights"):
        Sample(data).total("y")
    calibrated = Sample(data).post_stratify("cell", REFERENCE["shares"])
    with pytest.raises(ValueError, match="post-stratified"):
        calibrated.mean("level", inference="bootstrap")


def test_bootstrap_is_refused_for_stratified_samples(data: pd.DataFrame) -> None:
    sample = Sample(data, weights="w", strata="stratum", cluster="psu")
    with pytest.raises(ValueError, match="stratified"):
        sample.ratio("y", "z", inference="bootstrap", seed=1)


def test_weighted_bootstrap_is_close_to_linearization(data: pd.DataFrame) -> None:
    sample = Sample(data, weights="w", cluster="psu")
    analytic = sample.ratio("y", "z")
    bootstrap = sample.ratio("y", "z", inference="bootstrap", seed=1)
    assert bootstrap.standard_error == pytest.approx(analytic.standard_error, rel=0.25)


def test_unweighted_total_uses_post_stratified_weights(data: pd.DataFrame) -> None:
    """Population size times the calibrated mean, not the raw mean."""
    calibrated = Sample(data).post_stratify("cell", REFERENCE["shares"])
    total = calibrated.total("y", population_size=10_000)
    mean = calibrated.mean("y")
    assert total.estimate == pytest.approx(10_000 * mean.estimate)
    assert total.standard_error == pytest.approx(10_000 * mean.standard_error)
    assert total.estimate != pytest.approx(10_000 * data["y"].mean())


def test_second_post_stratification_is_refused(data: pd.DataFrame) -> None:
    calibrated = Sample(data).post_stratify("cell", REFERENCE["shares"])
    with pytest.raises(ValueError, match="already post-stratified"):
        calibrated.post_stratify("stratum", {"a": 0.4, "b": 0.3, "c": 0.3})
