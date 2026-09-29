from pathlib import Path

import numpy as np
import pytest

from src.CI_tests.common import CIResult
from src.Experiments.DAG_Generation.fundamental_topology_generator import FundamentalCase
from src.Experiments.Experimental_cases.common import (
    append_trial_result,
    build_trial_result,
    classify_structure,
    discretize_for_chi2,
)


def test_discretize_for_chi2_gives_equiprobable_bins() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    bins = discretize_for_chi2(x, max_bins=5)

    assert bins.min() == 0
    assert bins.max() == 4
    counts = np.bincount(bins)
    assert counts.min() > 150  # ~200/bin expected; equiprobable, not skewed


@pytest.mark.parametrize(
    "marginal_reject,conditional_reject,expected",
    [
        (False, False, FundamentalCase.INDEPENDENCE),
        (True, True, FundamentalCase.DIRECT_DEPENDENCE),
        (False, True, FundamentalCase.COLLIDER),
        (True, False, FundamentalCase.CONFOUNDER),
    ],
)
def test_classify_structure_matches_truth_table(
    marginal_reject: bool, conditional_reject: bool, expected: FundamentalCase
) -> None:
    assert classify_structure(marginal_reject, conditional_reject) == expected


def test_append_trial_result_round_trips_through_csv(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    marginal = CIResult(statistic=1.0, p_value=0.6, reject=False)
    conditional = CIResult(statistic=5.0, p_value=0.01, reject=True)

    result = build_trial_result(
        case=FundamentalCase.COLLIDER,
        z_size=2,
        n_samples=500,
        rule="LINEAR_GAUSSIAN",
        type_mix="continuous",
        ci_test="fisher_z",
        rep=0,
        seed=42,
        marginal=marginal,
        conditional=conditional,
    )
    assert result.predicted_case == FundamentalCase.COLLIDER.value
    assert result.correct

    append_trial_result(path, result)
    append_trial_result(path, result)

    lines = path.read_text().splitlines()
    assert len(lines) == 3  # header + 2 rows
    assert "predicted_case" in lines[0]
