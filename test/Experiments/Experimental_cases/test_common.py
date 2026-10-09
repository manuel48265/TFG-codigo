from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.CI_tests.common import CIResult
from src.Experiments.DAG_Generation.fundamental_topology_generator import FundamentalCase
from src.Experiments.Experimental_cases.common import (
    append_trial_result,
    build_trial_result,
    classify_structure,
    discretize_continuous_columns,
    discretize_for_chi2,
    has_constant_column,
    remove_unused_categories,
    to_float_columns,
)


def test_discretize_for_chi2_gives_equiprobable_bins() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    bins = discretize_for_chi2(x, max_bins=5)

    assert bins.min() == 0
    assert bins.max() == 4
    counts = np.bincount(bins)
    assert counts.min() > 150  # ~200/bin expected; equiprobable, not skewed


def test_discretize_continuous_columns_only_touches_float_columns() -> None:
    rng = np.random.default_rng(1)
    n = 500
    df = pd.DataFrame(
        {
            "continuous": rng.standard_normal(n),
            "binary": rng.integers(0, 2, n),
            "nominal": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        }
    )

    result = discretize_continuous_columns(df)

    assert sorted(result["continuous"].unique()) == [0, 1, 2, 3, 4]
    pd.testing.assert_series_equal(result["binary"], df["binary"])
    pd.testing.assert_series_equal(result["nominal"], df["nominal"])
    assert result.index.equals(df.index)


def test_to_float_columns_encodes_categories_and_casts_the_rest() -> None:
    df = pd.DataFrame(
        {
            "continuous": [0.5, 1.5, 2.5],
            "binary": [0, 1, 1],
            "nominal": pd.Categorical(["b", "a", "c"]),
            "text": ["y", "x", "y"],
        }
    )

    result = to_float_columns(df)

    assert (result.dtypes == np.float64).all()
    assert result["binary"].tolist() == [0.0, 1.0, 1.0]
    assert result["nominal"].tolist() == [1.0, 0.0, 2.0]
    assert result["text"].tolist() == [1.0, 0.0, 1.0]


def test_to_float_columns_keeps_the_values_of_numeric_categories() -> None:
    df = pd.DataFrame({"cylinders": pd.Categorical([4, 8, 6, 3], ordered=True)})

    result = to_float_columns(df)

    assert result["cylinders"].tolist() == [4.0, 8.0, 6.0, 3.0]


def test_remove_unused_categories_only_touches_categorical_columns() -> None:
    df = pd.DataFrame(
        {
            "nominal": pd.Categorical(["a", "b", "a"], categories=["a", "b", "c"]),
            "number": [1.0, 2.0, 3.0],
        }
    )

    result = remove_unused_categories(df)

    assert list(result["nominal"].cat.categories) == ["a", "b"]
    pd.testing.assert_series_equal(result["number"], df["number"])
    assert result["nominal"].tolist() == df["nominal"].tolist()


def test_has_constant_column_detects_a_single_valued_column_of_any_type() -> None:
    varied = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": pd.Categorical(["x", "y", "x"])})

    assert not has_constant_column(varied)
    assert has_constant_column(varied.assign(c=1.0))
    assert has_constant_column(
        varied.assign(c=pd.Categorical(["x", "x", "x"], categories=["x", "y"]))
    )


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
