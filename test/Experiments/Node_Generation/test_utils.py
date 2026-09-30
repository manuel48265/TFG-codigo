import numpy as np
import pandas as pd
import pytest

from src.Experiments.Node_Generation.causal_models import (
    LINEAR_GAUSSIAN,
    LOCAL_SCALE_NOISE_MODELS,
)
from src.Experiments.Node_Generation.utils import (
    VariableType,
    categorical_num_terms,
    mean_term_generator,
    robust_scale,
    split_scale_parent,
    standardize_scale,
    variable_type,
)


def test_variable_type_follows_dtype_before_the_number_of_values() -> None:
    assert variable_type(pd.Series([0.1, 0.5, 0.9])) == VariableType.CONTINUOUS
    assert variable_type(pd.Series([0, 1, 0, 1])) == VariableType.BINARY
    assert variable_type(pd.Series(pd.Categorical([0, 1, 2], ordered=True))) == VariableType.ORDINAL
    assert variable_type(pd.Series(pd.Categorical(["a", "b", "a"]))) == VariableType.NOMINAL
    # a category column with a single observed value is still nominal, not binary
    observed_one = pd.Series(pd.Categorical(["a", "a"], categories=["a", "b"]))
    assert variable_type(observed_one) == VariableType.NOMINAL


def test_split_scale_parent_is_only_used_by_heteroscedastic_models() -> None:
    rng = np.random.default_rng(0)
    parents = ["a", "b", "c"]

    assert split_scale_parent(rng, LINEAR_GAUSSIAN, parents) == (None, parents)
    assert split_scale_parent(rng, LOCAL_SCALE_NOISE_MODELS, []) == (None, [])
    scale_parent, mean_parents = split_scale_parent(rng, LOCAL_SCALE_NOISE_MODELS, parents)
    assert scale_parent in parents
    assert sorted([scale_parent, *mean_parents]) == parents


def test_mean_term_generator_draws_signed_values_with_bounded_magnitude() -> None:
    values = np.asarray(mean_term_generator(np.random.default_rng(1), 1000))

    assert np.abs(values).min() >= 1 / 3
    assert np.abs(values).max() <= 3.0
    assert (values > 0).any() and (values < 0).any()


def test_categorical_num_terms_is_between_three_and_five() -> None:
    rng = np.random.default_rng(2)

    assert {categorical_num_terms(rng) for _ in range(200)} == {3, 4, 5}


def test_standardize_scale_sets_the_largest_absolute_value_to_the_bound() -> None:
    x = np.array([-2.0, 1.0, 4.0])

    assert standardize_scale(x, bound=5.0) == pytest.approx(x * 5.0 / 4.0)
    assert standardize_scale(np.zeros(3)) == pytest.approx(np.zeros(3))


def test_robust_scale_equals_the_standard_deviation_for_normal_data_and_ignores_outliers() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal(20000)

    assert robust_scale(x) == pytest.approx(1.0, abs=0.03)
    assert robust_scale(np.append(x, 1e9)) == pytest.approx(1.0, abs=0.03)


def test_robust_scale_falls_back_to_the_standard_deviation_when_half_the_values_are_equal() -> None:
    x = np.array([0.0] * 7 + [1.0] * 3)

    assert robust_scale(x) == pytest.approx(x.std())
    assert robust_scale(np.ones(5)) == 0.0
