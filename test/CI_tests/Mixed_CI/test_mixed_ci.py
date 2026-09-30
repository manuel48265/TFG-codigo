import math

import numpy as np
import pandas as pd
import pytest

from src.CI_tests.Mixed_CI.mixed_ci import MixedCI, VariableType, mixed_ci_test


def _dataframe(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    z = rng.normal(size=n)
    x = 0.6 * z + rng.normal(size=n)
    y = 0.6 * z + 0.8 * x + rng.normal(size=n)
    return pd.DataFrame({"x": x, "y": y, "z": z})


@pytest.mark.parametrize(
    "column,expected",
    [
        (pd.Series([0.1, 0.5, 0.9, 1.3]), VariableType.CONTINUOUS),
        (pd.Series([0, 1, 2, 0, 1, 2]), VariableType.CONTINUOUS),  # integer codes are numeric
        (pd.Series([0, 1, 0, 1]), VariableType.BINARY),
        (pd.Series(pd.Categorical(["a", "b", "a", "b"])), VariableType.NOMINAL),  # dtype first
        (pd.Series(pd.Categorical([0, 1, 0, 1], ordered=True)), VariableType.ORDINAL),
        (pd.Series(pd.Categorical(["a", "b", "c", "a"])), VariableType.NOMINAL),
        (pd.Series(pd.Categorical(["a", "b", "c", "a"], ordered=True)), VariableType.ORDINAL),
    ],
)
def test_variable_type_follows_dtype_rules(column: pd.Series, expected: VariableType) -> None:
    assert MixedCI()._variable_type(column) == expected


def test_mm_combination_of_two_p_values() -> None:
    """MM = min(2 * min(p1, p2), max(p1, p2)): 2 * 0.01 = 0.02 < 0.04, and
    2 * 0.03 = 0.06 > 0.04 so the maximum wins.
    """
    combined = MixedCI()._combine_mm_log
    assert math.exp(combined(math.log(0.01), math.log(0.04))) == pytest.approx(0.02)
    assert math.exp(combined(math.log(0.03), math.log(0.04))) == pytest.approx(0.04)


def test_rejects_dependence_and_keeps_independence() -> None:
    df = _dataframe()
    df["w"] = np.random.default_rng(1).normal(size=len(df))

    assert mixed_ci_test(df, "x", "y", ["z"]).reject
    assert not mixed_ci_test(df, "x", "w", ["z"]).reject


def test_function_matches_class_method() -> None:
    df = _dataframe()

    assert mixed_ci_test(df, "x", "y", ["z"]) == MixedCI().test(df, "x", "y", ["z"])


def test_empty_z_list_is_the_marginal_test() -> None:
    df = _dataframe()

    assert mixed_ci_test(df, "x", "y", []) == mixed_ci_test(df, "x", "y")


def test_integer_coded_conditioning_column_gives_same_result_as_float() -> None:
    df_int = _dataframe().assign(c=lambda d: (d["z"] > 0).astype(int) + (d["z"] > 1))
    df_float = df_int.astype({"c": np.float64})

    assert mixed_ci_test(df_int, "x", "y", ["c"]) == mixed_ci_test(df_float, "x", "y", ["c"])


def test_categorical_conditioning_column_is_expanded_into_dummies() -> None:
    """A 3-level `category` column in Z adds two dummy columns; the result must match
    building those dummies by hand.
    """
    df = _dataframe().assign(c=lambda d: pd.Categorical(pd.cut(d["z"], 3, labels=["a", "b", "c"])))
    dummies = pd.get_dummies(df["c"], drop_first=True, dtype=float)
    by_hand = pd.concat([df[["x", "y"]], dummies], axis=1)

    assert mixed_ci_test(df, "x", "y", ["c"]) == mixed_ci_test(by_hand, "x", "y", ["b", "c"])


def test_two_level_categorical_columns_are_usable_as_x_and_y() -> None:
    """A `category` column with two levels goes through the nominal model."""
    rng = np.random.default_rng(6)
    n = 400
    latent = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x": pd.Categorical(np.where(latent > 0, "a", "b")),
            "y": pd.Categorical(np.where(latent + rng.normal(size=n) > 0, "c", "d")),
            "w": pd.Categorical(np.where(rng.normal(size=n) > 0, "c", "d")),
        }
    )

    assert mixed_ci_test(df, "x", "y").reject
    assert not mixed_ci_test(df, "x", "w").reject


def _with_unused_category(df: pd.DataFrame, column: str) -> pd.DataFrame:
    codes = np.digitize(df["z"], np.quantile(df["z"], [1 / 3, 2 / 3]))
    return df.assign(**{column: pd.Categorical(codes, categories=range(4), ordered=False)})


@pytest.mark.parametrize("role", ["x", "y", "z"])
def test_unused_categories_raise_a_clear_error_and_removing_them_fixes_it(role: str) -> None:
    df = _with_unused_category(_dataframe(), "c")
    columns = {"x": ("c", "y", []), "y": ("x", "c", []), "z": ("x", "y", ["c"])}
    x, y, z = columns[role]

    with pytest.raises(ValueError, match="remove_unused_categories"):
        mixed_ci_test(df, x, y, z)

    cleaned = df.assign(c=df["c"].cat.remove_unused_categories())
    assert np.isfinite(mixed_ci_test(cleaned, x, y, z).p_value)
