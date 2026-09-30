import numpy as np
import pandas as pd
import pytest

from src.CI_tests.Fisher_Z.fisher_z import FisherZ, fisher_z_test


def test_matches_ols_residual_correlation() -> None:
    """Check the precision-matrix partial correlation against the
    algorithmically independent route: regress x and y on z with OLS and
    Fisher-transform the Pearson correlation of the residuals.
    """
    rng = np.random.default_rng(0)
    n = 300
    z = rng.standard_normal((n, 2))
    x = z[:, 0] + 0.5 * z[:, 1] + rng.standard_normal(n)
    y = z[:, 0] - z[:, 1] + rng.standard_normal(n)

    z_design = np.column_stack([np.ones(n), z])
    resid_x = x - z_design @ np.linalg.lstsq(z_design, x, rcond=None)[0]
    resid_y = y - z_design @ np.linalg.lstsq(z_design, y, rcond=None)[0]
    r_expected = np.corrcoef(resid_x, resid_y)[0, 1]
    expected_statistic = np.arctanh(r_expected) * np.sqrt(n - 2 - 3)

    result = FisherZ().test(x, y, z)

    assert result.statistic == pytest.approx(expected_statistic)


def test_unconditional_reduces_to_pearson_correlation() -> None:
    """With z=None, the partial correlation must reduce to the plain
    Pearson correlation of x and y.
    """
    rng = np.random.default_rng(1)
    n = 200
    x = rng.standard_normal(n)
    y = 0.6 * x + rng.standard_normal(n)

    r_expected = np.corrcoef(x, y)[0, 1]
    expected_statistic = np.arctanh(r_expected) * np.sqrt(n - 3)

    result = FisherZ().test(x, y)

    assert result.statistic == pytest.approx(expected_statistic)


def test_does_not_reject_independence() -> None:
    rng = np.random.default_rng(2)
    n = 300
    z = rng.standard_normal((n, 1))
    x = z[:, 0] + rng.standard_normal(n)
    y = z[:, 0] + rng.standard_normal(n)

    result = FisherZ().test(x, y, z)

    assert not result.reject


def test_rejects_dependence() -> None:
    rng = np.random.default_rng(2)
    n = 300
    z = rng.standard_normal((n, 1))
    x = z[:, 0] + rng.standard_normal(n)
    y = z[:, 0] + x + rng.standard_normal(n)

    result = FisherZ().test(x, y, z)

    assert result.reject


def test_raises_when_sample_size_too_small_for_conditioning_set() -> None:
    rng = np.random.default_rng(3)
    n = 5
    z = rng.standard_normal((n, 3))
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)

    with pytest.raises(ValueError, match="Sample size too small"):
        FisherZ().test(x, y, z)


def test_accepts_large_conditioning_set_when_n_is_large_enough() -> None:
    """n - |z| - 3 > 0 is the only size requirement: |z|=20 with n=600 runs and,
    with x and y independent given z, does not reject.
    """
    rng = np.random.default_rng(4)
    n, dim_z = 600, 20
    z = rng.standard_normal((n, dim_z))
    x = z.sum(axis=1) + rng.standard_normal(n)
    y = z.sum(axis=1) + rng.standard_normal(n)

    assert not FisherZ().test(x, y, z).reject


def _dataframe(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    z1 = rng.standard_normal(n)
    z2 = rng.integers(0, 3, n)
    x = z1 + 0.5 * z2 + rng.standard_normal(n)
    y = z1 - z2 + rng.standard_normal(n)
    return pd.DataFrame({"x": x, "y": y, "z1": z1, "z2": z2})


def test_dataframe_signature_matches_array_call() -> None:
    df = _dataframe()

    result = fisher_z_test(df, "x", "y", ["z1", "z2"])

    expected = FisherZ().test(
        df["x"].to_numpy(), df["y"].to_numpy(), df[["z1", "z2"]].to_numpy(dtype=np.float64)
    )
    assert result == expected


def test_empty_z_list_is_the_marginal_test() -> None:
    df = _dataframe()

    assert fisher_z_test(df, "x", "y", []) == fisher_z_test(df, "x", "y")


def test_integer_coded_conditioning_column_gives_same_result_as_float() -> None:
    df_int = _dataframe()
    df_float = df_int.astype({"z2": np.float64})

    assert fisher_z_test(df_int, "x", "y", ["z2"]) == fisher_z_test(df_float, "x", "y", ["z2"])


def test_non_numeric_conditioning_column_raises_value_error() -> None:
    df = _dataframe().assign(z2=lambda d: np.where(d["z2"] == 0, "a", "b"))

    with pytest.raises(ValueError):
        fisher_z_test(df, "x", "y", ["z2"])
