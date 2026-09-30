import numpy as np
import pandas as pd
import pytest

from src.CI_tests.GCM.gcm import GCM, gcm_test
from src.CI_tests.GCM.gcm_utils import compute_residuals


class _MeanRegressor:
    """Predicts the training mean, so the residuals are the centered variable."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_MeanRegressor":
        self.mean = y.mean()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.mean)


def _dataframe(direct_effect: float, n: int = 250) -> pd.DataFrame:
    """X and Y both depend on Z; Y also depends on X with weight ``direct_effect``."""
    rng = np.random.default_rng(1)
    z = 4 * rng.standard_normal(n)
    x = 2 * np.sin(z) + rng.standard_normal(n)
    y = 2 * np.sin(z) + direct_effect * x + rng.standard_normal(n)
    return pd.DataFrame({"x": x, "y": y, "z": z})


def test_statistic_matches_formula_with_precomputed_residuals() -> None:
    """The statistic is deterministic given the residuals (no regression or RNG involved)."""
    rng = np.random.default_rng(0)
    resid_x = rng.standard_normal(50)
    resid_y = rng.standard_normal(50)
    r = resid_x * resid_y
    expected_statistic = np.sqrt(50) * r.mean() / np.sqrt((r**2).mean() - r.mean() ** 2)

    result = GCM().test_from_residuals(resid_x, resid_y)

    assert result.statistic == pytest.approx(expected_statistic)
    assert 0.0 <= result.p_value <= 1.0


def test_marginal_test_centers_the_variables() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal(80) + 5
    y = 0.5 * x + rng.standard_normal(80)

    assert GCM().test(x, y) == GCM().test_from_residuals(x - x.mean(), y - y.mean())


def test_marginal_test_ignores_the_means_of_independent_variables() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal(300) + 5
    y = rng.standard_normal(300) + 3

    assert not GCM().test(x, y).reject


def test_test_regresses_both_variables_on_z() -> None:
    """With a regressor that predicts the mean, the residuals are the centered variables."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(60) + 5
    y = rng.standard_normal(60) - 2
    z = rng.standard_normal((60, 1))

    result = GCM(estimator=_MeanRegressor()).test(x, y, z)

    assert result == GCM().test_from_residuals(x - x.mean(), y - y.mean())


def test_compute_residuals_handles_matrices_column_by_column() -> None:
    rng = np.random.default_rng(4)
    v = rng.standard_normal((40, 3))
    z = rng.standard_normal((40, 1))

    residuals = compute_residuals(v, z, _MeanRegressor())

    assert residuals.shape == v.shape
    assert residuals == pytest.approx(v - v.mean(axis=0))


def test_does_not_reject_independence() -> None:
    assert not gcm_test(_dataframe(0.0), "x", "y", ["z"]).reject


def test_rejects_dependence() -> None:
    assert gcm_test(_dataframe(1.0), "x", "y", ["z"]).reject


def test_multivariate_rejects_dependence() -> None:
    rng = np.random.default_rng(1)
    n = 300
    z = (2 * rng.standard_normal(n)).reshape(n, 1)
    x = np.column_stack(
        [np.sin(z[:, 0]) + rng.standard_normal(n), z[:, 0] + rng.standard_normal(n)]
    )
    y = np.column_stack([np.cos(z[:, 0]) + rng.standard_normal(n), rng.standard_normal(n)])
    y[:, 0] += x[:, 0]

    assert GCM(nsim=199).test(x, y, z).reject


def test_function_matches_class_method() -> None:
    df = _dataframe(1.0)

    expected = GCM().test(df["x"].to_numpy(), df["y"].to_numpy(), df[["z"]].to_numpy())

    assert gcm_test(df, "x", "y", ["z"]) == expected


def test_empty_z_list_is_the_marginal_test() -> None:
    df = _dataframe(1.0)

    assert gcm_test(df, "x", "y", []) == gcm_test(df, "x", "y")


@pytest.mark.parametrize("constant", ["x", "y"])
@pytest.mark.parametrize("conditioning", [["z"], []])
def test_constant_tested_variable_is_independent(constant: str, conditioning: list[str]) -> None:
    df = _dataframe(1.0).assign(**{constant: 1.0})

    result = gcm_test(df, "x", "y", conditioning)

    assert (result.statistic, result.p_value, result.reject) == (0.0, 1.0, False)


def test_constant_column_of_a_multivariate_variable_is_ignored() -> None:
    rng = np.random.default_rng(6)
    n = 200
    z = rng.standard_normal((n, 1))
    x = np.sin(z[:, 0]) + rng.standard_normal(n)
    y = x + rng.standard_normal(n)
    x_with_constant = np.column_stack([np.ones(n), x])

    result = GCM(estimator=_MeanRegressor()).test(x_with_constant, y, z)

    assert result == GCM(estimator=_MeanRegressor()).test(x, y, z)
    assert result.reject
