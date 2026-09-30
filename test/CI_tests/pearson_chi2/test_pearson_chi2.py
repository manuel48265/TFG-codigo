import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, chi2_contingency

from src.CI_tests.pearson_chi2.pearson_chi2 import PearsonChi2, pearson_chi2_test


def test_matches_stratified_chi2_contingency() -> None:
    """Check the stratified sum against the algorithmically independent route:
    per-stratum `scipy.stats.chi2_contingency`, summed by hand.
    """
    rng = np.random.default_rng(0)
    n = 600
    z = rng.integers(0, 3, size=n)
    x = np.where(rng.random(n) < 0.5 + 0.1 * z, "a", "b")
    y = np.where(rng.random(n) < 0.5 + 0.1 * (x == "a"), "c", "d")
    df = pd.DataFrame({"x": x, "y": y, "z": z})

    stat_expected = 0.0
    dof_expected = 0.0
    for _, stratum in df.groupby("z"):
        table = pd.crosstab(stratum["x"], stratum["y"]).reindex(
            index=["a", "b"], columns=["c", "d"], fill_value=0
        )
        stat, _, dof, _ = chi2_contingency(table, correction=False)
        stat_expected += stat
        dof_expected += dof

    result = PearsonChi2().test(df["x"], df["y"], df[["z"]])

    assert result.statistic == pytest.approx(stat_expected)
    assert result.p_value == pytest.approx(float(chi2.sf(stat_expected, dof_expected)))


def test_unconditional_matches_chi2_contingency() -> None:
    """With z=None, the single stratum must match a plain `chi2_contingency`
    on the full x/y table.
    """
    rng = np.random.default_rng(1)
    n = 400
    x = pd.Series(rng.choice(["a", "b", "c"], size=n))
    y = pd.Series(np.where(rng.random(n) < 0.5 + 0.15 * (x == "a"), "yes", "no"))

    table = pd.crosstab(x, y)
    stat_expected, p_expected, _, _ = chi2_contingency(table, correction=False)

    result = PearsonChi2().test(x, y)

    assert result.statistic == pytest.approx(stat_expected)
    assert result.p_value == pytest.approx(p_expected)


def test_does_not_reject_independence() -> None:
    rng = np.random.default_rng(2)
    n = 500
    z = pd.Series(rng.integers(0, 3, size=n), name="z")
    x = pd.Series(rng.choice(["a", "b"], size=n))
    y = pd.Series(rng.choice(["c", "d"], size=n))

    result = PearsonChi2().test(x, y, z.to_frame())

    assert not result.reject


def test_rejects_dependence() -> None:
    rng = np.random.default_rng(3)
    n = 500
    z = pd.Series(rng.integers(0, 3, size=n), name="z")
    x = pd.Series(rng.choice(["a", "b"], size=n))
    y = pd.Series(np.where(rng.random(n) < 0.9, x, np.where(x == "a", "b", "a")))

    result = PearsonChi2().test(x, y, z.to_frame())

    assert result.reject


def test_returns_no_reject_when_a_variable_is_constant() -> None:
    n = 20
    x = pd.Series(["a"] * n)
    y = pd.Series(["c", "d"] * (n // 2))

    result = PearsonChi2().test(x, y)

    assert result.p_value == 1.0
    assert not result.reject


def _dataframe(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    z1 = rng.integers(0, 3, n)
    z2 = rng.choice(["p", "q"], size=n)
    x = np.where(rng.random(n) < 0.4 + 0.1 * z1, "a", "b")
    y = np.where(rng.random(n) < 0.4 + 0.1 * z1, "c", "d")
    return pd.DataFrame({"x": x, "y": y, "z1": z1, "z2": z2})


def test_dataframe_signature_matches_series_call() -> None:
    df = _dataframe()

    result = pearson_chi2_test(df, "x", "y", ["z1", "z2"])

    assert result == PearsonChi2().test(df["x"], df["y"], df[["z1", "z2"]])


def test_empty_z_list_is_the_marginal_test() -> None:
    df = _dataframe()

    assert pearson_chi2_test(df, "x", "y", []) == pearson_chi2_test(df, "x", "y")


def test_conditioning_columns_may_be_named_like_the_internal_frame() -> None:
    """Conditioning columns named "x" or "y" must give the same result as the
    same columns under other names.
    """
    df = _dataframe().rename(columns={"x": "a", "y": "b", "z1": "x", "z2": "y"})
    renamed = df.rename(columns={"x": "z1", "y": "z2"})

    assert pearson_chi2_test(df, "a", "b", ["x", "y"]) == pearson_chi2_test(
        renamed, "a", "b", ["z1", "z2"]
    )
