import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, chi2_contingency

from src.CI_tests.pearson_chi2.pearson_chi2 import pearson_chi2_test


def test_matches_stratified_chi2_contingency() -> None:
    """🇬🇧 Checks the stratified sum against the algorithmically independent
    way to get it: per-stratum `scipy.stats.chi2_contingency`, summed by hand.

    🇪🇸 Comprueba la suma estratificada contra la vía algorítmicamente
    independiente de obtenerla: `scipy.stats.chi2_contingency` por estrato,
    sumado a mano.
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

    result = pearson_chi2_test(df["x"], df["y"], df[["z"]])

    assert result.statistic == pytest.approx(stat_expected)
    assert result.p_value == pytest.approx(float(chi2.sf(stat_expected, dof_expected)))


def test_unconditional_matches_chi2_contingency() -> None:
    """🇬🇧 With z=None, the single stratum must match a plain
    `chi2_contingency` on the full x/y table.

    🇪🇸 Con z=None, el único estrato debe coincidir con un
    `chi2_contingency` simple sobre la tabla completa de x/y.
    """
    rng = np.random.default_rng(1)
    n = 400
    x = pd.Series(rng.choice(["a", "b", "c"], size=n))
    y = pd.Series(np.where(rng.random(n) < 0.5 + 0.15 * (x == "a"), "yes", "no"))

    table = pd.crosstab(x, y)
    stat_expected, p_expected, _, _ = chi2_contingency(table, correction=False)

    result = pearson_chi2_test(x, y)

    assert result.statistic == pytest.approx(stat_expected)
    assert result.p_value == pytest.approx(p_expected)


def test_does_not_reject_independence() -> None:
    rng = np.random.default_rng(2)
    n = 500
    z = pd.Series(rng.integers(0, 3, size=n), name="z")
    x = pd.Series(rng.choice(["a", "b"], size=n))
    y = pd.Series(rng.choice(["c", "d"], size=n))

    result = pearson_chi2_test(x, y, z.to_frame())

    assert not result.reject


def test_rejects_dependence() -> None:
    rng = np.random.default_rng(3)
    n = 500
    z = pd.Series(rng.integers(0, 3, size=n), name="z")
    x = pd.Series(rng.choice(["a", "b"], size=n))
    y = pd.Series(np.where(rng.random(n) < 0.9, x, np.where(x == "a", "b", "a")))

    result = pearson_chi2_test(x, y, z.to_frame())

    assert result.reject


def test_returns_no_reject_when_a_variable_is_constant() -> None:
    n = 20
    x = pd.Series(["a"] * n)
    y = pd.Series(["c", "d"] * (n // 2))

    result = pearson_chi2_test(x, y)

    assert result.p_value == 1.0
    assert not result.reject
