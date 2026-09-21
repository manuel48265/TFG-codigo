import numpy as np
import pytest

from src.CI_tests.GCM.gcm import gcm_test


def test_statistic_matches_formula_with_precomputed_residuals() -> None:
    """🇬🇧 With residuals given directly, the statistic is fully
    deterministic (equation 3, no regression/RNG involved) — check it
    against the formula computed by hand.

    🇪🇸 Con los residuos dados directamente, el estadístico es totalmente
    determinista (ecuación 3, sin regresión ni aleatoriedad de por medio)
    — se comprueba contra la fórmula calculada a mano.
    """
    rng = np.random.default_rng(0)
    resid_x = rng.standard_normal(50)
    resid_y = rng.standard_normal(50)
    r = resid_x * resid_y
    expected_statistic = np.sqrt(50) * r.mean() / np.sqrt((r**2).mean() - r.mean() ** 2)

    result = gcm_test(resid_x_on_z=resid_x, resid_y_on_z=resid_y)

    assert result.statistic == pytest.approx(expected_statistic)
    assert 0.0 <= result.p_value <= 1.0


def test_univariate_does_not_reject_independence() -> None:
    rng = np.random.default_rng(1)
    n = 250
    z = (4 * rng.standard_normal(n)).reshape(n, 1)
    x = 2 * np.sin(z[:, 0]) + rng.standard_normal(n)
    y = 2 * np.sin(z[:, 0]) + rng.standard_normal(n)

    result = gcm_test(x, y, z)

    assert not result.reject


def test_univariate_rejects_dependence() -> None:
    rng = np.random.default_rng(1)
    n = 250
    z = (4 * rng.standard_normal(n)).reshape(n, 1)
    x = 2 * np.sin(z[:, 0]) + rng.standard_normal(n)
    y = 2 * np.sin(z[:, 0]) + x + rng.standard_normal(n)

    result = gcm_test(x, y, z)

    assert result.reject


def test_multivariate_rejects_dependence() -> None:
    rng = np.random.default_rng(1)
    n = 300
    z = (2 * rng.standard_normal(n)).reshape(n, 1)
    x = np.column_stack(
        [np.sin(z[:, 0]) + rng.standard_normal(n), z[:, 0] + rng.standard_normal(n)]
    )
    y = np.column_stack([np.cos(z[:, 0]) + rng.standard_normal(n), rng.standard_normal(n)])
    y[:, 0] += x[:, 0]

    result = gcm_test(x, y, z, nsim=199)

    assert result.reject


def test_raises_without_x_or_precomputed_residuals() -> None:
    with pytest.raises(ValueError):
        gcm_test(y=np.zeros(10), z=np.zeros((10, 1)))
