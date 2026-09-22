import numpy as np
import pytest

from src.CI_tests.Fisher_Z.fisher_z import fisher_z_test


def test_matches_ols_residual_correlation() -> None:
    """🇬🇧 Checks the precision-matrix partial correlation against the
    algorithmically independent way to get it: regress x and y on z with
    OLS, Fisher-transform the Pearson correlation of the residuals.

    🇪🇸 Comprueba la correlación parcial vía matriz de precisión contra la
    vía algorítmicamente independiente de obtenerla: regresar x e y sobre z
    con OLS, transformar por Fisher la correlación de Pearson de los
    residuos.
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

    result = fisher_z_test(x, y, z)

    assert result.statistic == pytest.approx(expected_statistic)


def test_unconditional_reduces_to_pearson_correlation() -> None:
    """🇬🇧 With z=None, the partial correlation must reduce to the plain
    Pearson correlation of x and y.

    🇪🇸 Con z=None, la correlación parcial debe reducirse a la correlación
    de Pearson simple de x e y.
    """
    rng = np.random.default_rng(1)
    n = 200
    x = rng.standard_normal(n)
    y = 0.6 * x + rng.standard_normal(n)

    r_expected = np.corrcoef(x, y)[0, 1]
    expected_statistic = np.arctanh(r_expected) * np.sqrt(n - 3)

    result = fisher_z_test(x, y)

    assert result.statistic == pytest.approx(expected_statistic)


def test_does_not_reject_independence() -> None:
    rng = np.random.default_rng(2)
    n = 300
    z = rng.standard_normal((n, 1))
    x = z[:, 0] + rng.standard_normal(n)
    y = z[:, 0] + rng.standard_normal(n)

    result = fisher_z_test(x, y, z)

    assert not result.reject


def test_rejects_dependence() -> None:
    rng = np.random.default_rng(2)
    n = 300
    z = rng.standard_normal((n, 1))
    x = z[:, 0] + rng.standard_normal(n)
    y = z[:, 0] + x + rng.standard_normal(n)

    result = fisher_z_test(x, y, z)

    assert result.reject


def test_raises_when_sample_size_too_small_for_conditioning_set() -> None:
    rng = np.random.default_rng(3)
    n = 5
    z = rng.standard_normal((n, 3))
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)

    with pytest.raises(ValueError, match="Sample size too small"):
        fisher_z_test(x, y, z)
