# Algorithm independently reimplemented from Shah & Peters (2020); its
# structure was checked against the official R package
# github.com/cran/GeneralisedCovarianceMeasure (Peters & Shah, GPL-2),
# vendored for reference (not redistributed) at Referencias/GCM_R_source/.

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
from xgboost import XGBRegressor

from src.CI_tests.common import CIResult, col, n_cols
from src.CI_tests.GCM.gcm_utils import Regressor, compute_residuals


def gcm_test(
    x: NDArray[np.float64] | None = None,
    y: NDArray[np.float64] | None = None,
    z: NDArray[np.float64] | None = None,
    alpha: float = 0.05,
    estimator: Regressor | None = None,
    nsim: int = 499,
    resid_x_on_z: NDArray[np.float64] | None = None,
    resid_y_on_z: NDArray[np.float64] | None = None,
) -> CIResult:
    """Test conditional independence via the Generalised Covariance Measure.

    🇬🇧 Regresses x on z and y on z, then tests whether the residual
    product has zero mean (asymptotically standard normal under
    H0: x⊥y|z). If z is None, no regression is performed and the raw
    x/y are used as residuals (unconditional independence test).

    🇪🇸 Regresa x sobre z e y sobre z, y comprueba si el producto de los
    residuos tiene media cero (asintóticamente normal estándar bajo
    H0: x⊥y|z). Si z es None, no se hace ninguna regresión y se usan
    x/y directamente como residuos (test de independencia incondicional).

    Args:
        x: Test variable, shape (n,) — Variable a testear, forma (n,).
        y: Test variable, shape (n,) — Variable a testear, forma (n,).
        z: Conditioning set, shape (n, dz), or None for an unconditional
            test — Conjunto de condicionamiento, forma (n, dz), o None
            para un test incondicional.
        alpha: Significance level — Nivel de significación.
        estimator: Regressor used for both x ~ z and y ~ z; defaults to
            `XGBRegressor()` if None — Regresor usado para x ~ z e y ~ z;
            por defecto `XGBRegressor()` si es None.
        nsim: Number of multiplier-bootstrap draws, only used when x or y
            is multivariate — Número de repeticiones del bootstrap
            multiplicador, solo se usa si x o y son multivariantes.
        resid_x_on_z: Precomputed residuals of x on z, skips the
            regression if given — Residuos precalculados de x sobre z,
            se salta la regresión si se proporciona.
        resid_y_on_z: Precomputed residuals of y on z, skips the
            regression if given — Residuos precalculados de y sobre z,
            se salta la regresión si se proporciona.

    Returns:
        CIResult with the test statistic, p-value and reject decision —
        CIResult con el estadístico, el p-valor y la decisión de rechazo.
    """
    if z is None:
        if resid_x_on_z is None:
            resid_x_on_z = x
        if resid_y_on_z is None:
            resid_y_on_z = y
    else:
        if estimator is None:
            estimator = XGBRegressor()
        resid_x_on_z = compute_residuals(x, z, estimator, resid_x_on_z)
        resid_y_on_z = compute_residuals(y, z, estimator, resid_y_on_z)

    if resid_x_on_z is None or resid_y_on_z is None:
        raise ValueError("Either x/y or their precomputed residuals must be provided.")

    n = resid_x_on_z.shape[0]

    if n_cols(resid_x_on_z) > 1 or n_cols(resid_y_on_z) > 1:
        d_x, d_y = n_cols(resid_x_on_z), n_cols(resid_y_on_z)

        r_mat = np.array(
            [col(resid_x_on_z, j) * col(resid_y_on_z, k) for j in range(d_x) for k in range(d_y)]
        )

        row_mean = r_mat.mean(axis=1, keepdims=True)
        row_std = np.sqrt((r_mat**2).mean(axis=1, keepdims=True) - row_mean**2)
        r_mat = r_mat / row_std

        test_statistic = float(np.max(np.abs(r_mat.mean(axis=1))) * np.sqrt(n))

        multipliers = np.random.default_rng().standard_normal((n, nsim))
        sim_draws = np.max(np.abs(r_mat @ multipliers), axis=0) / np.sqrt(n)
        p_value = float((np.sum(sim_draws >= test_statistic) + 1) / (nsim + 1))
    else:
        r = resid_x_on_z * resid_y_on_z
        mean_r = r.mean()
        test_statistic = float(np.sqrt(n) * mean_r / np.sqrt((r**2).mean() - mean_r**2))
        p_value = float(2 * norm.sf(abs(test_statistic)))

    return CIResult(statistic=test_statistic, p_value=p_value, reject=(p_value < alpha))
