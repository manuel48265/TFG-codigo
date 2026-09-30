# Algorithm independently reimplemented from Shah & Peters (2020); its
# structure was checked against the official R package
# github.com/cran/GeneralisedCovarianceMeasure (Peters & Shah, GPL-2),
# vendored for reference (not redistributed) at Referencias/GCM_R_source/.

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm
from xgboost import XGBRegressor

from src.CI_tests.common import CIResult, ci_test, col, n_cols
from src.CI_tests.GCM.gcm_utils import Regressor, compute_residuals


def _drop_constant_columns(a: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Return ``a`` without its constant columns, or None if all of them are constant."""
    if a.ndim == 1:
        return a if np.ptp(a) > 0 else None
    kept = a[:, np.ptp(a, axis=0) > 0]
    return kept if kept.shape[1] else None


class GCM:
    """Conditional independence test based on the Generalised Covariance Measure.

    X and Y are each regressed on Z, and the test checks whether the product of
    their residuals has zero mean (Shah & Peters 2020). If X and Y are
    univariate the statistic is asymptotically standard normal under the null
    hypothesis; if either is multivariate, the maximum over all pairs of
    columns is used and its p-value comes from a multiplier bootstrap.

    Parameters
    ----------
    estimator : Regressor or None, optional (default: None)
        Regressor used for both ``x ~ z`` and ``y ~ z``. If None,
        ``XGBRegressor()`` is used. It is refitted on every call.

    nsim : int, optional (default: 499)
        Number of multiplier-bootstrap draws. Used only if X or Y is
        multivariate.
    """

    def __init__(self, estimator: Regressor | None = None, nsim: int = 499) -> None:
        self.estimator = estimator
        self.nsim = nsim

    def test_from_residuals(
        self,
        resid_x_on_z: NDArray[np.float64],
        resid_y_on_z: NDArray[np.float64],
        alpha: float = 0.05,
    ) -> CIResult:
        """Test whether the product of two residual samples has zero mean.

        Parameters
        ----------
        resid_x_on_z : ndarray of shape (n,) or (n, d_x)
            Residuals of X regressed on Z.

        resid_y_on_z : ndarray of shape (n,) or (n, d_y)
            Residuals of Y regressed on Z.

        alpha : float, optional (default: 0.05)
            Significance level; the test rejects if ``p_value < alpha``.

        Returns
        -------
        result : CIResult
            ``statistic`` is the GCM statistic and ``p_value`` the two-sided
            normal p-value, or the multiplier-bootstrap p-value if X or Y is
            multivariate.
        """
        n = resid_x_on_z.shape[0]

        if n_cols(resid_x_on_z) > 1 or n_cols(resid_y_on_z) > 1:
            d_x, d_y = n_cols(resid_x_on_z), n_cols(resid_y_on_z)

            r_mat = np.array(
                [
                    col(resid_x_on_z, j) * col(resid_y_on_z, k)
                    for j in range(d_x)
                    for k in range(d_y)
                ]
            )

            row_mean = r_mat.mean(axis=1, keepdims=True)
            row_std = np.sqrt((r_mat**2).mean(axis=1, keepdims=True) - row_mean**2)
            r_mat = r_mat / row_std

            test_statistic = float(np.max(np.abs(r_mat.mean(axis=1))) * np.sqrt(n))

            multipliers = np.random.default_rng().standard_normal((n, self.nsim))
            sim_draws = np.max(np.abs(r_mat @ multipliers), axis=0) / np.sqrt(n)
            p_value = float((np.sum(sim_draws >= test_statistic) + 1) / (self.nsim + 1))
        else:
            r = resid_x_on_z.reshape(n) * resid_y_on_z.reshape(n)
            mean_r = r.mean()
            variance = (r**2).mean() - mean_r**2
            if variance <= 0:
                return CIResult(statistic=0.0, p_value=1.0, reject=False)
            test_statistic = float(np.sqrt(n) * mean_r / np.sqrt(variance))
            p_value = float(2 * norm.sf(abs(test_statistic)))

        return CIResult(statistic=test_statistic, p_value=p_value, reject=(p_value < alpha))

    def test(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64] | None = None,
        alpha: float = 0.05,
    ) -> CIResult:
        """Test whether X and Y are conditionally independent given Z.

        Parameters
        ----------
        x : ndarray of shape (n,) or (n, d_x)
            Sample of X.

        y : ndarray of shape (n,) or (n, d_y)
            Sample of Y.

        z : ndarray of shape (n, d_z) or None, optional (default: None)
            Sample of the conditioning set. If None, the test is marginal:
            ``x`` and ``y`` are centered and used as residuals.

        alpha : float, optional (default: 0.05)
            Significance level; the test rejects if ``p_value < alpha``.

        Returns
        -------
        result : CIResult
            Constant columns of X and Y are ignored; if all the columns of X, or
            all those of Y, are constant, the variables are trivially
            independent and the result is ``statistic=0``, ``p_value=1``.
            Otherwise ``statistic`` is the GCM statistic and ``p_value`` the
            two-sided normal p-value, or the multiplier-bootstrap p-value if X
            or Y is multivariate.
        """
        x_var, y_var = _drop_constant_columns(x), _drop_constant_columns(y)
        if x_var is None or y_var is None:
            return CIResult(statistic=0.0, p_value=1.0, reject=False)

        if z is None:
            return self.test_from_residuals(
                x_var - x_var.mean(axis=0), y_var - y_var.mean(axis=0), alpha
            )

        estimator = XGBRegressor() if self.estimator is None else self.estimator
        return self.test_from_residuals(
            compute_residuals(x_var, z, estimator), compute_residuals(y_var, z, estimator), alpha
        )


@ci_test
def gcm_test(
    data: pd.DataFrame,
    x: str,
    y: str,
    z: list[str] | None = None,
    alpha: float = 0.05,
) -> CIResult:
    """Test conditional independence of two columns given others with the GCM test.

    Wrapper that selects the columns and runs ``GCM().test`` with its default
    parameters.

    Parameters
    ----------
    data : DataFrame
        Observations in rows, numeric variables in columns.

    x : str
        Name of the first tested column.

    y : str
        Name of the second tested column.

    z : list of str or None, optional (default: None)
        Names of the conditioning columns. If None or empty, the test is marginal.

    alpha : float, optional (default: 0.05)
        Significance level; the test rejects if ``p_value < alpha``.

    Returns
    -------
    result : CIResult
        ``statistic`` is the GCM statistic, ``p_value`` the two-sided normal
        p-value and ``reject`` the decision at level ``alpha``.
    """
    z_values = data[z].to_numpy(dtype=np.float64) if z else None
    return GCM().test(
        data[x].to_numpy(dtype=np.float64), data[y].to_numpy(dtype=np.float64), z_values, alpha
    )
