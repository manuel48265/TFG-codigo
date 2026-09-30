# Z = atanh(r_XY.S) * sqrt(n - |S| - 3), the standard PC-algorithm CI test
# for Gaussian data (Kalisch & Bühlmann 2007, "Estimating High-Dimensional
# Directed Acyclic Graphs with the PC-Algorithm," JMLR).

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from src.CI_tests.common import CIResult, ci_test, col, n_cols


class FisherZ:
    """Conditional independence test based on Fisher's Z-transformed partial correlation.

    The partial correlation of X and Y given Z is read from the precision
    matrix (inverse correlation matrix) of [X, Y, Z], Fisher-transformed and
    tested against N(0, 1). It assumes (X, Y, Z) are jointly Gaussian or
    linearly related. Any size of Z is accepted while ``n - |Z| - 3 > 0``.
    """

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
        x : ndarray of shape (n,)
            Sample of X.

        y : ndarray of shape (n,)
            Sample of Y.

        z : ndarray of shape (n, d_z) or None, optional (default: None)
            Sample of the conditioning set. If None, the test is marginal.

        alpha : float, optional (default: 0.05)
            Significance level; the test rejects if ``p_value < alpha``.

        Returns
        -------
        result : CIResult
            ``statistic`` is the Fisher Z statistic, ``p_value`` the two-sided
            p-value under N(0, 1) and ``reject`` the decision at level ``alpha``.

        Raises
        ------
        ValueError
            If ``n - d_z - 3 <= 0`` or the correlation matrix is singular.
        """
        n = x.shape[0]
        dim_z = n_cols(z) if z is not None else 0
        dof = n - dim_z - 3
        if dof <= 0:
            raise ValueError(f"Sample size too small for |z|={dim_z}: n - |z| - 3 = {dof} <= 0.")

        columns = [x, y]
        if z is not None:
            columns += [col(z, j) for j in range(dim_z)]
        corr = np.corrcoef(np.column_stack(columns), rowvar=False)
        try:
            precision = np.linalg.inv(corr)
        except np.linalg.LinAlgError as err:
            raise ValueError(
                "Correlation matrix is singular, cannot run the Fisher-Z test."
            ) from err

        r = -precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])
        r = np.clip(r, -1.0 + np.finfo(float).eps, 1.0 - np.finfo(float).eps)

        z_stat = float(np.arctanh(r) * np.sqrt(dof))
        p_value = float(2 * (1 - norm.cdf(abs(z_stat))))

        return CIResult(statistic=z_stat, p_value=p_value, reject=(p_value < alpha))


@ci_test
def fisher_z_test(
    data: pd.DataFrame,
    x: str,
    y: str,
    z: list[str] | None = None,
    alpha: float = 0.05,
) -> CIResult:
    """Test conditional independence of two columns given others with the Fisher-Z test.

    Wrapper that selects the columns and runs ``FisherZ.test``.

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
        ``statistic`` is the Fisher Z statistic, ``p_value`` the two-sided
        p-value under N(0, 1) and ``reject`` the decision at level ``alpha``.
    """
    z_values = data[z].to_numpy(dtype=np.float64) if z else None
    return FisherZ().test(
        data[x].to_numpy(dtype=np.float64), data[y].to_numpy(dtype=np.float64), z_values, alpha
    )
