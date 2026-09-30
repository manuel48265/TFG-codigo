# Calls the installed causal-learn dependency at runtime — no source from
# that package is copied or redistributed here.

import numpy as np
from causallearn.utils.KCI.KCI import KCI_CInd
from numpy.typing import NDArray

from src.CI_tests.common import CIResult


def kci_test_causallearn(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    alpha: float = 0.05,
    **kwargs: object,
) -> CIResult:
    """Call causal-learn's `KCI_CInd.compute_pvalue`.

    A bridge, not a reimplementation: it hands the same inputs to causal-learn's
    `KCI_CInd` and wraps its output in the shared `CIResult` contract.

    Parameters
    ----------
    x : ndarray of shape (n, d_x)
        Sample of X.

    y : ndarray of shape (n, d_y)
        Sample of Y.

    z : ndarray of shape (n, d_z)
        Sample of the conditioning set.

    alpha : float, optional (default: 0.05)
        Significance level; the test rejects if ``p_value < alpha``.

    **kwargs
        Forwarded to `KCI_CInd.__init__`.

    Returns
    -------
    result : CIResult
        causal-learn's statistic, p-value and decision at level ``alpha``.
    """
    p_value, test_stat = KCI_CInd(**kwargs).compute_pvalue(x, y, z)
    return CIResult(statistic=float(test_stat), p_value=float(p_value), reject=(p_value < alpha))
