# Calls the official R package GeneralisedCovarianceMeasure (Peters & Shah,
# GPL-2, github.com/cran/GeneralisedCovarianceMeasure) at runtime via rpy2 —
# no source from that package is copied or redistributed here.

import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri as numpy2ri
from numpy.typing import NDArray
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from src.CI_tests.common import CIResult

# Drops empty site-library paths (e.g. /usr/lib/R/site-library on a fresh
# install) before loading any package — silences R's harmless but noisy
# "libraries ... contain no packages" warning on every import.
ro.r(".libPaths(Filter(function(p) length(list.files(p)) > 0, .libPaths()), include.site = FALSE)")

_conversion = default_converter + numpy2ri.converter
_gcm_r = importr("GeneralisedCovarianceMeasure")


def gcm_test_r(
    x: NDArray[np.float64] | None = None,
    y: NDArray[np.float64] | None = None,
    z: NDArray[np.float64] | None = None,
    alpha: float = 0.05,
    nsim: int = 499,
    resid_x_on_z: NDArray[np.float64] | None = None,
    resid_y_on_z: NDArray[np.float64] | None = None,
) -> CIResult:
    """Call the official R package `GeneralisedCovarianceMeasure::gcm.test`.

    A bridge, not a reimplementation: it hands the same inputs to the R package
    (via rpy2) and wraps its output in the shared `CIResult` contract. If both
    precomputed residuals are given, `x`, `y` and `z` are ignored.

    Parameters
    ----------
    x : ndarray of shape (n,) or None, optional (default: None)
        Sample of X.

    y : ndarray of shape (n,) or None, optional (default: None)
        Sample of Y.

    z : ndarray of shape (n, d_z) or None, optional (default: None)
        Sample of the conditioning set.

    alpha : float, optional (default: 0.05)
        Significance level; the test rejects if ``p_value < alpha``.

    nsim : int, optional (default: 499)
        Number of multiplier-bootstrap draws (multivariate case only).

    resid_x_on_z : ndarray or None, optional (default: None)
        Precomputed residuals of X on Z.

    resid_y_on_z : ndarray or None, optional (default: None)
        Precomputed residuals of Y on Z.

    Returns
    -------
    result : CIResult
        The R package's statistic, p-value and decision.
    """
    r_args: dict[str, object] = {"alpha": alpha, "nsim": nsim}
    if resid_x_on_z is not None and resid_y_on_z is not None:
        r_args["resid.XonZ"] = resid_x_on_z
        r_args["resid.YonZ"] = resid_y_on_z
    else:
        r_args["X"] = x
        r_args["Y"] = y
        r_args["Z"] = z

    with localconverter(_conversion):
        result = _gcm_r.gcm_test(**r_args)
        return CIResult(
            statistic=float(result.getbyname("test.statistic")[0]),
            p_value=float(result.getbyname("p.value")[0]),
            reject=bool(result.getbyname("reject")[0]),
        )
