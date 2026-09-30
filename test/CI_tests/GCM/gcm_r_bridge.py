# Calls the official R package GeneralisedCovarianceMeasure (Peters & Shah,
# GPL-2, github.com/cran/GeneralisedCovarianceMeasure) at runtime via rpy2 —
# no source from that package is copied or redistributed here.

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri as numpy2ri
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from src.CI_tests.common import CIResult, frame_to_arrays
from src.CI_tests.GCM.gcm_utils import compute_residuals
from test.CI_tests.GCM.ols_regressor import OLSRegressor

# Drops empty site-library paths (e.g. /usr/lib/R/site-library on a fresh
# install) before loading any package — silences R's harmless but noisy
# "libraries ... contain no packages" warning on every import.
ro.r(".libPaths(Filter(function(p) length(list.files(p)) > 0, .libPaths()), include.site = FALSE)")

_conversion = default_converter + numpy2ri.converter
_gcm_r = importr("GeneralisedCovarianceMeasure")


def gcm_test_r(
    data: pd.DataFrame,
    x: str,
    y: str,
    z: list[str] | None = None,
    alpha: float = 0.05,
) -> CIResult:
    """Call the official R package `GeneralisedCovarianceMeasure::gcm.test`.

    A bridge, not a reimplementation: it hands the residuals to the R package
    (via rpy2) and wraps its output in the shared `CIResult` contract. The
    residuals of X and Y on Z are computed here with `OLSRegressor`, so they are
    the same ones a Python test using that regressor sees. If Z is empty, they
    are the centered variables.

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
        The R package's statistic, p-value and decision at level ``alpha``.
    """
    x_values, y_values, z_values = frame_to_arrays(data, x, y, z)
    if z_values is None:
        z_values = np.empty((x_values.shape[0], 0))

    with localconverter(_conversion):
        result = _gcm_r.gcm_test(
            **{
                "resid.XonZ": compute_residuals(x_values, z_values, OLSRegressor()),
                "resid.YonZ": compute_residuals(y_values, z_values, OLSRegressor()),
                "alpha": alpha,
            }
        )
        return CIResult(
            statistic=float(result.getbyname("test.statistic")[0]),
            p_value=float(result.getbyname("p.value")[0]),
            reject=bool(result.getbyname("reject")[0]),
        )
