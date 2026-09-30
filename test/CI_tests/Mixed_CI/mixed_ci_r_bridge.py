# Calls the installed MXM dependency at runtime — no source from that
# package is copied or redistributed here.

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import default_converter, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from src.CI_tests.common import CIResult

# Same empty-site-library cleanup as the GCM bridge — silences R's harmless
# "libraries ... contain no packages" warning on every import.
ro.r(".libPaths(Filter(function(p) length(list.files(p)) > 0, .libPaths()), include.site = FALSE)")

_conversion = default_converter + pandas2ri.converter
_mxm = importr("MXM")


def mixed_ci_test_r(
    data: pd.DataFrame,
    x: str,
    y: str,
    z: list[str] | None = None,
    alpha: float = 0.05,
) -> CIResult:
    """Call the official R package `MXM::ci.mm`.

    A bridge, not a reimplementation: it hands the same data to the R package
    (via rpy2) and wraps its output in the shared `CIResult` contract.

    Parameters
    ----------
    data : DataFrame
        Observations in rows, variables in columns. Categorical columns need
        dtype ``category``.

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
        ``ci.mm``'s p-value and decision at level ``alpha``; ``statistic`` is None.
    """
    # ci.mm fits `lm(y ~ ., data = ds1)`, so a column literally named "y" (or "x")
    # would shadow the local variable and silently fit the wrong model; the
    # columns are renamed to collision-free tags before crossing into R.
    cs_rename = {col: f"r_cs{i}" for i, col in enumerate(z)} if z else {}
    rename = {x: "r_ind1", y: "r_ind2"} | cs_rename
    r_ready = data.rename(columns=rename)

    with localconverter(_conversion):
        r_df = ro.conversion.get_conversion().py2rpy(r_ready)
        cs = ro.StrVector([rename[c] for c in z]) if z else ro.NULL
        result = _mxm.ci_mm("r_ind1", "r_ind2", cs, r_df)
        _, log_p, _ = list(result)

    p_value = float(np.exp(log_p))
    return CIResult(statistic=None, p_value=p_value, reject=(p_value < alpha))
