# Calls the installed causal-learn dependency at runtime — no source from
# that package is copied or redistributed here.

import pandas as pd
from causallearn.utils.KCI.KCI import KCI_CInd, KCI_UInd

from src.CI_tests.common import CIResult, frame_to_arrays


def kci_test_causallearn(
    data: pd.DataFrame,
    x: str,
    y: str,
    z: list[str] | None = None,
    alpha: float = 0.05,
    approx: bool = True,
    nullss: int = 5000,
) -> CIResult:
    """Call causal-learn's `KCI_CInd.compute_pvalue`, or `KCI_UInd` if there is no Z.

    A bridge, not a reimplementation: it hands the same data to causal-learn and
    wraps its output in the shared `CIResult` contract.

    Parameters
    ----------
    data : DataFrame
        Observations in rows, numeric variables in columns.

    x : str
        Name of the first tested column.

    y : str
        Name of the second tested column.

    z : list of str or None, optional (default: None)
        Names of the conditioning columns. If None or empty, the unconditional
        `KCI_UInd` is used.

    alpha : float, optional (default: 0.05)
        Significance level; the test rejects if ``p_value < alpha``.

    approx : bool, optional (default: True)
        Whether to use the Gamma approximation instead of Monte Carlo draws.

    nullss : int, optional (default: 5000)
        Number of Monte Carlo draws, used only if ``approx`` is False.

    Returns
    -------
    result : CIResult
        causal-learn's statistic, p-value and decision at level ``alpha``.
    """
    x_values, y_values, z_values = frame_to_arrays(data, x, y, z)
    x_column, y_column = x_values.reshape(-1, 1), y_values.reshape(-1, 1)

    if z_values is None:
        p_value, test_stat = KCI_UInd(approx=approx, null_ss=nullss).compute_pvalue(
            x_column, y_column
        )
    else:
        p_value, test_stat = KCI_CInd(approx=approx, nullss=nullss).compute_pvalue(
            x_column, y_column, z_values
        )
    return CIResult(statistic=float(test_stat), p_value=float(p_value), reject=(p_value < alpha))
