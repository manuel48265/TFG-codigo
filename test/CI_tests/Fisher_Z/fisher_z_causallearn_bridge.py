# Calls the installed causal-learn dependency at runtime — no source from
# that package is copied or redistributed here.

import numpy as np
import pandas as pd
from causallearn.utils.cit import FisherZ

from src.CI_tests.common import CIResult, frame_to_arrays


def fisher_z_test_causallearn(
    data: pd.DataFrame,
    x: str,
    y: str,
    z: list[str] | None = None,
    alpha: float = 0.05,
) -> CIResult:
    """Call causal-learn's `FisherZ` CI test.

    A bridge, not a reimplementation: it hands the same data to causal-learn's
    `FisherZ` and wraps its output in the shared `CIResult` contract. Its public
    `__call__` only returns a p-value, so ``statistic`` is None.

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
        causal-learn's p-value and decision at level ``alpha``.
    """
    x_values, y_values, z_values = frame_to_arrays(data, x, y, z)
    columns = [x_values, y_values] + ([] if z_values is None else list(z_values.T))
    dim_z = len(columns) - 2

    p_value = float(FisherZ(np.column_stack(columns))(0, 1, tuple(range(2, 2 + dim_z))))
    return CIResult(statistic=None, p_value=p_value, reject=(p_value < alpha))
