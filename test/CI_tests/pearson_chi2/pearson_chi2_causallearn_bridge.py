# Calls the installed causal-learn dependency at runtime — no source from
# that package is copied or redistributed here.

import pandas as pd
from causallearn.utils.cit import CIT

from src.CI_tests.common import CIResult


def pearson_chi2_test_causallearn(
    data: pd.DataFrame,
    x: str,
    y: str,
    z: list[str] | None = None,
    alpha: float = 0.05,
) -> CIResult:
    """Call causal-learn's `chisq` CI test.

    A bridge, not a reimplementation: it hands the same data to causal-learn's
    `chisq` test and wraps its output in the shared `CIResult` contract. Its
    public `__call__` only returns a p-value, so ``statistic`` is None.

    Parameters
    ----------
    data : DataFrame
        Observations in rows, categorical variables in columns.

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
    columns = [x, y, *(z or [])]
    cit = CIT(data[columns].to_numpy(dtype=object), "chisq")
    p_value = float(cit(0, 1, tuple(range(2, len(columns)))))
    return CIResult(statistic=None, p_value=p_value, reject=(p_value < alpha))
