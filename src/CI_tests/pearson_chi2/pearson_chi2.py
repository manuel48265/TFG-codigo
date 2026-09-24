import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.CI_tests.common import CIResult


def pearson_chi2_test(
    x: pd.Series,
    y: pd.Series,
    z: pd.DataFrame | None = None,
    alpha: float = 0.05,
) -> CIResult:
    x_categories = x.unique()
    y_categories = y.unique()

    frame = pd.concat([x.rename("x"), y.rename("y")] + ([z] if z is not None else []), axis=1)
    strata = [frame] if z is None else [g for _, g in frame.groupby(list(z.columns))]

    stat = 0.0
    dof = 0
    for stratum in strata:
        table = pd.crosstab(stratum["x"], stratum["y"]).reindex(
            index=x_categories, columns=y_categories, fill_value=0
        )
        n = table.to_numpy().sum()
        if n == 0:
            continue
        row_sums = table.sum(axis=1).to_numpy()
        col_sums = table.sum(axis=0).to_numpy()
        expected = np.outer(row_sums, col_sums) / n
        observed = table.to_numpy()
        mask = expected > 0
        stat += float(((observed[mask] - expected[mask]) ** 2 / expected[mask]).sum())

        zero_rows = int((row_sums == 0).sum())
        zero_cols = int((col_sums == 0).sum())
        dof += (len(x_categories) - 1 - zero_rows) * (len(y_categories) - 1 - zero_cols)

    if dof == 0:
        return CIResult(statistic=stat, p_value=1.0, reject=False)

    p_value = float(chi2.sf(stat, dof))
    return CIResult(statistic=stat, p_value=p_value, reject=(p_value < alpha))
