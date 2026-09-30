# Pearson chi-squared CI test for categorical data: the chi-squared statistics
# of the X-Y contingency tables of each stratum of Z are summed and compared
# against a chi-squared distribution with the summed degrees of freedom.
# See P. Averin, "Conditional Independence Tests for Constraint-Based Causal
# Discovery: A Survey," TMLR (07/2026), Sec. 5.2, which cites A. Agresti,
# "Categorical Data Analysis," 2nd ed., Wiley, 2002. Degrees of freedom here
# exclude the empty rows and columns of each stratum.

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.CI_tests.common import CIResult, ci_test


class PearsonChi2:
    """Conditional independence test based on Pearson's chi-squared statistic.

    For each configuration of Z, the chi-squared statistic and degrees of
    freedom of the contingency table of X and Y are computed; both are summed
    over configurations and the p-value is read from the chi-squared
    distribution. Variables must be categorical. With Z empty, the test is a
    plain chi-squared test of independence. Constant strata add nothing, and
    if no degrees of freedom remain the p-value is 1.
    """

    def test(
        self,
        x: pd.Series,
        y: pd.Series,
        z: pd.DataFrame | None = None,
        alpha: float = 0.05,
    ) -> CIResult:
        """Test whether X and Y are conditionally independent given Z.

        Parameters
        ----------
        x : Series of length n
            Categorical sample of X.

        y : Series of length n
            Categorical sample of Y.

        z : DataFrame of n rows or None, optional (default: None)
            Categorical conditioning columns. If None, the test is marginal.

        alpha : float, optional (default: 0.05)
            Significance level; the test rejects if ``p_value < alpha``.

        Returns
        -------
        result : CIResult
            ``statistic`` is the summed chi-squared statistic, ``p_value`` its
            upper tail under the summed degrees of freedom and ``reject`` the
            decision at level ``alpha``.
        """
        x_categories = x.unique()
        y_categories = y.unique()

        frame = pd.DataFrame({"x": x, "y": y})
        strata = (
            [frame]
            if z is None
            else [group for _, group in frame.groupby([z[name] for name in z.columns])]
        )

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


@ci_test
def pearson_chi2_test(
    data: pd.DataFrame,
    x: str,
    y: str,
    z: list[str] | None = None,
    alpha: float = 0.05,
) -> CIResult:
    """Test conditional independence of two columns given others with Pearson's chi-squared test.

    Wrapper that selects the columns and runs ``PearsonChi2.test``.

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
        ``statistic`` is the summed chi-squared statistic, ``p_value`` its
        upper tail under the summed degrees of freedom and ``reject`` the
        decision at level ``alpha``.
    """
    return PearsonChi2().test(data[x], data[y], data[z] if z else None, alpha)
