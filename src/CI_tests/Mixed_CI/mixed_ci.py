# M. Tsagris, G. Borboudakis, V. Lagani, and I. Tsamardinos, "Constraint-based
# causal discovery with mixed data," International Journal of Data Science and
# Analytics 6:19-30 (2018). Likelihood-ratio tests of nested regressions
# (Sec. 2), made symmetric by combining the two directions with the MM rule
# (Sec. 4.1, Eq. 1). The "Fast" prioritization (Sec. 4.2) and the minimum and
# maximum combinations (Eqs. 2-3) are not implemented.

import math
from enum import StrEnum

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.discrete_model import (
    BinaryResultsWrapper,
    MultinomialResultsWrapper,
)
from statsmodels.regression.linear_model import RegressionResultsWrapper

from src.CI_tests.common import CIResult, ci_test


def _fit_discrete(model: sm.Logit | sm.MNLogit) -> BinaryResultsWrapper | MultinomialResultsWrapper:
    """Fit a logistic or multinomial model by maximum likelihood.

    Under quasi-complete separation (some predictor level occurs with a single outcome
    value) the likelihood approaches its supremum while the coefficients diverge, so the
    Hessian at the optimum is singular and statsmodels raises when it inverts it for the
    covariance. The tests use only the likelihood and the predictions, so in that case the
    same Newton fit is repeated without computing the covariance.
    """
    try:
        return model.fit(disp=0)
    except np.linalg.LinAlgError:
        return model.fit(disp=0, cov_params_func=lambda *_: None)


class VariableType(StrEnum):
    """Type of a variable, which selects the regression model used for it."""

    BINARY = "binary"
    CONTINUOUS = "continuous"
    ORDINAL = "ordinal"
    NOMINAL = "nominal"


class MixedCI:
    """Conditional independence test for mixed continuous and categorical data.

    X and Y are tested in both directions: each is regressed on Z, with and
    without the other as an additional predictor, and the two likelihood-ratio
    tests are combined into one p-value. The regression model depends on the
    type of the regressed variable: linear (continuous), logistic (binary),
    multinomial (nominal) or generalized ordered logit (ordinal).

    Variable types are read from the column dtype: a ``category`` column is
    ordinal if ordered and nominal otherwise, even with only two levels; any
    other column with exactly two distinct values is binary; any other
    column, including integer-coded categories, is continuous. Predictor
    columns of dtype ``category`` or ``object`` are expanded into dummy
    variables.

    ``category`` columns must not have unused categories: drop them beforehand
    with ``Series.cat.remove_unused_categories()``, otherwise the test raises
    a ``ValueError``.
    """

    def _check_no_unused_categories(self, series: pd.Series, name: str) -> None:
        """Raise a ``ValueError`` if a ``category`` column has categories with no observations."""
        if isinstance(series.dtype, pd.CategoricalDtype) and series.nunique() < len(
            series.cat.categories
        ):
            raise ValueError(
                f"Column '{name}' has categories with no observations; drop them first with "
                "Series.cat.remove_unused_categories()."
            )

    # TODO: Think about how to manage Binary variables.
    def _variable_type(self, series: pd.Series) -> VariableType:
        """Return the `VariableType` of a column, following the dtype rules of the class."""
        if isinstance(series.dtype, pd.CategoricalDtype):
            return VariableType.ORDINAL if series.dtype.ordered else VariableType.NOMINAL
        elif series.nunique() == 2:
            return VariableType.BINARY
        else:
            return VariableType.CONTINUOUS

    def _f_test_log_pvalue(
        self, mod0: RegressionResultsWrapper, mod1: RegressionResultsWrapper
    ) -> tuple[float, float, int]:
        """Return the F test of two nested OLS models as ``(log p-value, statistic, dof)``."""
        f_stat, _, dof = mod1.compare_f_test(mod0)
        log_p = float(stats.f.logsf(f_stat, dof, mod1.df_resid))
        return log_p, float(f_stat), int(dof)

    def _chi2_log_pvalue(
        self, mod0: RegressionResultsWrapper, mod1: RegressionResultsWrapper
    ) -> tuple[float, float, int]:
        """Return the likelihood-ratio test of two nested models as
        ``(log p-value, statistic, dof)``, with ``stat = 2 * (llf1 - llf0)``.
        """
        stat = float(2 * (mod1.llf - mod0.llf))
        dof = int(mod1.df_model - mod0.df_model)
        log_p = float(stats.chi2.logsf(stat, dof))
        return log_p, stat, dof

    def continuous_test(
        self,
        outcome: pd.Series,
        design0: pd.DataFrame | None,
        design1: pd.DataFrame,
    ) -> tuple[float, float, int]:
        """F test of nested OLS models for a continuous outcome.

        Parameters
        ----------
        outcome : Series
            Regressed variable.

        design0 : DataFrame or None
            Predictors of the reduced model, besides the intercept. If None, the
            reduced model has only the intercept.

        design1 : DataFrame
            Predictors of the full model, besides the intercept.

        Returns
        -------
        log_p_value : float
            Natural logarithm of the p-value.

        statistic : float
            F statistic of the comparison.

        dof : int
            Degrees of freedom of the test.
        """
        n = outcome.shape[0]
        exog0 = (
            np.ones((n, 1)) if design0 is None else sm.add_constant(design0, has_constant="skip")
        )
        exog1 = sm.add_constant(design1, has_constant="skip")

        mod0: RegressionResultsWrapper = sm.OLS(outcome, exog0).fit()
        mod1: RegressionResultsWrapper = sm.OLS(outcome, exog1).fit()

        return self._f_test_log_pvalue(mod0, mod1)

    def binary_test(
        self,
        outcome: pd.Series,
        design0: pd.DataFrame | None,
        design1: pd.DataFrame,
    ) -> tuple[float, float, int]:
        """Likelihood-ratio test of nested logistic models for a binary outcome.

        Parameters
        ----------
        outcome : Series
            Regressed variable.

        design0 : DataFrame or None
            Predictors of the reduced model, besides the intercept. If None, the
            reduced model has only the intercept.

        design1 : DataFrame
            Predictors of the full model, besides the intercept.

        Returns
        -------
        log_p_value : float
            Natural logarithm of the p-value.

        statistic : float
            Likelihood-ratio statistic.

        dof : int
            Degrees of freedom of the test.
        """
        n = outcome.shape[0]
        exog0 = (
            np.ones((n, 1)) if design0 is None else sm.add_constant(design0, has_constant="skip")
        )
        exog1 = sm.add_constant(design1, has_constant="skip")

        mod0: BinaryResultsWrapper = _fit_discrete(sm.Logit(outcome, exog0))
        mod1: BinaryResultsWrapper = _fit_discrete(sm.Logit(outcome, exog1))

        return self._chi2_log_pvalue(mod0, mod1)

    def nominal_test(
        self,
        outcome: pd.Series,
        design0: pd.DataFrame | None,
        design1: pd.DataFrame,
    ) -> tuple[float, float, int]:
        """Likelihood-ratio test of nested multinomial logistic models for a nominal outcome.

        Parameters
        ----------
        outcome : Series
            Regressed variable.

        design0 : DataFrame or None
            Predictors of the reduced model, besides the intercept. If None, the
            reduced model has only the intercept.

        design1 : DataFrame
            Predictors of the full model, besides the intercept.

        Returns
        -------
        log_p_value : float
            Natural logarithm of the p-value.

        statistic : float
            Likelihood-ratio statistic.

        dof : int
            Degrees of freedom of the test.
        """
        n = outcome.shape[0]
        exog0 = (
            np.ones((n, 1)) if design0 is None else sm.add_constant(design0, has_constant="skip")
        )
        exog1 = sm.add_constant(design1, has_constant="skip")

        mod0: MultinomialResultsWrapper = _fit_discrete(sm.MNLogit(outcome, exog0))
        mod1: MultinomialResultsWrapper = _fit_discrete(sm.MNLogit(outcome, exog1))

        return self._chi2_log_pvalue(mod0, mod1)

    def ordinal_test(
        self,
        outcome: pd.Series,
        design0: pd.DataFrame | None,
        design1: pd.DataFrame,
    ) -> tuple[float, float, int]:
        """Likelihood-ratio test of nested generalized ordered logit models for an ordinal outcome.

        Parameters
        ----------
        outcome : Series
            Regressed variable.

        design0 : DataFrame or None
            Predictors of the reduced model, besides the intercept. If None, the
            reduced model has only the intercept.

        design1 : DataFrame
            Predictors of the full model, besides the intercept.

        Returns
        -------
        log_p_value : float
            Natural logarithm of the p-value.

        statistic : float
            Difference of deviances between the two models.

        dof : int
            Degrees of freedom of the test.
        """
        n = outcome.shape[0]
        exog0 = (
            np.ones((n, 1)) if design0 is None else sm.add_constant(design0, has_constant="skip")
        )
        exog1 = sm.add_constant(design1, has_constant="skip")

        # One binary logistic model per threshold (y >= j), each with its own
        # coefficients: no proportional-odds assumption. Category probabilities
        # come from differences of the cumulative probabilities; rows with a
        # negative probability are replaced by their absolute values, rescaled
        # to sum to one.
        codes = outcome.cat.codes.to_numpy()
        k = len(outcome.cat.categories)
        n_thresholds = k - 1

        def deviance(exog: np.ndarray | pd.DataFrame) -> float:
            cump = np.empty((n, n_thresholds))
            for j in range(1, n_thresholds + 1):
                y_bin = (codes >= j).astype(int)
                mod: BinaryResultsWrapper = _fit_discrete(sm.Logit(y_bin, exog))
                cump[:, j - 1] = 1.0 - mod.predict(exog)

            p = np.empty((n, k))
            p[:, 0] = cump[:, 0]
            p[:, 1:-1] = np.diff(cump, axis=1)
            p[:, -1] = 1.0 - cump[:, -1]

            negative_rows = np.any(p < 0, axis=1)
            if negative_rows.any():
                abs_p = np.abs(p[negative_rows])
                p[negative_rows] = abs_p / abs_p.sum(axis=1, keepdims=True)

            chosen = p[np.arange(n), codes]
            return float(-2 * np.log(chosen).sum())

        devi0 = deviance(exog0)
        devi1 = deviance(exog1)

        stat = devi0 - devi1
        dof = n_thresholds * (exog1.shape[1] - exog0.shape[1])
        return float(stats.chi2.logsf(stat, dof)), stat, dof

    def _as_numeric_design(self, data: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Return the predictors as a numeric frame, expanding categorical columns into
        dummy variables and dropping the first level of each.
        """
        frame = data.to_frame() if isinstance(data, pd.Series) else data
        return pd.get_dummies(frame, drop_first=True, dtype=float)

    def _get_log_pvalue(
        self,
        outcome: pd.Series,
        added: pd.Series,
        z: pd.DataFrame | None,
    ) -> tuple[float, float, int]:
        """Test whether ``added`` improves the prediction of ``outcome`` beyond ``z``.

        Fits the reduced model (``z`` only) and the full model (``z`` and ``added``) with the
        regression that matches the type of ``outcome``, and returns the likelihood-ratio
        test as ``(log p-value, statistic, dof)``.
        """
        outcome_type: VariableType = self._variable_type(outcome)
        design0 = None if z is None else self._as_numeric_design(z)
        added_design = self._as_numeric_design(added)
        design1: pd.DataFrame = (
            added_design if design0 is None else pd.concat([design0, added_design], axis=1)
        )

        test_by_type = {
            VariableType.CONTINUOUS: self.continuous_test,
            VariableType.BINARY: self.binary_test,
            VariableType.NOMINAL: self.nominal_test,
            VariableType.ORDINAL: self.ordinal_test,
        }

        return test_by_type[outcome_type](outcome, design0, design1)

    def _combine_mm_log(self, log_p1: float, log_p2: float) -> float:
        """Combine two dependent p-values with the MM rule (Eq. 1) in log scale.

        Returns ``log(min(2 * min(p1, p2), max(p1, p2)))`` from ``log p1`` and ``log p2``.
        """
        return min(math.log(2) + min(log_p1, log_p2), max(log_p1, log_p2))

    def test(
        self, data: pd.DataFrame, x: str, y: str, z: list[str] | None = None, alpha: float = 0.05
    ) -> CIResult:
        """Test whether X and Y are conditionally independent given Z.

        Parameters
        ----------
        data : DataFrame
            Observations in rows, variables in columns. Categorical columns need dtype
            ``category`` (ordered for ordinal variables) to be treated as such, and must
            not have unused categories.

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
            ``p_value`` is the MM combination of the two directions. ``statistic`` is the
            statistic of the direction with the smaller p-value.

        Raises
        ------
        ValueError
            If a ``category`` column among X, Y and Z has unused categories.
        """
        data_x: pd.Series = data[x]
        data_y: pd.Series = data[y]
        data_z: pd.DataFrame | None = data[z] if z else None
        for name in [x, y, *(z or [])]:
            self._check_no_unused_categories(data[name], name)

        log_p1, stat1, _ = self._get_log_pvalue(outcome=data_y, added=data_x, z=data_z)
        log_p2, stat2, _ = self._get_log_pvalue(outcome=data_x, added=data_y, z=data_z)
        p_value = np.exp(self._combine_mm_log(log_p1, log_p2))
        return CIResult(
            statistic=stat1 if log_p1 < log_p2 else stat2, p_value=p_value, reject=(p_value < alpha)
        )


@ci_test
def mixed_ci_test(
    data: pd.DataFrame,
    x: str,
    y: str,
    z: list[str] | None = None,
    alpha: float = 0.05,
) -> CIResult:
    """Test conditional independence of two columns given others with the mixed-data test.

    Wrapper that runs ``MixedCI.test``.

    Parameters
    ----------
    data : DataFrame
        Observations in rows, variables in columns. Categorical columns need dtype
        ``category`` (ordered for ordinal variables) to be treated as such, and must
        not have unused categories.

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
        ``p_value`` is the MM combination of the two directions. ``statistic`` is the
        statistic of the direction with the smaller p-value.

    Raises
    ------
    ValueError
        If a ``category`` column among X, Y and Z has unused categories.
    """
    return MixedCI().test(data, x, y, z, alpha)
