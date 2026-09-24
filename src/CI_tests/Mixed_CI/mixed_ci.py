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

from src.CI_tests.common import CIResult


class VariableType(StrEnum):
    BINARY = "binary"
    CONTINUOUS = "continuous"
    ORDINAL = "ordinal"
    NOMINAL = "nominal"


class MixedCI:
    def __init__(self) -> None:
        return None

    def _variable_type(self, series: pd.Series) -> VariableType:
        if series.nunique() == 2:
            return VariableType.BINARY
        elif isinstance(series.dtype, pd.CategoricalDtype):
            return VariableType.ORDINAL if series.dtype.ordered else VariableType.NOMINAL
        return VariableType.CONTINUOUS

    def _f_test_log_pvalue(
        self, mod0: RegressionResultsWrapper, mod1: RegressionResultsWrapper
    ) -> tuple[float, float, int]:
        f_stat, _, dof = mod1.compare_f_test(mod0)
        log_p = float(stats.f.logsf(f_stat, dof, mod1.df_resid))
        return log_p, float(f_stat), int(dof)

    def _chi2_log_pvalue(
        self, mod0: RegressionResultsWrapper, mod1: RegressionResultsWrapper
    ) -> tuple[float, float, int]:

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
        n = outcome.shape[0]
        exog0 = (
            np.ones((n, 1)) if design0 is None else sm.add_constant(design0, has_constant="skip")
        )
        exog1 = sm.add_constant(design1, has_constant="skip")

        mod0: BinaryResultsWrapper = sm.Logit(outcome, exog0).fit(disp=0)
        mod1: BinaryResultsWrapper = sm.Logit(outcome, exog1).fit(disp=0)

        return self._chi2_log_pvalue(mod0, mod1)

    def nominal_test(
        self,
        outcome: pd.Series,
        design0: pd.DataFrame | None,
        design1: pd.DataFrame,
    ) -> tuple[float, float, int]:
        n = outcome.shape[0]
        exog0 = (
            np.ones((n, 1)) if design0 is None else sm.add_constant(design0, has_constant="skip")
        )
        exog1 = sm.add_constant(design1, has_constant="skip")

        mod0: MultinomialResultsWrapper = sm.MNLogit(outcome, exog0).fit(disp=0)
        mod1: MultinomialResultsWrapper = sm.MNLogit(outcome, exog1).fit(disp=0)

        return self._chi2_log_pvalue(mod0, mod1)

    def ordinal_test(
        self,
        outcome: pd.Series,
        design0: pd.DataFrame | None,
        design1: pd.DataFrame,
    ) -> tuple[float, float, int]:
        n = outcome.shape[0]
        exog0 = (
            np.ones((n, 1)) if design0 is None else sm.add_constant(design0, has_constant="skip")
        )
        exog1 = sm.add_constant(design1, has_constant="skip")

        codes = outcome.cat.codes.to_numpy()
        k = len(outcome.cat.categories)
        n_thresholds = k - 1

        def deviance(exog: np.ndarray | pd.DataFrame) -> float:
            cump = np.empty((n, n_thresholds))
            for j in range(1, n_thresholds + 1):
                y_bin = (codes >= j).astype(int)
                mod: BinaryResultsWrapper = sm.Logit(y_bin, exog).fit(disp=0)
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
        frame = data.to_frame() if isinstance(data, pd.Series) else data
        return pd.get_dummies(frame, drop_first=True, dtype=float)

    def _get_log_pvalue(
        self,
        outcome: pd.Series,
        added: pd.Series,
        z: pd.DataFrame | None,
    ) -> tuple[float, float, int]:
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

    # Equation 1 (in log scale as ci.mm.R)
    def _combine_mm_log(self, log_p1: float, log_p2: float) -> float:
        return min(math.log(2) + min(log_p1, log_p2), max(log_p1, log_p2))

    def mixed_ci_test(
        self, data: pd.DataFrame, x: str, y: str, z: list[str] | None = None, alpha: float = 0.05
    ) -> CIResult:
        data_x: pd.Series = data[x]
        data_y: pd.Series = data[y]
        data_z: pd.DataFrame | None = data[z] if z is not None else None

        log_p1, stat1, _ = self._get_log_pvalue(outcome=data_y, added=data_x, z=data_z)
        log_p2, stat2, _ = self._get_log_pvalue(outcome=data_x, added=data_y, z=data_z)
        p_value = np.exp(self._combine_mm_log(log_p1, log_p2))
        return CIResult(
            statistic=stat1 if log_p1 < log_p2 else stat2, p_value=p_value, reject=(p_value < alpha)
        )
