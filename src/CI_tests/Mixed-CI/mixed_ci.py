"""
# Una dirección del test: ¿aporta `added` al modelo de `outcome`, dado `z`? (§2.3.1/§2.3.2)
def _direction_log_pvalue(outcome, added, z) -> tuple[log_p, stat, dof]:
    design0 = z if z is not None else intercept_only     # M0: outcome ~ z
    design1 = column_stack(z, added)                      # M1: outcome ~ z + added
    outcome_type = _variable_type(outcome)

    if outcome_type == "binary":
        mod0, mod1 = fit_logistic(outcome, design0), fit_logistic(outcome, design1)
    elif outcome_type == "nominal":
        mod0, mod1 = fit_multinomial_logistic(outcome, design0), fit_multinomial_logistic(outcome,
        design1)
    elif outcome_type == "ordinal":
        # proportional odds
        mod0, mod1 = fit_ordinal_logistic(outcome, design0), fit_ordinal_logistic(outcome, design1)
    else:  # continuous
        mod0, mod1 = ols(outcome, design0), ols(outcome, design1)

    # F-test si continuous (§2.3.1), LR-test (χ²) si no (§2.3.2)
    stat, dof = likelihood_ratio_or_f_stat(mod0, mod1)
    log_p = log_survival_function(stat, dof, family_of(outcome_type))
    return log_p, stat, dof

# función pública
def mixed_ci_test(x, y, z=None, alpha=0.05) -> CIResult:
    log_p1, stat1, dof1 = _direction_log_pvalue(outcome=y, added=x, z=z)   # H1: Y~Z+X vs Y~Z
    log_p2, stat2, dof2 = _direction_log_pvalue(outcome=x, added=y, z=z)   # H2: X~Z+Y vs X~Z
    p_value = exp(_combine_mm_log(log_p1, log_p2))
    return CIResult(
        statistic=stat1 if log_p1 < log_p2 else stat2, p_value=p_value, reject=(p_value < alpha)
    )
"""

import math
from enum import StrEnum

import pandas as pd

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

    # Equation 1 (in log scale as ci.mm.R)
    def _combine_mm_log(log_p1: float, log_p2: float) -> float:
        return min(math.log(2) + min(log_p1, log_p2), max(log_p1, log_p2))

    def mixed_ci_test(
        self, data: pd.DataFrame, x: str, y: str, z: list[str] | None = None, alpha: float = 0.05
    ) -> CIResult:
        return CIResult(statistic=0, p_value=0, reject=True)
