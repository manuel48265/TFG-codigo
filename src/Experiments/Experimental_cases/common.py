import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.CI_tests.common import CIResult
from src.Experiments.DAG_Generation.fundamental_topology_generator import FundamentalCase


def discretize_for_chi2(x: NDArray[np.float64], max_bins: int = 5) -> NDArray[np.intp]:
    return pd.qcut(x, q=max_bins, labels=False, duplicates="drop").astype(np.intp)


def discretize_continuous_columns(data: pd.DataFrame, max_bins: int = 5) -> pd.DataFrame:
    """Return a copy of ``data`` with its continuous columns replaced by quantile-bin codes.

    A column is continuous if its dtype is floating point; every other column
    is left unchanged. Each continuous column is split by ``discretize_for_chi2``
    into at most ``max_bins`` equiprobable bins, coded ``0, 1, ...``.

    Parameters
    ----------
    data : DataFrame
        Observations in rows, variables in columns.

    max_bins : int, optional (default: 5)
        Maximum number of bins per continuous column.

    Returns
    -------
    discretized : DataFrame
        Same columns and index as ``data``.
    """
    return data.apply(
        lambda column: (
            pd.Series(discretize_for_chi2(column.to_numpy(), max_bins), index=column.index)
            if pd.api.types.is_float_dtype(column)
            else column
        )
    )


def to_float_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``data`` with every column as ``float64``.

    Non-numeric columns (``category``, ``object``, strings) are replaced by
    their integer codes, numbered by the sorted order of the categories; the
    numeric ones are cast.

    Parameters
    ----------
    data : DataFrame
        Observations in rows, variables in columns.

    Returns
    -------
    floats : DataFrame
        Same columns and index as ``data``.
    """
    return data.apply(
        lambda column: (
            column.astype(np.float64)
            if pd.api.types.is_numeric_dtype(column)
            else column.astype("category").cat.codes.astype(np.float64)
        )
    )


def classify_structure(marginal_reject: bool, conditional_reject: bool) -> FundamentalCase:
    if not marginal_reject and not conditional_reject:
        return FundamentalCase.INDEPENDENCE
    if marginal_reject and conditional_reject:
        return FundamentalCase.DIRECT_DEPENDENCE
    if not marginal_reject and conditional_reject:
        return FundamentalCase.COLLIDER
    return FundamentalCase.CONFOUNDER


@dataclass
class TrialResult:
    case: str
    z_size: int
    n_samples: int
    rule: str
    type_mix: str
    ci_test: str
    rep: int
    seed: int
    marginal_statistic: float | None
    marginal_p_value: float
    marginal_reject: bool
    conditional_statistic: float | None
    conditional_p_value: float
    conditional_reject: bool
    predicted_case: str
    correct: bool


def build_trial_result(
    case: FundamentalCase,
    z_size: int,
    n_samples: int,
    rule: str,
    type_mix: str,
    ci_test: str,
    rep: int,
    seed: int,
    marginal: CIResult,
    conditional: CIResult,
) -> TrialResult:
    predicted = classify_structure(marginal.reject, conditional.reject)
    return TrialResult(
        case=case.value,
        z_size=z_size,
        n_samples=n_samples,
        rule=rule,
        type_mix=type_mix,
        ci_test=ci_test,
        rep=rep,
        seed=seed,
        marginal_statistic=marginal.statistic,
        marginal_p_value=marginal.p_value,
        marginal_reject=marginal.reject,
        conditional_statistic=conditional.statistic,
        conditional_p_value=conditional.p_value,
        conditional_reject=conditional.reject,
        predicted_case=predicted.value,
        correct=predicted == case,
    )


def append_trial_result(path: Path, result: TrialResult) -> None:
    row = asdict(result)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
