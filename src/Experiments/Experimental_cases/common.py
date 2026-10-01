import csv
import io
import os
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


def remove_unused_categories(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``data`` without the unobserved categories of its ``category`` columns.

    Mixed-CI raises an error for a ``category`` column with categories that have no
    observations, so the data are cleaned with this function before running it. Columns
    that are not categorical are left unchanged.

    Parameters
    ----------
    data : DataFrame
        Observations in rows, variables in columns.

    Returns
    -------
    cleaned : DataFrame
        Same columns and index as ``data``.
    """
    return data.apply(
        lambda column: (
            column.cat.remove_unused_categories()
            if isinstance(column.dtype, pd.CategoricalDtype)
            else column
        )
    )


def has_constant_column(data: pd.DataFrame) -> bool:
    """Return whether some column of ``data`` has a single distinct value.

    A constant variable carries no information, and a CI test on it is degenerate, so a
    dataset with one is discarded and generated again with another seed.

    Parameters
    ----------
    data : DataFrame
        Observations in rows, variables in columns.

    Returns
    -------
    constant : bool
        True if at least one column takes a single value.
    """
    return bool((data.nunique() <= 1).any())


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
    dataset_id: str = ""
    attempts: int = 1
    alpha: float = 0.05
    x_type: str = ""
    y_type: str = ""
    z_types: str = ""
    roles: str = ""
    marginal_seconds: float = float("nan")
    conditional_seconds: float = float("nan")
    n_warnings: int = 0
    error: str = ""
    code_version: str = ""


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


def append_trial_results(path: Path, results: list[TrialResult]) -> None:
    """Append several results to a CSV file with a single write.

    The header is written if the file is new or empty. Writing all the rows at once
    keeps a dataset's rows together if the process is interrupted.

    Parameters
    ----------
    path : Path
        CSV file.

    results : list of TrialResult
        Rows to append; there must be at least one.

    Raises
    ------
    ValueError
        If the file already has a header that differs from the fields of ``TrialResult``.
    """
    rows = [asdict(result) for result in results]
    fieldnames = list(rows[0].keys())
    is_new = not path.exists() or path.stat().st_size == 0
    if not is_new:
        with path.open(newline="") as f:
            header = next(csv.reader(f))
        if header != fieldnames:
            raise ValueError(f"{path} has a different header than TrialResult; use another file.")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    if is_new:
        writer.writeheader()
    writer.writerows(rows)
    with path.open("a", newline="") as f:
        f.write(buffer.getvalue())
        f.flush()
        os.fsync(f.fileno())


def append_trial_result(path: Path, result: TrialResult) -> None:
    """Append one result to a CSV file (see ``append_trial_results``)."""
    append_trial_results(path, [result])
