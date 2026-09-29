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
