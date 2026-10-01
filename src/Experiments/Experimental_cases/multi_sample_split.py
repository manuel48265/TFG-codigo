import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from statsmodels.stats.multitest import multipletests

from src.CI_tests.common import CIResult
from src.Experiments.Experimental_cases.dagitty_bridge import Independence


def multi_split_p_values(
    data: pd.DataFrame,
    statements: list[Independence],
    test: Callable[..., CIResult],
    rng: np.random.Generator,
    n_splits: int = 50,
    screen_alpha: float = 0.05,
    gamma_min: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Adjusted p-value of every statement with multi sample splitting.

    Follows Meinshausen, Meier and Bühlmann (2009), "P-values for high-dimensional
    regression", JASA 104(488), where the screening is the selection of statements. In each
    split the data are divided into two random halves. The statements with a p-value of at
    most ``screen_alpha`` on the first half are selected and tested on the second half, with
    Holm's correction over the selected ones; the others get 1. The p-values of the splits
    are aggregated with the quantile rule ``min(1, (1 - log(gamma_min)) * inf_gamma
    q_gamma(p / gamma))``, so the result controls the family-wise error rate over all the
    statements. The statements that are rejected on a small selected set pay a smaller
    correction than with Holm over all of them.

    Parameters
    ----------
    data : DataFrame
        Observations in rows, in the format the test accepts.

    statements : list of Independence
        Statements ``x ⫫ y | z`` to test.

    test : callable
        CI test with the common signature ``test(data, x, y, z)``.

    rng : Generator
        Source of the random splits.

    n_splits : int, optional (default: 50)
        Number of random splits ``B``.

    screen_alpha : float, optional (default: 0.05)
        Level at which a statement is selected on the first half.

    gamma_min : float, optional (default: 0.05)
        Smallest quantile level of the aggregation.

    Returns
    -------
    p_values : ndarray of shape (n_statements,)
        Aggregated adjusted p-values.

    selected_fraction : ndarray of shape (n_statements,)
        Fraction of the splits in which each statement was selected.
    """
    n = len(data)
    adjusted = np.ones((n_splits, len(statements)))
    selected = np.zeros((n_splits, len(statements)), dtype=bool)
    for b in range(n_splits):
        order = rng.permutation(n)
        first, second = data.iloc[order[: n // 2]], data.iloc[order[n // 2 :]]
        screened = [
            j
            for j, s in enumerate(statements)
            if test(first, s.x, s.y, list(s.z)).p_value <= screen_alpha
        ]
        if screened:
            p = [
                test(second, statements[j].x, statements[j].y, list(statements[j].z)).p_value
                for j in screened
            ]
            selected[b, screened] = True
            adjusted[b, screened] = multipletests(p, method="holm")[1]
    ks = np.arange(math.ceil(gamma_min * n_splits), n_splits + 1)
    sorted_p = np.sort(adjusted, axis=0)
    quantile_rule = np.minimum(1.0, sorted_p[ks - 1] / (ks / n_splits)[:, None]).min(axis=0)
    p_values = np.minimum(1.0, (1 - math.log(gamma_min)) * quantile_rule)
    return p_values, selected.mean(axis=0)


def selection_frequency(
    data: pd.DataFrame,
    statements: list[Independence],
    test: Callable[..., CIResult],
    rng: np.random.Generator,
    n_splits: int = 50,
    alpha: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Fraction of random half samples in which each statement is rejected.

    It is the screening step of ``multi_split_p_values`` on its own: in each of the
    ``n_splits`` splits the data are divided into two random halves and every statement is
    tested on the first one, without any correction. A statement rejected in most halves has
    a strong signal that does not depend on a few observations. The fraction describes how
    robust a violation is within the sample; it is not a p-value and it is not a replication.

    Parameters
    ----------
    data : DataFrame
        Observations in rows, in the format the test accepts.

    statements : list of Independence
        Statements ``x ⫫ y | z`` to test.

    test : callable
        CI test with the common signature ``test(data, x, y, z)``. A statement whose test
        raises an error counts as not rejected in that half, and the failure is counted.

    rng : Generator
        Source of the random splits. Two generators created with the same seed give the same
        splits, so several tests can be compared on identical halves.

    n_splits : int, optional (default: 50)
        Number of random splits.

    alpha : float, optional (default: 0.05)
        Level at which a statement is rejected in a half.

    Returns
    -------
    frequency : ndarray of shape (n_statements,)
        Fraction of the splits in which each statement is rejected.

    n_failed : ndarray of int, shape (n_statements,)
        Number of splits in which the test raised an error for each statement. Those splits
        count as not rejected, so failures can only lower the frequency.
    """
    n = len(data)
    rejected = np.zeros(len(statements))
    failed = np.zeros(len(statements), dtype=int)
    for _ in range(n_splits):
        half = data.iloc[rng.permutation(n)[: n // 2]]
        for j, s in enumerate(statements):
            try:
                rejected[j] += test(half, s.x, s.y, list(s.z)).p_value <= alpha
            except Exception:
                failed[j] += 1
    return rejected / n_splits, failed
