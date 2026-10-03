import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import CIT_Base, register_ci_test
from numpy.typing import NDArray

from src.CI_tests.common import CIResult

PC_METHOD = "project_ci_test"


@dataclass(frozen=True)
class PCRun:
    """Outcome of one run of the PC algorithm.

    Attributes
    ----------
    edges : str
        Edges of the estimated CPDAG, see ``graph_to_text``.

    n_ci_calls : int
        Number of CI tests the algorithm asked for.

    seconds : float
        Wall-clock time of the run.

    cond_size_counts : str
        Histogram of the conditioning set sizes as ``size:count`` pairs separated by ``;``,
        in increasing size. The largest size and the mean follow from it.
    """

    edges: str
    n_ci_calls: int
    seconds: float
    cond_size_counts: str


class _ProjectCITest(CIT_Base):  # type: ignore[misc]
    """Adapter that lets causal-learn call a CI test of this project on a DataFrame."""

    def __init__(self, data: NDArray[np.float64], **kwargs: object) -> None:
        super().__init__(data, **kwargs)
        self.method = PC_METHOD  # causal-learn reads this attribute on every call
        self.frame = cast(pd.DataFrame, kwargs["frame"])
        self.test = cast(Callable[..., CIResult], kwargs["test"])
        self.extra = cast(dict[str, object], kwargs["extra"])
        self.cond_sizes = cast(Counter[int], kwargs["cond_sizes"])

    def __call__(self, X: int, Y: int, condition_set: list[int] | None = None) -> float:
        self.cond_sizes[len(condition_set) if condition_set else 0] += 1
        names = list(self.frame.columns)
        z = [names[i] for i in condition_set] if condition_set else None
        return self.test(self.frame, names[X], names[Y], z, **self.extra).p_value


register_ci_test(PC_METHOD, _ProjectCITest)


def graph_to_text(matrix: NDArray[np.int_], names: list[str]) -> str:
    """Write the edges of a causal-learn graph matrix as text.

    Edges are separated by ``;``: ``a->b`` is directed, ``a--b`` undirected and ``a<->b``
    a conflict between two orientations. A pair without edge does not appear.

    Parameters
    ----------
    matrix : ndarray of shape (d, d)
        Endpoint matrix of causal-learn: ``matrix[j, i] == 1`` and ``matrix[i, j] == -1``
        mean ``i -> j``, both ``-1`` mean ``i -- j``, both ``1`` mean ``i <-> j``.

    names : list of str
        Name of each variable.

    Returns
    -------
    text : str
        The edges, in the order of the variables.
    """
    edges = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            marks = (int(matrix[i, j]), int(matrix[j, i]))
            if marks == (-1, 1):
                edges.append(f"{names[i]}->{names[j]}")
            elif marks == (1, -1):
                edges.append(f"{names[j]}->{names[i]}")
            elif marks == (-1, -1):
                edges.append(f"{names[i]}--{names[j]}")
            elif marks == (1, 1):
                edges.append(f"{names[i]}<->{names[j]}")
    return ";".join(edges)


def run_pc(
    data: pd.DataFrame,
    test: Callable[..., CIResult],
    alpha: float = 0.05,
    **test_kwargs: object,
) -> PCRun:
    """Run PC-stable with a CI test of this project and return the estimated CPDAG.

    The order of the variables does not change the skeleton (stable version), and the
    conflicts between collider orientations are resolved with a fixed rule, so the result
    depends only on the data and on the test.

    Parameters
    ----------
    data : DataFrame
        One column per variable, in the format the test accepts.

    test : callable
        CI test with the common signature ``test(data, x, y, z, **kwargs)``.

    alpha : float, optional (default: 0.05)
        Significance level of every CI test.

    **test_kwargs
        Extra keyword arguments passed to ``test`` in every call.

    Returns
    -------
    run : PCRun
        Edges of the CPDAG, number of CI tests, time and conditioning set sizes.
    """
    cond_sizes: Counter[int] = Counter()
    start = time.perf_counter()
    graph = pc(
        np.zeros(data.shape),  # causal-learn only needs the shape; the test reads the frame
        alpha,
        indep_test=PC_METHOD,
        stable=True,
        uc_rule=0,
        uc_priority=2,
        show_progress=False,
        frame=data,
        test=test,
        extra=test_kwargs,
        cond_sizes=cond_sizes,
    )
    seconds = time.perf_counter() - start
    return PCRun(
        graph_to_text(graph.G.graph, list(data.columns)),
        sum(cond_sizes.values()),
        seconds,
        ";".join(f"{size}:{count}" for size, count in sorted(cond_sizes.items())),
    )
