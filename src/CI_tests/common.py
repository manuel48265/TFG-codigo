from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class CIResult:
    """Common result contract for any CI test (shared across methods).
    `statistic` is None when a reference implementation's public API
    doesn't expose one (e.g. causal-learn's FisherZ only returns a p-value).
    """

    statistic: float | None
    p_value: float
    reject: bool


def n_cols(a: NDArray[np.float64]) -> int:
    """Number of columns, treating a 1D array as a single column.
    Mirrors R's `NCOL()`: a plain (n,) vector counts as 1 column.
    """
    return 1 if a.ndim == 1 else a.shape[1]


def col(a: NDArray[np.float64], j: int) -> NDArray[np.float64]:
    """Get column j, treating a 1D array as its own single column.
    Companion to `n_cols`: lets multivariate code index a variable
    uniformly whether it's a plain vector (j is ignored) or a matrix.
    """
    return a if a.ndim == 1 else a[:, j]


def frame_to_arrays(
    data: pd.DataFrame, x: str, y: str, z: list[str] | None
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Select the tested and conditioning columns of a DataFrame as float arrays.

    Parameters
    ----------
    data : DataFrame
        Observations in rows, numeric variables in columns.

    x : str
        Name of the first tested column.

    y : str
        Name of the second tested column.

    z : list of str or None
        Names of the conditioning columns. If None or empty, ``z_values`` is None.

    Returns
    -------
    x_values : ndarray of shape (n,)
        Sample of X.

    y_values : ndarray of shape (n,)
        Sample of Y.

    z_values : ndarray of shape (n, d_z) or None
        Sample of the conditioning set.

    Raises
    ------
    ValueError
        If a selected column cannot be converted to float.
    """
    z_values = data[z].to_numpy(dtype=np.float64) if z else None
    return data[x].to_numpy(dtype=np.float64), data[y].to_numpy(dtype=np.float64), z_values


class CITest(Protocol):
    """Common signature of a conditional independence test on a DataFrame.

    Any callable with this signature can be plugged into a PC wrapper.
    """

    def __call__(
        self, data: pd.DataFrame, x: str, y: str, z: list[str] | None = None, alpha: float = 0.05
    ) -> CIResult:
        """Test whether column ``x`` is independent of column ``y`` given ``z``.

        Parameters
        ----------
        data : DataFrame
            Observations in rows, variables in columns.

        x : str
            Name of the first tested column.

        y : str
            Name of the second tested column.

        z : list of str or None, optional (default: None)
            Names of the conditioning columns. If None, the test is marginal.

        alpha : float, optional (default: 0.05)
            Significance level; ``reject`` is ``p_value < alpha``.

        Returns
        -------
        result : CIResult
            Statistic, p-value and decision of the test.
        """
        ...


F = TypeVar("F", bound=CITest)


def ci_test(f: F) -> F:
    return f
