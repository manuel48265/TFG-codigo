# Algorithm independently reimplemented from Shah & Peters (2020); its
# structure was checked against the official R package
# github.com/cran/GeneralisedCovarianceMeasure (Peters & Shah, GPL-2),
# vendored for reference (not redistributed) at Referencias/GCM_R_source/.

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Regressor(Protocol):
    """Any regressor with scikit-learn style ``fit`` and ``predict`` methods."""

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "Regressor": ...
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...


def residuals_of(
    v: NDArray[np.float64],
    z: NDArray[np.float64],
    estimator: Regressor,
) -> NDArray[np.float64]:
    """Regress a variable on Z and return the residuals ``v - E[v | z]``.

    Parameters
    ----------
    v : ndarray of shape (n,)
        Variable to regress.

    z : ndarray of shape (n, d_z)
        Sample of the conditioning set.

    estimator : Regressor
        Regressor fitted on ``(z, v)``. It is fitted in place.

    Returns
    -------
    residuals : ndarray of shape (n,)
        Residuals of the regression.
    """
    estimator.fit(z, v)
    return v - estimator.predict(z)


def compute_residuals(
    v: NDArray[np.float64],
    z: NDArray[np.float64],
    estimator: Regressor,
) -> NDArray[np.float64]:
    """Regress each column of a variable on Z and return the residuals.

    Parameters
    ----------
    v : ndarray of shape (n,) or (n, d_v)
        Variable to regress. A matrix is regressed column by column.

    z : ndarray of shape (n, d_z)
        Sample of the conditioning set.

    estimator : Regressor
        Regressor fitted once per column of ``v``.

    Returns
    -------
    residuals : ndarray of shape (n,) or (n, d_v)
        Residuals, with the same shape as ``v``.
    """
    if v.ndim > 1:
        return np.column_stack([residuals_of(v[:, j], z, estimator) for j in range(v.shape[1])])
    return residuals_of(v, z, estimator)
