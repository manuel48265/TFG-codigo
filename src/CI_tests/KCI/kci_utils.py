# Kernel centering: same O(n^2) identity as causal-learn's
# Kernel.center_kernel_matrix and pywhy-stats' _fast_centering.
# Z-regularization: Zhang, Peters, Janzing, Schölkopf (2011), Eq. 10;
# numerical floor and fallback from Marchildon & Zingg (2023, Eq. 25-26)
# and Foster et al. (2009); condition check via Hager (1984)/Higham (1988).

import warnings

import numpy as np
from numpy.linalg import pinv
from numpy.typing import NDArray
from scipy import stats
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.linalg.lapack import dpocon
from scipy.spatial.distance import cdist, pdist, squareform

# Zhang et al. (2011), Sec. 3.5: eps = 1e-3 works well when dim(Z) is small.
DEFAULT_EPSILON = 1e-3

# Marchildon & Zingg (2023) treat kappa_max as a user-set threshold, not a
# fixed constant; 1e8 is a conservative default for float64.
DEFAULT_KAPPA_MAX = 1e8


def center_kernel_matrix(K: NDArray[np.float64]) -> NDArray[np.float64]:
    """Center a kernel matrix as ``H @ K @ H`` with ``H = I - ones(n, n) / n``.

    The product is expanded into row, column and total sums, which costs
    O(n^2) instead of the O(n^3) of building ``H``.

    Parameters
    ----------
    K : ndarray of shape (n, n)
        Kernel matrix to center.

    Returns
    -------
    centered : ndarray of shape (n, n)
        Centered kernel matrix.
    """
    n = K.shape[0]
    K_colsums = K.sum(axis=0)
    K_allsum = K_colsums.sum()
    return K - (K_colsums[None, :] + K_colsums[:, None]) / n + (K_allsum / n**2)


def center_kernel_matrix_regression(
    K: NDArray[np.float64],
    Kz: NDArray[np.float64],
    epsilon: float = DEFAULT_EPSILON,
    kappa_max: float = DEFAULT_KAPPA_MAX,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Residualize a kernel matrix against Z as ``R_z @ K @ R_z``.

    ``R_z = eps * (Kz + eps * I)^-1`` (Zhang et al. 2011, Eq. 10-12), where
    ``eps`` is the larger of ``epsilon`` and ``n / (kappa_max - 1)``, which
    keeps ``cond(Kz + eps * I) <= kappa_max`` (Marchildon & Zingg 2023,
    Eq. 25-26). The system is solved by Cholesky factorization; if it fails or
    the estimated condition number exceeds ``kappa_max``, a pseudo-inverse is
    used instead and a warning is issued.

    Parameters
    ----------
    K : ndarray of shape (n, n)
        Kernel matrix to residualize.

    Kz : ndarray of shape (n, n)
        Centered kernel matrix of Z.

    epsilon : float, optional (default: 1e-3)
        Regularization parameter, before applying the numerical floor.

    kappa_max : float, optional (default: 1e8)
        Maximum tolerated condition number of ``Kz + eps * I``.

    Returns
    -------
    residualized : ndarray of shape (n, n)
        ``R_z @ K @ R_z``.

    Rz : ndarray of shape (n, n)
        The matrix ``R_z``, so it can be reused to residualize another matrix
        against the same Z.
    """
    n = Kz.shape[0]
    eps_numerical = n / (kappa_max - 1)
    eps = max(epsilon, eps_numerical)

    regularized = 0.5 * (Kz + Kz.T) + eps * np.eye(n)
    try:
        cho, lower = cho_factor(regularized)
        anorm = np.linalg.norm(regularized, 1)
        rcond, info = dpocon(cho, anorm, uplo="L" if lower else "U")
        if info != 0 or rcond < 1.0 / kappa_max:
            raise LinAlgError(f"Estimated condition number too high: rcond={rcond:.2e}")
        inv_regularized = cho_solve((cho, lower), np.eye(n))
    except LinAlgError as err:
        warnings.warn(f"Falling back to pinv for Z-regularization: {err}", stacklevel=2)
        inv_regularized = pinv(regularized)

    Rz = eps * inv_regularized
    return Rz.dot(K).dot(Rz), Rz


class GaussianKernel:
    """Gaussian kernel ``k(u, v) = exp(-0.5 * width * ||u - v||^2)``.

    Attributes
    ----------
    width : float
        Inverse squared bandwidth of the kernel (default: 1.0).
    """

    def __init__(self) -> None:
        self.width: float = 1.0

    def kernel(
        self, X: NDArray[np.float64], Y: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Return the kernel matrix of X with itself, or between X and Y.

        Parameters
        ----------
        X : ndarray of shape (n, d)
            First sample.

        Y : ndarray of shape (m, d) or None, optional (default: None)
            Second sample. If None, X is compared against itself.

        Returns
        -------
        kernel : ndarray of shape (n, n) or (n, m)
            Kernel matrix.
        """
        sq_dists = squareform(pdist(X, "sqeuclidean")) if Y is None else cdist(X, Y, "sqeuclidean")
        return np.exp(-0.5 * self.width * sq_dists)

    def set_width_empirical(self, n: int, dim_z: int) -> None:
        """Set ``width`` from the sample size and the dimension of Z.

        The base value is 1.2 for ``n < 200``, 0.7 for ``n < 1200`` and 0.4
        otherwise, and the width is ``1 / (base**2 * dim_z)``.

        Parameters
        ----------
        n : int
            Sample size.

        dim_z : int
            Number of conditioning columns, at least 1.
        """
        if n < 200:
            width = 1.2
        elif n < 1200:
            width = 0.7
        else:
            width = 0.4
        self.width = 1.0 / (width**2 * dim_z)

    def set_width_empirical_marginal(self, n: int, dim: int) -> None:
        """Set ``width`` for the marginal test from the sample size and the variable's dimension.

        The base value is 0.8 for ``n < 200``, 0.5 for ``n < 1200`` and 0.3
        otherwise (Zhang et al. 2011, Sec. 3.5), and the width is
        ``dim / base**2``.

        Parameters
        ----------
        n : int
            Sample size.

        dim : int
            Number of columns of the variable the kernel is built on.
        """
        if n < 200:
            width = 0.8
        elif n < 1200:
            width = 0.5
        else:
            width = 0.3
        self.width = dim / width**2


def normalize_unit_variance(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scale each column to zero mean and unit variance (Zhang et al. 2011, Sec. 3.5).

    Constant columns, whose z-score is undefined, are set to 0.

    Parameters
    ----------
    data : ndarray of shape (n, d)
        Data to normalize.

    Returns
    -------
    normalized : ndarray of shape (n, d)
        Normalized data.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        normalized = stats.zscore(data, ddof=1, axis=0)
    normalized[np.isnan(normalized)] = 0.0
    return normalized
