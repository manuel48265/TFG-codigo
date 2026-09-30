# K. Zhang, J. Peters, D. Janzing, and B. Schölkopf,
# "A kernel-based conditional independence test and application in causal discovery," In UAI 2011.

import numpy as np
import pandas as pd
from numpy.linalg import eigh, eigvalsh
from numpy.typing import NDArray
from scipy import stats

from src.CI_tests.common import CIResult, ci_test
from src.CI_tests.KCI.kci_utils import (
    GaussianKernel,
    center_kernel_matrix,
    center_kernel_matrix_regression,
    normalize_unit_variance,
)


def scaled_eigenvecs(K: NDArray[np.float64], thresh: float) -> NDArray[np.float64]:
    """Return the truncated, eigenvalue-scaled eigenvectors of a kernel matrix.

    Computes ``V @ diag(sqrt(w))`` from the eigendecomposition of the
    symmetrized ``K``, keeping only the eigenvectors whose eigenvalue exceeds
    ``thresh`` times the largest one (Zhang et al. 2011, Sec. 3.3).

    Parameters
    ----------
    K : ndarray of shape (n, n)
        Centered kernel matrix.

    thresh : float
        Relative eigenvalue cutoff.

    Returns
    -------
    scaled_eigenvecs : ndarray of shape (n, k)
        Retained eigenvectors, each scaled by the square root of its eigenvalue.
    """
    w, v = eigh(0.5 * (K + K.T))
    idx = np.argsort(-w)
    w, v = w[idx], v[:, idx]
    mask = w > np.max(w) * thresh
    return v[:, mask].dot(np.diag(np.sqrt(w[mask])))


class KCI:
    """Kernel-based conditional independence test (Zhang et al. 2011).

    Only the Gaussian kernel is supported. The kernel of X is built over X
    concatenated with half of Z, and the kernels of X and Y are residualized
    against the kernel of Z, following the original implementation
    (CInd_test_new_withGP.m). The p-value comes either from a Gamma
    approximation of the null distribution or from Monte Carlo draws of it.

    Parameters
    ----------
    nullss : int, optional (default: 5000)
        Number of Monte Carlo draws of the null distribution. Used only if
        ``approx`` is False.

    approx : bool, optional (default: True)
        If True, approximate the null distribution by a Gamma distribution
        (Prop. 6.ii); if False, sample it (Eq. 14).

    thresh : float, optional (default: 1e-5)
        Eigenvalues below ``thresh`` times the largest one are discarded.

    epsilon_x : float, optional (default: 1e-3)
        Regularization used to residualize the kernel of X against Z.

    epsilon_y : float, optional (default: 1e-3)
        Regularization used to residualize the kernel of Y against Z.

    kwidthx : float or None, optional (default: None)
        Width of the kernel of (X, Z). If None, it is set from the sample size
        and the dimension of Z.

    kwidthy : float or None, optional (default: None)
        Width of the kernel of Y. If None, it is set as for ``kwidthx``.

    kwidthz : float or None, optional (default: None)
        Width of the kernel of Z. If None, it is set as for ``kwidthx``.
    """

    def __init__(
        self,
        nullss: int = 5000,
        approx: bool = True,
        thresh: float = 1e-5,
        epsilon_x: float = 1e-3,
        epsilon_y: float = 1e-3,
        kwidthx: float | None = None,
        kwidthy: float | None = None,
        kwidthz: float | None = None,
    ) -> None:
        self.nullss = nullss
        self.approx = approx
        self.thresh = thresh
        self.epsilon_x = epsilon_x
        self.epsilon_y = epsilon_y
        self.kwidthx = kwidthx
        self.kwidthy = kwidthy
        self.kwidthz = kwidthz

    def kernel_matrix(
        self,
        data_x: NDArray[np.float64],
        data_y: NDArray[np.float64],
        data_z: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Build the centered Gaussian kernel matrices of (X, Z), Y and Z.

        Each sample is first scaled to unit variance. The kernel of X is
        computed over ``[X, Z / 2]`` (Zhang et al. 2011, Sec. 3.3).

        Parameters
        ----------
        data_x : ndarray of shape (n, d_x)
            Sample of X.

        data_y : ndarray of shape (n, d_y)
            Sample of Y.

        data_z : ndarray of shape (n, d_z)
            Sample of Z; ``d_z`` may be 0.

        Returns
        -------
        Kx : ndarray of shape (n, n)
            Kernel matrix of (X, Z).

        Ky : ndarray of shape (n, n)
            Kernel matrix of Y.

        Kzx : ndarray of shape (n, n)
            Kernel matrix of Z, used to residualize ``Kx``.

        Kzy : ndarray of shape (n, n)
            Kernel matrix of Z, used to residualize ``Ky``. It is the same
            object as ``Kzx``.
        """
        data_x = normalize_unit_variance(data_x)
        data_y = normalize_unit_variance(data_y)
        data_z = normalize_unit_variance(data_z)

        n, dim_z = data_z.shape
        x_z = np.column_stack([data_x, 0.5 * data_z])

        kernel_x, kernel_y, kernel_z = GaussianKernel(), GaussianKernel(), GaussianKernel()
        for kernel, manual_width in (
            (kernel_x, self.kwidthx),
            (kernel_y, self.kwidthy),
            (kernel_z, self.kwidthz),
        ):
            if manual_width is not None:
                kernel.width = manual_width
            else:
                kernel.set_width_empirical(n, dim_z)

        Kx = center_kernel_matrix(kernel_x.kernel(x_z))
        Ky = center_kernel_matrix(kernel_y.kernel(data_y))
        Kz = center_kernel_matrix(kernel_z.kernel(data_z))
        return Kx, Ky, Kz, Kz

    def get_eigenvects_prod(
        self, Kx: NDArray[np.float64], Ky: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], int]:
        """Return the matrix whose eigenvalues weight the null distribution.

        Built from the tensor product of the scaled eigenvectors of ``Kx`` and
        ``Ky``: ``w w^T`` or its dual ``w^T w``, whichever is smaller (Zhang et
        al. 2011, Prop. 5).

        Parameters
        ----------
        Kx : ndarray of shape (n, n)
            Residualized kernel matrix of (X, Z).

        Ky : ndarray of shape (n, n)
            Residualized kernel matrix of Y.

        Returns
        -------
        eigenvects_prod : ndarray of shape (m, m)
            ``m`` is ``n`` or the number of tensor components, whichever is smaller.

        n_components : int
            Number of tensor components, the product of the numbers of
            retained eigenvectors of ``Kx`` and ``Ky``.
        """
        scaled_eigenvecs_x = scaled_eigenvecs(Kx, self.thresh)
        scaled_eigenvecs_y = scaled_eigenvecs(Ky, self.thresh)

        n: int = Kx.shape[0]
        n_components: int = scaled_eigenvecs_x.shape[1] * scaled_eigenvecs_y.shape[1]

        tensor_features = (scaled_eigenvecs_x[:, :, None] * scaled_eigenvecs_y[:, None, :]).reshape(
            n, n_components
        )

        eigenvects_prod: NDArray[np.float64] = (
            tensor_features.dot(tensor_features.T)
            if n_components > n
            else tensor_features.T.dot(tensor_features)
        )

        return eigenvects_prod, n_components

    def get_kappa(self, eigenvecs_product: NDArray[np.float64]) -> tuple[np.float64, np.float64]:
        """Return the Gamma parameters that match the mean and variance of the null distribution.

        Moment matching of Zhang et al. (2011, Prop. 6.ii).

        Parameters
        ----------
        eigenvecs_product : ndarray of shape (m, m)
            Matrix returned by ``get_eigenvects_prod``.

        Returns
        -------
        k : float
            Shape parameter of the Gamma distribution.

        theta : float
            Scale parameter of the Gamma distribution.
        """
        approx_mean = np.trace(eigenvecs_product)
        approx_var = 2 * np.trace(eigenvecs_product.dot(eigenvecs_product))
        approx_k = approx_mean**2 / approx_var
        approx_theta = approx_var / approx_mean
        return approx_k, approx_theta

    def null_sample(
        self, eigenvecs_prod: NDArray[np.float64], n_components: int, sample_size: int
    ) -> NDArray[np.float64]:
        """Draw samples from the null distribution (Zhang et al. 2011, Eq. 14).

        Each draw is a weighted sum of independent chi-square(1) variables,
        weighted by the eigenvalues of ``eigenvecs_prod``. At most
        ``min(sample_size, n_components)`` eigenvalues are kept, and only
        those above ``thresh`` times the largest one. Uses the global NumPy
        random state.

        Parameters
        ----------
        eigenvecs_prod : ndarray of shape (m, m)
            Matrix returned by ``get_eigenvects_prod``.

        n_components : int
            Number of tensor components returned by ``get_eigenvects_prod``.

        sample_size : int
            Number of observations ``n``.

        Returns
        -------
        null_distribution : ndarray of shape (nullss,)
            Draws from the null distribution.
        """
        eigen = eigvalsh(eigenvecs_prod)
        eigen = -np.sort(-eigen)
        eigen = eigen[0 : np.min((sample_size, n_components))]
        eigen = eigen[eigen > np.max(eigen) * self.thresh]

        chi_distribution = np.random.chisquare(1, (eigen.shape[0], self.nullss))
        null_distribution = eigen.T.dot(chi_distribution)

        return null_distribution

    def KCI_V_statistic(
        self,
        Kx: NDArray[np.float64],
        Ky: NDArray[np.float64],
        Kzx: NDArray[np.float64],
        Kzy: NDArray[np.float64],
    ) -> tuple[np.float64, NDArray[np.float64], NDArray[np.float64]]:
        """Residualize ``Kx`` and ``Ky`` against Z and return the V-statistic.

        The statistic is ``sum(KxR * KyR)`` (Zhang et al. 2011, Eq. 13).
        If ``Kzx`` and ``Kzy`` are the same matrix and ``epsilon_x ==
        epsilon_y``, the residualizing matrix of ``Kx`` is reused for ``Ky``.

        Parameters
        ----------
        Kx : ndarray of shape (n, n)
            Centered kernel matrix of (X, Z).

        Ky : ndarray of shape (n, n)
            Centered kernel matrix of Y.

        Kzx : ndarray of shape (n, n)
            Centered kernel matrix of Z used to residualize ``Kx``.

        Kzy : ndarray of shape (n, n)
            Centered kernel matrix of Z used to residualize ``Ky``.

        Returns
        -------
        v_statistic : float
            Test statistic.

        KxR : ndarray of shape (n, n)
            Residualized kernel matrix of (X, Z).

        KyR : ndarray of shape (n, n)
            Residualized kernel matrix of Y.
        """
        KxR, Rzx = center_kernel_matrix_regression(Kx, Kzx, self.epsilon_x)
        if self.epsilon_x != self.epsilon_y:
            KyR, _ = center_kernel_matrix_regression(Ky, Kzy, self.epsilon_y)
        else:
            KyR = Rzx.dot(Ky).dot(Rzx)

        Vstat = np.sum(KxR * KyR)
        return Vstat, KxR, KyR

    def test(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64] | None = None,
        alpha: float = 0.05,
    ) -> CIResult:
        """Test whether X and Y are conditionally independent given Z.

        Parameters
        ----------
        x : ndarray of shape (n,) or (n, d_x)
            Sample of X.

        y : ndarray of shape (n,) or (n, d_y)
            Sample of Y.

        z : ndarray of shape (n,) or (n, d_z), or None, optional (default: None)
            Sample of the conditioning set. If None, the test is marginal.

        alpha : float, optional (default: 0.05)
            Significance level; the test rejects if ``p_value < alpha``.

        Returns
        -------
        result : CIResult
            If all the columns of X, or all those of Y, are constant, the
            variables are trivially independent and the result is
            ``statistic=0``, ``p_value=1``. Otherwise ``statistic`` is the
            V-statistic. ``p_value`` comes from the Gamma
            approximation if ``approx`` is True, otherwise it is the fraction
            of the ``nullss`` Monte Carlo draws above the statistic.
        """
        n = x.shape[0]
        data_x = x.reshape(n, -1)
        data_y = y.reshape(n, -1)
        data_z = np.empty((n, 0)) if z is None else z.reshape(n, -1)

        if not np.ptp(data_x, axis=0).any() or not np.ptp(data_y, axis=0).any():
            return CIResult(statistic=0.0, p_value=1.0, reject=False)

        Kx, Ky, Kzx, Kzy = self.kernel_matrix(data_x, data_y, data_z)
        test_stat, KxR, KyR = self.KCI_V_statistic(Kx, Ky, Kzx, Kzy)
        eigenvects_prod, n_components = self.get_eigenvects_prod(KxR, KyR)

        if self.approx:
            approx_k, approx_theta = self.get_kappa(eigenvects_prod)
            p_value = float(1 - stats.gamma.cdf(test_stat, approx_k, 0, approx_theta))
        else:
            null_distribution = self.null_sample(eigenvects_prod, n_components, n)
            p_value = float(np.sum(null_distribution > test_stat) / self.nullss)

        return CIResult(statistic=float(test_stat), p_value=p_value, reject=(p_value < alpha))


@ci_test
def kci_test(
    data: pd.DataFrame,
    x: str,
    y: str,
    z: list[str] | None = None,
    alpha: float = 0.05,
) -> CIResult:
    """Test conditional independence of two columns given others with the KCI test.

    Wrapper that selects the columns and runs ``KCI().test`` with its default
    parameters.

    Parameters
    ----------
    data : DataFrame
        Observations in rows, numeric variables in columns.

    x : str
        Name of the first tested column.

    y : str
        Name of the second tested column.

    z : list of str or None, optional (default: None)
        Names of the conditioning columns. If None or empty, the test is marginal.

    alpha : float, optional (default: 0.05)
        Significance level; the test rejects if ``p_value < alpha``.

    Returns
    -------
    result : CIResult
        ``statistic`` is the V-statistic, ``p_value`` the one from the Gamma
        approximation and ``reject`` the decision at level ``alpha``.
    """
    z_values = data[z].to_numpy(dtype=np.float64) if z else None
    return KCI().test(
        data[x].to_numpy(dtype=np.float64), data[y].to_numpy(dtype=np.float64), z_values, alpha
    )
