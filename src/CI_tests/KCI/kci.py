# References
# K. Zhang, J. Peters, D. Janzing, and B. Schölkopf,
# "A kernel-based conditional independence test and application in causal discovery," In UAI 2011.

import numpy as np
from numpy.linalg import eigh, eigvalsh
from numpy.typing import NDArray

from src.CI_tests.KCI.kci_utils import center_kernel_matrix, center_kernel_matrix_regression
from src.CI_tests.common import CIResult

def scaled_eigenvecs(
    K: NDArray[np.float64], 
    thresh: float
) -> NDArray[np.float64]:
    
    w, v = eigh(0.5 * (K + K.T))
    idx = np.argsort(-w)
    w, v = w[idx], v[:, idx]
    mask = w > np.max(w) * thresh
    return v[:, mask].dot(np.diag(np.sqrt(w[mask])))


def get_eigenvects_prod(
    Kx: NDArray[np.float64], 
    Ky: NDArray[np.float64],
    thresh : float
) -> tuple[NDArray[np.float64], int]:
    scaled_eigenvecs_x = scaled_eigenvecs(Kx, thresh)
    scaled_eigenvecs_y = scaled_eigenvecs(Ky, thresh)

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


def get_kappa(
    eigenvecs_product: NDArray[np.float64]
) -> tuple[np.float64, np.float64]:
    
    approx_mean = np.trace(eigenvecs_product)
    approx_var = 2 * np.trace(eigenvecs_product.dot(eigenvecs_product))
    approx_k = approx_mean**2 / approx_var
    approx_theta = approx_var / approx_mean
    return approx_k, approx_theta


def null_sample(
    eigenvecs_prod: NDArray[np.float64], 
    n_components: int, 
    sample_size: int, 
    thresh : float,
    nullss : int
) -> NDArray[np.float64]:

    eigen = eigvalsh(eigenvecs_prod)
    eigen = -np.sort(-eigen)
    eigen = eigen[0 : np.min((sample_size, n_components))]
    eigen = eigen[eigen > np.max(eigen) * thresh]

    chi_distribution = np.random.chisquare(1, (eigen.shape[0], nullss))
    null_distribution = eigen.T.dot(chi_distribution)

    return null_distribution

def compute_pvalue():
    return

def kernel_matrix():
    return

def KCI_V_statistic(
    Kx : NDArray[np.float64],
    Ky : NDArray[np.float64],
    Kzx : NDArray[np.float64],
    Kzy : NDArray[np.float64],
    epsilon_x : float,
    epsilon_y : float
) -> tuple[np.float64,NDArray[np.float64],NDArray[np.float64]]:
    KxR, Rzx = center_kernel_matrix_regression(kx,kzx,epsilon_x)
    return


def KCI(
    kernelX,
    kernelY,
    kernelZ,
    nullss,
    est_width,
    use_gp,
    approx,
    kwidthx=None,
    kwidthy=None,
    kwidthz=None
) -> CIResult:
    return CIResult(statistic=0, p_value=0, reject=True)
