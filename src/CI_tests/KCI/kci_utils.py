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
    """Center a kernel matrix: K_c = H @ K @ H, H = I - (1/n)*ones(n, n).

    🇬🇧 Expands the product algebraically into row/column/total sums,
    O(n^2), instead of materializing H, O(n^3).

    🇪🇸 Expande el producto algebraicamente en sumas por fila/columna/total,
    O(n^2), en vez de materializar H, O(n^3).

    Args:
        K: Kernel matrix to center, shape (n, n) — Matriz de kernel a
            centrar, forma (n, n).

    Returns:
        Centered kernel matrix, shape (n, n) — Matriz de kernel centrada,
        forma (n, n).
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
    """Residualize a kernel matrix against Z: R_z @ K @ R_z.

    🇬🇧 R_z = eps*(Kz + eps*I)^-1 (Zhang et al. 2011, Eq. 10-12). `eps` is
    the larger of the paper's statistical default and a numerical floor,
    `n / (kappa_max - 1)`, that keeps cond(Kz + eps*I) <= kappa_max
    (Marchildon & Zingg 2023, Eq. 25-26). Solved via Cholesky; falls back to
    a pseudo-inverse and warns if the estimated condition number exceeds
    `kappa_max` or the factorization fails.

    🇪🇸 R_z = eps*(Kz + eps*I)^-1 (Zhang et al. 2011, Ec. 10-12). `eps` es
    el mayor entre el valor estadístico por defecto del paper y una cota
    numérica, `n / (kappa_max - 1)`, que mantiene cond(Kz + eps*I) <=
    kappa_max (Marchildon & Zingg 2023, Ec. 25-26). Se resuelve con
    Cholesky; recurre a la pseudo-inversa y avisa si el número de
    condición estimado supera `kappa_max` o falla la factorización.

    Args:
        K: Kernel matrix to residualize, shape (n, n) — Matriz de kernel a
            residualizar, forma (n, n).
        Kz: Centered kernel matrix of Z, shape (n, n) — Matriz de kernel
            centrada de Z, forma (n, n).
        epsilon: Statistical regularizer floor — Cota estadística del
            regularizador.
        kappa_max: Maximum tolerated condition number of (Kz + eps*I) —
            Número de condición máximo tolerado de (Kz + eps*I).

    Returns:
        (R_z @ K @ R_z, R_z) — R_z is also returned so callers residualizing
        another matrix against the same Z can reuse it — (R_z @ K @ R_z,
        R_z) — se devuelve también R_z para que quien residualice otra
        matriz contra la misma Z pueda reutilizarla.
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
    """Gaussian kernel: k(u, v) = exp(-0.5 * theta * ||u - v||^2)."""

    def __init__(self) -> None:
        self.width: float = 1.0

    def kernel(
        self, X: NDArray[np.float64], Y: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Kernel matrix for X, or between X and Y.

        Args:
            X: Data, shape (n, d) — Datos, forma (n, d).
            Y: Data, shape (m, d), or None to compute X against itself —
                Datos, forma (m, d), o None para calcular X contra sí misma.

        Returns:
            Kernel matrix, shape (n, n) or (n, m) — Matriz de kernel, forma
            (n, n) o (n, m).
        """
        sq_dists = squareform(pdist(X, "sqeuclidean")) if Y is None else cdist(X, Y, "sqeuclidean")
        return np.exp(-0.5 * self.width * sq_dists)

    def set_width_empirical(self, n: int, dim_z: int) -> None:
        """Set the width from the sample size and dim(Z) (CInd_test_new_withGP.m).

        🇬🇧 The n<200/n<1200 boundaries follow causal-learn's port
        (`set_width_empirical_kci`), not the original MATLAB (`T<=200`),
        since causal-learn is what this is validated against at n=200.

        🇪🇸 Los límites n<200/n<1200 siguen el port de causal-learn
        (`set_width_empirical_kci`), no el MATLAB original (`T<=200`), ya
        que es contra causal-learn contra quien se valida esto en n=200.

        Args:
            n: Sample size — Tamaño muestral.
            dim_z: Dimensionality of the conditioning set Z — Dimensionalidad
                del conjunto de condicionamiento Z.
        """
        if n < 200:
            width = 1.2
        elif n < 1200:
            width = 0.7
        else:
            width = 0.4
        self.width = 1.0 / (width**2 * dim_z)


def normalize_unit_variance(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Z-score each column to unit variance (Zhang et al. 2011, Sec. 3.5).

    🇬🇧 A constant column has zero variance, which zscore turns into NaN;
    those are reset to 0 (a constant column carries no information for a
    distance-based kernel anyway).

    🇪🇸 Una columna constante tiene varianza cero, que zscore convierte en
    NaN; se ponen a 0 (una columna constante no aporta información a un
    kernel basado en distancias).

    Args:
        data: Data, shape (n, d) — Datos, forma (n, d).

    Returns:
        Normalized data, shape (n, d) — Datos normalizados, forma (n, d).
    """
    normalized = stats.zscore(data, ddof=1, axis=0)
    normalized[np.isnan(normalized)] = 0.0
    return normalized
