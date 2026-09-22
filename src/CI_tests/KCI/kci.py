# K. Zhang, J. Peters, D. Janzing, and B. Schölkopf,
# "A kernel-based conditional independence test and application in causal discovery," In UAI 2011.

import numpy as np
from numpy.linalg import eigh, eigvalsh
from numpy.typing import NDArray
from scipy import stats

from src.CI_tests.common import CIResult
from src.CI_tests.KCI.kci_utils import (
    GaussianKernel,
    center_kernel_matrix,
    center_kernel_matrix_regression,
    normalize_unit_variance,
)


def scaled_eigenvecs(K: NDArray[np.float64], thresh: float) -> NDArray[np.float64]:
    """Truncated, eigenvalue-scaled eigenvectors of a centered kernel matrix.

    🇬🇧 V*sqrt(Λ) from the eigendecomposition of K, keeping only eigenvectors
    whose eigenvalue exceeds `thresh` relative to the largest one (Zhang et
    al. 2011, Sec. 3.3). Symmetrizes K first, since `eigh` only reads one
    triangle.

    🇪🇸 V*sqrt(Λ) a partir de la descomposición espectral de K, quedándose
    solo con los autovectores cuyo autovalor supera `thresh` en relación al
    mayor (Zhang et al. 2011, Sec. 3.3). Simetriza K antes, ya que `eigh`
    solo lee un triángulo.

    Args:
        K: Centered kernel matrix, shape (n, n) — Matriz de kernel centrada,
            forma (n, n).
        thresh: Relative eigenvalue cutoff — Umbral relativo de autovalores.

    Returns:
        Truncated, scaled eigenvectors, shape (n, k) — Autovectores
        truncados y escalados, forma (n, k).
    """
    w, v = eigh(0.5 * (K + K.T))
    idx = np.argsort(-w)
    w, v = w[idx], v[:, idx]
    mask = w > np.max(w) * thresh
    return v[:, mask].dot(np.diag(np.sqrt(w[mask])))


class KCI:
    """Kernel-based Conditional Independence test (Zhang et al. 2011).

    🇬🇧 Only the Gaussian kernel is supported (the paper's own default,
    Sec. 2). Use the module-level `kci_test` function to run the test.

    🇪🇸 Solo se soporta el kernel gaussiano (el que usa el paper por
    defecto, Sec. 2). Usa la función `kci_test` del módulo para correr el
    test.
    """

    def __init__(
        self,
        nullss: int = 5000,
        approx: bool = True,
        thresh: float = 1e-5,
        alpha: float = 0.05,
        epsilon_x: float = 1e-3,
        epsilon_y: float = 1e-3,
        kwidthx: float | None = None,
        kwidthy: float | None = None,
        kwidthz: float | None = None,
    ) -> None:
        self.nullss = nullss
        self.approx = approx
        self.thresh = thresh
        self.alpha = alpha
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
        """Build and center the Gaussian kernel matrices for (X,Z), Y and Z.

        🇬🇧 The X kernel is built over (X, Z/2) concatenated, not X alone
        (Zhang et al. 2011, Sec. 3.3: Ẍ ≜ (X,Z); the Z/2 halving and the
        rest of this construction follow CInd_test_new_withGP.m, the
        original implementation, rather than the published text). All three
        kernels share one width, from `self.kwidthx/y/z` if given, otherwise
        from the empirical heuristic on (n, dim(Z)).

        🇪🇸 El kernel de X se construye sobre (X, Z/2) concatenados, no sobre
        X sola (Zhang et al. 2011, Sec. 3.3: Ẍ ≜ (X,Z); el que Z se divida
        entre 2 y el resto de esta construcción siguen
        CInd_test_new_withGP.m, la implementación original, no el texto
        publicado). Los tres kernels comparten un único ancho, tomado de
        `self.kwidthx/y/z` si se dan, si no de la heurística empírica sobre
        (n, dim(Z)).

        Args:
            data_x: X sample, shape (n, dx) — Muestra de X, forma (n, dx).
            data_y: Y sample, shape (n, dy) — Muestra de Y, forma (n, dy).
            data_z: Z sample, shape (n, dz) — Muestra de Z, forma (n, dz).

        Returns:
            (Kx, Ky, Kzx, Kzy), all centered; Kzx and Kzy are the same
            matrix — (Kx, Ky, Kzx, Kzy), todas centradas; Kzx y Kzy son la
            misma matriz.
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
        """Tensor-product eigenvector features of two residualized kernels.

        🇬🇧 w̆w̆ᵀ (or its dual w̆ᵀw̆, whichever is smaller) from the scaled
        eigenvectors of Kx and Ky; its eigenvalues are the null
        distribution's mixing weights (Zhang et al. 2011, Prop. 5).

        🇪🇸 w̆w̆ᵀ (o su dual w̆ᵀw̆, el que sea más pequeño) a partir de los
        autovectores escalados de Kx y Ky; sus autovalores son los pesos de
        la mezcla de la distribución nula (Zhang et al. 2011, Prop. 5).

        Args:
            Kx: Residualized kernel matrix, shape (n, n) — Matriz de kernel
                residualizada, forma (n, n).
            Ky: Residualized kernel matrix, shape (n, n) — Matriz de kernel
                residualizada, forma (n, n).

        Returns:
            (eigenvects_prod, n_components) — Matriz y número de
            componentes tensoriales combinados.
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
        """Gamma(k, theta) approximating the null distribution by moment matching.

        🇬🇧 Zhang et al. (2011, Prop. 6.ii). `k`/`theta` are named after the
        shape/scale parameters `scipy.stats.gamma.cdf` expects.

        🇪🇸 Zhang et al. (2011, Prop. 6.ii). `k`/`theta` se llaman así por
        los parámetros de forma/escala que espera `scipy.stats.gamma.cdf`.

        Args:
            eigenvecs_product: Output of `get_eigenvects_prod` — Salida de
                `get_eigenvects_prod`.

        Returns:
            (k, theta) — forma y escala de la distribución Gamma.
        """
        approx_mean = np.trace(eigenvecs_product)
        approx_var = 2 * np.trace(eigenvecs_product.dot(eigenvecs_product))
        approx_k = approx_mean**2 / approx_var
        approx_theta = approx_var / approx_mean
        return approx_k, approx_theta

    def null_sample(
        self, eigenvecs_prod: NDArray[np.float64], n_components: int, sample_size: int
    ) -> NDArray[np.float64]:
        """Monte Carlo draws from the null distribution (Zhang et al. 2011, Eq. 14).

        🇬🇧 Weights i.i.d. chi-square(1) draws with the eigenvalues of
        `eigenvecs_prod`, truncated to at most `sample_size` components
        (its rank can't exceed the sample size) and to those above
        `self.thresh` relative to the largest.

        🇪🇸 Pondera muestras chi-cuadrado(1) i.i.d. con los autovalores de
        `eigenvecs_prod`, truncados a como mucho `sample_size` componentes
        (su rango no puede superar el tamaño muestral) y a los que superan
        `self.thresh` en relación al mayor.

        Args:
            eigenvecs_prod: Output of `get_eigenvects_prod` — Salida de
                `get_eigenvects_prod`.
            n_components: Its second output — Su segunda salida.
            sample_size: Number of observations n — Número de observaciones
                n.

        Returns:
            `self.nullss` draws from the null distribution — `self.nullss`
            muestras de la distribución nula.
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
        """Residualize Kx/Ky against Z and compute the V-statistic (Zhang et al. 2011, Eq. 13).

        🇬🇧 Reuses R_z for Ky instead of recomputing it when Kzx and Kzy are
        the same matrix and `self.epsilon_x == self.epsilon_y`, since R_z
        depends only on Kz and epsilon.

        🇪🇸 Reutiliza R_z para Ky en vez de recalcularlo cuando Kzx y Kzy son
        la misma matriz y `self.epsilon_x == self.epsilon_y`, ya que R_z
        depende solo de Kz y epsilon.

        Args:
            Kx: Centered kernel matrix of (X,Z), shape (n, n) — Matriz de
                kernel centrada de (X,Z), forma (n, n).
            Ky: Centered kernel matrix of Y, shape (n, n) — Matriz de kernel
                centrada de Y, forma (n, n).
            Kzx: Centered kernel matrix of Z used to residualize Kx —
                Matriz de kernel centrada de Z usada para residualizar Kx.
            Kzy: Centered kernel matrix of Z used to residualize Ky —
                Matriz de kernel centrada de Z usada para residualizar Ky.

        Returns:
            (V_stat, KxR, KyR) — el estadístico del test y ambos kernels
            residualizados.
        """
        KxR, Rzx = center_kernel_matrix_regression(Kx, Kzx, self.epsilon_x)
        if self.epsilon_x != self.epsilon_y:
            KyR, _ = center_kernel_matrix_regression(Ky, Kzy, self.epsilon_y)
        else:
            KyR = Rzx.dot(Ky).dot(Rzx)

        Vstat = np.sum(KxR * KyR)
        return Vstat, KxR, KyR

    def compute_pvalue(
        self,
        data_x: NDArray[np.float64],
        data_y: NDArray[np.float64],
        data_z: NDArray[np.float64],
    ) -> CIResult:
        """Run the KCI conditional-independence test."""
        Kx, Ky, Kzx, Kzy = self.kernel_matrix(data_x, data_y, data_z)
        test_stat, KxR, KyR = self.KCI_V_statistic(Kx, Ky, Kzx, Kzy)
        eigenvects_prod, n_components = self.get_eigenvects_prod(KxR, KyR)

        if self.approx:
            approx_k, approx_theta = self.get_kappa(eigenvects_prod)
            p_value = float(1 - stats.gamma.cdf(test_stat, approx_k, 0, approx_theta))
        else:
            null_distribution = self.null_sample(eigenvects_prod, n_components, Kx.shape[0])
            p_value = float(np.sum(null_distribution > test_stat) / self.nullss)

        return CIResult(statistic=float(test_stat), p_value=p_value, reject=(p_value < self.alpha))


def kci_test(
    data_x: NDArray[np.float64],
    data_y: NDArray[np.float64],
    data_z: NDArray[np.float64],
    **kwargs: object,
) -> CIResult:
    """Test conditional independence via the Kernel Conditional Independence test.

    Args:
        data_x: X sample, shape (n, dx) — Muestra de X, forma (n, dx).
        data_y: Y sample, shape (n, dy) — Muestra de Y, forma (n, dy).
        data_z: Z sample, shape (n, dz) — Muestra de Z, forma (n, dz).
        **kwargs: Forwarded to `KCI.__init__` — Reenviados a `KCI.__init__`.

    Returns:
        CIResult with the test statistic, p-value and reject decision —
        CIResult con el estadístico, el p-valor y la decisión de rechazo.
    """
    return KCI(**kwargs).compute_pvalue(data_x, data_y, data_z)  # type: ignore[arg-type]
