# Kernel centering: same O(n^2) algebraic identity as causal-learn's
# Kernel.center_kernel_matrix and pywhy-stats' _fast_centering (py-why).
# Z-regularization: ridge floor from Zhang, Peters, Janzing, Schölkopf (2011,
# UAI, "Kernel-based Conditional Independence Test..."), Eq. 10; numerical
# floor and Cholesky-vs-pinv fallback from Marchildon & Zingg (2023,
# arXiv:2307.05855, Eq. 25-26) and Foster et al. (2009, JMLR 10:857-882).
# Condition-number check: Hager's estimator (1984, SIAM J. Sci. Stat.
# Comput.), refined by Higham (1988, ACM TOMS) — the same algorithm behind
# LAPACK's ?pocon/?gecon and MATLAB/R's `rcond`.

import warnings

import numpy as np
from numpy.linalg import pinv
from numpy.typing import NDArray
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.linalg.lapack import dpocon

# Zhang et al. (2011), Sec. 3.5: eps = 1e-3 works well when dim(Z) is small.
DEFAULT_EPSILON = 1e-3

# Marchildon & Zingg (2023) treat kappa_max as a user-set threshold, not a
# fixed constant; 1e8 is a conservative default for float64. Also used as
# the acceptance threshold for the runtime condition-number check below.
DEFAULT_KAPPA_MAX = 1e8


def center_kernel_matrix(K: NDArray[np.float64]) -> NDArray[np.float64]:
    """Center a kernel matrix in O(n^2).

    🇬🇧 Computes K_c = H @ K @ H, with H = I - (1/n)*ones(n, n), by expanding
    the product algebraically into row/column/total sums instead of
    materializing H and multiplying matrices (O(n^3)).

    🇪🇸 Calcula K_c = H @ K @ H, con H = I - (1/n)*unos(n, n), expandiendo el
    producto algebraicamente en sumas por fila/columna/total en vez de
    materializar H y multiplicar matrices (O(n^3)).

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
    """Residualize a kernel matrix against Z.

    🇬🇧 Computes R_z = eps*(Kz + eps*I)^-1 and returns R_z @ K @ R_z, the
    kernel matrix of the part of K not explained by Z (Zhang et al. 2011,
    Eq. 10-12). `eps` is the larger of two floors: the statistical default
    from the paper, and a numerical floor, `n / (kappa_max - 1)`, sufficient
    to keep cond(Kz + eps*I) <= kappa_max — since Kz is PSD with unit
    diagonal (a correlation-type kernel matrix, tr(Kz) = n), its largest
    eigenvalue is at most tr(Kz) = n (Marchildon & Zingg 2023, Eq. 25-26).
    The regularized system is solved via Cholesky, since Kz + eps*I is SPD.

    The ideal correctness check would rebuild the full inverse and verify
    max|A@inv - I| directly, but that is an O(n^3) matrix product — as
    expensive as the solve itself, and measured to roughly double the total
    cost. Instead this estimates cond(Kz + eps*I) in O(n^2) with LAPACK's
    Hager-Higham estimator (`dpocon`, the same algorithm behind MATLAB/R's
    `rcond`) and combines it with the standard backward-stability bound for
    Cholesky solves, `||x̂-x||/||x|| ≲ cond(A)*u`. This is the standard
    production trade-off (used throughout LAPACK-based numerical software),
    not an exact check: `dpocon` is a well-established but approximate
    estimator, and in rare pathological cases can underestimate the true
    condition number. When the estimate exceeds `kappa_max`, or Cholesky
    itself fails, this falls back to a pseudo-inverse and warns instead of
    failing silently. Benchmarked against always using pinv: this is at
    worst a statistical tie for tiny n (~10), and 5-7x faster from n=100
    upward.

    🇪🇸 Calcula R_z = eps*(Kz + eps*I)^-1 y devuelve R_z @ K @ R_z, la matriz
    de kernel de la parte de K no explicada por Z (Zhang et al. 2011,
    Ec. 10-12). `eps` es el mayor de dos cotas: el valor estadístico por
    defecto del paper, y una cota numérica, `n / (kappa_max - 1)`,
    suficiente para mantener cond(Kz + eps*I) <= kappa_max — ya que Kz es
    semidefinida positiva con diagonal unidad (matriz de kernel tipo
    correlación, tr(Kz) = n), su autovalor máximo es como mucho tr(Kz) = n
    (Marchildon & Zingg 2023, Ec. 25-26). El sistema regularizado se
    resuelve con Cholesky, ya que Kz + eps*I es SDP.

    El chequeo de corrección ideal reconstruiría la inversa completa y
    comprobaría max|A@inv - I| directamente, pero eso es un producto de
    matrices O(n^3) — tan caro como la propia resolución, y en la práctica
    duplica aproximadamente el coste total. En su lugar, esto estima
    cond(Kz + eps*I) en O(n^2) con el estimador de Hager-Higham de LAPACK
    (`dpocon`, el mismo algoritmo detrás del `rcond` de MATLAB/R) y lo
    combina con la cota estándar de estabilidad hacia atrás de las
    resoluciones por Cholesky, `||x̂-x||/||x|| ≲ cond(A)*u`. Es la
    concesión estándar en producción (usada en todo el software numérico
    basado en LAPACK), no un chequeo exacto: `dpocon` es un estimador bien
    establecido pero aproximado, y en casos patológicos raros puede
    subestimar el número de condición real. Cuando la estimación supera
    `kappa_max`, o el propio Cholesky falla, se recurre a la pseudo-inversa
    y se avisa en vez de fallar en silencio. Medido contra usar siempre
    pinv: en el peor caso es un empate estadístico para n diminuto (~10), y
    5-7x más rápido a partir de n=100.

    Args:
        K: Kernel matrix to residualize, shape (n, n) — Matriz de kernel a
            residualizar, forma (n, n).
        Kz: Centered kernel matrix of the conditioning set Z, shape (n, n) —
            Matriz de kernel centrada del conjunto de condicionamiento Z,
            forma (n, n).
        epsilon: Statistical regularizer floor — Cota estadística del
            regularizador.
        kappa_max: Maximum tolerated condition number of (Kz + eps*I) —
            Número de condición máximo tolerado de (Kz + eps*I).

    Returns:
        (R_z @ K @ R_z, R_z), both shape (n, n) — R_z is also returned
        because it depends only on Kz and eps, not on K: callers residualizing
        a second matrix against the same Z (e.g. KCI's Ky against the same
        Kz used for Kx) can reuse it directly instead of recomputing the
        regularized inverse.
        (R_z @ K @ R_z, R_z), ambas de forma (n, n) — se devuelve también
        R_z porque depende solo de Kz y eps, no de K: quien necesite
        residualizar una segunda matriz contra la misma Z (p. ej. Ky de KCI
        contra la misma Kz que Kx) puede reutilizarla directamente en vez de
        recalcular la inversa regularizada.
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
