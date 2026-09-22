# Z = atanh(r_XY.S) * sqrt(n - |S| - 3), the standard PC-algorithm CI test
# for Gaussian data (Kalisch & Bühlmann 2007, "Estimating High-Dimensional
# Directed Acyclic Graphs with the PC-Algorithm," JMLR).

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from src.CI_tests.common import CIResult, col, n_cols


def fisher_z_test(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64] | None = None,
    alpha: float = 0.05,
) -> CIResult:
    """Test conditional independence via Fisher's Z-transformed partial correlation.

    🇬🇧 Computes the partial correlation of x and y given z from the
    precision matrix (inverse correlation matrix) of [x, y, z], applies
    Fisher's z-transform, and tests it against N(0,1). Assumes (x, y, z)
    are jointly Gaussian/linearly related — GCM is the nonparametric
    analogue when that doesn't hold.

    🇪🇸 Calcula la correlación parcial de x e y dado z a partir de la matriz
    de precisión (inversa de la matriz de correlación) de [x, y, z], aplica
    la transformación z de Fisher, y la testea contra N(0,1). Asume que
    (x, y, z) son conjuntamente gaussianas/relacionadas linealmente — GCM
    es el análogo no paramétrico cuando eso no se cumple.

    Args:
        x: Test variable, shape (n,) — Variable a testear, forma (n,).
        y: Test variable, shape (n,) — Variable a testear, forma (n,).
        z: Conditioning set, shape (n, dz), or None for an unconditional
            test — Conjunto de condicionamiento, forma (n, dz), o None
            para un test incondicional.
        alpha: Significance level — Nivel de significación.

    Returns:
        CIResult with the test statistic, p-value and reject decision —
        CIResult con el estadístico, el p-valor y la decisión de rechazo.
    """
    n = x.shape[0]
    dim_z = n_cols(z) if z is not None else 0
    dof = n - dim_z - 3
    if dof <= 0:
        raise ValueError(f"Sample size too small for |z|={dim_z}: n - |z| - 3 = {dof} <= 0.")

    columns = [x, y]
    if z is not None:
        columns += [col(z, j) for j in range(dim_z)]
    corr = np.corrcoef(np.column_stack(columns), rowvar=False)
    try:
        precision = np.linalg.inv(corr)
    except np.linalg.LinAlgError as err:
        raise ValueError("Correlation matrix is singular, cannot run the Fisher-Z test.") from err

    r = -precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])
    r = np.clip(r, -1.0 + np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    z_stat = float(np.arctanh(r) * np.sqrt(dof))
    p_value = float(2 * (1 - norm.cdf(abs(z_stat))))

    return CIResult(statistic=z_stat, p_value=p_value, reject=(p_value < alpha))
