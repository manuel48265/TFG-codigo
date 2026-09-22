# Calls the installed causal-learn dependency at runtime — no source from
# that package is copied or redistributed here.

import numpy as np
from causallearn.utils.cit import FisherZ
from numpy.typing import NDArray

from src.CI_tests.common import CIResult, col, n_cols


def fisher_z_test_causallearn(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64] | None = None,
    alpha: float = 0.05,
) -> CIResult:
    """Call causal-learn's `FisherZ` CI test.

    🇬🇧 A bridge, not a reimplementation: hands the same inputs to
    causal-learn's actual `FisherZ` and wraps its output in the shared
    `CIResult` contract. Its public `__call__` only returns a p-value (no
    signed statistic), so `statistic` is left as None rather than
    recomputed here from its internals.

    🇪🇸 Un puente, no una reimplementación: le pasa las mismas entradas al
    `FisherZ` real de causal-learn y envuelve su salida en el contrato
    compartido `CIResult`. Su `__call__` público solo devuelve un p-valor
    (sin estadístico con signo), así que `statistic` se deja en None en
    vez de recalcularlo aquí a partir de sus internos.

    Args:
        x: Test variable, shape (n,) — Variable a testear, forma (n,).
        y: Test variable, shape (n,) — Variable a testear, forma (n,).
        z: Conditioning set, shape (n, dz), or None — Conjunto de
            condicionamiento, forma (n, dz), o None.
        alpha: Significance level — Nivel de significación.

    Returns:
        CIResult with causal-learn's p-value and reject decision —
        CIResult con el p-valor y la decisión de rechazo de causal-learn.
    """
    dim_z = n_cols(z) if z is not None else 0
    columns = [x, y]
    if z is not None:
        columns += [col(z, j) for j in range(dim_z)]
    data = np.column_stack(columns)

    p_value = float(FisherZ(data)(0, 1, tuple(range(2, 2 + dim_z))))
    return CIResult(statistic=None, p_value=p_value, reject=(p_value < alpha))
