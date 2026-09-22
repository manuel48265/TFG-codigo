# Calls the installed causal-learn dependency at runtime — no source from
# that package is copied or redistributed here.

import numpy as np
from causallearn.utils.KCI.KCI import KCI_CInd
from numpy.typing import NDArray

from src.CI_tests.common import CIResult


def kci_test_causallearn(
    data_x: NDArray[np.float64],
    data_y: NDArray[np.float64],
    data_z: NDArray[np.float64],
    alpha: float = 0.05,
    **kwargs: object,
) -> CIResult:
    """Call causal-learn's `KCI_CInd.compute_pvalue`.

    🇬🇧 A bridge, not a reimplementation: hands the same inputs to
    causal-learn's actual `KCI_CInd` and wraps its output in the shared
    `CIResult` contract, so it can be compared against `kci_test` (the
    Python port) using identical call sites in tests.

    🇪🇸 Un puente, no una reimplementación: le pasa las mismas entradas al
    `KCI_CInd` real de causal-learn y envuelve su salida en el contrato
    compartido `CIResult`, para poder compararlo con `kci_test` (el port a
    Python) usando exactamente las mismas llamadas en los tests.

    Args:
        data_x: X sample, shape (n, dx) — Muestra de X, forma (n, dx).
        data_y: Y sample, shape (n, dy) — Muestra de Y, forma (n, dy).
        data_z: Z sample, shape (n, dz) — Muestra de Z, forma (n, dz).
        alpha: Significance level — Nivel de significación.
        **kwargs: Forwarded to `KCI_CInd.__init__` — Reenviados a
            `KCI_CInd.__init__`.

    Returns:
        CIResult with causal-learn's statistic, p-value and reject decision
        — CIResult con el estadístico, p-valor y decisión de rechazo de
        causal-learn.
    """
    p_value, test_stat = KCI_CInd(**kwargs).compute_pvalue(data_x, data_y, data_z)
    return CIResult(statistic=float(test_stat), p_value=float(p_value), reject=(p_value < alpha))
