from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class CIResult:
    """🇬🇧 Common result contract for any CI test (shared across methods).

    🇪🇸 Contrato de resultado común para cualquier test de CI (compartido
    entre métodos).
    """

    statistic: float
    p_value: float
    reject: bool


def n_cols(a: NDArray[np.float64]) -> int:
    """Number of columns, treating a 1D array as a single column.

    🇬🇧 Mirrors R's `NCOL()`: a plain (n,) vector counts as 1 column.

    🇪🇸 Replica el `NCOL()` de R: un vector plano (n,) cuenta como 1 columna.
    """
    return 1 if a.ndim == 1 else a.shape[1]


def col(a: NDArray[np.float64], j: int) -> NDArray[np.float64]:
    """Get column j, treating a 1D array as its own single column.

    🇬🇧 Companion to `n_cols`: lets multivariate code index a variable
    uniformly whether it's a plain vector (j is ignored) or a matrix.

    🇪🇸 Complementa a `n_cols`: permite indexar una variable de forma
    uniforme, ya sea un vector plano (j se ignora) o una matriz.
    """
    return a if a.ndim == 1 else a[:, j]
