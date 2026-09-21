from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ResidualScenario:
    """A base assumption shared by every residual-based CI test.

    🇬🇧 Generates a pair of residual arrays (as if already regressed on
    some Z) — the deterministic input every residual-based CI test
    (GCM, and any future method built the same way) can be fed
    directly, skipping the regression step entirely so cross-language
    comparisons aren't polluted by RNG/regressor differences.

    🇪🇸 Genera un par de arrays de residuos (como si ya se hubiera
    regresado sobre algún Z) — la entrada determinista que cualquier
    test de CI basado en residuos (GCM, y cualquier método futuro
    construido igual) puede recibir directamente, saltándose la
    regresión por completo para que las comparaciones entre lenguajes
    no se vean afectadas por diferencias de aleatoriedad/regresor.
    """

    name: str
    generate: Callable[[np.random.Generator], tuple[NDArray[np.float64], NDArray[np.float64]]]


def _independent(rng: np.random.Generator) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = 200
    return rng.standard_normal(n), rng.standard_normal(n)


def _dependent(rng: np.random.Generator) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = 200
    resid_x = rng.standard_normal(n)
    resid_y = 0.8 * resid_x + 0.3 * rng.standard_normal(n)
    return resid_x, resid_y


SCENARIOS = [
    ResidualScenario("independent", _independent),
    ResidualScenario("dependent", _dependent),
]
