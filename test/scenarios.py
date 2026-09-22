from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Scenario:
    """A conditional-independence data generator, shared across every registered method.

    🇬🇧 Generates raw (x, y, z) under a known ground truth (independent or
    dependent given z). New scenarios go here and are automatically run
    against every method in `METHODS` — each method's own bridge adapts
    (x, y, z) into whatever its call signature needs.

    🇪🇸 Genera (x, y, z) crudos bajo una verdad conocida (independientes o
    dependientes dado z). Los escenarios nuevos van aquí y se ejecutan
    automáticamente contra todos los métodos de `METHODS` — el bridge de
    cada método adapta (x, y, z) a lo que necesite su firma.
    """

    name: str
    generate: Callable[
        [np.random.Generator],
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ]


def _independent(
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    n = 250
    z = rng.standard_normal((n, 1))
    x = np.sin(z[:, 0]) + rng.standard_normal(n)
    y = np.cos(z[:, 0]) + rng.standard_normal(n)
    return x, y, z


def _dependent(
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    x, y, z = _independent(rng)
    return x, y + x, z


def _zhang_case_base(
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """D=1 construction shared by Case I and Case II (Zhang et al. 2011,
    Sec. 4.1; exp/simulation1/test_effect_D_caseI.m and caseII.m)."""
    n = 200
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    z = rng.standard_normal(n)

    zz1 = 0.7 * (z**3 / 5 + z / 2)
    x = zz1 + np.tanh(x)
    x = x + x**3 / 3 + np.tanh(x / 3) / 2

    zz2 = (z**3 / 4 + z) / 3
    y = y + zz2
    y = y + np.tanh(y / 3)

    x = (x - x.mean()) / x.std(ddof=1)
    y = (y - y.mean()) / y.std(ddof=1)
    z = (z - z.mean()) / z.std(ddof=1)
    return x, y, z


def _zhang_case_independent(
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    x, y, z = _zhang_case_base(rng)
    return x, y, z.reshape(-1, 1)


def _zhang_case_dependent(
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    x, y, z = _zhang_case_base(rng)
    common = 0.5 * rng.standard_normal(x.shape[0])
    return x + common, y + common, z.reshape(-1, 1)


SCENARIOS = [
    Scenario("independent", _independent),
    Scenario("dependent", _dependent),
    Scenario("zhang_case_independent", _zhang_case_independent),
    Scenario("zhang_case_dependent", _zhang_case_dependent),
]
