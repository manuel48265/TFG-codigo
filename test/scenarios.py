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
    clean_seed: int = 0
    """A seed pre-verified (see test_scenarios.py) to generate no near-tie
    (`has_near_tie`) in x/y/z. For tests that need exact tie-break
    agreement with an external reference implementation that does its own
    independent random tie-breaking (e.g. CMIknn vs tigramite,
    test/CI_tests/CMIknn/test_cmiknn.py) rather than a formula match at
    every seed, since near-ties make that agreement a coin flip on both
    sides, not a property of either formula. `test_scenarios.py` re-checks
    this on every run, so a scenario whose generator changes and
    invalidates its stored clean_seed fails loudly instead of silently."""


def has_near_tie(
    *arrays: NDArray[np.float64], safety_margin: float = 10.0, noise_scale_factor: float = 1e-6
) -> bool:
    """🇬🇧 Flags data with a gap smaller than `safety_margin` times
    `CMIknn._rank_transform`'s default std-relative tie-break noise scale.
    Below that, which of two distinct values sorts first is decided by the
    tie-break noise, not by the data — and since we and an external
    reference (e.g. tigramite) draw that noise from independent RNGs, a
    mismatch there reflects the two RNGs disagreeing on an arbitrary coin
    flip, not a disagreement about the underlying computation (found
    empirically, see experiments/verification/verify_cmiknn.py —
    coarsening the data instead, e.g. rounding, was tried and made things
    far worse by creating mass *exact* ties, each independently
    coin-flipped by both sides' RNGs).

    🇪🇸 Marca datos con un hueco menor que `safety_margin` veces la escala
    de ruido de desempate por defecto de `CMIknn._rank_transform` (relativa
    a std). Por debajo de eso, cuál de dos valores distintos ordena primero
    lo decide el ruido de desempate, no el dato — y como nosotros y una
    referencia externa (p.ej. tigramite) sacamos ese ruido de RNGs
    independientes, un desajuste ahí refleja que los dos RNGs discreparon
    en una moneda al aire arbitraria, no un desacuerdo sobre el cálculo en
    sí (comprobado empíricamente, ver
    experiments/verification/verify_cmiknn.py — hacer los datos más
    gruesos, p.ej. redondeando, se probó y empeoró mucho las cosas, al
    crear empates *exactos* masivos, cada uno decidido por separado por el
    RNG de cada lado).
    """
    for a in arrays:
        flat = np.asarray(a).ravel()
        gaps = np.diff(np.sort(flat))
        gaps = gaps[gaps > 0]
        if gaps.size == 0:
            continue
        if gaps.min() < safety_margin * noise_scale_factor * flat.std():
            return True
    return False


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
    Scenario("independent", _independent, clean_seed=1),
    Scenario("dependent", _dependent, clean_seed=1),
    Scenario("zhang_case_independent", _zhang_case_independent, clean_seed=1),
    Scenario("zhang_case_dependent", _zhang_case_dependent, clean_seed=2),
]
