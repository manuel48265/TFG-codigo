from collections.abc import Callable
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from src.CI_tests.Mixed_CI.mixed_ci import VariableType

N_SAMPLES = 300
N_SAMPLES_CONTINUOUS = 250
CONTINUOUS, BINARY, NOMINAL = VariableType.CONTINUOUS, VariableType.BINARY, VariableType.NOMINAL


@dataclass
class DataFrameScenario:
    """A data generator returning a DataFrame with columns ``x``, ``y`` and the conditioning ones.

    Attributes
    ----------
    name : str
        Identifier of the scenario.

    generate : callable
        Maps a random generator to the DataFrame. Continuous variables are
        ``float64`` columns, binary ones ``int64`` columns with values 0 and 1,
        and nominal ones unordered ``category`` columns with three levels.

    conditioning : list of str
        Names of the conditioning columns.

    clean_seed : int, optional (default: 0)
        Seed whose data has no near-tie (see ``has_near_tie``). Only set for the
        continuous scenarios, for tests that need exact tie-break agreement with
        a reference that breaks ties with its own random generator.
    """

    name: str
    generate: Callable[[np.random.Generator], pd.DataFrame]
    conditioning: list[str]
    clean_seed: int = 0


def _observe(latent: np.ndarray, kind: VariableType) -> pd.Series | pd.Categorical | np.ndarray:
    """Turn a continuous latent variable into a variable of the given type, by quantiles."""
    if kind == CONTINUOUS:
        return latent
    if kind == BINARY:
        return pd.qcut(latent, 2, labels=False).astype(np.int64)
    return pd.Categorical(np.array(["a", "b", "c"])[pd.qcut(latent, 3, labels=False)])


def _mixed_scenario(
    x_kind: VariableType,
    y_kind: VariableType,
    z_kinds: tuple[VariableType, ...],
    dependent: bool,
) -> DataFrameScenario:
    """X and Y share a latent dependence on Z; Y also depends on X if ``dependent``."""

    def generate(rng: np.random.Generator) -> pd.DataFrame:
        z_latent = rng.standard_normal((N_SAMPLES, len(z_kinds)))
        common = 0.6 * z_latent.sum(axis=1)
        x_latent = common + rng.standard_normal(N_SAMPLES)
        y_latent = common + (0.8 * x_latent if dependent else 0.0) + rng.standard_normal(N_SAMPLES)

        columns = {"x": _observe(x_latent, x_kind), "y": _observe(y_latent, y_kind)}
        for j, kind in enumerate(z_kinds):
            columns[f"z{j}"] = _observe(z_latent[:, j], kind)
        return pd.DataFrame(columns)

    letters = {CONTINUOUS: "C", BINARY: "B", NOMINAL: "N"}
    name = (
        f"x={letters[x_kind]},y={letters[y_kind]},"
        f"z={''.join(letters[k] for k in z_kinds) or 'none'},"
        f"{'dependent' if dependent else 'independent'}"
    )
    return DataFrameScenario(
        name=name,
        generate=generate,
        conditioning=[f"z{j}" for j in range(len(z_kinds))],
    )


def has_near_tie(
    *arrays: np.ndarray, safety_margin: float = 10.0, noise_scale_factor: float = 1e-6
) -> bool:
    """Return whether any array has two distinct values closer than the tie-break noise.

    A gap below ``safety_margin * noise_scale_factor * std`` is comparable to the
    noise ``CMIknn._rank_transform`` adds to break ties, so the order of the two
    values would be decided by that noise.

    Parameters
    ----------
    *arrays : ndarray
        Samples to check, each flattened.

    safety_margin : float, optional (default: 10.0)
        Multiple of the noise scale below which a gap counts as a near-tie.

    noise_scale_factor : float, optional (default: 1e-6)
        Noise scale relative to the standard deviation of each array.

    Returns
    -------
    has_tie : bool
        True if some array has a gap below the threshold.
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


def _continuous_frame(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"x": x, "y": y, "z0": z})


def _independent(rng: np.random.Generator) -> pd.DataFrame:
    z = rng.standard_normal(N_SAMPLES_CONTINUOUS)
    x = np.sin(z) + rng.standard_normal(N_SAMPLES_CONTINUOUS)
    y = np.cos(z) + rng.standard_normal(N_SAMPLES_CONTINUOUS)
    return _continuous_frame(x, y, z)


def _dependent(rng: np.random.Generator) -> pd.DataFrame:
    df = _independent(rng)
    return df.assign(y=df["y"] + df["x"])


def _zhang_case_base(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the D=1 construction shared by Case I and Case II of Zhang et al. (2011, Sec. 4.1)."""
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


def _zhang_case_independent(rng: np.random.Generator) -> pd.DataFrame:
    return _continuous_frame(*_zhang_case_base(rng))


def _zhang_case_dependent(rng: np.random.Generator) -> pd.DataFrame:
    x, y, z = _zhang_case_base(rng)
    common = 0.5 * rng.standard_normal(x.shape[0])
    return _continuous_frame(x + common, y + common, z)


CONTINUOUS_SCENARIOS = [
    DataFrameScenario("independent", _independent, ["z0"], clean_seed=1),
    DataFrameScenario("dependent", _dependent, ["z0"], clean_seed=1),
    DataFrameScenario("zhang_case_independent", _zhang_case_independent, ["z0"], clean_seed=1),
    DataFrameScenario("zhang_case_dependent", _zhang_case_dependent, ["z0"], clean_seed=2),
]

_KINDS = (CONTINUOUS, BINARY, NOMINAL)
_Z_KINDS: list[tuple[VariableType, ...]] = [
    (),
    (CONTINUOUS,),
    (BINARY,),
    (NOMINAL,),
    (CONTINUOUS, NOMINAL),
]

DF_SCENARIOS = CONTINUOUS_SCENARIOS + [
    _mixed_scenario(x_kind, y_kind, z_kinds, dependent)
    for x_kind, y_kind, z_kinds in product(_KINDS, _KINDS, _Z_KINDS)
    for dependent in (False, True)
]
