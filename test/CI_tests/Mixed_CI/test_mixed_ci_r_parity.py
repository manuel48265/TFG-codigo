"""Parity tests: `MixedCI.mixed_ci_test` (Python) vs `MXM::ci.mm` (R), one
scenario per outcome type (continuous/binary/nominal/ordinal), each with a
dependent and an independent case.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from src.CI_tests.Mixed_CI.mixed_ci import MixedCI
from test.CI_tests.Mixed_CI.mixed_ci_r_bridge import mixed_ci_test_r

LOG_EPSILON = 1e-3
N_SEEDS = 5  # R/rpy2 is slow per call; keeps the sweep's cost bounded


@dataclass
class MixedScenario:
    """A mixed-CI data generator, paired with whether p-values must match exactly."""

    name: str
    build: Callable[[np.random.Generator], pd.DataFrame]
    strict: bool = field(default=True)


def _continuous_df(rng: np.random.Generator, dependent: bool) -> pd.DataFrame:
    n = 400
    z = rng.normal(size=n)
    x = 0.6 * z + rng.normal(size=n)
    y = 0.6 * z + (0.8 * x if dependent else 0.0) + rng.normal(size=n)
    return pd.DataFrame({"x": x, "y": y, "z": z})


def _binary_df(rng: np.random.Generator, dependent: bool) -> pd.DataFrame:
    n = 400
    z = rng.normal(size=n)
    x = 0.6 * z + rng.normal(size=n)
    logit = 0.6 * z + (1.2 * x if dependent else 0.0)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"x": x, "y": y, "z": z})


def _categorical_df(rng: np.random.Generator, dependent: bool, *, ordered: bool) -> pd.DataFrame:
    n = 400
    z = rng.normal(size=n)
    x = 0.6 * z + rng.normal(size=n)
    latent = 0.6 * z + (1.2 * x if dependent else 0.0) + rng.normal(size=n)
    y = pd.Categorical(pd.cut(latent, bins=3, labels=["a", "b", "c"]), ordered=ordered)
    return pd.DataFrame({"x": x, "y": y, "z": z})


SCENARIOS = [
    MixedScenario("continuous_dependent", lambda rng: _continuous_df(rng, dependent=True)),
    MixedScenario("continuous_independent", lambda rng: _continuous_df(rng, dependent=False)),
    MixedScenario("binary_dependent", lambda rng: _binary_df(rng, dependent=True)),
    MixedScenario("binary_independent", lambda rng: _binary_df(rng, dependent=False)),
    MixedScenario(
        "nominal_dependent", lambda rng: _categorical_df(rng, dependent=True, ordered=False)
    ),
    MixedScenario(
        "nominal_independent", lambda rng: _categorical_df(rng, dependent=False, ordered=False)
    ),
    MixedScenario(
        "ordinal_dependent", lambda rng: _categorical_df(rng, dependent=True, ordered=True)
    ),
    MixedScenario(
        "ordinal_independent", lambda rng: _categorical_df(rng, dependent=False, ordered=True)
    ),
]


@pytest.mark.parametrize("seed", range(N_SEEDS))
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_python_matches_r_reference(scenario: MixedScenario, seed: int) -> None:
    """Every scenario's reject decision must match R, across `N_SEEDS` draws
    each — not just one, since mixed_ci_test has no internal randomness of
    its own (unlike CMIknn's shuffle test) so there's no obstacle to
    requiring agreement on more than a single lucky/unlucky seed. p-value
    must match too when the two sides use the same underlying model
    (`scenario.strict`).
    """
    df = scenario.build(np.random.default_rng(seed))

    py = MixedCI().mixed_ci_test(df, "x", "y", ["z"])
    ref = mixed_ci_test_r(df, "x", "y", ["z"])

    assert py.reject == ref.reject
    if scenario.strict:
        assert np.log(py.p_value) == pytest.approx(np.log(ref.p_value), abs=LOG_EPSILON)
