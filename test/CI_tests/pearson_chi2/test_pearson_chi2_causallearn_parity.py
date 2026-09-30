"""Parity tests: `PearsonChi2` (Python) vs causal-learn's `chisq`."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from src.CI_tests.pearson_chi2.pearson_chi2 import PearsonChi2
from test.CI_tests.pearson_chi2.pearson_chi2_causallearn_bridge import (
    pearson_chi2_test_causallearn,
)


def _df(rng: np.random.Generator, dependent: bool, *, cond_dims: int) -> pd.DataFrame:
    n = 600
    z = rng.integers(0, 3, size=(n, cond_dims)) if cond_dims else np.empty((n, 0), dtype=int)
    z_effect = z.sum(axis=1) if cond_dims else 0
    x = np.where(rng.random(n) < 0.5 + 0.05 * z_effect, "a", "b")
    logit = 0.05 * z_effect + (0.9 * (x == "a") if dependent else 0.0)
    y = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "c", "d")

    df = pd.DataFrame({"x": x, "y": y})
    for j in range(cond_dims):
        df[f"z{j}"] = z[:, j]
    return df


SCENARIOS: list[tuple[str, Callable[[np.random.Generator], pd.DataFrame]]] = [
    ("dependent_one_z", lambda rng: _df(rng, dependent=True, cond_dims=1)),
    ("independent_one_z", lambda rng: _df(rng, dependent=False, cond_dims=1)),
    ("dependent_two_z", lambda rng: _df(rng, dependent=True, cond_dims=2)),
    ("independent_two_z", lambda rng: _df(rng, dependent=False, cond_dims=2)),
    ("dependent_unconditional", lambda rng: _df(rng, dependent=True, cond_dims=0)),
    ("independent_unconditional", lambda rng: _df(rng, dependent=False, cond_dims=0)),
]

N_SEEDS = 20  # pure Python on both sides, no R/rpy2 cost -- can afford more than Mixed-CI


@pytest.mark.parametrize("seed", range(N_SEEDS))
@pytest.mark.parametrize("name,build", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_python_matches_causallearn(
    name: str, build: Callable[[np.random.Generator], pd.DataFrame], seed: int
) -> None:
    """Swept over `N_SEEDS` draws, not just one: PearsonChi2 has no
    internal randomness of its own, so nothing stops requiring agreement
    with causal-learn on more than a single lucky/unlucky seed.
    """
    df = build(np.random.default_rng(seed))
    z_cols = [c for c in df.columns if c not in ("x", "y")]
    z = df[z_cols] if z_cols else None

    py = PearsonChi2().test(df["x"], df["y"], z)
    ref = pearson_chi2_test_causallearn(df["x"], df["y"], z)

    assert py.p_value == pytest.approx(ref.p_value)
    assert py.reject == ref.reject
