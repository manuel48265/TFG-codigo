"""Cases whose answer is known in advance, to catch implementation errors in the generator.

Each test builds data with the real ``node_generator`` and checks what the theory says
must happen: the conditional independences of a chain, a confounder and a collider, the
R2 of a linear node, the kurtosis of its noise, and the dependence of a variable on a
parent that acts only on its scale. A wrong construction (for example a nonlinear term in
a linear model) breaks one of them.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.CI_tests.Fisher_Z.fisher_z import FisherZ
from src.CI_tests.KCI.kci import KCI
from src.CI_tests.pearson_chi2.pearson_chi2 import pearson_chi2_test
from src.Experiments.Experimental_cases.common import discretize_continuous_columns
from src.Experiments.Node_Generation import factory
from src.Experiments.Node_Generation.causal_models import (
    ADDITIVE_NOISE_MODELS,
    LINEAR_GAUSSIAN,
    LINEAR_NON_GAUSSIAN,
    POST_NON_LINEAR,
    Model,
)
from src.Experiments.Node_Generation.node_generator import (
    MAX_TARGET_R2,
    MIN_TARGET_R2,
    node_generator,
)
from src.Experiments.Node_Generation.utils import VariableType, standardize_scale

CONTINUOUS = VariableType.CONTINUOUS
ALPHA_TOLERANCE = 0.13  # a test at level 0.05 rejects more than this very rarely in 40 runs


def _root(rng: np.random.Generator, n: int) -> pd.Series:
    return pd.Series(standardize_scale(rng.standard_normal(n)))


def _child(rng: np.random.Generator, rules: Model, parents: list[pd.Series]) -> pd.Series:
    data = pd.DataFrame({f"p{j}": p for j, p in enumerate(parents)})
    return node_generator(rng, data, list(data.columns), rules, CONTINUOUS)


def _structure(kind: str, rng: np.random.Generator, rules: Model, n: int) -> dict[str, np.ndarray]:
    """Three variables a, b, c with the structure ``kind``."""
    if kind == "chain":  # a -> b -> c
        a = _root(rng, n)
        b = _child(rng, rules, [a])
        c = _child(rng, rules, [b])
    elif kind == "confounder":  # a <- b -> c
        b = _root(rng, n)
        a = _child(rng, rules, [b])
        c = _child(rng, rules, [b])
    else:  # collider: a -> b <- c
        a, c = _root(rng, n), _root(rng, n)
        b = _child(rng, rules, [a, c])
    return {k: v.to_numpy(dtype=np.float64) for k, v in {"a": a, "b": b, "c": c}.items()}


def _rejection_rate(
    kind: str, rules: Model, test: str, given_b: bool, repetitions: int, n: int
) -> float:
    rejections = 0
    for seed in range(repetitions):
        v = _structure(kind, np.random.default_rng(seed), rules, n)
        z = v["b"].reshape(-1, 1) if given_b else None
        ci = FisherZ() if test == "fisher_z" else KCI()
        rejections += ci.test(v["a"], v["c"], z).reject
    return rejections / repetitions


# (structure, conditioning on b, must be independent)
_LINEAR_CASES = [
    ("chain", False, False),  # a and c are dependent
    ("chain", True, True),  # a _||_ c | b
    ("confounder", False, False),  # a and c are dependent
    ("confounder", True, True),  # a _||_ c | b
    ("collider", False, True),  # a _||_ c
    ("collider", True, False),  # conditioning on the collider makes them dependent
]


@pytest.mark.parametrize("kind,given_b,independent", _LINEAR_CASES)
def test_linear_gaussian_structures_follow_partial_correlation(
    kind: str, given_b: bool, independent: bool
) -> None:
    """A linear Gaussian model is exactly linear, so the partial correlation test is exact."""
    rate = _rejection_rate(kind, LINEAR_GAUSSIAN, "fisher_z", given_b, 60, 400)

    if independent:
        assert rate <= ALPHA_TOLERANCE
    else:
        assert rate >= 0.8


@pytest.mark.parametrize("rules", [ADDITIVE_NOISE_MODELS, POST_NON_LINEAR])
@pytest.mark.parametrize("kind,given_b,independent", _LINEAR_CASES[:4])
def test_nonlinear_structures_follow_the_kernel_test(
    rules: Model, kind: str, given_b: bool, independent: bool
) -> None:
    """Nonlinear and post-nonlinear models keep the conditional independences of the graph."""
    rate = _rejection_rate(kind, rules, "kci", given_b, 30, 300)

    if independent:
        assert rate <= ALPHA_TOLERANCE + 0.05
    else:
        assert rate >= 0.8


def test_the_r2_of_a_linear_node_is_inside_the_target_band() -> None:
    n = 4000
    r2 = []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        parent = _root(rng, n)
        child = _child(rng, LINEAR_GAUSSIAN, [parent])
        r2.append(np.corrcoef(parent, child)[0, 1] ** 2)

    assert min(r2) > MIN_TARGET_R2 - 0.03
    assert max(r2) < MAX_TARGET_R2 + 0.02
    assert np.ptp(r2) > 0.2  # the R2 varies from node to node


@pytest.mark.parametrize(
    "rules,expected_kurtosis", [(LINEAR_GAUSSIAN, 0.0), (LINEAR_NON_GAUSSIAN, 3.0)]
)
def test_the_noise_of_a_linear_node_has_the_kurtosis_of_the_model(
    rules: Model, expected_kurtosis: float
) -> None:
    n = 20000
    kurtosis = []
    for seed in range(15):
        rng = np.random.default_rng(seed)
        parent = _root(rng, n)
        child = _child(rng, rules, [parent])
        slope, intercept = np.polyfit(parent, child, 1)
        kurtosis.append(stats.kurtosis(child - slope * parent - intercept))

    assert np.median(kurtosis) == pytest.approx(expected_kurtosis, abs=0.4)


_LINEAR_HETEROSCEDASTIC = Model(
    factory.LinearShapeFactory,
    factory.CategoricalVariablesMeanFactory,
    factory.HeteroscedasticShapeFactory,
    factory.CategoricalVariablesScaleFactory,
    factory.IdentityTransformFactory,
    factory.GaussianNoiseFactory,
)


def test_a_scale_parent_changes_the_size_of_the_noise_and_a_mean_parent_does_not() -> None:
    """With a heteroscedastic model the only parent acts on the scale, so the size of the
    child depends on it; with a homoscedastic one the size of the residual does not.
    """
    n, repetitions = 300, 30
    hetero = homo = 0
    for seed in range(repetitions):
        rng = np.random.default_rng(seed)
        parent = _root(rng, n)
        scale_child = _child(rng, _LINEAR_HETEROSCEDASTIC, [parent])
        mean_child = _child(rng, LINEAR_GAUSSIAN, [parent])
        slope, intercept = np.polyfit(parent, mean_child, 1)
        residual = mean_child - slope * parent - intercept
        parent_values = parent.to_numpy(dtype=np.float64)
        hetero += (
            KCI().test(parent_values, np.abs(scale_child.to_numpy() - scale_child.median())).reject
        )
        homo += KCI().test(parent_values, np.abs(residual.to_numpy())).reject

    assert hetero / repetitions >= 0.9
    assert homo / repetitions <= ALPHA_TOLERANCE


def test_ordinal_and_nominal_nodes_have_at_most_five_categories() -> None:
    for seed in range(200):
        rng = np.random.default_rng(seed)
        data = pd.DataFrame({"p": _root(rng, 200)})
        for kind in (VariableType.ORDINAL, VariableType.NOMINAL):
            node = node_generator(rng, data, ["p"], LINEAR_GAUSSIAN, kind)

            assert 3 <= len(node.cat.categories) <= 5


def test_a_nominal_node_depends_on_its_parent_and_not_on_an_unrelated_variable() -> None:
    n, repetitions = 800, 30
    related = unrelated = 0
    for seed in range(repetitions):
        rng = np.random.default_rng(seed)
        parent, other = _root(rng, n), _root(rng, n)
        child = node_generator(
            rng, pd.DataFrame({"p": parent}), ["p"], LINEAR_GAUSSIAN, VariableType.NOMINAL
        )
        data = discretize_continuous_columns(
            pd.DataFrame(
                {"p": parent.to_numpy(dtype=np.float64), "o": other.to_numpy(dtype=np.float64)}
            )
        ).assign(y=child.to_numpy())
        related += pearson_chi2_test(data, "p", "y", []).reject
        unrelated += pearson_chi2_test(data, "o", "y", []).reject

    assert related / repetitions >= 0.9
    assert unrelated / repetitions <= ALPHA_TOLERANCE


def test_an_ordinal_node_is_monotonically_related_to_its_parent() -> None:
    n = 1000
    correlations = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        parent = _root(rng, n)
        child = node_generator(
            rng, pd.DataFrame({"p": parent}), ["p"], LINEAR_GAUSSIAN, VariableType.ORDINAL
        )
        correlations.append(abs(stats.spearmanr(parent, child.cat.codes)[0]))

    assert min(correlations) > 0.3
