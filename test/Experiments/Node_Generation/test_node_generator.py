from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.CI_tests.Fisher_Z.fisher_z import FisherZ
from src.CI_tests.KCI.kci import KCI
from src.Experiments.Node_Generation import factory
from src.Experiments.Node_Generation.causal_models import (
    ADDITIVE_NOISE_MODELS,
    LINEAR_GAUSSIAN,
    LINEAR_NON_GAUSSIAN,
    LOCAL_SCALE_NOISE_MODELS,
    POST_NON_LINEAR,
    Model,
)
from src.Experiments.Node_Generation.factory import NonLinearShapeFactory
from src.Experiments.Node_Generation.node_generator import (
    MAX_TARGET_R2,
    MIN_PARENT_SHARE,
    MIN_TARGET_R2,
    compress_and_standardize,
    discretization,
    draw_function,
    evaluate_scale,
    generate_nominal,
    generate_root,
    get_mean_functions,
    get_scale_function,
    latent_to_variable,
    mean_coefficients,
    node_generator,
    node_generator_with_no_parents,
    noise_std,
    noise_std_at,
    noise_std_bounds,
    r2_to_position,
    scale_position,
    standardized_term,
)
from src.Experiments.Node_Generation.utils import (
    VariableType,
    apply_function,
    standardize_scale,
)


def _categorical(rng: np.random.Generator, n: int, categories: int) -> pd.Series:
    return pd.Series(pd.Categorical(rng.integers(0, categories, n), categories=range(categories)))


@pytest.mark.parametrize("seed", range(30))
def test_standardized_term_has_zero_mean_and_unit_variance_for_any_function(seed: int) -> None:
    rng = np.random.default_rng(seed)
    parent = pd.Series(rng.standard_normal(500) * 3)

    term = standardized_term(NonLinearShapeFactory.sample(rng), parent)

    assert term.mean() == pytest.approx(0.0, abs=1e-9)
    assert term.std() == pytest.approx(1.0)


@pytest.mark.parametrize("parent_kind", ["binary", "ordinal", "nominal"])
def test_standardized_term_handles_discrete_parents(parent_kind: str) -> None:
    rng = np.random.default_rng(0)
    n = 400
    function: Any
    if parent_kind == "binary":
        parent = pd.Series(rng.integers(0, 2, n))
        function = NonLinearShapeFactory.sample(rng)
    elif parent_kind == "ordinal":
        codes = rng.integers(0, 4, n)
        parent = pd.Series(pd.Categorical(codes, categories=range(4), ordered=True))
        function = NonLinearShapeFactory.sample(rng)
    else:
        parent = _categorical(rng, n, 3)
        function = ADDITIVE_NOISE_MODELS.categorical_mean_functions.sample(rng, 3)

    term = standardized_term(function, parent)

    assert term.mean() == pytest.approx(0.0, abs=1e-9)
    assert term.std() == pytest.approx(1.0)


SHAPES = [
    "identity_shape",
    "sine_shape",
    "cosine_shape",
    "tanh_shape",
    "square_shape",
    "cube_shape",
    "exponential_shape",
]


PARENT_LAWS: dict[str, Callable[[np.random.Generator, int], np.ndarray]] = {
    "gaussian": lambda rng, n: rng.standard_normal(n),
    "laplace": lambda rng, n: rng.laplace(size=n),
}


@pytest.mark.parametrize("law", PARENT_LAWS)
@pytest.mark.parametrize("shape", SHAPES)
def test_at_least_a_fifth_of_the_observations_carry_the_variance_of_a_term(
    shape: str, law: str
) -> None:
    """The variance of a standardized term is not taken from a handful of observations.

    The effective number of observations is ``(sum d)^2 / sum d^2`` with ``d`` the
    squared term; it equals n when all observations weigh the same and is close to 0
    when a few dominate. The parents have the noise laws of the generator, including
    the heavy-tailed ones, where the raw exponential, cube and square shapes have
    less than 0.1 n.
    """
    n = 3000
    fractions = []
    for seed in range(50):
        rng = np.random.default_rng(seed)
        parent = pd.Series(standardize_scale(PARENT_LAWS[law](rng, n)))
        squared = standardized_term(getattr(factory, shape)(rng), parent) ** 2
        fractions.append(squared.sum() ** 2 / (squared**2).sum() / n)

    assert np.median(fractions) >= 0.2


@pytest.mark.parametrize(
    "shape", ["identity_shape", "square_shape", "cube_shape", "exponential_shape"]
)
def test_a_standardized_term_is_detectable_whatever_its_shape(shape: str) -> None:
    """A term with explained variance 0.15 is detected by the KCI test for every shape.

    The noise is set so that the term explains 15 % of the variance of the child. With
    the tails of the term dominating the variance, the cube and exponential shapes are
    detected in only 40-60 % of the datasets at this sample size.
    """
    n, r2, repetitions = 300, 0.15, 30
    noise_std = np.sqrt((1 - r2) / r2)
    rejections = 0
    for seed in range(repetitions):
        rng = np.random.default_rng(7000 + seed)
        parent_values = standardize_scale(rng.standard_normal(n))
        term = standardized_term(getattr(factory, shape)(rng), pd.Series(parent_values))
        child = term + noise_std * rng.standard_normal(n)
        rejections += KCI().test(parent_values, child).reject

    assert rejections / repetitions >= 0.9


def test_standardized_term_rejects_a_constant_term() -> None:
    parent = pd.Series(np.ones(50), name="c")

    with pytest.raises(ValueError, match="constant"):
        standardized_term(lambda u: u, parent)


@pytest.mark.parametrize("n_parents", range(1, 13))
def test_mean_coefficients_share_the_variance_respecting_the_minimum(n_parents: int) -> None:
    coefficients = mean_coefficients(np.random.default_rng(n_parents), n_parents, n_outputs=200)
    shares = coefficients**2

    assert coefficients.shape == (200, n_parents)
    assert shares.sum(axis=1) == pytest.approx(np.ones(200))
    assert shares.min() >= min(MIN_PARENT_SHARE, 1 / n_parents) - 1e-12


def test_mean_coefficients_have_both_signs() -> None:
    coefficients = mean_coefficients(np.random.default_rng(3), 4, n_outputs=50)

    assert (coefficients > 0).any()
    assert (coefficients < 0).any()


def test_mean_of_independent_terms_has_unit_variance() -> None:
    rng = np.random.default_rng(4)
    terms = rng.standard_normal((50000, 3))

    mean_signal = terms @ mean_coefficients(rng, 3)[0]

    assert mean_signal.var() == pytest.approx(1.0, abs=0.03)


def test_node_generator_weakest_parent_explains_at_least_the_minimum_share() -> None:
    """With independent parents and a linear Gaussian model, each parent's regression
    contribution to the node is bounded below by its variance share.

    The contributions are estimated by least squares on a finite sample, so each one
    fluctuates a few percent around its true value; the tolerance covers that.
    """
    from src.Experiments.Node_Generation.causal_models import LINEAR_GAUSSIAN

    n = 20000
    weakest = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        data = pd.DataFrame({f"p{j}": rng.standard_normal(n) for j in range(3)})
        node = node_generator(
            rng, data, list(data.columns), LINEAR_GAUSSIAN, VariableType.CONTINUOUS
        )
        design = np.column_stack([np.ones(n), data.to_numpy()])
        coefficients = np.linalg.lstsq(design, node.to_numpy(), rcond=None)[0][1:]
        contributions = (coefficients**2) * data.to_numpy().var(axis=0)
        weakest.append(contributions.min() / contributions.sum())

    assert min(weakest) > MIN_PARENT_SHARE * 0.9


@pytest.mark.parametrize("law", ["gaussian", "laplace"])
@pytest.mark.parametrize("shape", ["identity_shape", "square_shape", "cube_shape"])
def test_the_weakest_allowed_edge_is_detected_by_a_kernel_test(shape: str, law: str) -> None:
    """The edge that explains the least variance the generator allows can still be detected.

    The child has three parents and the lowest R2 of the generator (0.4); the weakest
    parent carries the minimum share of the explained variance (10 %), which is about
    6 % of the variance left by the other two. The KCI test, given the other two
    parents, rejects independence in at least 80 % of the datasets at n = 600.
    """
    n, r2, repetitions = 600, 0.4, 15
    shares = np.array([MIN_PARENT_SHARE, 0.45, 0.45])
    rejections = 0
    for seed in range(repetitions):
        rng = np.random.default_rng(100 + seed)
        parents = [standardize_scale(PARENT_LAWS[law](rng, n)) for _ in shares]
        terms = [
            standardized_term(getattr(factory, shape)(rng), pd.Series(values)) for values in parents
        ]
        signal = sum(np.sqrt(w) * t for w, t in zip(shares, terms))
        child = np.sqrt(r2) * signal + np.sqrt(1 - r2) * rng.standard_normal(n)
        rejections += KCI().test(parents[0], child, np.column_stack(parents[1:])).reject

    assert rejections / repetitions >= 0.8


def _parents(rng: np.random.Generator, n: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "continuous": rng.standard_normal(n),
            "binary": rng.integers(0, 2, n),
            "ordinal": pd.Categorical(rng.integers(0, 4, n), categories=range(4), ordered=True),
            "nominal": pd.Categorical(rng.integers(0, 3, n), categories=range(3)),
        }
    )


@pytest.mark.parametrize("parent", ["continuous", "binary", "ordinal"])
def test_numeric_parents_get_a_shape_function(parent: str) -> None:
    rng = np.random.default_rng(0)
    data = _parents(rng)

    function: Any = draw_function(
        rng,
        data[parent],
        LOCAL_SCALE_NOISE_MODELS.mean_functions,
        LOCAL_SCALE_NOISE_MODELS.categorical_mean_functions,
    )

    assert function(np.linspace(-1.0, 1.0, 5)).shape == (5,)


def test_nominal_parents_get_a_lookup_table_with_one_value_per_category() -> None:
    rng = np.random.default_rng(0)
    data = _parents(rng)

    function = draw_function(
        rng,
        data["nominal"],
        LOCAL_SCALE_NOISE_MODELS.scale_functions,
        LOCAL_SCALE_NOISE_MODELS.categorical_scale_function,
    )

    values = function(np.array([0, 1, 2]))
    assert values.shape == (3,)
    assert (values > 0).all()
    with pytest.raises(IndexError):
        function(np.array([3]))


def test_get_mean_functions_returns_one_function_per_parent() -> None:
    rng = np.random.default_rng(1)
    data = _parents(rng)

    functions = get_mean_functions(rng, data, ["nominal", "continuous"], ADDITIVE_NOISE_MODELS)

    assert len(functions) == 2
    assert functions[0](np.array([0, 2])).shape == (2,)


def test_get_scale_function_handles_the_absence_of_a_scale_parent() -> None:
    rng = np.random.default_rng(2)
    data = _parents(rng)

    without_parent: Any = get_scale_function(rng, data, None, LOCAL_SCALE_NOISE_MODELS)
    with_parent: Any = get_scale_function(rng, data, "continuous", LOCAL_SCALE_NOISE_MODELS)

    assert without_parent(np.zeros(4)).shape == (4,)
    assert with_parent(np.linspace(-1.0, 1.0, 4)).shape == (4,)


def test_noise_std_bounds_are_the_noise_levels_of_the_extreme_r2() -> None:
    variance = 2.5

    lowest, highest = noise_std_bounds(variance)

    assert variance / (variance + lowest**2) == pytest.approx(MAX_TARGET_R2)
    assert variance / (variance + highest**2) == pytest.approx(MIN_TARGET_R2)


def test_scale_position_keeps_the_shape_of_the_scale_function() -> None:
    """Away from the clipped ends, the position is an affine function of the scale values."""
    rng = np.random.default_rng(0)
    scale_values = np.sqrt(1 + 0.4 * rng.standard_normal(3000) ** 2)

    position = scale_position(rng, scale_values)

    inside = (position > 0.0) & (position < 1.0)
    assert inside.sum() > 2000
    correlation = np.corrcoef(position[inside], scale_values[inside])[0, 1]
    assert abs(correlation) == pytest.approx(1.0)


def test_scale_position_maps_the_extreme_percentiles_to_the_ends() -> None:
    rng = np.random.default_rng(1)
    scale_values = rng.standard_normal(4000) ** 2

    position = scale_position(rng, scale_values)

    low, high = np.percentile(scale_values, [5, 95])
    assert set(np.unique(position[scale_values <= low])) <= {0.0, 1.0}
    assert set(np.unique(position[scale_values >= high])) <= {0.0, 1.0}
    assert position.min() == 0.0
    assert position.max() == 1.0


def test_scale_position_ignores_a_few_extreme_observations() -> None:
    rng = np.random.default_rng(2)
    scale_values = rng.uniform(1.0, 2.0, 2000)
    with_outlier = np.append(scale_values, 1e6)

    position = scale_position(np.random.default_rng(5), with_outlier)

    assert 0.3 < np.std(position[:-1]) < 0.4  # still spread over the range, not squeezed


def test_scale_position_of_a_two_valued_scale_uses_both_ends() -> None:
    rng = np.random.default_rng(3)
    for minority in (0.3, 0.02):
        scale_values = np.where(rng.random(2000) < minority, 2.0, 1.0)

        position = scale_position(rng, scale_values)

        assert set(np.unique(position)) == {0.0, 1.0}


def test_scale_position_keeps_the_order_of_the_scale_function_or_reverses_it() -> None:
    scale_values = np.random.default_rng(4).standard_normal(300)
    signs = {
        np.sign(
            np.corrcoef(scale_position(np.random.default_rng(seed), scale_values), scale_values)[
                0, 1
            ]
        )
        for seed in range(40)
    }

    assert signs == {-1.0, 1.0}


def test_scale_position_of_a_constant_scale_is_one_level_inside_the_range() -> None:
    for seed in range(200):
        position = scale_position(np.random.default_rng(seed), np.ones(50))

        assert np.ptp(position) == 0.0
        assert 0.0 <= position[0] <= 1.0


def test_r2_to_position_maps_the_extreme_r2_to_the_ends_of_the_range() -> None:
    assert r2_to_position(MAX_TARGET_R2) == pytest.approx(0.0)
    assert r2_to_position(MIN_TARGET_R2) == pytest.approx(1.0)


def test_r2_of_a_homoscedastic_node_is_uniform_over_the_band() -> None:
    """The constant noise level is placed so that the R2 of the node is uniform."""
    variance = 1.3
    r2 = []
    for seed in range(1000):
        rng = np.random.default_rng(seed)
        std = noise_std(rng, variance, np.ones(10))
        r2.append(variance / (variance + std[0] ** 2))

    p_value = stats.kstest(r2, "uniform", args=(MIN_TARGET_R2, MAX_TARGET_R2 - MIN_TARGET_R2))[1]
    assert p_value > 0.01


@pytest.mark.parametrize(
    "shape",
    [
        "sqrt_quadratic_scale",
        "periodic_scale",
        "piecewise_linear_scale",
        "abs_linear_scale",
        "unit_scale",
    ],
)
def test_noise_std_keeps_the_local_and_global_r2_inside_the_band(shape: str) -> None:
    variance = 1.7
    for seed in range(20):
        rng = np.random.default_rng(seed)
        parent = pd.Series(standardize_scale(rng.standard_normal(2000)))
        scale_values = apply_function(getattr(factory, shape)(rng), parent)

        std = noise_std(rng, variance, scale_values)

        local_r2 = variance / (variance + std**2)
        assert local_r2.min() >= MIN_TARGET_R2 - 1e-12
        assert local_r2.max() <= MAX_TARGET_R2 + 1e-12
        global_r2 = variance / (variance + np.mean(std**2))
        assert MIN_TARGET_R2 <= global_r2 <= MAX_TARGET_R2


def _linear_heteroscedastic_node(
    rng: np.random.Generator, n: int, scale_shape: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two parents that act on the mean linearly and a third one that acts on the noise scale."""
    x1, x2, scale_parent = (standardize_scale(rng.standard_normal(n)) for _ in range(3))
    terms = np.column_stack(
        [standardized_term(factory.identity_shape(rng), pd.Series(values)) for values in (x1, x2)]
    )
    coefficients = mean_coefficients(rng, 2)[0]
    mean = terms @ coefficients
    scale_values = apply_function(getattr(factory, scale_shape)(rng), pd.Series(scale_parent))
    child = mean + noise_std(rng, float(np.var(mean)), scale_values) * rng.standard_normal(n)
    return x1, x2, scale_parent, child, coefficients


@pytest.mark.parametrize(
    "scale_shape",
    ["sqrt_quadratic_scale", "periodic_scale", "piecewise_linear_scale", "abs_linear_scale"],
)
def test_heteroscedasticity_does_not_hide_the_mean_nor_the_scale_effect(scale_shape: str) -> None:
    """On a linear mean, both the weakest mean edge and the scale edge are detected.

    The mean parents share the variance with the minimum allowed to the weakest one, and
    the noise level moves over the whole detectable range along the scale parent. The
    weakest mean edge is tested with Fisher-Z given the other mean parent, and the scale
    edge with KCI between the scale parent and the absolute residual of a linear fit.
    """
    n, repetitions = 300, 20
    weak_edge = scale_edge = 0
    for seed in range(repetitions):
        rng = np.random.default_rng(200 + seed)
        x1, x2, scale_parent, child, coefficients = _linear_heteroscedastic_node(
            rng, n, scale_shape
        )
        parents = [x1, x2]
        weak = int(np.argmin(coefficients**2))
        weak_edge += FisherZ().test(parents[weak], child, parents[1 - weak].reshape(-1, 1)).reject

        design = np.column_stack([np.ones(n), x1, x2])
        residual = child - design @ np.linalg.lstsq(design, child, rcond=None)[0]
        scale_edge += KCI().test(scale_parent, np.abs(residual)).reject

    assert weak_edge / repetitions >= 0.9
    assert scale_edge / repetitions >= 0.9


def test_compress_and_standardize_gives_unit_variance_and_keeps_the_order() -> None:
    rng = np.random.default_rng(0)
    values = rng.exponential(size=1000) ** 3  # heavy right tail

    result = compress_and_standardize(values)

    assert result.mean() == pytest.approx(0.0, abs=1e-9)
    assert result.std() == pytest.approx(1.0)
    assert np.array_equal(np.argsort(result), np.argsort(values))


def test_compress_and_standardize_tames_the_tails() -> None:
    rng = np.random.default_rng(1)
    values = rng.exponential(size=5000) ** 3
    plain = (values - values.mean()) / values.std()

    result = compress_and_standardize(values)

    assert np.abs(result).max() < 0.5 * np.abs(plain).max()


def test_evaluate_scale_is_constant_without_a_scale_parent_or_with_a_constant_one() -> None:
    data = pd.DataFrame({"constant": np.full(40, 3.0), "other": np.arange(40.0)})

    without_parent = evaluate_scale(data, None, lambda u: 1 + np.abs(u))
    with_constant_parent = evaluate_scale(data, "constant", lambda u: 1 + np.abs(u))

    assert np.ptp(without_parent) == 0.0
    assert np.ptp(with_constant_parent) == pytest.approx(0.0)


def test_evaluate_scale_gives_the_function_a_standardized_argument() -> None:
    rng = np.random.default_rng(2)
    data = pd.DataFrame({"parent": 40 + 7 * rng.standard_normal(500)})

    values = evaluate_scale(data, "parent", lambda u: u)

    assert values.mean() == pytest.approx(0.0, abs=1e-9)
    assert values.std() == pytest.approx(1.0)


def test_evaluate_scale_does_not_leave_a_hinge_function_constant_for_an_offset_parent() -> None:
    """A hinge at 0 is constant if the whole parent is negative; standardizing avoids that."""
    rng = np.random.default_rng(3)
    data = pd.DataFrame({"parent": -50 + rng.uniform(0, 3, 300)})
    hinge = factory.piecewise_linear_scale(rng)

    assert np.ptp(hinge(data["parent"].to_numpy())) == 0.0
    assert np.ptp(evaluate_scale(data, "parent", hinge)) > 0.5


def test_evaluate_scale_reads_a_nominal_parent_as_category_codes() -> None:
    codes = np.array([0, 1, 2, 1, 0, 2])
    data = pd.DataFrame({"parent": pd.Categorical(codes, categories=range(3))})
    table = np.array([1.0, 2.0, 4.0])

    values = evaluate_scale(data, "parent", lambda c: table[c])

    assert values == pytest.approx(table[codes])


def test_evaluate_scale_reads_binary_and_ordinal_parents_as_numbers() -> None:
    rng = np.random.default_rng(4)
    data = pd.DataFrame(
        {
            "binary": rng.integers(0, 2, 300),
            "ordinal": pd.Categorical(rng.integers(0, 4, 300), categories=range(4), ordered=True),
        }
    )

    for parent in ("binary", "ordinal"):
        values = evaluate_scale(data, parent, lambda u: 1 + np.abs(u))

        assert values.shape == (300,)
        assert np.ptp(values) > 0.0


def test_discretization_codes_follow_the_order_of_the_latent() -> None:
    rng = np.random.default_rng(0)
    latent = rng.standard_normal(2000)

    codes = discretization(rng, 4, latent)

    assert set(np.unique(codes)) <= {0, 1, 2, 3}
    order = np.argsort(latent)
    assert (np.diff(codes[order]) >= 0).all()


def test_latent_to_variable_builds_each_target_type() -> None:
    rng = np.random.default_rng(1)
    latent = rng.standard_normal(500)
    index = pd.RangeIndex(10, 510)

    continuous = latent_to_variable(rng, latent, VariableType.CONTINUOUS, index)
    binary = latent_to_variable(rng, latent, VariableType.BINARY, index)
    ordinal = latent_to_variable(rng, latent, VariableType.ORDINAL, index)

    assert continuous.index.equals(index)
    assert np.abs(continuous).max() == pytest.approx(5.0)
    assert set(binary.unique()) == {0, 1}
    assert isinstance(ordinal.dtype, pd.CategoricalDtype)
    assert ordinal.dtype.ordered
    assert 3 <= len(ordinal.cat.categories) <= 5


def test_latent_to_variable_rejects_a_nominal_target() -> None:
    target: Any = VariableType.NOMINAL

    with pytest.raises(ValueError, match="Unexpected target"):
        latent_to_variable(np.random.default_rng(0), np.arange(5.0), target, pd.RangeIndex(5))


_MODELS = [
    pytest.param(LINEAR_GAUSSIAN, id="linear_gaussian"),
    pytest.param(LINEAR_NON_GAUSSIAN, id="linear_non_gaussian"),
    pytest.param(ADDITIVE_NOISE_MODELS, id="additive_noise"),
    pytest.param(POST_NON_LINEAR, id="post_nonlinear"),
    pytest.param(LOCAL_SCALE_NOISE_MODELS, id="local_scale"),
]


@pytest.mark.parametrize("target", list(VariableType))
@pytest.mark.parametrize("rules", _MODELS)
def test_node_generator_produces_a_valid_variable_for_any_model_and_parents(
    rules: Model, target: VariableType
) -> None:
    """Every model and target type works with any mix of parent types, without NaNs or constants."""
    kinds = list(VariableType)
    n = 200
    for seed in range(8):
        rng = np.random.default_rng(seed)
        parent_kinds = [kinds[i] for i in rng.integers(0, 4, size=int(rng.integers(1, 5)))]
        parents = pd.DataFrame(
            {f"p{j}": generate_root(rng, n, rules, kind) for j, kind in enumerate(parent_kinds)}
        )

        node = node_generator(rng, parents, list(parents.columns), rules, target)

        assert node.index.equals(parents.index)
        assert not node.isna().any()
        assert node.nunique() >= 2
        if target == VariableType.CONTINUOUS:
            assert np.isfinite(node.to_numpy(dtype=np.float64)).all()
        elif target == VariableType.BINARY:
            assert set(node.unique()) == {0, 1}
        else:
            assert isinstance(node.dtype, pd.CategoricalDtype)
            assert node.dtype.ordered == (target == VariableType.ORDINAL)
            assert 3 <= len(node.cat.categories) <= 5


def test_roots_follow_the_noise_of_the_model() -> None:
    """Gaussian models give Gaussian roots; only the linear non-Gaussian one gives Laplace roots."""
    rng = np.random.default_rng(0)

    gaussian = node_generator_with_no_parents(rng, 100000, LINEAR_GAUSSIAN, VariableType.CONTINUOUS)
    laplace = node_generator_with_no_parents(
        rng, 100000, LINEAR_NON_GAUSSIAN, VariableType.CONTINUOUS
    )

    assert stats.kurtosis(gaussian) == pytest.approx(0.0, abs=0.1)
    assert stats.kurtosis(laplace) == pytest.approx(3.0, abs=0.3)


def test_noise_std_at_reaches_the_bounds_of_the_band() -> None:
    variance = 1.6
    lowest, highest = noise_std_bounds(variance)

    std = noise_std_at(variance, np.array([0.0, 0.5, 1.0]))

    assert std == pytest.approx([lowest, (lowest + highest) / 2, highest])


def test_noise_std_is_noise_std_at_the_scale_position() -> None:
    scale_values = np.random.default_rng(0).standard_normal(200)
    position = scale_position(np.random.default_rng(1), scale_values)

    expected = noise_std_at(1.3, position)

    assert noise_std(np.random.default_rng(1), 1.3, scale_values) == pytest.approx(expected)


def test_standardized_term_without_compression_keeps_a_linear_function_linear() -> None:
    rng = np.random.default_rng(0)
    parent = pd.Series(standardize_scale(rng.standard_normal(500)))

    term = standardized_term(lambda u: u, parent, compress=False)

    assert np.corrcoef(parent, term)[0, 1] ** 2 == pytest.approx(1.0)
    assert term.mean() == pytest.approx(0.0, abs=1e-9)
    assert term.std() == pytest.approx(1.0)


def test_a_linear_gaussian_node_is_linear_for_a_partial_correlation_test() -> None:
    """Two linear Gaussian children of Z are conditionally independent given Z, and the
    partial correlation test rejects at about its level, not far above it.
    """
    n, repetitions = 300, 150
    rejections = 0
    for seed in range(repetitions):
        rng = np.random.default_rng(seed)
        z_values = standardize_scale(rng.standard_normal(n))
        z = pd.Series(z_values, name="z")
        data = pd.DataFrame({"z": z})
        x = node_generator(rng, data, ["z"], LINEAR_GAUSSIAN, VariableType.CONTINUOUS)
        y = node_generator(rng, data, ["z"], LINEAR_GAUSSIAN, VariableType.CONTINUOUS)
        given = np.asarray(z_values, dtype=np.float64).reshape(-1, 1)
        rejections += FisherZ().test(x.to_numpy(), y.to_numpy(), given).reject

    assert rejections / repetitions < 0.12


def _utilities_means(n: int, num_categories: int = 4) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.standard_normal((n, num_categories))


def test_generate_nominal_returns_valid_codes_that_follow_the_mean() -> None:
    n = 4000
    mean_signal = _utilities_means(n)

    codes = generate_nominal(np.random.default_rng(0), mean_signal, np.ones(n))

    assert set(np.unique(codes)) <= {0, 1, 2, 3}
    assert np.mean(codes == mean_signal.argmax(axis=1)) > 0.4  # chance is 0.25


def test_generate_nominal_is_more_predictable_where_the_noise_is_lower() -> None:
    n = 6000
    mean_signal = _utilities_means(n)
    scale_values = np.linspace(0.0, 1.0, n)

    codes = generate_nominal(np.random.default_rng(0), mean_signal, scale_values)

    agreement = codes == mean_signal.argmax(axis=1)
    first, second = agreement[: n // 3].mean(), agreement[-(n // 3) :].mean()
    assert abs(first - second) > 0.15  # the orientation of the scale is random


def test_generate_nominal_transmits_a_variance_only_dependence_with_constant_means() -> None:
    """With constant category means, a scale parent changes the distribution of the category."""
    n = 6000
    base = np.array([1.0, 0.3, -0.4, -0.9])
    mean_signal = np.tile(base, (n, 1))
    scale_values = np.linspace(0.0, 1.0, n)

    codes = generate_nominal(np.random.default_rng(0), mean_signal, scale_values)

    top_share_first = np.mean(codes[: n // 3] == 0)
    top_share_last = np.mean(codes[-(n // 3) :] == 0)
    assert abs(top_share_first - top_share_last) > 0.1


def test_a_nominal_node_without_parents_on_the_mean_is_never_constant() -> None:
    """A scale parent alone still gives a variable with several categories."""
    n = 400
    for seed in range(100):
        rng = np.random.default_rng(seed)
        data = pd.DataFrame({"scale_parent": standardize_scale(rng.standard_normal(n))})

        node = node_generator(
            rng, data, ["scale_parent"], LOCAL_SCALE_NOISE_MODELS, VariableType.NOMINAL
        )

        assert node.nunique() >= 2
        assert node.value_counts(normalize=True).max() < 0.95


def test_nominal_temperature_band_goes_from_almost_deterministic_to_almost_uniform() -> None:
    """With four equally spread utilities, the modal category has about 0.93 at the lowest
    temperature and about 0.37 at the highest, as the docstring of ``generate_nominal`` says.
    """
    utilities = np.array([1.3, 0.4, -0.4, -1.3])
    spread = utilities.std()
    modal = []
    for temperature in (spread / 3.0, spread * 3.0):
        logits = utilities / temperature
        modal.append(np.exp(logits).max() / np.exp(logits).sum())

    assert modal[0] == pytest.approx(0.93, abs=0.02)
    assert modal[1] == pytest.approx(0.37, abs=0.02)


@pytest.mark.parametrize("n_outputs", [2, 3, 4, 5, 6])
def test_a_single_parent_never_gets_identical_coefficients_for_all_categories(
    n_outputs: int,
) -> None:
    for seed in range(500):
        coefficients = mean_coefficients(np.random.default_rng(seed), 1, n_outputs)

        assert coefficients.min() < 0.0 < coefficients.max()
