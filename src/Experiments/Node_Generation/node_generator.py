from typing import Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.Experiments.Node_Generation.causal_models import Model
from src.Experiments.Node_Generation.factory import (
    CategoricalFactory,
    FunctionFactory,
    GeneralNoiseFactory,
)
from src.Experiments.Node_Generation.utils import (
    NodeFunction,
    VariableType,
    apply_function,
    categorical_num_terms,
    mean_term_generator,
    rescale_to_band,
    robust_scale,
    sample_categorical_rows,
    softmax,
    split_scale_parent,
    standardize_scale,
    variable_type,
)

# Band of the R2 of the true mean that the noise is calibrated to (see
# adjust_signal_noise_ratio). Low-R2 regimes are excluded on purpose.
# TODO: the band is a provisional choice; tune it per domain (physical data tends to
# have higher R2 than biological data).
MIN_TARGET_R2 = 0.4
MAX_TARGET_R2 = 0.95


def adjust_signal_noise_ratio(
    rng: np.random.Generator,
    signal: NDArray[np.float64],
    raw_noise: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Scale the noise so that the signal explains a random share of the variance of the node.

    A target R2 is drawn uniformly from ``[MIN_TARGET_R2, MAX_TARGET_R2]``, and the
    noise is multiplied by the constant that makes ``var(signal) / (var(signal) +
    var(noise))`` equal to it. That is the R2 of the true mean function, which a
    linear regression reaches only if the mean is linear. The shape of the noise,
    including any dependence of its scale on a parent, is kept: only its overall
    level changes. If the signal is constant, the noise is returned unchanged.

    The noise has to be set relative to the signal. If its variance were much larger
    than the signal's, the node would be mostly noise and the dependence on its
    parents could not be told from it; the minimum R2 excludes that regime, since no
    CI test can be expected to detect a relationship that is indistinguishable from
    noise. The maximum keeps some noise in every node. The idea of normalizing the
    noise with respect to the signal comes from "A Scale-invariant Sorting criterion
    to find causal order additive noise Models".

    Parameters
    ----------
    rng : Generator
        Source of randomness.

    signal : ndarray of shape (n,)
        Mean of the node.

    raw_noise : ndarray of shape (n,)
        Noise before calibration.

    Returns
    -------
    noise : ndarray of shape (n,)
        ``raw_noise`` multiplied by a positive constant.
    """
    var_signal = np.var(signal)
    if var_signal == 0.0:
        return raw_noise

    target_r2 = rng.uniform(MIN_TARGET_R2, MAX_TARGET_R2)
    target_snr = target_r2 / (1 - target_r2)

    var_noise = np.var(raw_noise)
    coeff = np.sqrt(var_signal / (target_snr * var_noise))

    return coeff * raw_noise


def noise_std_bounds(signal_variance: float) -> tuple[float, float]:
    """Return the noise standard deviations that give the maximum and the minimum target R2.

    For a signal of variance ``v``, noise of variance ``v (1 - R2) / R2`` makes the
    signal explain a share ``R2`` of the variance. The smallest noise compatible with
    the band of target R2 is the one for ``MAX_TARGET_R2`` and the largest is the one
    for ``MIN_TARGET_R2``.

    Parameters
    ----------
    signal_variance : float
        Variance of the signal (mean) of the node.

    Returns
    -------
    lowest : float
        Noise standard deviation for ``MAX_TARGET_R2``.

    highest : float
        Noise standard deviation for ``MIN_TARGET_R2``.
    """
    lowest = np.sqrt(signal_variance * (1 - MAX_TARGET_R2) / MAX_TARGET_R2)
    highest = np.sqrt(signal_variance * (1 - MIN_TARGET_R2) / MIN_TARGET_R2)
    return float(lowest), float(highest)


def r2_to_position(r2: float) -> float:
    """Return the position in the range of noise levels whose R2 is ``r2``.

    The noise standard deviation that gives an R2 is proportional to
    ``sqrt((1 - R2) / R2)``, whatever the variance of the signal. Position 0 is the
    noise for ``MAX_TARGET_R2`` and position 1 the noise for ``MIN_TARGET_R2``, with
    the noise standard deviation varying linearly in between (see ``noise_std``).

    Parameters
    ----------
    r2 : float
        R2 in ``[MIN_TARGET_R2, MAX_TARGET_R2]``.

    Returns
    -------
    position : float
        Position in ``[0, 1]``.
    """

    def level(value: float) -> float:
        return float(np.sqrt((1 - value) / value))

    lowest, highest = level(MAX_TARGET_R2), level(MIN_TARGET_R2)
    return (level(r2) - lowest) / (highest - lowest)


def scale_position(
    rng: np.random.Generator, scale_values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return where each observation lies, between 0 and 1, in the range of the noise level.

    The scale function is mapped linearly onto the range, so the noise follows its
    shape: its 5th percentile goes to 0 and its 95th to 1, and values beyond are
    clipped. Percentiles are used instead of the minimum and maximum so that a few
    extreme observations do not squeeze the rest of the function into a corner; when
    they coincide (a two-valued parent with a rare category), the minimum and the
    maximum are used. Whether the largest value of the function is mapped to the
    highest noise level or to the lowest is drawn at random, which adds variability
    without leaving the range. If the scale function is constant there is no
    heteroscedasticity and no shape to follow, so all observations share one position,
    chosen so that the R2 of the node is drawn uniformly from ``[MIN_TARGET_R2,
    MAX_TARGET_R2]``: a homoscedastic node has a constant noise level somewhere inside
    the detectable range, different from one node to another.

    Parameters
    ----------
    rng : Generator
        Source of randomness.

    scale_values : ndarray of shape (n,)
        Scale function evaluated on the scale parent.

    Returns
    -------
    position : ndarray of shape (n,)
        Values in ``[0, 1]``, an increasing or decreasing affine function of
        ``scale_values`` away from the clipped ends.
    """
    flip = rng.random() < 0.5
    low, high = np.percentile(scale_values, [5, 95])
    if low == high:
        low, high = scale_values.min(), scale_values.max()
    if low == high:
        target_r2 = rng.uniform(MIN_TARGET_R2, MAX_TARGET_R2)
        return np.full(len(scale_values), r2_to_position(target_r2))
    position = np.clip((scale_values - low) / (high - low), 0.0, 1.0)
    return 1.0 - position if flip else position


def noise_std(
    rng: np.random.Generator, signal_variance: float, scale_values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the noise standard deviation of each observation of a node.

    The standard deviation moves linearly between the two bounds of
    ``noise_std_bounds``, following the position given by ``scale_position``. The local R2
    of every observation therefore stays inside the band ``[MIN_TARGET_R2,
    MAX_TARGET_R2]``: heteroscedasticity changes how much noise each observation
    has, but never takes the node out of the range where its mean can be detected.
    This replaces choosing a fixed noise level (or an arbitrary range of scales) with
    moving inside the range of noise levels known to be detectable. The R2 of the
    node is not drawn: it follows from how the observations are placed in the range.

    Parameters
    ----------
    rng : Generator
        Source of randomness.

    signal_variance : float
        Variance of the signal (mean) of the node.

    scale_values : ndarray of shape (n,)
        Scale function evaluated on the scale parent.

    Returns
    -------
    std : ndarray of shape (n,)
        Noise standard deviation of each observation.
    """
    lowest, highest = noise_std_bounds(signal_variance)
    position = scale_position(rng, scale_values)
    return lowest + (highest - lowest) * position


def discretization(
    rng: np.random.Generator, num_bins: int, latent_j: NDArray[np.float64]
) -> NDArray[np.intp]:
    cuts = np.sort(rng.uniform(0, 1, size=num_bins - 1))
    quantile_levels = np.concatenate(([0.0], cuts, [1.0]))
    X_j = pd.qcut(latent_j, q=quantile_levels, labels=False)
    return X_j


# Gain of the compression of ``standardized_term``: how many robust scales a value
# has to be away from the median for the compression to start acting.
COMPRESSION_GAIN = 2.0


def compress_and_standardize(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compress the tails of ``values`` with ``arcsinh`` and scale them to unit variance.

    The values are centered at their median and divided by a robust scale, so the bulk
    of the data sits at scale 1 whatever the tails are; ``arcsinh`` of
    ``COMPRESSION_GAIN`` times that is then taken and the result is centered and
    divided by its standard deviation. ``arcsinh`` is an invertible, monotone map that
    is close to the identity for small values and logarithmic for large ones, so it
    keeps the shape of the data where most of it lies and stops a few extreme values
    from dominating the variance or from squeezing the rest into a corner.

    Parameters
    ----------
    values : ndarray of shape (n,)
        Values to compress; they must not be constant.

    Returns
    -------
    compressed : ndarray of shape (n,)
        Values with mean 0 and variance 1.
    """
    compressed = np.arcsinh(COMPRESSION_GAIN * (values - np.median(values)) / robust_scale(values))
    return (compressed - compressed.mean()) / compressed.std()


def standardized_term(function: NodeFunction, parent_series: pd.Series) -> NDArray[np.float64]:
    """Return ``function`` applied to a parent, with its tails compressed, standardized.

    The mean of a node is a sum with one term per parent, and the terms must weigh
    according to their coefficient and not to their function. Two things prevent
    that if the raw values are used:

    - The scales differ: an exponential has far more variance than a sine, and a
      0/1 parent far less than a continuous one.
    - A function that grows fast (exponential, cube, square) takes its variance
      from the few observations in the tails of the parent, where it is large.
      Dividing by the standard deviation does not change that: most observations
      end up with a term close to zero, and the dependence on the parent is only
      visible in a handful of points.

    The second problem is solved by compressing the tails before standardizing,
    with ``compress_and_standardize``: the shape of the function is kept where most
    data lie and the tails stop dominating. The standardization, which is part of
    that function, takes care of the first problem.

    Parameters
    ----------
    function : callable
        Mean function of the parent: a shape function, or a lookup table over
        category codes if the parent is nominal.

    parent_series : Series
        Observations of the parent.

    Returns
    -------
    term : ndarray of shape (n,)
        Term with mean 0 and variance 1.

    Raises
    ------
    ValueError
        If the term is constant, since the parent would then not affect the node.
    """
    values = apply_function(function, parent_series)
    if np.ptp(values) == 0.0:
        raise ValueError(
            f"Parent '{parent_series.name}' has a constant term and cannot affect the node."
        )
    return compress_and_standardize(values)


# Smallest share of the explained variance that each parent carries (see mean_coefficients).
MIN_PARENT_SHARE = 0.10


def mean_coefficients(
    rng: np.random.Generator, n_parents: int, n_outputs: int = 1
) -> NDArray[np.float64]:
    """Draw the coefficients that combine the standardized terms into the mean.

    The squared coefficients of a row are the shares of the explained variance
    carried by each parent. They sum to 1, so the mean has variance 1 when the
    terms are uncorrelated. Every parent gets at least ``min(MIN_PARENT_SHARE,
    1 / n_parents)``, the rest is split at random (uniformly over the allowed
    shares), and the signs are random.

    A parent whose term explains only a tiny part of the variance cannot be told
    apart from the residual variance by a CI test at the sample sizes of the
    experiments, so the edge would exist in the graph but not in the data. The
    minimum share keeps every edge detectable, which is what lets the
    experiments measure the behavior of the tests and not the weakness of the
    generated effects. The bound ``1 / n_parents`` keeps the minimum feasible
    for nodes with more than ten parents, where it forces equal shares.

    Parameters
    ----------
    rng : Generator
        Source of randomness.

    n_parents : int
        Number of parents that contribute to the mean.

    n_outputs : int, optional (default: 1)
        Number of independent coefficient rows (one per category of a nominal node).

    Returns
    -------
    coefficients : ndarray of shape (n_outputs, n_parents)
        Signed coefficients.
    """
    floor = min(MIN_PARENT_SHARE, 1.0 / n_parents)
    free = 1.0 - n_parents * floor
    shares = floor + free * rng.dirichlet(np.ones(n_parents), size=n_outputs)
    return rng.choice((-1.0, 1.0), size=shares.shape) * np.sqrt(shares)


def _no_parent_mean_signal(rng: np.random.Generator, rules: Model, n: int) -> NDArray[np.float64]:
    mean_function = rules.mean_functions.sample(rng)
    raw = mean_function(rng.uniform(-5.0, 5.0, size=n))
    return standardize_scale(raw, bound=0.3)


def _no_parent_nominal(
    rng: np.random.Generator, raw_scale: NDArray[np.float64], num_categories: int
) -> NDArray[np.intp]:
    base_utility = cast(NDArray[np.float64], mean_term_generator(rng, num_categories)) * 2.0
    temperature = rescale_to_band(raw_scale, 0.05, 10.0)
    probs = softmax(base_utility[None, :] / temperature[:, None])
    return sample_categorical_rows(rng, probs)


def draw_function(
    rng: np.random.Generator,
    parent_series: pd.Series,
    continuous_factory: type[FunctionFactory],
    categorical_factory: type[CategoricalFactory],
) -> NodeFunction:
    """Draw the function that a parent applies, according to the type of the parent.

    A nominal parent has no order, so its function is a lookup table with one value
    per category, drawn by ``categorical_factory``. Any other parent (continuous,
    binary or ordinal) is numeric and gets a shape function from
    ``continuous_factory``. The same rule serves the mean and the scale, which differ
    only in the factories.

    Parameters
    ----------
    rng : Generator
        Source of randomness.

    parent_series : Series
        Observations of the parent.

    continuous_factory : type of FunctionFactory
        Factory of the functions of numeric parents.

    categorical_factory : type of CategoricalFactory
        Factory of the lookup tables of nominal parents.

    Returns
    -------
    function : callable
        Shape function, or lookup table over category codes for a nominal parent.
    """
    if variable_type(parent_series) == VariableType.NOMINAL:
        return categorical_factory.sample(rng, parent_series.cat.categories.size)
    return continuous_factory.sample(rng)


def get_mean_functions(
    rng: np.random.Generator, data: pd.DataFrame, mean_parents: list[str], rules: Model
) -> list[NodeFunction]:
    """Draw the mean function of each mean parent, in the order of ``mean_parents``."""
    return [
        draw_function(rng, data[parent], rules.mean_functions, rules.categorical_mean_functions)
        for parent in mean_parents
    ]


def get_scale_function(
    rng: np.random.Generator, data: pd.DataFrame, scale_parent: str | None, rules: Model
) -> NodeFunction:
    """Draw the scale function of the node.

    Without a scale parent, the function is drawn from the scale factory of the model;
    it is then evaluated on a constant, so the scale does not vary.
    """
    if scale_parent is None:
        return rules.scale_functions.sample(rng)
    return draw_function(
        rng, data[scale_parent], rules.scale_functions, rules.categorical_scale_function
    )


def latent_to_variable(
    rng: np.random.Generator,
    latent: NDArray[np.float64],
    target: Literal[VariableType.CONTINUOUS, VariableType.ORDINAL, VariableType.BINARY],
    index: pd.Index,
) -> pd.Series:
    if target == VariableType.ORDINAL:
        num_bins = categorical_num_terms(rng)
        codes = discretization(rng, num_bins, latent)
        return pd.Series(
            pd.Categorical(codes, categories=range(num_bins), ordered=True), index=index
        )
    if target == VariableType.BINARY:
        return pd.Series(discretization(rng, 2, latent), index=index)
    if target == VariableType.CONTINUOUS:
        return pd.Series(standardize_scale(latent), index=index)
    raise ValueError(f"Unexpected target: {target}")


def evaluate_scale(
    data: pd.DataFrame, scale_parent: str | None, scale_function: NodeFunction
) -> NDArray[np.float64]:
    """Evaluate the scale function of a node on its scale parent.

    A numeric scale parent is first compressed and standardized with
    ``compress_and_standardize``. The scale functions are written for an argument with
    the bulk of the data at scale 1 and centered at 0, where their shape (a hinge, a V,
    a period) acts on most observations; the raw parent has an arbitrary location and
    scale, and its tails can leave most of the data on one side of the hinge or in the
    flat part of the function, or even make the function constant without warning. A
    nominal parent is not transformed, since its function is a lookup table over the
    category codes. Without a scale parent, or with a constant one, the function is
    evaluated on a constant and the scale does not vary.

    Parameters
    ----------
    data : DataFrame
        Observations of the parents of the node.

    scale_parent : str or None
        Name of the scale parent, or None if there is none.

    scale_function : callable
        Scale function: a shape function, or a lookup table over category codes if
        the scale parent is nominal.

    Returns
    -------
    values : ndarray of shape (n,)
        Scale function evaluated at each observation.
    """
    if scale_parent is None:
        return apply_function(scale_function, pd.Series(np.zeros(len(data)), index=data.index))

    parent = data[scale_parent]
    if variable_type(parent) != VariableType.NOMINAL:
        numbers = np.asarray(parent, dtype=np.float64)
        if np.ptp(numbers) > 0.0:
            parent = pd.Series(compress_and_standardize(numbers), index=parent.index)
    return apply_function(scale_function, parent)


def node_generator(
    rng: np.random.Generator,
    data: pd.DataFrame,
    parents: list[str],
    rules: Model,
    target_variable: VariableType,
) -> pd.Series:

    # Just one scale parent
    scale_parent, mean_parents = split_scale_parent(rng, rules, parents)

    # Get functions
    mean_functions: list[NodeFunction] = get_mean_functions(rng, data, mean_parents, rules)
    scale_function: NodeFunction = get_scale_function(rng, data, scale_parent, rules)
    noise_sampler = rules.noise_functions.sample(rng)
    transform_function = rules.transform_functions.sample(rng)

    raw_scale_j = evaluate_scale(data, scale_parent, scale_function)
    if (
        target_variable == VariableType.CONTINUOUS
        or target_variable == VariableType.ORDINAL
        or target_variable == VariableType.BINARY
    ):
        noise_sample = noise_sampler(len(data))
        if mean_parents:
            terms = np.column_stack(
                [standardized_term(f, data[p]) for p, f in zip(mean_parents, mean_functions)]
            )
            mean_signal = terms @ mean_coefficients(rng, len(mean_parents))[0]
            signal_variance = float(np.var(mean_signal))
        else:
            # mean_signal here is just a small non-constant baseline, not a real
            # causal signal: the noise levels are set against a unit variance
            mean_signal = _no_parent_mean_signal(rng, rules, len(data))
            signal_variance = 1.0
        noise_term = noise_std(rng, signal_variance, raw_scale_j) * noise_sample

        # bound before the transform, not after: mean_functions like
        # exponential_shape/cube_shape are unbounded and can reach into the
        # thousands, which saturates tanh_transform to exact ties or overflows
        # exponential_transform to inf (both break discretization's pd.qcut).
        # OuterTransformFactory's kappa is calibrated for a sane input scale.
        latent_j = transform_function(standardize_scale(mean_signal + noise_term))
        output = latent_to_variable(rng, latent_j, target_variable, data.index)

    elif target_variable == VariableType.NOMINAL:
        # Rescaled to a guaranteed band: the sampled gamma/omega in
        # HeteroscedasticShapeFactory can otherwise make raw_scale_j barely vary,
        # making the scale_parent's effect too weak to detect. Collapses to a
        # constant when there is no scale parent - no spurious effect.
        raw_scale = rescale_to_band(raw_scale_j, 0.5, 2.5)
        num_categories = categorical_num_terms(rng)
        if mean_parents:
            terms = np.column_stack(
                [standardized_term(f, data[p]) for p, f in zip(mean_parents, mean_functions)]
            )
            # one independent linear combination (utility) per category
            mean_signal = terms @ mean_coefficients(rng, len(mean_parents), num_categories).T

            # argmax between categories dilutes the scale_parent's effect much
            # more than an additive combination does (a real mean_signal usually
            # already decides the winner); widen the band further here so the
            # noise has enough amplitude to still compete
            wide_raw_scale = rescale_to_band(raw_scale, 0.1, 20.0)
            utilities = np.empty((len(data), num_categories))
            for k in range(num_categories):
                # independent noise draw per category: a shared draw correlates
                # utilities across categories and mostly cancels out in the argmax
                raw_noise = wide_raw_scale * noise_sampler(len(data))
                noise_term = adjust_signal_noise_ratio(rng, mean_signal[:, k], raw_noise)
                # bound before the transform - see the comment on the same
                # pattern above for why (unbounded mean_functions + saturating
                # transforms otherwise overflow or tie multiple categories)
                utilities[:, k] = transform_function(
                    standardize_scale(mean_signal[:, k] + noise_term)
                )

            X_j = utilities.argmax(axis=1)
        else:
            # no real parent drives the mean (the sole parent was picked as
            # scale_parent): heteroscedastic multinomial logit instead of
            # argmax-with-shared-noise, which structurally can't transmit a
            # variance-only dependency (see _no_parent_nominal)
            X_j = _no_parent_nominal(rng, raw_scale, num_categories)

        output = pd.Series(
            pd.Categorical(X_j, categories=range(num_categories), ordered=False), index=data.index
        )

    else:
        raise ValueError(f"Unexpected target_variable: {target_variable}")

    return output


def node_generator_with_no_parents(
    rng: np.random.Generator,
    n: int,
    target_variable: VariableType,
) -> pd.Series:
    # Root node: no parents to condition a mean/scale/transform on, so it's just
    # the exogenous noise term (standard ANM benchmark convention, e.g. NOTEARS).
    if target_variable == VariableType.NOMINAL:
        num_categories = categorical_num_terms(rng)
        probs = rng.dirichlet(np.ones(num_categories))
        X_j = rng.choice(num_categories, size=n, p=probs)
        return pd.Series(pd.Categorical(X_j, categories=range(num_categories), ordered=False))

    noise_sample = GeneralNoiseFactory.sample(rng)(n)

    if target_variable == VariableType.CONTINUOUS:
        return pd.Series(standardize_scale(noise_sample))

    # pd.qcut is scale-invariant, so binary/ordinal discretize the raw noise directly.
    num_bins = categorical_num_terms(rng) if target_variable == VariableType.ORDINAL else 2
    X_j = discretization(rng, num_bins, noise_sample)
    if target_variable == VariableType.ORDINAL:
        return pd.Series(pd.Categorical(X_j, categories=range(num_bins), ordered=True))
    return pd.Series(X_j)


def generate_root(
    rng: np.random.Generator,
    n: int,
    rules: Model,
    target_variable: VariableType,
) -> pd.Series:
    if rng.random() < 0.5:
        return node_generator_with_no_parents(rng, n, target_variable)

    dummy_data = pd.DataFrame(
        {"__dummy_1": rng.standard_normal(n), "__dummy_2": rng.standard_normal(n)}
    )
    return node_generator(rng, dummy_data, ["__dummy_1", "__dummy_2"], rules, target_variable)
