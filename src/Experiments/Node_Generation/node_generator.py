"""Generation of the variables of a structural causal model, one node at a time.

A node is generated from its parents with the functions of a model family (see
``causal_models``). The pieces are designed so that every edge of the graph stays
detectable, which lets an experiment measure the behavior of the CI tests and not the
weakness of the generated effects:

- Each parent acts through a term standardized to unit variance, and the variance of the
  mean is shared among the parents with a minimum share for each of them
  (``standardized_term``, ``mean_coefficients``).
- The noise is set relative to the signal: its level stays inside the range that
  corresponds to an R2 between ``MIN_TARGET_R2`` and ``MAX_TARGET_R2``, and a parent that
  acts on the scale moves it inside that range following the shape of its function
  (``noise_std``).
- A continuous, binary or ordinal node is an outer transform of the standardized mean
  plus noise, converted to its type (``latent_to_variable``); a nominal node is drawn from
  a multinomial logit whose temperature plays the role of the noise level
  (``generate_nominal``).
- The noise is Gaussian, as in the literature on CI tests, except in the linear
  non-Gaussian model, where it is Laplace.
"""

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.Experiments.Node_Generation.causal_models import Model
from src.Experiments.Node_Generation.factory import (
    CategoricalFactory,
    FunctionFactory,
    LinearShapeFactory,
)
from src.Experiments.Node_Generation.utils import (
    NodeFunction,
    VariableType,
    apply_function,
    categorical_num_terms,
    mean_term_generator,
    robust_scale,
    split_scale_parent,
    standardize_scale,
    standardize_variance,
    variable_type,
)

# Band of the R2 of the true mean that the noise is calibrated to (see noise_std_bounds).
# Low-R2 regimes are excluded on purpose.
# TODO: the band is a provisional choice; tune it per domain (physical data tends to
# have higher R2 than biological data).
MIN_TARGET_R2 = 0.4
MAX_TARGET_R2 = 0.95


def noise_std_bounds(signal_variance: float) -> tuple[float, float]:
    """Return the noise standard deviations that give the maximum and the minimum target R2.

    For a signal of variance ``v``, noise of variance ``v (1 - R2) / R2`` makes the
    signal explain a share ``R2`` of the variance. The smallest noise compatible with
    the band of target R2 is the one for ``MAX_TARGET_R2`` and the largest is the one
    for ``MIN_TARGET_R2``.

    The noise has to be set relative to the signal. If its variance were much larger
    than the signal's, the node would be mostly noise and the dependence on its
    parents could not be told from it; the minimum R2 excludes that regime, since no
    CI test can be expected to detect a relationship that is indistinguishable from
    noise. The maximum keeps some noise in every node. The idea of normalizing the
    noise with respect to the signal comes from "A Scale-invariant Sorting criterion
    to find causal order additive noise Models".

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


def noise_std_at(signal_variance: float, position: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the noise standard deviation at given positions in the range of noise levels.

    The standard deviation moves linearly between the two bounds of
    ``noise_std_bounds``: position 0 gives the noise of ``MAX_TARGET_R2`` and position
    1 the noise of ``MIN_TARGET_R2``.

    Parameters
    ----------
    signal_variance : float
        Variance of the signal (mean) of the node.

    position : ndarray of shape (n,)
        Positions in ``[0, 1]``, as given by ``scale_position``.

    Returns
    -------
    std : ndarray of shape (n,)
        Noise standard deviation at each position.
    """
    lowest, highest = noise_std_bounds(signal_variance)
    return lowest + (highest - lowest) * position


def noise_std(
    rng: np.random.Generator, signal_variance: float, scale_values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the noise standard deviation of each observation of a node.

    The standard deviation moves between the two bounds of ``noise_std_bounds``
    following the position given by ``scale_position``. The local R2 of every
    observation therefore stays inside the band ``[MIN_TARGET_R2, MAX_TARGET_R2]``:
    heteroscedasticity changes how much noise each observation has, but never takes the
    node out of the range where its mean can be detected. This replaces choosing a fixed
    noise level (or an arbitrary range of scales) with moving inside the range of noise
    levels known to be detectable. The R2 of the node is not drawn: it follows from how
    the observations are placed in the range.

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
    return noise_std_at(signal_variance, scale_position(rng, scale_values))


# Range of the temperature of a nominal node, in units of the spread of the utilities
# of its categories: from spread / NOMINAL_TEMPERATURE_FACTOR to spread times it.
NOMINAL_TEMPERATURE_FACTOR = 3.0


def generate_nominal(
    rng: np.random.Generator,
    mean_signal: NDArray[np.float64],
    scale_values: NDArray[np.float64],
) -> NDArray[np.intp]:
    """Generate the categories of a nominal node from a multinomial logit with temperature.

    The probability of category ``k`` is the softmax of ``mean_signal[:, k] / T``, where
    the temperature ``T`` of each observation follows the scale function (through
    ``scale_position``, so it keeps the shape of the function): a scale parent makes the
    category more or less predictable, in the same way that it makes a continuous node
    more or less noisy. The categories are then sampled from those probabilities.

    The temperature ranges from ``s / NOMINAL_TEMPERATURE_FACTOR`` to ``s *
    NOMINAL_TEMPERATURE_FACTOR``, with ``s`` the spread (standard deviation across
    categories) of the utilities. With a factor of 3, in a node of four categories the
    most probable one has probability about 0.93 at the lowest temperature, where the
    category is almost determined by the parents, and about 0.37 at the highest one,
    where the categories are almost equally probable (0.25 would be exactly uniform).
    The band thus goes from almost deterministic to almost uniform, which is the widest
    change a scale parent can produce, and is measured in units of the utilities so it
    does not depend on their scale. The temperature moves geometrically in it, since it
    is a scale parameter.

    A band derived from the band of R2 of continuous nodes would be too narrow here: the
    categories would be dominated by the parents, and when no parent acts on the mean
    many nodes would be almost constant. The temperature is what lets a scale parent be
    detected, because a variable drawn by taking the largest of several noisy utilities
    changes little when the noise level changes.

    The outer transform of the model is not applied to nominal nodes: it would act on
    the logits of every category in the same way and would not change their ranking.

    Parameters
    ----------
    rng : Generator
        Source of randomness.

    mean_signal : ndarray of shape (n, n_categories)
        Utility of each category at each observation. When no parent acts on the mean,
        the same constant utilities for every observation.

    scale_values : ndarray of shape (n,)
        Scale function evaluated on the scale parent.

    Returns
    -------
    codes : ndarray of shape (n,)
        Category of each observation, from 0 to ``n_categories - 1``.
    """
    spread = float(mean_signal.std(axis=1).mean())
    position = scale_position(rng, scale_values)
    temperature = (spread / NOMINAL_TEMPERATURE_FACTOR) * NOMINAL_TEMPERATURE_FACTOR ** (
        2 * position
    )
    # Gumbel-max trick: the largest of the logits plus independent Gumbel noise is a
    # sample of the softmax of the logits.
    logits = mean_signal / temperature[:, None]
    return (logits + rng.gumbel(size=logits.shape)).argmax(axis=1)


def discretization(
    rng: np.random.Generator, num_bins: int, latent_j: NDArray[np.float64]
) -> NDArray[np.intp]:
    """Split a continuous variable into ordered classes at random quantiles.

    The cut levels are drawn uniformly, so the classes have random sizes; the codes
    follow the order of the variable.

    TODO: the classes can be very small (and, rarely, empty). A minimum share per class,
    like the one of the parents of a node, would avoid classes too rare to be detected.

    Parameters
    ----------
    rng : Generator
        Source of randomness.

    num_bins : int
        Number of classes.

    latent_j : ndarray of shape (n,)
        Variable to split.

    Returns
    -------
    codes : ndarray of shape (n,)
        Class of each observation, from 0 to ``num_bins - 1``.
    """
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


def standardized_term(
    function: NodeFunction, parent_series: pd.Series, compress: bool = True
) -> NDArray[np.float64]:
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

    The compression is nonlinear, so it makes even the identity function nonlinear. In
    a linear model that would break the model: a partial correlation test, which is exact
    for a linear Gaussian node, would reject conditional independence far more often
    than its level. With ``compress=False`` the term is only standardized and stays
    exactly linear if the function is.

    Parameters
    ----------
    function : callable
        Mean function of the parent: a shape function, or a lookup table over
        category codes if the parent is nominal.

    parent_series : Series
        Observations of the parent.

    compress : bool, optional (default: True)
        Whether to compress the tails before standardizing. It should be False for
        linear models.

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
    return compress_and_standardize(values) if compress else standardize_variance(values)


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

    With a single parent and several rows (the categories of a nominal node), each
    coefficient is ``+1`` or ``-1``. If all had the same sign the rows would be
    identical and the parent would not affect the node, so one sign is flipped in
    that case.

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
    signs = rng.choice((-1.0, 1.0), size=shares.shape)
    if n_parents == 1 and n_outputs > 1 and (signs == signs[0, 0]).all():
        signs[rng.integers(n_outputs), 0] *= -1.0
    return signs * np.sqrt(shares)


def _no_parent_mean_signal(rng: np.random.Generator, rules: Model, n: int) -> NDArray[np.float64]:
    """Return a small baseline with the shape of a mean function, for a node without mean parents.

    It is a mean function of the model applied to a uniform variable and bounded to a
    small range, so that a node whose only parent acts on the scale is not plain noise
    but has some shape. It is not a causal signal: no parent drives it.
    """
    mean_function = rules.mean_functions.sample(rng)
    raw = mean_function(rng.uniform(-5.0, 5.0, size=n))
    return standardize_scale(raw, bound=0.3)


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
    """Convert a continuous latent variable into a continuous, binary or ordinal variable.

    A continuous variable is the latent one scaled to a bounded range; a binary one is
    the latent variable split in two classes, and an ordinal one is split in 3 to 5
    ordered classes (see ``discretization``). Nominal variables are not built from a
    latent variable (see ``generate_nominal``).

    Parameters
    ----------
    rng : Generator
        Source of randomness.

    latent : ndarray of shape (n,)
        Latent variable.

    target : {VariableType.CONTINUOUS, VariableType.BINARY, VariableType.ORDINAL}
        Type of the resulting variable.

    index : Index
        Index of the resulting Series.

    Returns
    -------
    variable : Series
        The variable, with ``index``. Ordinal variables have dtype ``category`` with
        ordered categories.

    Raises
    ------
    ValueError
        If ``target`` is not one of the three types above.
    """
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
    """Generate a node from its parents.

    In a model with heteroscedastic noise, one parent chosen at random acts only on the
    scale of the noise and the others act on the mean (``split_scale_parent``).

    A continuous, binary or ordinal node is built in four steps. Each mean parent gives a
    term standardized to unit variance, and the terms are combined with coefficients that
    share the variance (``mean_coefficients``). The noise is drawn from the noise of the
    model and multiplied by ``noise_std``, which keeps it inside the detectable range.
    The sum is standardized to unit variance and goes through the outer transform of the
    model, which keeps the dependences (it is increasing) and makes the model
    non-additive. The result is converted to the type of the node (``latent_to_variable``).
    If no parent acts on the mean, the mean is a small baseline with the shape of a mean
    function (``_no_parent_mean_signal``) and the noise levels are set against a unit
    variance.

    A nominal node is drawn from a multinomial logit with temperature
    (``generate_nominal``). The utility of each category is a combination of the mean
    terms with its own coefficients; if no parent acts on the mean, the utilities are
    random constants, one per category, standardized to unit spread, so the temperature
    (and therefore the scale parent) alone changes the distribution of the category. The
    outer transform is not applied to nominal nodes.

    In the linear models the terms are only standardized, not compressed, so that a
    linear Gaussian node is exactly linear (see ``standardized_term``).

    Parameters
    ----------
    rng : Generator
        Source of randomness.

    data : DataFrame
        Observations of the variables generated so far, one column per variable.

    parents : list of str
        Names of the columns of ``data`` that are parents of the node. They may be empty.

    rules : Model
        Family of structural models the node belongs to.

    target_variable : VariableType
        Type of the node.

    Returns
    -------
    node : Series
        The node, with the index of ``data``. Ordinal nodes have ordered categories from
        3 to 5 and nominal nodes unordered categories from 3 to 5.

    Raises
    ------
    ValueError
        If ``target_variable`` is not a ``VariableType``, or if a mean parent has a
        constant term and therefore cannot affect the node.
    """
    # Just one scale parent
    scale_parent, mean_parents = split_scale_parent(rng, rules, parents)

    # Get functions
    mean_functions: list[NodeFunction] = get_mean_functions(rng, data, mean_parents, rules)
    scale_function: NodeFunction = get_scale_function(rng, data, scale_parent, rules)
    noise_sampler = rules.noise_functions.sample(rng)
    transform_function = rules.transform_functions.sample(rng)

    raw_scale_j = evaluate_scale(data, scale_parent, scale_function)
    # linear models keep exactly linear terms (see standardized_term)
    compress = rules.mean_functions is not LinearShapeFactory
    if (
        target_variable == VariableType.CONTINUOUS
        or target_variable == VariableType.ORDINAL
        or target_variable == VariableType.BINARY
    ):
        noise_sample = noise_sampler(len(data))
        if mean_parents:
            terms = np.column_stack(
                [
                    standardized_term(f, data[p], compress)
                    for p, f in zip(mean_parents, mean_functions)
                ]
            )
            mean_signal = terms @ mean_coefficients(rng, len(mean_parents))[0]
            signal_variance = float(np.var(mean_signal))
        else:
            # mean_signal here is just a small non-constant baseline, not a real
            # causal signal: the noise levels are set against a unit variance
            mean_signal = _no_parent_mean_signal(rng, rules, len(data))
            signal_variance = 1.0
        noise_term = noise_std(rng, signal_variance, raw_scale_j) * noise_sample

        # the outer transforms are calibrated for an input of unit variance (see the
        # kappa ranges in factory.py), so the node value is standardized before them
        latent_j = transform_function(standardize_variance(mean_signal + noise_term))
        output = latent_to_variable(rng, latent_j, target_variable, data.index)

    elif target_variable == VariableType.NOMINAL:
        num_categories = categorical_num_terms(rng)
        if mean_parents:
            terms = np.column_stack(
                [
                    standardized_term(f, data[p], compress)
                    for p, f in zip(mean_parents, mean_functions)
                ]
            )
            # one independent linear combination (utility) per category
            mean_signal = terms @ mean_coefficients(rng, len(mean_parents), num_categories).T
        else:
            base = standardize_variance(np.asarray(mean_term_generator(rng, num_categories)))
            mean_signal = np.tile(base, (len(data), 1))
        codes = generate_nominal(rng, mean_signal, raw_scale_j)
        output = pd.Series(
            pd.Categorical(codes, categories=range(num_categories), ordered=False),
            index=data.index,
        )

    else:
        raise ValueError(f"Unexpected target_variable: {target_variable}")

    return output


def node_generator_with_no_parents(
    rng: np.random.Generator,
    n: int,
    rules: Model,
    target_variable: VariableType,
) -> pd.Series:
    """Generate a root node as pure exogenous noise, without any function of parents.

    A continuous root is the noise of the model (Gaussian, or Laplace in the linear
    non-Gaussian model), scaled to a bounded range; a binary or ordinal root is that
    noise discretized; a nominal root has categories with random probabilities. This
    is the usual convention for the first variables of an additive-noise benchmark.

    Parameters
    ----------
    rng : Generator
        Source of randomness.

    n : int
        Number of observations.

    rules : Model
        Family of structural models; only its noise is used.

    target_variable : VariableType
        Type of the root.

    Returns
    -------
    root : Series of length n
        The root variable, with a default integer index.
    """
    if target_variable == VariableType.NOMINAL:
        num_categories = categorical_num_terms(rng)
        probs = rng.dirichlet(np.ones(num_categories))
        X_j = rng.choice(num_categories, size=n, p=probs)
        return pd.Series(pd.Categorical(X_j, categories=range(num_categories), ordered=False))

    noise_sample = rules.noise_functions.sample(rng)(n)

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
    """Generate a root node, either as pure noise or as a function of two dummy parents.

    With probability 1/2 the root is pure exogenous noise (``node_generator_with_no_parents``).
    Otherwise it is generated by ``node_generator`` from two independent standard normal
    variables that are not part of the data, so that the roots too follow the functions
    (shape, scale and transform) of the model family and are not all plain noise. The
    distribution of a root is therefore a mixture of the two.

    Parameters
    ----------
    rng : Generator
        Source of randomness.

    n : int
        Number of observations.

    rules : Model
        Family of structural models.

    target_variable : VariableType
        Type of the root.

    Returns
    -------
    root : Series of length n
        The root variable.
    """
    if rng.random() < 0.5:
        return node_generator_with_no_parents(rng, n, rules, target_variable)

    dummy_data = pd.DataFrame(
        {"__dummy_1": rng.standard_normal(n), "__dummy_2": rng.standard_normal(n)}
    )
    return node_generator(rng, dummy_data, ["__dummy_1", "__dummy_2"], rules, target_variable)
