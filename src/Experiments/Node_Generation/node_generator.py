from enum import StrEnum
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.Experiments.Node_Generation.causal_models import Model
from src.Experiments.Node_Generation.factory import (
    CategoricalFunction,
    GeneralNoiseFactory,
    HomoscedasticShapeFactory,
    ShapeFunction,
)


class VariableType(StrEnum):
    BINARY = "binary"
    CONTINUOUS = "continuous"
    ORDINAL = "ordinal"
    NOMINAL = "nominal"


def variable_type(series: pd.Series) -> VariableType:
    # dtype checked before nunique(): an ordinal/nominal series can realize
    # fewer categories than declared (small n, or an unlucky split leaving a
    # category empty) - expected, hard to prevent, and not an error. Checking
    # dtype first keeps it reported as its declared type instead of binary.
    if isinstance(series.dtype, pd.CategoricalDtype):
        return VariableType.ORDINAL if series.dtype.ordered else VariableType.NOMINAL
    elif series.nunique() == 2:
        return VariableType.BINARY
    return VariableType.CONTINUOUS


def split_scale_parent(
    rng: np.random.Generator, rules: Model, parents: list[str]
) -> tuple[str | None, list[str]]:
    if rules.scale_functions is HomoscedasticShapeFactory or not parents:
        return None, parents
    scale_parent = parents[rng.integers(len(parents))]
    mean_parents = [p for p in parents if p != scale_parent]
    return scale_parent, mean_parents


def mean_term_generator(
    rng: np.random.Generator, size: int | tuple[int, int] = 1
) -> NDArray[np.float64] | float:
    val = rng.uniform(1 / 3, 3, size=size)
    sign = rng.choice((-1.0, 1.0), size=size)
    return sign * val


# We need to normalize noise in terms of signal, because if noise variance
# is much bigger than signal ratio we are just adjusting noise.
# "A Scale-invariant Sorting criterion to find causal order aditive noise Models"
def adjust_signal_noise_ratio(
    rng: np.random.Generator,
    signal: NDArray[np.float64],
    raw_noise: NDArray[np.float64],
) -> NDArray[np.float64]:
    var_signal = np.var(signal)
    if var_signal == 0.0:
        return raw_noise

    # Target R2 kept moderate-to-high: low-R2 regimes are excluded on purpose,
    # no CI test can be expected to detect a relationship indistinguishable
    # from noise. Range is a placeholder, tune per domain later (physical
    # data tends to have higher R2 than biological data).
    target_r2 = rng.uniform(0.4, 0.95)
    target_snr = target_r2 / (1 - target_r2)

    var_noise = np.var(raw_noise)
    coeff = np.sqrt(var_signal / (target_snr * var_noise))

    return coeff * raw_noise


def categorical_num_terms(rng: np.random.Generator) -> int:
    return int(rng.integers(3, 7))


def standardize_scale(x: NDArray[np.float64], bound: float = 5.0) -> NDArray[np.float64]:
    max_abs = np.max(np.abs(x))
    if max_abs == 0.0:
        return x
    return (bound / max_abs) * x


def discretization(
    rng: np.random.Generator, num_bins: int, latent_j: NDArray[np.float64]
) -> NDArray[np.intp]:
    cuts = np.sort(rng.uniform(0, 1, size=num_bins - 1))
    quantile_levels = np.concatenate(([0.0], cuts, [1.0]))
    X_j = pd.qcut(latent_j, q=quantile_levels, labels=False)
    return X_j


NodeFunction = ShapeFunction | CategoricalFunction


def _apply(function: NodeFunction, parent_series: pd.Series) -> NDArray[np.float64]:
    if variable_type(parent_series) == VariableType.NOMINAL:
        codes = parent_series.cat.codes.to_numpy().astype(np.int64)
        return cast(CategoricalFunction, function)(codes)
    return cast(ShapeFunction, function)(parent_series.to_numpy())


def _no_parent_mean_signal(rng: np.random.Generator, rules: Model, n: int) -> NDArray[np.float64]:
    mean_function = rules.mean_functions.sample(rng)
    raw = mean_function(rng.uniform(-5.0, 5.0, size=n))
    return standardize_scale(raw, bound=0.3)


def _rescale_to_band(x: NDArray[np.float64], low: float, high: float) -> NDArray[np.float64]:
    xmin, xmax = x.min(), x.max()
    if xmax > xmin:
        return low + (high - low) * (x - xmin) / (xmax - xmin)
    return np.full_like(x, (low + high) / 2)


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _sample_categorical_rows(
    rng: np.random.Generator, probs: NDArray[np.float64]
) -> NDArray[np.intp]:
    cumulative = np.cumsum(probs, axis=1)
    u = rng.uniform(size=(probs.shape[0], 1))
    return (u > cumulative).sum(axis=1)


def _no_parent_nominal(
    rng: np.random.Generator, raw_scale: NDArray[np.float64], num_categories: int
) -> NDArray[np.intp]:
    base_utility = cast(NDArray[np.float64], mean_term_generator(rng, num_categories)) * 2.0
    temperature = _rescale_to_band(raw_scale, 0.05, 10.0)
    probs = _softmax(base_utility[None, :] / temperature[:, None])
    return _sample_categorical_rows(rng, probs)


def node_generator(
    rng: np.random.Generator,
    data: pd.DataFrame,
    parents: list[str],
    rules: Model,
    target_variable: VariableType,
) -> pd.Series:

    # Just one scale parent
    scale_parent, mean_parents = split_scale_parent(rng, rules, parents)

    # Get mean functions
    mean_functions: list[NodeFunction] = []
    for parent in mean_parents:
        parent_series = data[parent]
        kind_of_mean_parent = variable_type(parent_series)
        function: NodeFunction

        if kind_of_mean_parent == VariableType.NOMINAL:
            n_categories = parent_series.cat.categories.size
            function = rules.categorical_mean_functions.sample(rng, n_categories)
        elif (
            kind_of_mean_parent == VariableType.BINARY
            or kind_of_mean_parent == VariableType.ORDINAL
            or kind_of_mean_parent == VariableType.CONTINUOUS
        ):
            function = rules.mean_functions.sample(rng)
        else:
            raise ValueError(f"Unexpected kind_of_mean_parent: {kind_of_mean_parent}")

        mean_functions.append(function)

    scale_function: NodeFunction
    if scale_parent is None:
        scale_function = rules.scale_functions.sample(rng)
    else:
        parent_series = data[scale_parent]
        kind_of_scale_parent = variable_type(parent_series)

        if kind_of_scale_parent == VariableType.NOMINAL:
            n_categories = parent_series.cat.categories.size
            scale_function = rules.categorical_scale_function.sample(rng, n_categories)
        elif (
            kind_of_scale_parent == VariableType.BINARY
            or kind_of_scale_parent == VariableType.ORDINAL
            or kind_of_scale_parent == VariableType.CONTINUOUS
        ):
            scale_function = rules.scale_functions.sample(rng)
        else:
            raise ValueError(f"Unexpected kind_of_scale_parent: {kind_of_scale_parent}")

    noise_sampler = rules.noise_functions.sample(rng)
    transform_function = rules.transform_functions.sample(rng)

    # values for S_j(X_{i*}); if there is no scale_parent, this is a vector full of 1's
    raw_scale_j = _apply(
        scale_function,
        data[scale_parent]
        if scale_parent is not None
        else pd.Series(np.zeros(len(data)), index=data.index),
    )
    # Rescaled to a guaranteed band: the sampled gamma/omega in
    # HeteroscedasticShapeFactory can otherwise make raw_scale_j barely vary,
    # making the scale_parent's effect too weak to detect regardless of whether
    # there's also a real mean_parent. Collapses to a constant when scale_parent
    # is None (homoscedastic, or no scale parent at all) - no spurious effect.
    raw_scale = _rescale_to_band(raw_scale_j, 0.5, 2.5)

    if (
        target_variable == VariableType.CONTINUOUS
        or target_variable == VariableType.ORDINAL
        or target_variable == VariableType.BINARY
    ):
        noise_sample = noise_sampler(len(data))
        if mean_parents:
            raw_mean = np.column_stack(
                [_apply(f, data[p]) for p, f in zip(mean_parents, mean_functions)]
            )
            mean_coeffs = mean_term_generator(rng, len(mean_parents))
            mean_signal = (mean_coeffs * raw_mean).sum(axis=1)
            raw_noise = raw_scale * noise_sample
            noise_term = adjust_signal_noise_ratio(rng, mean_signal, raw_noise)
        else:
            # mean_signal here is just a small non-constant baseline, not a real
            # causal signal - calibrating noise's R2 against it would dilute the
            # A-dependent variance this branch exists to make detectable
            mean_signal = _no_parent_mean_signal(rng, rules, len(data))
            noise_term = raw_scale * noise_sample

        # bound before the transform, not after: mean_functions like
        # exponential_shape/cube_shape are unbounded and can reach into the
        # thousands, which saturates tanh_transform to exact ties or overflows
        # exponential_transform to inf (both break discretization's pd.qcut).
        # OuterTransformFactory's kappa is calibrated for a sane input scale.
        latent_j = transform_function(standardize_scale(mean_signal + noise_term))

        num_bins: int
        if target_variable == VariableType.ORDINAL:
            num_bins = categorical_num_terms(rng)
            X_j = discretization(rng, num_bins, latent_j)
            output = pd.Series(
                pd.Categorical(X_j, categories=range(num_bins), ordered=True), index=data.index
            )

        elif target_variable == VariableType.BINARY:
            num_bins = 2
            X_j = discretization(rng, num_bins, latent_j)
            output = pd.Series(X_j, index=data.index)

        elif target_variable == VariableType.CONTINUOUS:
            output = pd.Series(standardize_scale(latent_j), index=data.index)

    elif target_variable == VariableType.NOMINAL:
        num_categories = categorical_num_terms(rng)
        if mean_parents:
            raw_mean = np.column_stack(
                [_apply(f, data[p]) for p, f in zip(mean_parents, mean_functions)]
            )
            # one independent linear combination (utility) per category
            mean_coeffs = cast(
                NDArray[np.float64], mean_term_generator(rng, (num_categories, len(mean_parents)))
            )
            mean_signal = raw_mean @ mean_coeffs.T  # (n, num_categories)

            # argmax between categories dilutes the scale_parent's effect much
            # more than an additive combination does (a real mean_signal usually
            # already decides the winner); widen the band further here so the
            # noise has enough amplitude to still compete
            wide_raw_scale = _rescale_to_band(raw_scale, 0.1, 20.0)
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
