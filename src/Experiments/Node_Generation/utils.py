from enum import StrEnum
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.Experiments.Node_Generation.causal_models import Model
from src.Experiments.Node_Generation.factory import (
    CategoricalFunction,
    HomoscedasticShapeFactory,
    ShapeFunction,
)

NodeFunction = ShapeFunction | CategoricalFunction


class VariableType(StrEnum):
    """Type of a variable, which decides how it is generated and how it enters a mean."""

    BINARY = "binary"
    CONTINUOUS = "continuous"
    ORDINAL = "ordinal"
    NOMINAL = "nominal"


def variable_type(series: pd.Series) -> VariableType:
    """Return the ``VariableType`` of a column.

    A ``category`` column is ordinal if ordered and nominal otherwise, even when
    some of its categories are unobserved; any other column with exactly two
    distinct values is binary, and the rest are continuous.

    Parameters
    ----------
    series : Series
        Observations of the variable.

    Returns
    -------
    kind : VariableType
        Type of the variable.
    """
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
    """Pick the parent that drives the noise scale, and return the others as mean parents.

    With a homoscedastic model, or without parents, no parent is picked. Otherwise
    one parent, chosen at random, acts only on the scale of the noise and is
    excluded from the mean.

    Parameters
    ----------
    rng : Generator
        Source of randomness.

    rules : Model
        Family of structural models the node is generated from.

    parents : list of str
        Names of the parents of the node.

    Returns
    -------
    scale_parent : str or None
        Name of the scale parent, or None if there is none.

    mean_parents : list of str
        Names of the parents that enter the mean.
    """
    if rules.scale_functions is HomoscedasticShapeFactory or not parents:
        return None, parents
    scale_parent = parents[rng.integers(len(parents))]
    mean_parents = [p for p in parents if p != scale_parent]
    return scale_parent, mean_parents


def mean_term_generator(
    rng: np.random.Generator, size: int | tuple[int, int] = 1
) -> NDArray[np.float64] | float:
    """Draw signed values with magnitude in ``[1/3, 3]``.

    Parameters
    ----------
    rng : Generator
        Source of randomness.

    size : int or tuple of int, optional (default: 1)
        Shape of the draw.

    Returns
    -------
    values : ndarray or float
        Random signed values.
    """
    val = rng.uniform(1 / 3, 3, size=size)
    sign = rng.choice((-1.0, 1.0), size=size)
    return sign * val


def categorical_num_terms(rng: np.random.Generator) -> int:
    """Draw the number of categories of an ordinal or nominal variable, between 3 and 6."""
    return int(rng.integers(3, 7))


def standardize_scale(x: NDArray[np.float64], bound: float = 5.0) -> NDArray[np.float64]:
    """Scale ``x`` linearly so that its largest absolute value equals ``bound``.

    It changes the scale but not the shape of ``x``, so it bounds a variable without
    altering how its values are distributed. A constant-zero ``x`` is returned as is.

    Parameters
    ----------
    x : ndarray
        Values to scale.

    bound : float, optional (default: 5.0)
        Largest absolute value after scaling.

    Returns
    -------
    scaled : ndarray
        Scaled values, with the same shape as ``x``.
    """
    max_abs = np.max(np.abs(x))
    if max_abs == 0.0:
        return x
    return (bound / max_abs) * x


def apply_function(function: NodeFunction, parent_series: pd.Series) -> NDArray[np.float64]:
    """Apply a mean or scale function to a parent, reading nominal parents as category codes.

    Parameters
    ----------
    function : callable
        Shape function, or lookup table over category codes for a nominal parent.

    parent_series : Series
        Observations of the parent.

    Returns
    -------
    values : ndarray of shape (n,)
        Function evaluated on the parent.
    """
    if variable_type(parent_series) == VariableType.NOMINAL:
        codes = parent_series.cat.codes.to_numpy().astype(np.int64)
        return cast(CategoricalFunction, function)(codes)
    return cast(ShapeFunction, function)(parent_series.to_numpy())


def rescale_to_band(x: NDArray[np.float64], low: float, high: float) -> NDArray[np.float64]:
    """Map ``x`` linearly onto ``[low, high]`` (min-max); a constant ``x`` goes to the midpoint.

    Parameters
    ----------
    x : ndarray
        Values to rescale.

    low : float
        Lower end of the band.

    high : float
        Upper end of the band.

    Returns
    -------
    rescaled : ndarray
        Values in ``[low, high]``, with the same shape as ``x``.
    """
    xmin, xmax = x.min(), x.max()
    if xmax > xmin:
        return low + (high - low) * (x - xmin) / (xmax - xmin)
    return np.full_like(x, (low + high) / 2)


def softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the softmax of each row of ``logits``, computed in a numerically stable way."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def sample_categorical_rows(
    rng: np.random.Generator, probs: NDArray[np.float64]
) -> NDArray[np.intp]:
    """Draw one category per row of ``probs``, by inverse transform sampling.

    Parameters
    ----------
    rng : Generator
        Source of randomness.

    probs : ndarray of shape (n, k)
        Category probabilities of each row; every row sums to 1.

    Returns
    -------
    categories : ndarray of shape (n,)
        Sampled category codes.
    """
    cumulative = np.cumsum(probs, axis=1)
    u = rng.uniform(size=(probs.shape[0], 1))
    return (u > cumulative).sum(axis=1)


def robust_scale(x: NDArray[np.float64]) -> float:
    """Return a scale of ``x`` that the observations in its tails hardly affect.

    It is the median absolute deviation about the median, multiplied by 1.4826 so that
    it equals the standard deviation for normal data. If that deviation is 0 (more
    than half of the values are equal), the standard deviation is returned instead.

    Parameters
    ----------
    x : ndarray
        Values whose scale is measured.

    Returns
    -------
    scale : float
        Robust scale of ``x``; 0 only if ``x`` is constant.
    """
    scale = 1.4826 * np.median(np.abs(x - np.median(x)))
    return float(scale if scale > 0.0 else x.std())
