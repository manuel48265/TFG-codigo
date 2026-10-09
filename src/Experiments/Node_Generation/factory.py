from collections.abc import Callable
from typing import Concatenate, Generic, ParamSpec, TypeVar

import numpy as np
from numpy.typing import NDArray

ShapeFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]
NoiseSampler = Callable[[int], NDArray[np.float64]]
CategoricalFunction = Callable[[NDArray[np.int64]], NDArray[np.float64]]

# Every builder below takes a random generator, draws the parameters of one member
# of its family from it, and returns the resulting function (or sampler). The
# families are the "shape" of the mean, the "scale" of the standard deviation, the
# outer "transform" and the "noise" of the node generator.

## Shape ##
# Mean functions u -> f(u); the linear case is the identity. They are applied to a parent
# in robust scales from its median and the result is standardized (see standardized_term
# in node_generator.py). The parameter ranges keep every nonlinear shape clearly nonlinear
# at the low end (on a Gaussian parent, at least ~10% of the dependence is not captured by
# a line, like the cube's 15%): below them sine, tanh and exponential are almost linear.
# The top of the exponential is limited so that a few observations do not carry most of
# the variance.


def sine_shape(rng: np.random.Generator) -> ShapeFunction:
    omega = rng.uniform(1.0, 2.0)
    return lambda u: np.sin(omega * u)


def cosine_shape(rng: np.random.Generator) -> ShapeFunction:
    omega = rng.uniform(0.5, 1.5)
    return lambda u: np.cos(omega * u) - 1


def exponential_shape(rng: np.random.Generator) -> ShapeFunction:
    magnitude = rng.uniform(0.5, 1.0)
    sign = rng.choice((-1.0, 1.0))
    beta = sign * magnitude
    return lambda u: np.exp(beta * u) - 1


def square_shape(rng: np.random.Generator) -> ShapeFunction:
    return lambda u: u**2


def cube_shape(rng: np.random.Generator) -> ShapeFunction:
    return lambda u: u**3


def tanh_shape(rng: np.random.Generator) -> ShapeFunction:
    lam = rng.uniform(1.5, 3.0)
    return lambda u: np.tanh(lam * u)


def identity_shape(rng: np.random.Generator) -> ShapeFunction:
    return lambda u: u


## Scale ##
# Standard-deviation functions u -> s(u) > 0; the unit scale is homoscedastic. Only their
# shape matters: they are mapped linearly onto the range of noise levels (see
# scale_position in node_generator.py), so constants that only shift or rescale them
# change nothing.


def sqrt_quadratic_scale(rng: np.random.Generator) -> ShapeFunction:
    gamma = rng.uniform(1 / 25, 1 / 5)
    return lambda u: np.sqrt(1 + gamma * u**2)


def periodic_scale(rng: np.random.Generator) -> ShapeFunction:
    omega = rng.uniform(1.0, 2.0)
    return lambda u: 2 + np.sin(omega * u)


def piecewise_linear_scale(rng: np.random.Generator) -> ShapeFunction:
    return lambda u: 1 + np.maximum(0.0, u)


def abs_linear_scale(rng: np.random.Generator) -> ShapeFunction:
    return lambda u: 1 + abs(u)


def unit_scale(rng: np.random.Generator) -> ShapeFunction:
    return lambda u: np.ones_like(u)


## Transform ##
# Outer transforms t -> g(t) applied to the noisy mean (post-nonlinear models). They are
# an extra layer of difficulty for detection, not a way of making it impossible, so they
# must not change the shape of the data drastically. They act on an input of unit
# variance, which for Gaussian data spans about [-GAUSSIAN_RANGE, GAUSSIAN_RANGE], and
# kappa is set so that the transform stays inside [-TARGET_RANGE, TARGET_RANGE]: the
# argument kappa * t of tanh and asinh does, and so does the output of exp(kappa * t) - 1.
# Each transform draws kappa between half of that bound and the bound.

GAUSSIAN_RANGE = 3.0
TARGET_RANGE = 5.0
KAPPA_FOR_ARGUMENT = TARGET_RANGE / GAUSSIAN_RANGE
KAPPA_FOR_OUTPUT = float(np.log1p(TARGET_RANGE)) / GAUSSIAN_RANGE


def identity_transform(rng: np.random.Generator) -> ShapeFunction:
    return lambda t: t


def tanh_transform(rng: np.random.Generator) -> ShapeFunction:
    kappa = rng.uniform(KAPPA_FOR_ARGUMENT / 2, KAPPA_FOR_ARGUMENT)
    return lambda t: np.tanh(kappa * t)


def asinh_transform(rng: np.random.Generator) -> ShapeFunction:
    kappa = rng.uniform(KAPPA_FOR_ARGUMENT / 2, KAPPA_FOR_ARGUMENT)
    return lambda t: np.arcsinh(kappa * t)


def exponential_transform(rng: np.random.Generator) -> ShapeFunction:
    kappa = rng.uniform(KAPPA_FOR_OUTPUT / 2, KAPPA_FOR_OUTPUT)
    return lambda t: np.exp(kappa * t) - 1


## Noise ##
# Samplers n -> n draws of zero-mean, unit-variance noise.


def gaussian_noise(rng: np.random.Generator) -> NoiseSampler:
    return lambda n: rng.standard_normal(n)


def laplace_noise(rng: np.random.Generator) -> NoiseSampler:
    return lambda n: rng.laplace(0.0, 1.0 / np.sqrt(2.0), n)


## Factory ##


R = TypeVar("R")
P = ParamSpec("P")


class BaseFactory(Generic[P, R]):
    """Draw one builder uniformly from ``_BUILDERS`` and call it.

    Subclasses only set ``_BUILDERS``, the tuple of builders they choose from.
    """

    _BUILDERS: tuple[Callable[Concatenate[np.random.Generator, P], R], ...] = ()

    @classmethod
    def sample(cls, rng: np.random.Generator, *args: P.args, **kwargs: P.kwargs) -> R:
        """Return ``builder(rng, ...)`` for a builder chosen uniformly at random."""
        builder = cls._BUILDERS[rng.integers(len(cls._BUILDERS))]
        return builder(rng, *args, **kwargs)


class FunctionFactory(BaseFactory[[], ShapeFunction]):
    """Factory of functions of a continuous argument (shape, scale and transform)."""


class NoiseFactory(BaseFactory[[], NoiseSampler]):
    """Factory of noise samplers."""


class CategoricalFactory(BaseFactory[[int], CategoricalFunction]):
    """Factory of functions of a categorical code; ``sample`` takes the number of categories."""


# Mean function cases: linear (identity) or nonlinear (sine, cosine, exponential,
# square, cube, tanh).


class LinearShapeFactory(FunctionFactory):
    _BUILDERS = (identity_shape,)


class NonLinearShapeFactory(FunctionFactory):
    _BUILDERS = (
        sine_shape,
        cosine_shape,
        exponential_shape,
        square_shape,
        cube_shape,
        tanh_shape,
    )


# Scale function cases: homoscedastic (unit scale) or heteroscedastic (the scale
# depends on the argument).


class HomoscedasticShapeFactory(FunctionFactory):
    _BUILDERS = (unit_scale,)


class HeteroscedasticShapeFactory(FunctionFactory):
    _BUILDERS = (
        sqrt_quadratic_scale,
        periodic_scale,
        piecewise_linear_scale,
        abs_linear_scale,
    )


# Transform cases: identity or an outer nonlinear transform (tanh, asinh, exp).


class IdentityTransformFactory(FunctionFactory):
    _BUILDERS = (identity_transform,)


class OuterTransformFactory(FunctionFactory):
    _BUILDERS = (
        tanh_transform,
        asinh_transform,
        exponential_transform,
    )


# Noise cases: Gaussian, the standard in the literature on CI tests, or Laplace, the
# only non-Gaussian noise, used by the linear non-Gaussian model.


class GaussianNoiseFactory(NoiseFactory):
    _BUILDERS = (gaussian_noise,)


class NonGaussianNoiseFactory(NoiseFactory):
    _BUILDERS = (laplace_noise,)


# Nominal variables: the mean and the scale are lookup tables indexed by the
# category code, drawn once per variable.


def categorical_mean_function(rng: np.random.Generator, n_categories: int) -> CategoricalFunction:
    magnitude = rng.uniform(0.25, 4.0, size=n_categories)
    sign = rng.choice((-1.0, 1.0), size=n_categories)
    table = magnitude * sign
    return lambda t: table[t]


def categorical_shape_function(rng: np.random.Generator, n_categories: int) -> CategoricalFunction:
    magnitude = rng.uniform(0.5, 2.0, size=n_categories)
    table = magnitude
    return lambda t: table[t]


class CategoricalVariablesMeanFactory(CategoricalFactory):
    _BUILDERS = (categorical_mean_function,)


class CategoricalVariablesScaleFactory(CategoricalFactory):
    _BUILDERS = (categorical_shape_function,)
