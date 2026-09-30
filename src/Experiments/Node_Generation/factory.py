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
# Mean functions u -> f(u); the linear case is the identity.


def sine_shape(rng: np.random.Generator) -> ShapeFunction:
    omega = rng.uniform(0.5, 1.5)
    return lambda u: np.sin(omega * u)


def cosine_shape(rng: np.random.Generator) -> ShapeFunction:
    omega = rng.uniform(0.5, 1.5)
    return lambda u: np.cos(omega * u) - 1


def exponential_shape(rng: np.random.Generator) -> ShapeFunction:
    magnitude = rng.uniform(0.5, 2.0)
    sign = rng.choice((-1.0, 1.0))
    beta = sign * magnitude
    return lambda u: np.exp(beta * u) - 1


def square_shape(rng: np.random.Generator) -> ShapeFunction:
    return lambda u: u**2


def cube_shape(rng: np.random.Generator) -> ShapeFunction:
    return lambda u: u**3


def tanh_shape(rng: np.random.Generator) -> ShapeFunction:
    lam = rng.uniform(0.5, 2.0)
    return lambda u: np.tanh(lam * u)


def identity_shape(rng: np.random.Generator) -> ShapeFunction:
    return lambda u: u


## Scale ##
# Standard-deviation functions u -> s(u) > 0; the unit scale is homoscedastic.


def sqrt_quadratic_scale(rng: np.random.Generator) -> ShapeFunction:
    gamma = rng.uniform(0.25, 0.5)
    return lambda u: np.sqrt(1 + gamma * u**2)


def periodic_scale(rng: np.random.Generator) -> ShapeFunction:
    gamma = rng.uniform(0.25, 0.5)
    omega = rng.uniform(0.5, 1.0)
    return lambda u: 1 + gamma * (1 + np.sin(omega * u))


def piecewise_linear_scale(rng: np.random.Generator) -> ShapeFunction:
    gamma = rng.uniform(0.25, 0.5)
    return lambda u: 1 + gamma * np.maximum(0.0, u)


def unit_scale(rng: np.random.Generator) -> ShapeFunction:
    return lambda u: np.ones_like(u)


## Transform ##
# Outer transforms t -> g(t) applied to the noisy mean (post-nonlinear models).


def identity_transform(rng: np.random.Generator) -> ShapeFunction:
    return lambda t: t


def tanh_transform(rng: np.random.Generator) -> ShapeFunction:
    kappa = rng.uniform(0.25, 1.0)
    return lambda t: np.tanh(kappa * t)


def asinh_transform(rng: np.random.Generator) -> ShapeFunction:
    kappa = rng.uniform(0.25, 1.0)
    return lambda t: np.arcsinh(kappa * t)


def exponential_transform(rng: np.random.Generator) -> ShapeFunction:
    kappa = rng.uniform(0.25, 0.5)
    return lambda t: np.exp(kappa * t)


## Noise ##
# Samplers n -> n draws of zero-mean, unit-variance noise.


def gaussian_noise(rng: np.random.Generator) -> NoiseSampler:
    return lambda n: rng.standard_normal(n)


def laplace_noise(rng: np.random.Generator) -> NoiseSampler:
    return lambda n: rng.laplace(0.0, 1.0 / np.sqrt(2.0), n)


def exponential_noise(rng: np.random.Generator) -> NoiseSampler:
    return lambda n: rng.exponential(1.0, n) - 1.0


def gumbel_noise(rng: np.random.Generator) -> NoiseSampler:
    beta = np.sqrt(6.0) / np.pi
    return lambda n: rng.gumbel(-beta * np.euler_gamma, beta, n)


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


# Noise cases: Gaussian, non-Gaussian (Laplace) or any of the four samplers.


class GaussianNoiseFactory(NoiseFactory):
    _BUILDERS = (gaussian_noise,)


class NonGaussianNoiseFactory(NoiseFactory):
    _BUILDERS = (laplace_noise,)


class GeneralNoiseFactory(NoiseFactory):
    _BUILDERS = (gaussian_noise, laplace_noise, exponential_noise, gumbel_noise)


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
