from collections.abc import Callable
from typing import Concatenate, Generic, ParamSpec, TypeVar

import numpy as np
from numpy.typing import NDArray

ShapeFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]
NoiseSampler = Callable[[int], NDArray[np.float64]]
CategoricalFunction = Callable[[int], np.float64]

## FUNTIONS ##

## Shape ##


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


def gaussian_noise(rng: np.random.Generator) -> NoiseSampler:
    return lambda n: rng.standard_normal(n)


def laplace_noise(rng: np.random.Generator) -> NoiseSampler:
    return lambda n: rng.laplace(0.0, 1.0 / np.sqrt(2.0), n)


## Factory ##


R = TypeVar("R")
P = ParamSpec("P")


class BaseFactory(Generic[P, R]):
    _BUILDERS: tuple[Callable[Concatenate[np.random.Generator, P], R], ...] = ()

    @classmethod
    def sample(cls, rng: np.random.Generator, *args: P.args, **kwargs: P.kwargs) -> R:
        builder = cls._BUILDERS[rng.integers(len(cls._BUILDERS))]
        return builder(rng, *args, **kwargs)


class FunctionFactory(BaseFactory[[], ShapeFunction]):
    pass


class NoiseFactory(BaseFactory[[], NoiseSampler]):
    pass


class CategoricalFactory(BaseFactory[[int], CategoricalFunction]):
    pass


# Mean Function Cases


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


# Scale Function Cases


class HomoscedasticShapeFactory(FunctionFactory):
    _BUILDERS = (unit_scale,)


class HeteroscedasticShapeFactory(FunctionFactory):
    _BUILDERS = (
        sqrt_quadratic_scale,
        periodic_scale,
        piecewise_linear_scale,
    )


# Transform Function Cases


class IdentityTransformFactory(FunctionFactory):
    _BUILDERS = (identity_transform,)


class OuterTransformFactory(FunctionFactory):
    _BUILDERS = (
        tanh_transform,
        asinh_transform,
        exponential_transform,
    )


# Noise samplers


class GaussianNoiseFactory(NoiseFactory):
    _BUILDERS = (gaussian_noise,)


class NonGaussianNoiseFactory(NoiseFactory):
    _BUILDERS = (laplace_noise,)


# Nominal variables


def categorical_mean_function(rng: np.random.Generator, n_categories: int) -> CategoricalFunction:
    magnitude = rng.uniform(0.25, 4.0, size=n_categories)
    sign = rng.choice((-1.0, 1.0), size=n_categories)
    table = magnitude * sign
    return lambda t: table[t]


def categorical_shape_function(rng: np.random.Generator, n_categories: int) -> CategoricalFunction:
    magnitude = rng.uniform(0.25, 4.0, size=n_categories)
    table = magnitude
    return lambda t: table[t]


class CategoricalVariablesMeanFactory(CategoricalFactory):
    _BUILDERS = (categorical_mean_function,)


class CategoricalVariablesScaleFactory(CategoricalFactory):
    _BUILDERS = (categorical_shape_function,)
