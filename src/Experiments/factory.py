from collections.abc import Callable
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

ShapeFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]
ShapeBuilder = Callable[[np.random.Generator], ShapeFunction]

## FUNTIONS ##

## Shape ##


def _sine_shape(rng: np.random.Generator) -> ShapeFunction:
    omega = rng.uniform(0.5, 1.5)
    return lambda u: np.sin(omega * u)


def _cosine_shape(rng: np.random.Generator) -> ShapeFunction:
    omega = rng.uniform(0.5, 1.5)
    return lambda u: np.cos(omega * u) - 1


def _exponential_shape(rng: np.random.Generator) -> ShapeFunction:
    magnitude = rng.uniform(0.5, 2.0)
    sign = rng.choice((-1.0, 1.0))
    beta = sign * magnitude
    return lambda u: np.exp(beta * u) - 1


def _square_shape(rng: np.random.Generator) -> ShapeFunction:
    return lambda u: u**2


def _cube_shape(rng: np.random.Generator) -> ShapeFunction:
    return lambda u: u**3


def _tanh_shape(rng: np.random.Generator) -> ShapeFunction:
    lam = rng.uniform(0.5, 2.0)
    return lambda u: np.tanh(lam * u)


def _identity_shape(rng: np.random.Generator) -> ShapeFunction:
    return lambda u: u


## Scale ##


def _sqrt_quadratic_scale(rng: np.random.Generator) -> ShapeFunction:
    gamma = rng.uniform(0.25, 0.5)
    return lambda u: np.sqrt(1 + gamma * u**2)


def _periodic_scale(rng: np.random.Generator) -> ShapeFunction:
    gamma = rng.uniform(0.25, 0.5)
    omega = rng.uniform(0.5, 1.0)
    return lambda u: 1 + gamma * (1 + np.sin(omega * u))


def _piecewise_linear_scale(rng: np.random.Generator) -> ShapeFunction:
    gamma = rng.uniform(0.25, 0.5)
    return lambda u: 1 + gamma * np.maximum(0.0, u)


def _unit_scale(rng: np.random.Generator) -> ShapeFunction:
    return lambda u: np.ones_like(u)


## Transform ##


def _identity_transform(rng: np.random.Generator) -> ShapeFunction:
    return lambda t: t


def _tanh_transform(rng: np.random.Generator) -> ShapeFunction:
    kappa = rng.uniform(0.25, 1.0)
    return lambda t: np.tanh(kappa * t)


def _asinh_transform(rng: np.random.Generator) -> ShapeFunction:
    kappa = rng.uniform(0.25, 1.0)
    return lambda t: np.arcsinh(kappa * t)


def _exponential_transform(rng: np.random.Generator) -> ShapeFunction:
    kappa = rng.uniform(0.25, 0.5)
    return lambda t: np.exp(kappa * t)


## Noise ##


def normal_noise(rng: np.random.Generator, n: int) -> NDArray[np.float64]:
    return rng.standard_normal(n)


def laplace_noise(rng: np.random.Generator, n: int) -> NDArray[np.float64]:
    return rng.laplace(0.0, 1.0 / np.sqrt(2.0), n)


## Factory ##


class FunctionFactory:
    _BUILDERS: ClassVar[tuple[ShapeBuilder, ...]] = ()

    @classmethod
    def sample(cls, rng: np.random.Generator) -> ShapeFunction:
        builder = cls._BUILDERS[rng.integers(len(cls._BUILDERS))]
        return builder(rng)


class MeanShapeFactory(FunctionFactory):
    _BUILDERS = (
        _sine_shape,
        _cosine_shape,
        _exponential_shape,
        _square_shape,
        _cube_shape,
        _tanh_shape,
    )


class ScaleShapeFactory(FunctionFactory):
    _BUILDERS = (
        _sqrt_quadratic_scale,
        _periodic_scale,
        _piecewise_linear_scale,
    )


class OuterTransformFactory(FunctionFactory):
    _BUILDERS = (
        _tanh_transform,
        _asinh_transform,
        _exponential_transform,
    )
