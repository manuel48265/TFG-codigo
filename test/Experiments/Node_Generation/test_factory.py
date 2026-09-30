import numpy as np
import pytest

from src.Experiments.Node_Generation import factory
from src.Experiments.Node_Generation.factory import GAUSSIAN_RANGE, TARGET_RANGE

TRANSFORMS = ["tanh_transform", "asinh_transform", "exponential_transform"]


@pytest.mark.parametrize("name", TRANSFORMS)
def test_outer_transforms_stay_inside_the_target_range_on_the_range_of_a_gaussian(
    name: str,
) -> None:
    t = np.linspace(-GAUSSIAN_RANGE, GAUSSIAN_RANGE, 601)
    for seed in range(200):
        out = getattr(factory, name)(np.random.default_rng(seed))(t)

        assert np.isfinite(out).all()
        assert out.min() >= -TARGET_RANGE
        assert out.max() <= TARGET_RANGE + 1e-9


@pytest.mark.parametrize("name", TRANSFORMS)
def test_outer_transforms_are_increasing_and_keep_the_origin(name: str) -> None:
    t = np.linspace(-GAUSSIAN_RANGE, GAUSSIAN_RANGE, 601)
    for seed in range(50):
        out = getattr(factory, name)(np.random.default_rng(seed))(t)

        assert (np.diff(out) > 0).all()
        assert out[300] == pytest.approx(0.0, abs=1e-12)


def test_the_outer_transforms_are_not_straight_lines() -> None:
    """Each transform bends a standard Gaussian, so it adds a layer of difficulty."""
    x = np.random.default_rng(0).standard_normal(5000)
    for name in TRANSFORMS:
        r2 = [
            np.corrcoef(x, getattr(factory, name)(np.random.default_rng(seed))(x))[0, 1] ** 2
            for seed in range(50)
        ]
        print(r2)
        assert np.median(r2) < 0.999
