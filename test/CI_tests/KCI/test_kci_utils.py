import numpy as np
import pytest

from src.CI_tests.KCI import kci_utils
from src.CI_tests.KCI.kci_utils import (
    center_kernel_matrix,
    center_kernel_matrix_regression,
)


def test_center_kernel_matrix_matches_literal_HKH() -> None:
    """Checks the O(n^2) algebraic shortcut against the literal, slow
    H @ K @ H definition it's derived from.
    """
    rng = np.random.default_rng(0)
    n = 30
    A = rng.normal(size=(n, n))
    K = A @ A.T

    H = np.eye(n) - np.ones((n, n)) / n
    expected = H @ K @ H

    result = center_kernel_matrix(K)

    assert result == pytest.approx(expected)
    assert result.sum(axis=0) == pytest.approx(np.zeros(n), abs=1e-8)


def test_center_kernel_matrix_regression_matches_naive_pinv() -> None:
    """For a well-conditioned Kz, the Cholesky path should match the
    naive pinv-based formula from Zhang et al. (2011, Eq. 10-12) exactly.
    """
    rng = np.random.default_rng(1)
    n = 20
    A = rng.normal(size=(n, n))
    K = A @ A.T
    B = rng.normal(size=(n, n))
    Kz = B @ B.T
    epsilon = 1e-3

    Rz_expected = epsilon * np.linalg.pinv(Kz + epsilon * np.eye(n))
    expected = Rz_expected @ K @ Rz_expected

    result, Rz = center_kernel_matrix_regression(K, Kz, epsilon=epsilon)

    assert result == pytest.approx(expected, abs=1e-6)
    assert Rz == pytest.approx(Rz_expected, abs=1e-6)


def test_numerical_floor_bounds_condition_number() -> None:
    """With a near-singular Kz and a tiny statistical epsilon, the
    numerical floor n/(kappa_max-1) should take over and keep
    cond(Kz + eps*I) at or below kappa_max (Marchildon & Zingg 2023, Eq. 26).
    """
    n = 15
    rank_deficient = np.zeros((n, n))
    rank_deficient[0, 0] = 1.0  # a single nonzero eigenvalue -> ill-conditioned
    kappa_max = 1e4

    center_kernel_matrix_regression(np.eye(n), rank_deficient, epsilon=1e-12, kappa_max=kappa_max)

    eps_used = max(1e-12, n / (kappa_max - 1))
    cond = np.linalg.cond(rank_deficient + eps_used * np.eye(n))

    assert cond <= kappa_max * 1.01


def test_falls_back_to_pinv_when_cholesky_solve_is_unreliable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forces the Cholesky residual check to fail and checks the
    function warns (not silently, per Foster et al. 2009) and still returns
    the mathematically correct result via the pinv fallback.
    """
    rng = np.random.default_rng(2)
    n = 10
    A = rng.normal(size=(n, n))
    K = A @ A.T
    B = rng.normal(size=(n, n))
    Kz = B @ B.T
    epsilon = 1e-3

    # force the dpocon-based condition check to always report "too ill-conditioned"
    monkeypatch.setattr(kci_utils, "dpocon", lambda cho, anorm, uplo: (0.0, 0))

    Rz_expected = epsilon * np.linalg.pinv(Kz + epsilon * np.eye(n))
    expected = Rz_expected @ K @ Rz_expected

    with pytest.warns(UserWarning, match="Falling back to pinv"):
        result, Rz = center_kernel_matrix_regression(K, Kz, epsilon=epsilon)

    assert result == pytest.approx(expected, abs=1e-6)
    assert Rz == pytest.approx(Rz_expected, abs=1e-6)
