import numpy as np
import pytest

from src.CI_tests.KCI.kci import KCI, kci_test
from src.CI_tests.KCI.kci_utils import center_kernel_matrix_regression


def _random_kernel(rng: np.random.Generator, n: int) -> np.ndarray:
    A = rng.normal(size=(n, n))
    return A @ A.T


def test_get_eigenvects_prod_matches_hadamard_product() -> None:
    """🇬🇧 With the full (untruncated) eigenbasis, eigenvects_prod's
    reconstruction should equal the Hadamard product Kx*Ky.

    🇪🇸 Con la base propia completa (sin truncar), la reconstrucción de
    eigenvects_prod debe igualar el producto de Hadamard Kx*Ky.
    """
    rng = np.random.default_rng(0)
    n = 8
    Kx = _random_kernel(rng, n)
    Ky = _random_kernel(rng, n)
    kci = KCI(thresh=1e-12)

    eigenvects_prod, n_components = kci.get_eigenvects_prod(Kx, Ky)

    assert n_components >= n  # full-rank bases -> tensor space is larger than n
    assert eigenvects_prod == pytest.approx(Kx * Ky, abs=1e-8)


def test_KCI_V_statistic_reuse_matches_independent_recompute() -> None:
    """🇬🇧 When Kzx and Kzy are the same matrix and epsilon matches, the
    "reuse Rzx for Ky" shortcut must give exactly the same KyR as
    recomputing center_kernel_matrix_regression independently for Ky — it's
    an exact algebraic reuse (Rz depends only on Kz/eps), not an
    approximation.

    🇪🇸 Cuando Kzx y Kzy son la misma matriz y epsilon coincide, el atajo de
    "reutilizar Rzx para Ky" debe dar exactamente el mismo KyR que
    recalcular center_kernel_matrix_regression de forma independiente para
    Ky — es una reutilización algebraica exacta (Rz depende solo de
    Kz/eps), no una aproximación.
    """
    rng = np.random.default_rng(2)
    n = 12
    Kx = _random_kernel(rng, n)
    Ky = _random_kernel(rng, n)
    Kz = _random_kernel(rng, n)
    epsilon = 1e-3
    kci = KCI(use_gp=False, epsilon_x=epsilon, epsilon_y=epsilon)

    _, KxR, KyR_reused = kci.KCI_V_statistic(Kx, Ky, Kz, Kz)

    KxR_expected, _ = center_kernel_matrix_regression(Kx, Kz, epsilon)
    KyR_expected, _ = center_kernel_matrix_regression(Ky, Kz, epsilon)

    assert KxR == pytest.approx(KxR_expected, abs=1e-8)
    assert KyR_reused == pytest.approx(KyR_expected, abs=1e-8)


def test_KCI_V_statistic_recomputes_when_epsilon_differs() -> None:
    """🇬🇧 With different epsilon_x/epsilon_y, the reuse shortcut must not
    fire — KyR should match an independent recompute with epsilon_y, not a
    reuse of Rzx (which was built with epsilon_x).

    🇪🇸 Con epsilon_x/epsilon_y distintos, el atajo de reutilización no debe
    activarse — KyR debe coincidir con un recálculo independiente con
    epsilon_y, no con una reutilización de Rzx (que se construyó con
    epsilon_x).
    """
    rng = np.random.default_rng(3)
    n = 10
    Kx = _random_kernel(rng, n)
    Ky = _random_kernel(rng, n)
    Kz = _random_kernel(rng, n)
    kci = KCI(epsilon_x=1e-3, epsilon_y=1e-2)

    _, _, KyR = kci.KCI_V_statistic(Kx, Ky, Kz, Kz)

    KyR_expected, _ = center_kernel_matrix_regression(Ky, Kz, epsilon=1e-2)

    assert KyR == pytest.approx(KyR_expected, abs=1e-8)


def test_kernel_matrix_builds_symmetric_centered_matrices() -> None:
    """🇬🇧 Kx/Ky/Kzx are symmetric and centered (zero row sums); Kzx and Kzy
    are the same matrix.

    🇪🇸 Kx/Ky/Kzx son simétricas y están centradas (suma por fila cero);
    Kzx y Kzy son la misma matriz.
    """
    rng = np.random.default_rng(0)
    n = 20
    data_x = rng.standard_normal((n, 1))
    data_y = rng.standard_normal((n, 1))
    data_z = rng.standard_normal((n, 1))
    kci = KCI()

    Kx, Ky, Kzx, Kzy = kci.kernel_matrix(data_x, data_y, data_z)

    for K in (Kx, Ky, Kzx):
        assert K == pytest.approx(K.T)
        assert K.sum(axis=0) == pytest.approx(np.zeros(n), abs=1e-8)
    assert Kzx is Kzy


def test_kernel_matrix_respects_manual_width() -> None:
    """🇬🇧 A manual `kwidthz` skips the empirical heuristic entirely.

    🇪🇸 Un `kwidthz` manual salta por completo la heurística empírica.
    """
    rng = np.random.default_rng(1)
    n = 20
    data_x = rng.standard_normal((n, 1))
    data_y = rng.standard_normal((n, 1))
    data_z = rng.standard_normal((n, 1))

    _, _, Kz_manual, _ = KCI(kwidthz=2.0).kernel_matrix(data_x, data_y, data_z)
    _, _, Kz_empirical, _ = KCI().kernel_matrix(data_x, data_y, data_z)

    assert not np.allclose(Kz_manual, Kz_empirical)


def test_kci_test_does_not_reject_independence() -> None:
    """🇬🇧 X and Y both depend on Z but not on each other given Z.

    🇪🇸 X e Y dependen ambas de Z pero no entre sí dado Z.
    """
    rng = np.random.default_rng(1)
    n = 300
    z = rng.standard_normal((n, 1))
    x = np.sin(z[:, 0]) + rng.standard_normal(n)
    y = np.cos(z[:, 0]) + rng.standard_normal(n)

    result = kci_test(x.reshape(-1, 1), y.reshape(-1, 1), z)

    assert not result.reject


def test_kci_test_rejects_dependence() -> None:
    """🇬🇧 Same as above, but Y also depends directly on X.

    🇪🇸 Igual que arriba, pero Y también depende directamente de X.
    """
    rng = np.random.default_rng(1)
    n = 300
    z = rng.standard_normal((n, 1))
    x = np.sin(z[:, 0]) + rng.standard_normal(n)
    y = np.cos(z[:, 0]) + x + rng.standard_normal(n)

    result = kci_test(x.reshape(-1, 1), y.reshape(-1, 1), z)

    assert result.reject
