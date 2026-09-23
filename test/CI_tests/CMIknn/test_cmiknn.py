import numpy as np

from src.CI_tests.CMIknn.CMIknn import CMIknn


def test_ranks_1d_data_descending_without_ties() -> None:
    """🇬🇧 30 is the largest -> rank 0 ("x_i is the r-th largest", paper
    §2.2); 10 is the smallest -> rank 2. No ties, so the tie-break noise
    can't change the outcome.

    🇪🇸 30 es el mayor -> rango 0 ("x_i es el r-ésimo mayor", paper §2.2);
    10 es el menor -> rango 2. Sin empates, el ruido de desempate no puede
    cambiar el resultado.
    """
    x = np.array([30.0, 10.0, 20.0])

    result = CMIknn()._rank_transform(x)

    assert result.shape == x.shape
    assert result.tolist() == [2.0, 0.0, 1.0]


def test_breaks_ties_into_a_valid_permutation() -> None:
    """🇬🇧 1.0 appears twice; the tie-break noise must still turn the
    result into a genuine 0..n-1 permutation (no repeated ranks), even
    though which of the two tied points gets which rank is arbitrary.

    🇪🇸 1.0 aparece dos veces; el ruido de desempate debe seguir dando una
    permutación genuina de 0..n-1 (sin rangos repetidos), aunque cuál de
    los dos puntos empatados se lleve cuál rango sea arbitrario.
    """
    x = np.array([1.0, 1.0, 2.0, 5.0])

    result = CMIknn()._rank_transform(x)

    assert result[3] == 3.0
    assert result[2] == 2.0


def test_ranks_each_column_of_2d_data_independently() -> None:
    """🇬🇧 Column 0 and column 1 have unrelated orderings; rank_transform
    must rank each one on its own, not mix them.

    🇪🇸 La columna 0 y la columna 1 tienen órdenes sin relación entre sí;
    rank_transform debe rankear cada una por su cuenta, sin mezclarlas.
    """
    z = np.array([[30.0, 1.0], [10.0, 3.0], [20.0, 2.0]])

    result = CMIknn()._rank_transform(z)

    assert result.shape == z.shape
    assert result[:, 0].tolist() == [2.0, 0.0, 1.0]  # [30, 10, 20]
    assert result[:, 1].tolist() == [0.0, 2.0, 1.0]  # [1, 3, 2]
