import numpy as np
import pytest
from tigramite.independence_tests.cmiknn import CMIknn as TigramiteCMIknn

from src.CI_tests.CMIknn.CMIknn import CMIknn, cmiknn_test
from test.scenarios import SCENARIOS, Scenario


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


@pytest.mark.parametrize(
    "knn_param,sample_size,expected",
    [
        (0.2, 100, 20),  # fraction branch: 0.2*100
        (0.2, 10, 2),  # fraction branch, smaller n
        (0.05, 10, 1),  # fraction branch rounds to 0 -> floored to 1
        (5, 100, 5),  # absolute branch: >=1 taken as-is
        (1, 100, 1),  # boundary: knn==1 is NOT knn<1, so absolute branch
    ],
)
def test_get_knn_matches_precomputed_values(
    knn_param: float, sample_size: int, expected: int
) -> None:
    assert CMIknn(knn=knn_param)._get_knn(sample_size) == expected


def test_local_permutation_neighbors_matches_brute_force_knn() -> None:
    """🇬🇧 Compares the KD-tree-based `_local_permutation_neighbors`
    against a brute-force nearest-neighbor computation (plain sorting by
    distance, an algorithmically independent path, not the same formula
    reimplemented) on 1-D data simple enough to reason about by hand: two
    well-separated clusters, so each point's k_perm=2 nearest neighbors
    are unambiguous (no ties at the cluster boundary).

    🇪🇸 Compara el `_local_permutation_neighbors` basado en KD-tree contra
    un cálculo de vecinos más cercanos por fuerza bruta (ordenar por
    distancia sin más, una vía algorítmicamente independiente, no la misma
    fórmula reimplementada) sobre datos 1-D lo bastante simples para
    razonarlos a mano: dos clusters bien separados, así que los k_perm=2
    vecinos más cercanos de cada punto no son ambiguos (sin empates en el
    borde entre clusters).
    """
    z = np.array([[0.0], [1.0], [2.0], [100.0], [101.0], [102.0]])
    k_perm = 2

    neighbors = CMIknn(k_perm=k_perm)._local_permutation_neighbors(z)

    for i in range(len(z)):
        distances = np.abs(z[:, 0] - z[i, 0])
        distances[i] = np.inf  # exclude self from "brute force nearest OTHER points"
        brute_force_nearest = set(np.argsort(distances)[: k_perm - 1])
        # self is always among the k nearest neighbors of itself (distance 0);
        # the other k_perm-1 slots must match the brute-force computation.
        assert i in neighbors[i]
        assert set(neighbors[i]) - {i} == brute_force_nearest


def test_local_permutation_output_always_drawn_from_declared_neighbors() -> None:
    """🇬🇧 Hard invariant of `_local_permutation` (Algorithm 1): whatever
    index it assigns to a point, that index must be one of the point's own
    declared neighbors -- true regardless of collisions with `used`.

    🇪🇸 Invariante duro de `_local_permutation` (Algoritmo 1): cualquier
    índice que asigne a un punto tiene que ser uno de los vecinos
    declarados de ese punto -- cierto pase lo que pase con las colisiones
    de `used`.
    """
    rng = np.random.default_rng(0)
    n, k_perm = 30, 4
    neighbors = rng.integers(0, n, size=(n, k_perm)).astype(np.int32)

    permutation = CMIknn(k_perm=k_perm, seed=0)._local_permutation(neighbors.copy())

    for i in range(n):
        assert permutation[i] in neighbors[i]


def test_local_permutation_is_a_valid_permutation_when_k_perm_covers_all_points() -> None:
    """🇬🇧 When every row lists all n indices as candidates (k_perm=n), a
    pigeonhole argument guarantees no collision: by the time the last
    not-yet-assigned point is processed, at most n-1 indices are in `used`,
    and its row of n distinct candidates must contain one that isn't. So
    the output must be a genuine bijection on 0..n-1 -- not just "drawn
    from declared neighbors" (the weaker invariant above), a full
    collision-free permutation.

    🇪🇸 Cuando cada fila lista los n índices como candidatos (k_perm=n), un
    argumento de palomar garantiza que no hay colisión: para cuando se
    procesa el último punto sin asignar, `used` tiene como mucho n-1
    índices, y su fila de n candidatos distintos tiene que contener uno que
    no esté. Así que la salida tiene que ser una biyección genuina en
    0..n-1 -- no solo "sacado de los vecinos declarados" (el invariante más
    débil de arriba), una permutación completa sin colisiones.
    """
    n = 20
    neighbors = np.tile(np.arange(n, dtype=np.int32), (n, 1))

    permutation = CMIknn(k_perm=n, seed=0)._local_permutation(neighbors)

    assert sorted(permutation.tolist()) == list(range(n))


def test_unconditional_z_none_does_not_crash_and_distinguishes_dependence() -> None:
    """🇬🇧 Regression test: z=None used to crash `_neighbor_counts` with
    `ValueError: data must be of shape (n, m)`, since a bare 1D x/y falls
    back to itself instead of a column vector (fixed by `.reshape(-1, 1)`).

    🇪🇸 Test de regresión: z=None hacía fallar `_neighbor_counts` con
    `ValueError: data must be of shape (n, m)`, porque un x/y 1D caía de
    vuelta a sí mismo en vez de un vector columna (arreglado con
    `.reshape(-1, 1)`).
    """
    rng = np.random.default_rng(0)
    n = 300
    x_dep = rng.standard_normal(n)
    y_dep = 0.9 * x_dep + 0.3 * rng.standard_normal(n)

    x_indep = rng.standard_normal(n)
    y_indep = rng.standard_normal(n)

    assert cmiknn_test(x_dep, y_dep, None).reject
    assert not cmiknn_test(x_indep, y_indep, None).reject


def test_gamma_null_fit_distinguishes_dependence_with_fewer_draws() -> None:
    """🇬🇧 null_fit="gamma" fits a Gamma to the (small) null sample and
    reads the p-value off its CDF instead of counting exceedances, so it
    should still separate dependence from independence with far fewer
    permutation draws than the default empirical count needs.

    🇪🇸 null_fit="gamma" ajusta una Gamma a la (pequeña) muestra nula y lee
    el p-valor de su CDF en vez de contar cuántas la superan, así que debe
    seguir separando dependencia de independencia con muchas menos
    permutaciones de las que necesita el conteo empírico por defecto.
    """
    rng = np.random.default_rng(0)
    n = 500
    z = rng.standard_normal((n, 3))

    x_dep = rng.standard_normal(n)
    y_dep = np.sin(2 * x_dep) + 0.1 * rng.standard_normal(n)

    x_indep = rng.standard_normal(n)
    y_indep = rng.standard_normal(n)

    assert cmiknn_test(x_dep, y_dep, z, sig_samples=50, null_fit="gamma").reject
    assert not cmiknn_test(x_indep, y_indep, z, sig_samples=50, null_fit="gamma").reject


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_cmi_matches_tigramite_given_the_same_permutation(scenario: Scenario) -> None:
    """🇬🇧 Isolates the actual CMI computation (Eq. 5 + neighbor counting
    under a permuted Y) from the shuffle-test orchestration around it,
    which is what makes end-to-end p_value matching against tigramite
    infeasible (see the "cmiknn" entry in test/methods.py). Here both
    sides are handed the *same* permuted y — generated once, outside
    either implementation's own RNG — so nothing about which permutation
    gets drawn can cause a mismatch; a mismatch here could only mean a
    formula itself is wrong. Checks two separate sections against their
    tigramite counterpart, each called directly (bypassing tigramite's own
    shuffle loop, `get_shuffle_significance`, and its RNG entirely):
    `_neighbor_counts` against `_get_nearest_neighbors` (the k_xz/k_yz/k_z
    counts themselves, exact integers), then `_cmi_estimator` against the
    rest of `get_dependence_measure` (Eq. 5 applied to those counts). A
    failure pinpoints which of the two sections diverged. Uses
    `scenario.clean_seed` (pre-verified free of near-ties, see
    test/scenarios.py's `has_near_tie` and test/test_scenarios.py) instead
    of the scenario's arbitrary default seed — near-tied data makes which
    of two values sorts first a coin flip decided by tie-break noise, and
    since we and tigramite draw that noise from independent RNGs, a
    mismatch there would reflect the two RNGs disagreeing on that coin
    flip, not a formula bug (verified over thousands of cases in
    experiments/verification/verify_cmiknn.py: 0 mismatches once near-ties
    are excluded).

    🇪🇸 Aísla el cálculo de CMI en sí (Eq. 5 + conteo de vecinos bajo una Y
    permutada) de la orquestación del shuffle test alrededor, que es lo
    que hace inviable casar el p_value de principio a fin contra tigramite
    (ver la entrada "cmiknn" en test/methods.py). Aquí a ambos lados se
    les da el *mismo* y permutado — generado una vez, fuera del RNG propio
    de cualquiera de las dos implementaciones — así que nada sobre qué
    permutación sale puede causar un desajuste; un desajuste aquí solo
    podría significar que alguna fórmula está mal. Comprueba dos secciones
    por separado contra su equivalente en tigramite, cada una llamada
    directamente (evitando por completo el bucle de shuffle propio de
    tigramite, `get_shuffle_significance`, y su RNG): `_neighbor_counts`
    contra `_get_nearest_neighbors` (los conteos k_xz/k_yz/k_z en sí,
    enteros exactos), y luego `_cmi_estimator` contra el resto de
    `get_dependence_measure` (Eq. 5 aplicada a esos conteos). Un fallo
    señala cuál de las dos secciones divergió. Usa `scenario.clean_seed`
    (verificado sin casi-empates, ver `has_near_tie` de test/scenarios.py y
    test/test_scenarios.py) en vez de la semilla arbitraria por defecto del
    escenario — con datos casi empatados, cuál de dos valores ordena
    primero es una moneda al aire decidida por el ruido de desempate, y
    como nosotros y tigramite sacamos ese ruido de RNGs independientes, un
    desajuste ahí reflejaría que los dos RNGs discreparon en esa moneda al
    aire, no un bug de fórmula (verificado sobre miles de casos en
    experiments/verification/verify_cmiknn.py: 0 desajustes una vez se
    excluyen los casi-empates).
    """
    x, y, z = scenario.generate(np.random.default_rng(scenario.clean_seed))
    x2, y2, z2 = x.reshape(-1, 1), y.reshape(-1, 1), z

    ours = CMIknn(seed=0)
    x_ranked, y_ranked, z_ranked = (
        ours._rank_transform(x2),
        ours._rank_transform(y2),
        ours._rank_transform(z2),
    )
    knn = ours._get_knn(x_ranked.shape[0])

    tigramite_cmiknn = TigramiteCMIknn(
        significance="shuffle_test", transform="ranks", null_fit=None, permute="Y", seed=0
    )
    array = np.vstack((x2.T, y2.T, z2.T))
    xyz = np.array([0, 1, 2])

    rng = np.random.default_rng(123)
    for _ in range(20):
        perm = rng.permutation(x_ranked.shape[0])

        array_shuffled = array.copy()
        array_shuffled[1] = array[1, perm]  # row 1 = Y, per xyz above

        # Section 1: neighbor counting alone.
        k_xz, k_yz, k_z = ours._neighbor_counts(x_ranked, y_ranked[perm], z_ranked, knn)
        k_xz_ref, k_yz_ref, k_z_ref = tigramite_cmiknn._get_nearest_neighbors(
            array_shuffled, xyz, knn
        )
        assert np.array_equal(k_xz, k_xz_ref)
        assert np.array_equal(k_yz, k_yz_ref)
        assert np.array_equal(k_z, k_z_ref)

        # Section 2: Eq. 5 applied to those (already-matched) counts.
        cmi_ours = ours._cmi_estimator(knn, k_z, k_xz, k_yz)
        cmi_tigramite = tigramite_cmiknn.get_dependence_measure(array_shuffled, xyz)
        assert cmi_ours == pytest.approx(cmi_tigramite, abs=1e-9)
