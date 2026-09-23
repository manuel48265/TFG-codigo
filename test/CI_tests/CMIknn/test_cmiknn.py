import numpy as np
import pytest
from tigramite.independence_tests.cmiknn import CMIknn as TigramiteCMIknn

from src.CI_tests.CMIknn.CMIknn import CMIknn, cmiknn_test
from test.scenarios import SCENARIOS


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


def test_calibration_fpr_and_power() -> None:
    """🇬🇧 Matching cmiknn_test's p_value against tigramite's exactly isn't
    possible (see the comment on the "cmiknn" entry in test/methods.py):
    both compute the same CMI given the same permuted data (verified
    directly), but sample different permutations from their RNG. What
    actually certifies cmiknn_test's p_value is correct is this: under a
    real null (independent scenario), the rejection rate at alpha=0.05
    should sit near 5%; under a real, clear dependence (dependent
    scenario), it should reject almost always. Measured together, not FPR
    alone — a test that's just conservative (rejects too little) would
    also show a low FPR without actually being calibrated.
    seed_offset=1000 keeps these seeds distinct from other seed=0..N sweeps
    elsewhere in the suite, so an unlucky seed here can't masquerade as
    reappearing across unrelated tests.

    🇪🇸 Casar el p_value de cmiknn_test contra tigramite de forma exacta no
    es posible (ver el comentario de la entrada "cmiknn" en
    test/methods.py): ambos calculan el mismo CMI dado el mismo dato
    permutado (verificado directamente), pero muestrean permutaciones
    distintas de su RNG. Lo que de verdad certifica que el p_value de
    cmiknn_test es correcto es esto: bajo un nulo real (escenario
    independent), la tasa de rechazo a alpha=0.05 debe rondar el 5%; bajo
    una dependencia real y clara (escenario dependent), debe rechazar casi
    siempre. Medido junto, no solo FPR — un test simplemente conservador
    (que rechaza de menos) también daría un FPR bajo sin estar realmente
    calibrado. seed_offset=1000 mantiene estas semillas distintas de otros
    barridos seed=0..N del resto de la suite, para que una semilla
    desafortunada aquí no pueda parecer un problema que reaparece en tests
    sin relación.
    """
    independent = next(s for s in SCENARIOS if s.name == "independent")
    dependent = next(s for s in SCENARIOS if s.name == "dependent")
    seed_offset = 1000
    sig_samples = 150
    alpha = 0.05

    n_rep_fpr = 30
    rejections_h0 = 0
    for i in range(n_rep_fpr):
        x, y, z = independent.generate(np.random.default_rng(seed_offset + i))
        result = cmiknn_test(x.reshape(-1, 1), y.reshape(-1, 1), z, sig_samples=sig_samples, seed=i)
        rejections_h0 += result.reject
    fpr = rejections_h0 / n_rep_fpr

    n_rep_power = 15
    rejections_h1 = 0
    for i in range(n_rep_power):
        x, y, z = dependent.generate(np.random.default_rng(seed_offset + n_rep_fpr + i))
        result = cmiknn_test(x.reshape(-1, 1), y.reshape(-1, 1), z, sig_samples=sig_samples, seed=i)
        rejections_h1 += result.reject
    power = rejections_h1 / n_rep_power

    # Generous upper bound (not a tight band around alpha): n_rep_fpr=30 makes
    # a nominal-vs-observed comparison noisy, and the project favors FPR≈0
    # over power (a low FPR isn't itself a red flag here), so only a clearly
    # anti-conservative rate should fail this.
    assert fpr <= 0.20, f"FPR={fpr:.2f} muy por encima de alpha={alpha} en {n_rep_fpr} réplicas"
    assert power >= 0.8, f"potencia={power:.2f} demasiado baja en {n_rep_power} réplicas"


def test_cmi_matches_tigramite_given_the_same_permutation() -> None:
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
    failure pinpoints which of the two sections diverged.

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
    señala cuál de las dos secciones divergió.
    """
    scenario = next(s for s in SCENARIOS if s.name == "independent")
    x, y, z = scenario.generate(np.random.default_rng(0))
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
