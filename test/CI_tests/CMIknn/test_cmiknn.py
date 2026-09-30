import numpy as np
import pytest
from tigramite.independence_tests.cmiknn import CMIknn as TigramiteCMIknn

from src.CI_tests.CMIknn.CMIknn import CMIknn, cmiknn_test
from test.scenarios import SCENARIOS, Scenario


def test_ranks_1d_data_descending_without_ties() -> None:
    """30 is the largest -> rank 0 ("x_i is the r-th largest", paper
    §2.2); 10 is the smallest -> rank 2. No ties, so the tie-break noise
    can't change the outcome.
    """
    x = np.array([30.0, 10.0, 20.0])

    result = CMIknn()._rank_transform(x)

    assert result.shape == x.shape
    assert result.tolist() == [2.0, 0.0, 1.0]


def test_breaks_ties_into_a_valid_permutation() -> None:
    """1.0 appears twice; the tie-break noise must still turn the
    result into a genuine 0..n-1 permutation (no repeated ranks), even
    though which of the two tied points gets which rank is arbitrary.
    """
    x = np.array([1.0, 1.0, 2.0, 5.0])

    result = CMIknn()._rank_transform(x)

    assert result[3] == 3.0
    assert result[2] == 2.0


def test_ranks_each_column_of_2d_data_independently() -> None:
    """Column 0 and column 1 have unrelated orderings; rank_transform
    must rank each one on its own, not mix them.
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


def test_local_permutation_output_always_drawn_from_declared_neighbors() -> None:
    """Hard invariant of `_local_permutation` (Algorithm 1): whatever
    index it assigns to a point, that index must be one of the point's own
    declared neighbors -- true regardless of collisions with `used`.
    """
    rng = np.random.default_rng(0)
    n, k_perm = 30, 4
    neighbors = rng.integers(0, n, size=(n, k_perm)).astype(np.int32)

    permutation = CMIknn(k_perm=k_perm, seed=0)._local_permutation(neighbors.copy())

    for i in range(n):
        assert permutation[i] in neighbors[i]


def test_local_permutation_is_a_valid_permutation_when_k_perm_covers_all_points() -> None:
    """When every row lists all n indices as candidates (k_perm=n), a
    pigeonhole argument guarantees no collision: by the time the last
    not-yet-assigned point is processed, at most n-1 indices are in `used`,
    and its row of n distinct candidates must contain one that isn't. So
    the output must be a genuine bijection on 0..n-1 -- not just "drawn
    from declared neighbors" (the weaker invariant above), a full
    collision-free permutation.
    """
    n = 20
    neighbors = np.tile(np.arange(n, dtype=np.int32), (n, 1))

    permutation = CMIknn(k_perm=n, seed=0)._local_permutation(neighbors)

    assert sorted(permutation.tolist()) == list(range(n))


def test_unconditional_z_none_does_not_crash_and_distinguishes_dependence() -> None:
    """Regression test: z=None used to crash `_neighbor_counts` with
    `ValueError: data must be of shape (n, m)`, since a bare 1D x/y falls
    back to itself instead of a column vector (fixed by `.reshape(-1, 1)`).
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
    """null_fit="gamma" fits a Gamma to the (small) null sample and
    reads the p-value off its CDF instead of counting exceedances, so it
    should still separate dependence from independence with far fewer
    permutation draws than the default empirical count needs.
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
    """Isolates the actual CMI computation (Eq. 5 + neighbor counting
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
