"""
# Algoritmo 1 del paper
def _local_permutation(z (n,dz)|None, k_perm, rng) -> perm (n,):
    if z is None:
        return rng.permutation(n)   # sin Z, permutación global normal
    neighbor_lists = cKDTree(z).query(z, k=k_perm, p=inf)[índices]  # incl. self como vecino 0
    shuffle cada fila de neighbor_lists
    used = set()
    perm = array vacío de tamaño n
    for i in rng.permutation(n):          # orden aleatorio de visita
        for candidato in neighbor_lists[i]:
            if candidato not in used:
                perm[i] = candidato; used.add(candidato); break
        else:
            perm[i] = neighbor_lists[i][0]   # agotados los k_perm-1 intentos: con reemplazo
    return perm

# Eq. 6: p-valor por Monte Carlo
def _shuffle_pvalue(x, y, z, k, k_perm, n_perm, observed, rng) -> float:
    null_dist = []
    for b in range(n_perm):
        perm = _local_permutation(z, k_perm, rng)
        x_perm = x[perm]                  # el paper permuta X; tigramite por defecto permuta Y
        k_xz, k_yz, k_z = _neighbor_counts(x_perm, y, z, k)
        null_dist.append(_cmi_estimate(k, k_xz, k_yz, k_z))
    return mean(null_dist >= observed)    # Eq. 6 tal cual; +1/(B+1) es un suavizado habitual,
    no está en el paper

# función pública, firma al estilo fisher_z_test
def cmiknn_test(x, y, z=None, k=0.2, k_perm=5, n_perm=1000, alpha=0.05, seed=None) -> CIResult:
    rng = default_rng(seed)
    x, y, z = _rank_transform(x), _rank_transform(y), _rank_transform(z) if z is not None else None
    k_here = max(1, int(k*n)) if k < 1 else int(k)
    k_xz, k_yz, k_z = _neighbor_counts(x, y, z, k_here)
    value = _cmi_estimate(k_here, k_xz, k_yz, k_z)
    p_value = _shuffle_pvalue(x, y, z, k_here, k_perm, n_perm, value, rng)
    return CIResult(statistic=value, p_value=p_value, reject=(p_value < alpha))

"""

import numpy as np
from numpy.typing import NDArray
from scipy import spatial, special

from src.CI_tests.common import CIResult


class CMIknn:
    def __init__(self, knn: int | float, seed: int) -> None:

        self.random_state = np.random.default_rng(seed=seed)
        self.knn = knn

    ######################################################################################
    # Rank transform
    # we rank-transform the samples individually in
    # each dimension: Firstly, to avoid points with equal
    # distance, small amplitude random noise is added to break ties.

    # Then, for all n values x1 , . . . , xn , we
    # replace xi with the transformed value r, where r is
    # defined such that xi is the rth largest among all x values.

    # In the oficial implementation they have chosen ascensing order
    # intead of descending as they say in the paper.
    def _rank_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        data = data + (
            1e-6 * data.std(axis=0, keepdims=True) * self.random_state.random(data.shape)
        )
        return data.argsort(axis=0).argsort(axis=0).astype(np.float64)

    ######################################################################################
    # easier way to get the exact amount of neighbor we are going to use
    def _get_knn(self, sample_size: int) -> int:
        if self.knn < 1:
            return max(1, int(self.knn * sample_size))
        else:
            return max(1, int(self.knn))

    ######################################################################################
    # Equation 5 in the paper
    def cmi_estimator(
        k: int, k_z: NDArray[np.int64], k_xz: NDArray[np.int64], k_yz: NDArray[np.int64]
    ) -> float:
        return special.digamma(k) + np.mean(
            special.digamma(k_z) - special.digamma(k_xz) - special.digamma(k_yz)
        )

    ######################################################################################
    # The main computational cost comes from
    # searching nearest neighbors in the high dimensional
    # subspaces which we speed up using KD-tree neighbor
    # search (Maneewongvatana and Mount, 1999). Hence,
    # the computational complexity will typically scale
    # less than quadratically with the sample size.

    def _neighbor_counts(
        x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64] | None, knn: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        size: int = x.shape[0]
        joint: NDArray[np.float64] = np.stack(x, y, z) if z is not None else np.stack(x, y)
        tree = spatial.cKDTree(joint)
        eps = tree.query(joint, k=[knn + 1], p=np.inf)

        # From the original implementation
        # To search neighbors < eps instead of <= eps, we need to reduce eps by a bit
        eps = np.multiply(eps, 0.999999999)

        k_xz = spatial.cKDTree(np.stack(x, z)).query_ball_point(
            np.stack(x, z), r=eps, p=np.inf, return_length=True
        )
        k_yz = spatial.cKDTree(np.stack(y, z)).query_ball_point(
            np.stack(y, z), r=eps, p=np.inf, return_length=True
        )

        if z is not None:
            k_z = spatial.cKDTree(z).query_ball_point(z, r=eps, p=np.inf, return_length=True)
        else:
            k_z = np.full(size, size, dtype=np.float64)

        return k_xz, k_yz, k_z

    ######################################################################################


def CMIknn_test(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64] | None = None,
) -> CIResult:
    pass
