import numpy as np
from numpy.typing import NDArray
from scipy import spatial, special

from src.CI_tests.common import CIResult


class CMIknn:
    def __init__(
        self, knn: int | float = 0.2, k_perm: int = 5, sig_samples: int = 500, seed: int = 0
    ) -> None:

        self.random_state = np.random.default_rng(seed=seed)
        self.knn = knn
        self.k_perm = k_perm
        self.sig_samples = sig_samples

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
    def _cmi_estimator(
        self, k: int, k_z: NDArray[np.int64], k_xz: NDArray[np.int64], k_yz: NDArray[np.int64]
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
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64] | None,
        knn: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        size: int = x.shape[0]
        joint: NDArray[np.float64] = (
            np.column_stack((x, y, z)) if z is not None else np.column_stack((x, y))
        )
        tree = spatial.cKDTree(joint)
        eps = tree.query(joint, k=[knn + 1], p=np.inf)[0][:, 0].astype(np.float64)

        # From the original implementation
        # To search neighbors < eps instead of <= eps, we need to reduce eps by a bit
        eps = np.multiply(eps, 0.999999999)

        joint_x_z = np.column_stack((x, z)) if z is not None else x
        k_xz = spatial.cKDTree(joint_x_z).query_ball_point(
            joint_x_z, r=eps, p=np.inf, return_length=True
        )

        joint_y_z = np.column_stack((y, z)) if z is not None else y
        k_yz = spatial.cKDTree(joint_y_z).query_ball_point(
            joint_y_z, r=eps, p=np.inf, return_length=True
        )

        if z is not None:
            k_z = spatial.cKDTree(z).query_ball_point(z, r=eps, p=np.inf, return_length=True)
        else:
            k_z = np.full(size, size, dtype=np.float64)

        return k_xz, k_yz, k_z

    ######################################################################################
    # algorithm 1:
    def _local_permutation(self, z: NDArray[np.float64]) -> NDArray[np.int32]:

        neighbors = (
            spatial.cKDTree(z).query(z, k=self.k_perm, p=np.inf, eps=0.0)[1].astype(np.int32)
        )

        for i in range(len(neighbors)):
            self.random_state.shuffle(neighbors[i])

        used = set()
        permutation = np.zeros(z.shape[0], dtype=np.int32)
        for sample_index in self.random_state.permutation(z.shape[0]).astype(np.int32):
            m = 0
            use = neighbors[sample_index, m]

            while (use in used) and (m < self.k_perm - 1):
                m += 1
                use = neighbors[sample_index, m]

            permutation[sample_index] = use
            used.add(use)

        return permutation

    ######################################################################################
    # Equation 6:
    def _calc_pvalue(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64] | None,
        value: float,
        knn: int,
    ) -> float:
        sample_size = y.shape[0]
        null_dist = np.zeros(self.sig_samples)

        for b in range(self.sig_samples):
            if z is None:
                perm = self.random_state.permutation(sample_size)
            else:
                perm = self._local_permutation(z)

            y_perm = y[perm]
            k_xz, k_yz, k_z = self._neighbor_counts(x, y_perm, z, knn)
            null_dist[b] = self._cmi_estimator(knn, k_z, k_xz, k_yz)

        # Empirical Monte Carlo
        pval = float(np.sum(null_dist >= value) + 1) / (self.sig_samples + 1)

        return pval

    ######################################################################################
    def CMIknn_test(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64] | None = None,
        alpha: float = 0.05,
    ) -> CIResult:
        x, y, z = (
            self._rank_transform(x),
            self._rank_transform(y),
            self._rank_transform(z) if z is not None else None,
        )
        sample_size = y.shape[0]
        knn = self._get_knn(sample_size)
        k_xz, k_yz, k_z = self._neighbor_counts(x, y, z, knn)
        value = self._cmi_estimator(knn, k_z, k_xz, k_yz)
        p_value = self._calc_pvalue(x, y, z, value, knn)
        return CIResult(statistic=value, p_value=p_value, reject=(p_value < alpha))
