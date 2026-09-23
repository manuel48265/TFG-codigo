# Test de permutaciones
# Calculo de CMI 
# Null distribution

"""

# helper: aplicar tie-break noise + rank transform a una matriz de columnas
def _rank_transform(data (n, d)) -> (n, d):
    data += ruido_gaussiano_pequeño(escala ~ 1e-6 * std por columna)
    return rank(data, por columna)  # argsort().argsort()

# paso 1+2 del §2.2: contar vecinos en los subespacios
def _neighbor_counts(x (n,dx), y (n,dy), z (n,dz)|None, k) -> (k_xz, k_yz, k_z):
    joint = stack(x, y, z)                       # si z es None, joint = stack(x, y)
    eps = cKDTree(joint).query(joint, k=k+1, p=inf)[k-ésima columna]  # excluye self
    eps *= 0.999999999                           # para "<" estricto

    k_xz = cKDTree(stack(x, z)).query_ball_point(stack(x,z), r=eps, p=inf, return_length=True)
    k_yz = cKDTree(stack(y, z)).query_ball_point(stack(y,z), r=eps, p=inf, return_length=True)
    if z is not None:
        k_z = cKDTree(z).query_ball_point(z, r=eps, p=inf, return_length=True)
    else:
        k_z = full(n, n)   # sin Z, todos son "vecinos" en el espacio trivial
    return k_xz, k_yz, k_z

# Eq. 5
def _cmi_estimate(k, k_xz, k_yz, k_z) -> float:
    return digamma(k) + mean(digamma(k_z) - digamma(k_xz) - digamma(k_yz))

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
    return mean(null_dist >= observed)    # Eq. 6 tal cual; +1/(B+1) es un suavizado habitual, no está en el paper

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
from src.CI_tests.common import CIResult
from scipy import spatial,special

# Equation 5 in the paper
def cmi_estimator(
    k: int,
    k_z : NDArray[np.int64],
    k_xz : NDArray[np.int64],
    k_yz : NDArray[np.int64]
) -> float:
    return special.digamma(k) + np.mean(special.digamma(k_z) - special.digamma(k_xz) - special.digamma(k_yz))


def CMIknn(    
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64] | None = None,

)-> CIResult:
    pass 