import numpy as np


def erdos_renyi_topology(
    rng: np.random.Generator,
    n_nodes: int,
    edge_prob: float,
    max_in_degree: int | None = None,
) -> list[list[int]]:
    parents: list[list[int]] = [[] for _ in range(n_nodes)]
    for j in range(1, n_nodes):
        candidates = np.flatnonzero(rng.random(j) < edge_prob)
        if max_in_degree is not None and candidates.size > max_in_degree:
            candidates = np.sort(rng.choice(candidates, size=max_in_degree, replace=False))
        parents[j] = candidates.tolist()
    return parents
