from dataclasses import dataclass
from enum import StrEnum


class FundamentalCase(StrEnum):
    INDEPENDENCE = "independence"
    DIRECT_DEPENDENCE = "direct_dependence"
    COLLIDER = "collider"
    CONFOUNDER = "confounder"


@dataclass
class FundamentalTopology:
    topology: list[list[int]]
    x_idx: int
    y_idx: int
    z_idxs: list[int]


def fundamental_topology(case: FundamentalCase, z_size: int) -> FundamentalTopology:
    if case == FundamentalCase.INDEPENDENCE:
        n = 2 + z_size
        return FundamentalTopology(
            topology=[[] for _ in range(n)], x_idx=0, y_idx=1, z_idxs=list(range(2, n))
        )

    elif case == FundamentalCase.DIRECT_DEPENDENCE:
        n = 2 + z_size
        topology: list[list[int]] = [[] for _ in range(n)]
        topology[1] = [0]
        return FundamentalTopology(topology=topology, x_idx=0, y_idx=1, z_idxs=list(range(2, n)))

    elif case == FundamentalCase.COLLIDER:
        n = 2 + z_size
        topology = [[] for _ in range(n)]
        for k in range(2, n):
            topology[k] = [0, 1]
        return FundamentalTopology(topology=topology, x_idx=0, y_idx=1, z_idxs=list(range(2, n)))

    else:  # CONFOUNDER
        z_idxs = list(range(z_size))
        x_idx, y_idx = z_size, z_size + 1
        topology = [[] for _ in range(z_size + 2)]
        topology[x_idx] = z_idxs.copy()
        topology[y_idx] = z_idxs.copy()
        return FundamentalTopology(topology=topology, x_idx=x_idx, y_idx=y_idx, z_idxs=z_idxs)
