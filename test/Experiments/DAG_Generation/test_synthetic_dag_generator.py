import numpy as np

from src.Experiments.DAG_Generation.synthetic_dag_generator import erdos_renyi_topology


def test_topology_is_acyclic_by_construction() -> None:
    rng = np.random.default_rng(0)
    parents = erdos_renyi_topology(rng, n_nodes=50, edge_prob=0.3)

    assert len(parents) == 50
    assert parents[0] == []
    for j, node_parents in enumerate(parents):
        assert all(p < j for p in node_parents)


def test_max_in_degree_is_respected() -> None:
    rng = np.random.default_rng(1)
    parents = erdos_renyi_topology(rng, n_nodes=30, edge_prob=1.0, max_in_degree=2)

    for j, node_parents in enumerate(parents):
        assert len(node_parents) == min(j, 2)


def test_edge_prob_matches_expected_density() -> None:
    rng = np.random.default_rng(2)
    n_nodes, edge_prob = 200, 0.1
    parents = erdos_renyi_topology(rng, n_nodes=n_nodes, edge_prob=edge_prob)

    n_edges = sum(len(p) for p in parents)
    expected = edge_prob * n_nodes * (n_nodes - 1) / 2

    assert expected * 0.8 < n_edges < expected * 1.2
