import pytest

from src.Experiments.DAG_Generation.real_dag_topologies import BNLEARN_NETWORKS, load_bif_topology

# (n_nodes, n_edges) as reported in Referencias/survey.pdf, section 9.5.
_EXPECTED_SIZE = {
    "asia": (8, 8),
    "sachs": (11, 17),
    "alarm": (37, 46),
    "insurance": (27, 52),
}


@pytest.mark.parametrize("name", sorted(BNLEARN_NETWORKS))
def test_topology_matches_survey_size(name: str) -> None:
    parents = load_bif_topology(BNLEARN_NETWORKS[name])
    n_edges = sum(len(p) for p in parents)

    expected_nodes, expected_edges = _EXPECTED_SIZE[name]
    assert len(parents) == expected_nodes
    assert n_edges == expected_edges


@pytest.mark.parametrize("name", sorted(BNLEARN_NETWORKS))
def test_topology_is_acyclic_by_construction(name: str) -> None:
    parents = load_bif_topology(BNLEARN_NETWORKS[name])
    for j, node_parents in enumerate(parents):
        assert all(p < j for p in node_parents)
