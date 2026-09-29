import pytest

from src.Experiments.DAG_Generation.fundamental_topology_generator import (
    FundamentalCase,
    fundamental_topology,
)


@pytest.mark.parametrize("case", list(FundamentalCase))
@pytest.mark.parametrize("z_size", [1, 2, 3])
def test_topology_is_acyclic_by_construction(case: FundamentalCase, z_size: int) -> None:
    result = fundamental_topology(case, z_size)
    for j, parents in enumerate(result.topology):
        assert all(p < j for p in parents)


@pytest.mark.parametrize("z_size", [1, 2, 3])
def test_independence_has_no_edges(z_size: int) -> None:
    result = fundamental_topology(FundamentalCase.INDEPENDENCE, z_size)
    assert all(parents == [] for parents in result.topology)


@pytest.mark.parametrize("z_size", [1, 2, 3])
def test_direct_dependence_is_x_to_y_only(z_size: int) -> None:
    result = fundamental_topology(FundamentalCase.DIRECT_DEPENDENCE, z_size)
    assert result.topology[result.y_idx] == [result.x_idx]
    assert result.topology[result.x_idx] == []
    assert all(result.topology[z] == [] for z in result.z_idxs)


@pytest.mark.parametrize("z_size", [1, 2, 3])
def test_collider_every_z_has_x_and_y_as_parents(z_size: int) -> None:
    result = fundamental_topology(FundamentalCase.COLLIDER, z_size)
    assert result.topology[result.x_idx] == []
    assert result.topology[result.y_idx] == []
    assert all(
        sorted(result.topology[z]) == sorted([result.x_idx, result.y_idx]) for z in result.z_idxs
    )


@pytest.mark.parametrize("z_size", [1, 2, 3])
def test_confounder_x_and_y_share_all_z_as_parents(z_size: int) -> None:
    result = fundamental_topology(FundamentalCase.CONFOUNDER, z_size)
    assert all(result.topology[z] == [] for z in result.z_idxs)
    assert sorted(result.topology[result.x_idx]) == sorted(result.z_idxs)
    assert sorted(result.topology[result.y_idx]) == sorted(result.z_idxs)


def test_z_size_matches_number_of_z_indices() -> None:
    for case in FundamentalCase:
        for z_size in (1, 2, 3):
            result = fundamental_topology(case, z_size)
            assert len(result.z_idxs) == z_size
            assert len(result.topology) == z_size + 2
