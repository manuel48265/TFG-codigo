from collections import Counter

import numpy as np
import pandas as pd
import pytest

from src.CI_tests.Fisher_Z.fisher_z import fisher_z_test
from src.Experiments.Experimental_cases import run_pc_experiment
from src.Experiments.Experimental_cases.pc_runner import graph_to_text, run_pc
from src.Experiments.Experimental_cases.run_pc_experiment import (
    SCENARIOS,
    PCCase,
    _generate,
    assign_rules,
    run_pc_dataset,
)


def test_graph_to_text_writes_each_kind_of_edge() -> None:
    matrix = np.zeros((4, 4), dtype=int)
    matrix[0, 1], matrix[1, 0] = -1, 1  # 0 -> 1
    matrix[2, 1], matrix[1, 2] = 1, -1  # 1 -> 2 written from the larger index
    matrix[0, 2], matrix[2, 0] = -1, -1  # 0 -- 2
    matrix[1, 3], matrix[3, 1] = 1, 1  # 1 <-> 3
    assert graph_to_text(matrix, ["a", "b", "c", "d"]) == "a->b;a--c;b->c;b<->d"


def test_run_pc_recovers_a_collider() -> None:
    rng = np.random.default_rng(0)
    a, c = rng.normal(size=(2, 2000))
    data = pd.DataFrame({"a": a, "b": a + c + rng.normal(size=2000), "c": c})
    run = run_pc(data, fisher_z_test)
    assert run.edges == "a->b;c->b"
    assert run.n_ci_calls > 0
    sizes = {int(k): int(v) for k, v in (p.split(":") for p in run.cond_size_counts.split(";"))}
    assert sum(sizes.values()) == run.n_ci_calls


@pytest.mark.parametrize("n_nodes", [15, 8, 11])
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_assign_rules_counts_are_exact(scenario: str, n_nodes: int) -> None:
    shares = SCENARIOS[scenario]
    rules = assign_rules(np.random.default_rng(1), n_nodes, shares)
    counts = Counter(rules)
    assert len(rules) == n_nodes
    assert set(counts) == set(shares)
    for rule, share in shares.items():
        assert abs(counts[rule] - share * n_nodes) < 1


def test_assign_rules_baseline_with_15_nodes_is_11_and_4() -> None:
    # 10.5 and 4.5: the extra node goes to the first family on a tie
    counts = Counter(assign_rules(np.random.default_rng(1), 15, SCENARIOS["baseline"]))
    assert counts == {"linear_gaussian": 11, "linear_non_gaussian": 4}


def test_sample_sizes_are_nested_and_scenarios_share_the_graph() -> None:
    small = PCCase("er15", "baseline", 500, 2)
    large = PCCase("er15", "baseline", 1000, 2)
    other = PCCase("er15", "nonlinear", 500, 2)
    assert small.seed == large.seed != other.seed
    assert small.graph_seed == other.graph_seed

    generated = [_generate(case) for case in (small, large, other)]
    assert all(g is not None for g in generated)
    (
        (data_s, top_s, types_s, rules_s, _),
        (data_l, top_l, types_l, rules_l, _),
        (
            _,
            top_o,
            types_o,
            rules_o,
            _,
        ),
    ) = generated  # type: ignore[misc]
    assert (top_s, types_s, rules_s) == (top_l, types_l, rules_l)
    assert (top_s, types_s) == (top_o, types_o)
    assert rules_s != rules_o
    assert len(data_l.data) == 1000
    pd.testing.assert_frame_equal(data_s.data, data_l.data)  # same 1000 rows, cut afterwards


def test_run_pc_dataset_writes_truth_and_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_pc_experiment, "TESTS", [t for t in run_pc_experiment.TESTS if t.name == "fisher_z"]
    )
    (row,) = run_pc_dataset(PCCase("asia", "baseline", 500, 0))
    assert row.error == ""
    assert row.ci_test == "fisher_z"
    assert row.true_dag.count("->") == 8  # Asia has 8 edges
    assert len(row.node_types.split(";")) == 8
    assert len(row.node_models.split(";")) == 8
    assert row.estimated_graph != ""
    assert row.n_ci_calls > 0
