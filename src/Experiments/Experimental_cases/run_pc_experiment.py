"""Experiment on structure recovery: how well PC recovers a graph with each CI test.

Every case is a dataset generated from a graph (a random Erdős–Rényi graph or the topology of
a bnlearn network), with a mix of variable types and of structural models. PC-stable is run
on it once per CI test, and the estimated CPDAG is written next to the true graph so that
the metrics can be computed afterwards.

The datasets are nested: the dataset with 500 observations is the first 500 rows of the one
with 1000, so the effect of the sample size is measured on the same data. The graph and the
types of the variables depend only on the topology and the repetition, so the scenarios are
compared on the same graph. The runs are reproducible and can be resumed in the same way as
``run_experiment``.

Run it with ``python -m src.Experiments.Experimental_cases.run_pc_experiment``.
"""

import argparse
import hashlib
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from src.Experiments.DAG_Generation.real_dag_topologies import (
    BNLEARN_NETWORKS,
    load_bif_topology,
)
from src.Experiments.DAG_Generation.synthetic_dag_generator import erdos_renyi_topology
from src.Experiments.Experimental_cases.assemble import GeneratedDataset, assemble_dataset
from src.Experiments.Experimental_cases.common import has_constant_column
from src.Experiments.Experimental_cases.pc_runner import run_pc
from src.Experiments.Experimental_cases.run_experiment import (
    MAX_ATTEMPTS,
    RULES,
    TESTS,
    run_experiment,
)
from src.Experiments.Node_Generation.utils import VariableType

TOPOLOGIES = ("er15", "sachs", "asia")
ER_NODES = 15
ER_EDGE_PROB = 0.2
N_SAMPLES = (500, 1000)
ALPHA = 0.05
TYPE_PROBABILITIES = {
    VariableType.CONTINUOUS: 0.35,
    VariableType.ORDINAL: 0.35,
    VariableType.BINARY: 0.15,
    VariableType.NOMINAL: 0.15,
}
SCENARIOS: dict[str, dict[str, float]] = {
    "baseline": {"linear_gaussian": 0.7, "linear_non_gaussian": 0.3},
    "nonlinear": {"linear_gaussian": 1 / 3, "additive_noise": 1 / 3, "post_nonlinear": 1 / 3},
    "heteroscedastic": {"linear_gaussian": 1 / 3, "additive_noise": 1 / 3, "local_scale": 1 / 3},
}
DEFAULT_OUTPUT = Path("src/Experiments/Results/experiment_pc.csv")


def _hash_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") >> 1


@dataclass(frozen=True)
class PCCase:
    """One dataset of the experiment: a graph, a scenario and a sample size.

    Attributes
    ----------
    topology : str
        Element of ``TOPOLOGIES``.

    scenario : str
        Key of ``SCENARIOS``: how the structural models are shared among the variables.

    n_samples : int
        Number of observations; the first ones of the dataset with ``max(N_SAMPLES)``.

    rep : int
        Repetition number.
    """

    topology: str
    scenario: str
    n_samples: int
    rep: int

    @property
    def dataset_id(self) -> str:
        """Unique, readable identifier of the case."""
        return f"{self.topology}-{self.scenario}-n{self.n_samples}-r{self.rep}"

    @property
    def graph_seed(self) -> int:
        """Seed of the graph and of the types, shared by the scenarios and sample sizes."""
        return _hash_seed(f"{self.topology}-r{self.rep}")

    @property
    def seed(self) -> int:
        """Seed of the models and of the data, shared by the sample sizes."""
        return _hash_seed(f"{self.topology}-{self.scenario}-r{self.rep}")


@dataclass
class PCTrialResult:
    dataset_id: str
    topology: str
    scenario: str
    n_samples: int
    ci_test: str
    rep: int
    seed: int
    attempts: int
    alpha: float
    true_dag: str
    node_types: str
    node_models: str
    roles: str
    estimated_graph: str
    n_ci_calls: int
    seconds: float
    error: str = ""
    code_version: str = ""


def build_cases(
    reps: int = 20,
    topologies: tuple[str, ...] = TOPOLOGIES,
    scenarios: tuple[str, ...] = tuple(SCENARIOS),
    n_samples: tuple[int, ...] = N_SAMPLES,
) -> list[PCCase]:
    """List the cases of the experiment, with the repetition as the outermost loop.

    Parameters
    ----------
    reps : int, optional (default: 20)
        Repetitions of every combination of the other arguments.

    topologies, scenarios, n_samples : tuples
        Values of each condition.

    Returns
    -------
    cases : list of PCCase
    """
    return [
        PCCase(topology, scenario, n, rep)
        for rep in range(reps)
        for topology in topologies
        for scenario in scenarios
        for n in n_samples
    ]


def assign_rules(rng: np.random.Generator, n_nodes: int, shares: dict[str, float]) -> list[str]:
    """Share the nodes among the model families in the proportions of ``shares``.

    The counts are exact (largest remainder method), so every dataset of a scenario has the
    same number of nodes of each family; only which nodes receive it is random.

    Parameters
    ----------
    rng : Generator
        Source of randomness.

    n_nodes : int
        Number of variables.

    shares : dict
        Proportion of nodes of each family, by key of ``RULES``; they add up to 1.

    Returns
    -------
    rules : list of str
        Family of each node.
    """
    exact = {rule: share * n_nodes for rule, share in shares.items()}
    counts = {rule: int(value) for rule, value in exact.items()}
    by_remainder = sorted(exact, key=lambda rule: exact[rule] - counts[rule], reverse=True)
    for rule in by_remainder[: n_nodes - sum(counts.values())]:
        counts[rule] += 1
    rules = [rule for rule, count in counts.items() for _ in range(count)]
    return [rules[i] for i in rng.permutation(n_nodes)]


def _topology(name: str, rng: np.random.Generator) -> list[list[int]]:
    if name == "er15":
        return erdos_renyi_topology(rng, ER_NODES, ER_EDGE_PROB)
    return load_bif_topology(BNLEARN_NETWORKS[name])


def _generate(
    case: PCCase,
) -> tuple[GeneratedDataset, list[list[int]], list[VariableType], list[str], int] | None:
    """Generate the dataset with the largest sample size, again if a column is constant.

    A dataset is accepted only if none of its nested subsets has a constant column.
    """
    graph_rng = np.random.default_rng(case.graph_seed)
    topology = _topology(case.topology, graph_rng)
    types = [
        list(TYPE_PROBABILITIES)[int(k)]
        for k in graph_rng.choice(
            len(TYPE_PROBABILITIES), size=len(topology), p=list(TYPE_PROBABILITIES.values())
        )
    ]
    rules = assign_rules(np.random.default_rng(case.seed), len(topology), SCENARIOS[case.scenario])
    for attempt in range(1, MAX_ATTEMPTS + 1):
        rng = np.random.default_rng(case.seed + attempt)
        dataset = assemble_dataset(
            rng, topology, types, [RULES[rule] for rule in rules], max(N_SAMPLES)
        )
        if not any(has_constant_column(dataset.data.iloc[:n]) for n in N_SAMPLES):
            return dataset, topology, types, rules, attempt
    return None


def _roles_text(dataset: GeneratedDataset) -> str:
    """Compact text of the roles: ``child:scale=parent|mean=parent,parent;...``."""
    return ";".join(
        f"{child}:scale={scale or '-'}|mean={','.join(mean) or '-'}"
        for child, (mean, scale) in dataset.roles.items()
    )


def run_pc_dataset(case: PCCase, code_version: str = "") -> list[PCTrialResult]:
    """Generate the dataset of a case and run PC with every CI test on it.

    A test that raises an error is recorded with the error and without a graph instead of
    stopping the experiment.

    Parameters
    ----------
    case : PCCase
        Dataset to generate.

    code_version : str, optional (default: "")
        Version of the code, written in every row.

    Returns
    -------
    results : list of PCTrialResult
        One per test in ``TESTS``, in that order.
    """
    generated = _generate(case)
    base = PCTrialResult(
        dataset_id=case.dataset_id,
        topology=case.topology,
        scenario=case.scenario,
        n_samples=case.n_samples,
        ci_test="",
        rep=case.rep,
        seed=case.seed,
        attempts=MAX_ATTEMPTS,
        alpha=ALPHA,
        true_dag="",
        node_types="",
        node_models="",
        roles="",
        estimated_graph="",
        n_ci_calls=0,
        seconds=float("nan"),
        code_version=code_version,
    )
    if generated is None:
        message = f"a column was constant in {MAX_ATTEMPTS} attempts"
        return [replace(base, ci_test=spec.name, error=message) for spec in TESTS]
    dataset, topology, types, rules, attempts = generated
    base.attempts = attempts
    base.true_dag = ";".join(f"v{p}->v{j}" for j, parents in enumerate(topology) for p in parents)
    base.node_types = ";".join(f"v{j}:{t.value}" for j, t in enumerate(types))
    base.node_models = ";".join(f"v{j}:{rule}" for j, rule in enumerate(rules))
    base.roles = _roles_text(dataset)
    data = dataset.data.iloc[: case.n_samples].reset_index(drop=True)

    results = []
    for spec in TESTS:
        row = replace(base, ci_test=spec.name)
        try:
            run = run_pc(spec.post_process(data), spec.test, ALPHA, **spec.kwargs(case.seed))
            row.estimated_graph = run.edges
            row.n_ci_calls = run.n_ci_calls
            row.seconds = run.seconds
        except Exception as error:
            row.error = f"{type(error).__name__}: {error}"
        results.append(row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--limit", type=int, default=None)
    arguments = parser.parse_args()
    run_experiment(
        arguments.output,
        build_cases(arguments.reps),
        arguments.workers,
        arguments.limit,
        runner=run_pc_dataset,
    )


if __name__ == "__main__":
    main()
