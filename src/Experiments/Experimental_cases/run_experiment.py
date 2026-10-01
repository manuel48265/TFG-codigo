"""Experiment on the four fundamental structures: which CI tests recover each of them.

Every case is a dataset generated from one structure (independence, direct dependence,
collider or confounder) with a model family, a sample size, a size of the conditioning
set and a mix of variable types. Each CI test is run on it, marginally and given the
conditioning set, and the pair of decisions classifies the structure.

The runs are reproducible and can be resumed at any moment. The seed of a dataset
depends only on its identity, not on the order of execution, and the results of a dataset
are written to the CSV file all at once; when the experiment is launched again, the
datasets already in the file are skipped and the rest are run.

Run it with ``python -m src.Experiments.Experimental_cases.run_experiment``.
"""

import argparse
import hashlib
import multiprocessing
import os
import subprocess
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.CI_tests.CMIknn.CMIknn import cmiknn_test
from src.CI_tests.common import CIResult
from src.CI_tests.Fisher_Z.fisher_z import fisher_z_test
from src.CI_tests.GCM.gcm import gcm_test
from src.CI_tests.KCI.kci import kci_test
from src.CI_tests.Mixed_CI.mixed_ci import mixed_ci_test
from src.CI_tests.pearson_chi2.pearson_chi2 import pearson_chi2_test
from src.Experiments.DAG_Generation.fundamental_topology_generator import (
    FundamentalCase,
    fundamental_topology,
)
from src.Experiments.Experimental_cases.assemble import GeneratedDataset, assemble_dataset
from src.Experiments.Experimental_cases.common import (
    TrialResult,
    append_trial_results,
    build_trial_result,
    discretize_continuous_columns,
    has_constant_column,
    remove_unused_categories,
    to_float_columns,
)
from src.Experiments.Node_Generation.causal_models import (
    ADDITIVE_NOISE_MODELS,
    LINEAR_GAUSSIAN,
    LINEAR_NON_GAUSSIAN,
    LOCAL_SCALE_NOISE_MODELS,
    POST_NON_LINEAR,
    Model,
)
from src.Experiments.Node_Generation.utils import VariableType

RULES: dict[str, Model] = {
    "linear_gaussian": LINEAR_GAUSSIAN,
    "linear_non_gaussian": LINEAR_NON_GAUSSIAN,
    "additive_noise": ADDITIVE_NOISE_MODELS,
    "post_nonlinear": POST_NON_LINEAR,
    "local_scale": LOCAL_SCALE_NOISE_MODELS,
}
TYPE_MIXES = ("continuous", "half_categorical", "categorical")
CATEGORICAL_TYPES = (VariableType.BINARY, VariableType.ORDINAL, VariableType.NOMINAL)
N_SAMPLES = (500, 1000)  # nested: the smaller datasets are the first rows of the largest
MAX_ATTEMPTS = 20  # datasets with a constant column are generated again, up to this many times
DEFAULT_OUTPUT = Path("src/Experiments/Results/experiment_confusion.csv")


@dataclass(frozen=True)
class Case:
    """One dataset of the experiment: a structure and the conditions it is generated under.

    Attributes
    ----------
    topology : FundamentalCase
        Structure of the three kinds of variables (x, y and the conditioning set).

    z_size : int
        Number of conditioning variables.

    n_samples : int
        Number of observations.

    rule : str
        Key of ``RULES``: the model family.

    type_mix : str
        Element of ``TYPE_MIXES``: how the type of each variable is drawn.

    rep : int
        Repetition number.
    """

    topology: FundamentalCase
    z_size: int
    n_samples: int
    rule: str
    type_mix: str
    rep: int

    @property
    def dataset_id(self) -> str:
        """Unique, readable identifier of the case."""
        return (
            f"{self.topology.value}-z{self.z_size}-n{self.n_samples}-"
            f"{self.rule}-{self.type_mix}-r{self.rep}"
        )

    @property
    def seed(self) -> int:
        """Seed derived from the identifier without the sample size, shared by the sample sizes.

        It does not depend on the order of the runs either.
        """
        text = self.dataset_id.replace(f"-n{self.n_samples}-", "-")
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") >> 1


@dataclass(frozen=True)
class CITestSpec:
    """A CI test with the post-processing it needs and its extra arguments.

    Attributes
    ----------
    name : str
        Name written in the results.

    test : callable
        Test with the common signature on a DataFrame, plus optional keyword arguments.

    post_process : callable
        Maps the generated data to the format the test accepts.

    kwargs : callable
        Maps the seed of the dataset to extra keyword arguments of the test.
    """

    name: str
    test: Callable[..., CIResult]
    post_process: Callable[[pd.DataFrame], pd.DataFrame]
    kwargs: Callable[[int], dict[str, object]] = lambda seed: {}


TESTS = [
    CITestSpec("fisher_z", fisher_z_test, to_float_columns),
    CITestSpec("kci", kci_test, to_float_columns),
    CITestSpec("gcm", gcm_test, to_float_columns),
    CITestSpec(
        "cmiknn", cmiknn_test, to_float_columns, lambda seed: {"workers": 1, "seed": seed % 2**31}
    ),
    CITestSpec("pearson_chi2", pearson_chi2_test, discretize_continuous_columns),
    CITestSpec("mixed_ci", mixed_ci_test, remove_unused_categories),
]


def build_cases(
    reps: int = 50,
    topologies: tuple[FundamentalCase, ...] = tuple(FundamentalCase),
    z_sizes: tuple[int, ...] = (1, 2, 3),
    n_samples: tuple[int, ...] = N_SAMPLES,
    rules: tuple[str, ...] = tuple(RULES),
    type_mixes: tuple[str, ...] = TYPE_MIXES,
) -> list[Case]:
    """List the cases of the experiment, with the repetition as the outermost loop.

    That order makes every combination of conditions have the same number of repetitions
    (give or take one) at any moment, so a partial run is balanced.

    Parameters
    ----------
    reps : int, optional (default: 50)
        Repetitions of every combination of the other arguments.

    topologies, z_sizes, n_samples, rules, type_mixes : tuples
        Values of each condition.

    Returns
    -------
    cases : list of Case
    """
    return [
        Case(topology, z_size, n, rule, mix, rep)
        for rep in range(reps)
        for topology in topologies
        for z_size in z_sizes
        for n in n_samples
        for rule in rules
        for mix in type_mixes
    ]


def sample_types(rng: np.random.Generator, n_nodes: int, type_mix: str) -> list[VariableType]:
    """Draw the type of each variable independently according to the mix.

    ``continuous``: all variables continuous. ``categorical``: each one binary, ordinal or
    nominal with equal probability. ``half_categorical``: continuous with probability 1/2,
    otherwise categorical as above.

    Parameters
    ----------
    rng : Generator
        Source of randomness.

    n_nodes : int
        Number of variables.

    type_mix : str
        Element of ``TYPE_MIXES``.

    Returns
    -------
    types : list of VariableType
    """
    types = []
    for _ in range(n_nodes):
        categorical = type_mix == "categorical" or (
            type_mix == "half_categorical" and rng.random() < 0.5
        )
        types.append(
            CATEGORICAL_TYPES[int(rng.integers(len(CATEGORICAL_TYPES)))]
            if categorical
            else VariableType.CONTINUOUS
        )
    return types


def _generate(case: Case) -> tuple[GeneratedDataset, list[VariableType], int] | None:
    """Generate the dataset with the largest sample size, again if a column is constant.

    The data of a case are the first rows of it, so the sample sizes are nested. A dataset is
    accepted only if none of its nested subsets has a constant column.
    """
    n_max = max(*N_SAMPLES, case.n_samples)
    topology = fundamental_topology(case.topology, case.z_size)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        rng = np.random.default_rng(case.seed + attempt - 1)
        types = sample_types(rng, len(topology.topology), case.type_mix)
        dataset = assemble_dataset(rng, topology.topology, types, RULES[case.rule], n_max)
        if not any(
            has_constant_column(dataset.data.iloc[:n]) for n in (*N_SAMPLES, case.n_samples)
        ):
            return dataset, types, attempt
    return None


def _labels(case: Case) -> dict[str, str]:
    """Names ``x``, ``y``, ``z0``, ... of the variables ``v0``, ``v1``, ..."""
    topology = fundamental_topology(case.topology, case.z_size)
    labels = {f"v{topology.x_idx}": "x", f"v{topology.y_idx}": "y"}
    labels.update({f"v{index}": f"z{k}" for k, index in enumerate(topology.z_idxs)})
    return labels


def _roles_text(dataset: GeneratedDataset, labels: dict[str, str]) -> str:
    """Compact text of the roles: ``child:scale=parent|mean=parent,parent;...``."""
    parts = []
    for child, (mean_parents, scale_parent) in dataset.roles.items():
        scale = labels[scale_parent] if scale_parent is not None else "-"
        mean = ",".join(labels[p] for p in mean_parents) or "-"
        parts.append(f"{labels[child]}:scale={scale}|mean={mean}")
    return ";".join(parts)


def run_dataset(case: Case, code_version: str = "") -> list[TrialResult]:
    """Generate the dataset of a case and run every CI test on it.

    Each test is run marginally and given the conditioning set, and the pair of decisions
    gives the predicted structure. A test that raises an error is recorded with the error
    and without p-values instead of stopping the experiment; the warnings raised while a
    test runs are counted and not shown.

    Parameters
    ----------
    case : Case
        Dataset to generate.

    code_version : str, optional (default: "")
        Version of the code, written in every row.

    Returns
    -------
    results : list of TrialResult
        One per test in ``TESTS``, in that order.
    """
    generated = _generate(case)
    if generated is None:
        message = f"a column was constant in {MAX_ATTEMPTS} attempts"
        return [_failure(case, spec.name, MAX_ATTEMPTS, message, code_version) for spec in TESTS]
    dataset, types, attempts = generated
    topology = fundamental_topology(case.topology, case.z_size)
    x, y = f"v{topology.x_idx}", f"v{topology.y_idx}"
    z = [f"v{index}" for index in topology.z_idxs]
    x_type, y_type = types[topology.x_idx].value, types[topology.y_idx].value
    z_types = ";".join(types[index].value for index in topology.z_idxs)
    roles = _roles_text(dataset, _labels(case))

    results = []
    for spec in TESTS:
        try:
            prepared = spec.post_process(dataset.data.iloc[: case.n_samples].reset_index(drop=True))
            extra = spec.kwargs(case.seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                start = time.perf_counter()
                marginal = spec.test(prepared, x, y, [], **extra)
                middle = time.perf_counter()
                conditional = spec.test(prepared, x, y, z, **extra)
                end = time.perf_counter()
            result = build_trial_result(
                case.topology,
                case.z_size,
                case.n_samples,
                case.rule,
                case.type_mix,
                spec.name,
                case.rep,
                case.seed,
                marginal,
                conditional,
            )
            result = replace(
                result,
                marginal_seconds=middle - start,
                conditional_seconds=end - middle,
                n_warnings=len(caught),
            )
        except Exception as error:
            result = _failure(case, spec.name, attempts, f"{type(error).__name__}: {error}", "")
        results.append(
            replace(
                result,
                dataset_id=case.dataset_id,
                attempts=attempts,
                x_type=x_type,
                y_type=y_type,
                z_types=z_types,
                roles=roles,
                code_version=code_version,
            )
        )
    return results


def _failure(
    case: Case, test_name: str, attempts: int, message: str, code_version: str
) -> TrialResult:
    nan = float("nan")
    return TrialResult(
        case=case.topology.value,
        z_size=case.z_size,
        n_samples=case.n_samples,
        rule=case.rule,
        type_mix=case.type_mix,
        ci_test=test_name,
        rep=case.rep,
        seed=case.seed,
        marginal_statistic=None,
        marginal_p_value=nan,
        marginal_reject=False,
        conditional_statistic=None,
        conditional_p_value=nan,
        conditional_reject=False,
        predicted_case="error",
        correct=False,
        dataset_id=case.dataset_id,
        attempts=attempts,
        error=message,
        code_version=code_version,
    )


def repair_results(path: Path) -> set[str]:
    """Prepare a results file to be resumed and return the datasets it already holds.

    A last line cut by an interruption is removed, and so are the rows of datasets that
    do not have exactly one row per test (a dataset is written in a single operation, but
    a write can still be cut), so those datasets are run again from scratch.

    Parameters
    ----------
    path : Path
        CSV file of results; it may not exist.

    Returns
    -------
    done : set of str
        Identifiers of the complete datasets in the file.
    """
    if not path.exists() or path.stat().st_size == 0:
        return set()
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        path.write_bytes(raw[: raw.rfind(b"\n") + 1])
        if path.stat().st_size == 0:
            return set()
    data = pd.read_csv(path, dtype={"dataset_id": str})
    counts = data["dataset_id"].value_counts()
    complete = set(counts[counts == len(TESTS)].index)
    if len(complete) * len(TESTS) != len(data):
        temporary = path.with_suffix(".tmp")
        data[data["dataset_id"].isin(complete)].to_csv(temporary, index=False)
        os.replace(temporary, path)
    return complete


def code_version() -> str:
    """Return the short hash of the current commit, with ``+dirty`` if there are changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("+dirty" if changes else "")


def run_experiment(
    output: Path,
    cases: Sequence[Any],
    workers: int = 1,
    limit: int | None = None,
    report_every: int = 25,
    runner: Callable[..., Sequence[Any]] = run_dataset,
) -> int:
    """Run the pending cases and append their results to ``output``.

    The datasets already in the file are skipped, so the experiment can be stopped and
    launched again with the same arguments. With several workers the datasets are run in
    parallel processes; at most one dataset per worker is lost if it is interrupted.

    Parameters
    ----------
    output : Path
        CSV file of results.

    cases : sequence of Case
        Cases of the experiment, in the order they are run; each one needs a ``dataset_id``.

    workers : int, optional (default: 1)
        Number of processes.

    limit : int or None, optional (default: None)
        Run at most this many pending cases.

    report_every : int, optional (default: 25)
        Print the progress every this many datasets.

    runner : callable, optional (default: run_dataset)
        Module-level function ``runner(case, code_version=...)`` that returns the rows of a
        case, one per CI test, as dataclass instances. It must be picklable.

    Returns
    -------
    executed : int
        Number of datasets run in this call.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    done = repair_results(output)
    pending = [case for case in cases if case.dataset_id not in done]
    if limit is not None:
        pending = pending[:limit]
    print(f"{len(done)} datasets already done, {len(pending)} to run", flush=True)
    if not pending:
        print("Experiment finished: every case is already in the file.", flush=True)

    version = code_version()
    run = partial(runner, code_version=version)
    started = time.perf_counter()

    def write(count: int, rows: Sequence[Any]) -> None:
        append_trial_results(output, rows)
        if count % report_every == 0 or count == len(pending):
            elapsed = time.perf_counter() - started
            remaining = elapsed / count * (len(pending) - count)
            print(
                f"{count}/{len(pending)} datasets, {elapsed / 60:.1f} min, "
                f"about {remaining / 60:.1f} min left",
                flush=True,
            )

    if workers <= 1:
        for count, case in enumerate(pending, start=1):
            write(count, run(case))
    else:
        # one thread per process, so that the processes do not compete (XGBoost, BLAS)
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[variable] = "1"
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            for count, rows in enumerate(pool.imap_unordered(run, pending), start=1):
                write(count, rows)
    return len(pending)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--limit", type=int, default=None)
    arguments = parser.parse_args()
    run_experiment(
        arguments.output, build_cases(arguments.reps), arguments.workers, arguments.limit
    )


if __name__ == "__main__":
    main()
