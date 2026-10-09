"""Experiment on real data: do the independencies implied by a reference graph hold.

For each real data set (Sachs, Auto MPG, APM) the independencies implied by its reference
graph are listed with dagitty, as Kook does: one or more per pair of non-adjacent nodes, with
a non-empty conditioning set. The statements that involve a variable without data are left
out, as Eulig et al. do. Every CI test is run on each remaining statement and the p-values
are corrected with Holm's method for each pair of data set and test. A graph is rejected by
a test if any statement is rejected.

Besides, the robustness of every statement is measured: the fraction of random half samples
in which a test rejects it. The statements robust in at least one test are ranked, one per
pair of nodes, to choose the cases to study. The robustness is a measure of how strong a
violation is inside the sample, not a replication.

Only the observational data are used and they are not transformed. The runs can be resumed:
a (data set, test) pair already in the results file is skipped. Run it with
``python -m src.Experiments.Experimental_cases.run_real_experiment``.
"""

import argparse
import hashlib
import multiprocessing
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from src.Experiments.Experimental_cases.dagitty_bridge import Independence, implied_independencies
from src.Experiments.Experimental_cases.multi_sample_split import selection_frequency
from src.Experiments.Experimental_cases.run_experiment import TESTS, CITestSpec, code_version
from src.Experiments.Real_Data.datasets import (
    RealDataset,
    load_apm,
    load_auto_mpg,
    load_sachs,
)

LOADERS: dict[str, Callable[[], RealDataset]] = {
    "sachs": load_sachs,
    "apm": load_apm,
    "auto_mpg": load_auto_mpg,
}
# Slowest tests first, so the workers finish at about the same time.
TEST_ORDER = ("cmiknn", "kci", "pearson_chi2", "gcm", "mixed_ci", "fisher_z")
ALPHA = 0.05
N_SPLITS = 50
ROBUST = 0.8
DEFAULT_OUTPUT = Path("src/Experiments/Results/experiment_real.csv")


def build_statements(dataset: RealDataset) -> tuple[list[Independence], int]:
    """List the independencies of the graph that can be tested with the data.

    Parameters
    ----------
    dataset : RealDataset
        Data and reference graph.

    Returns
    -------
    statements : list of Independence
        Implied independencies with a non-empty conditioning set whose variables all have
        a column in the data.

    n_omitted : int
        Number of statements with a non-empty conditioning set left out because a variable
        has no data.
    """
    observed = set(dataset.data.columns)
    with_condition = [s for s in implied_independencies(dataset.nodes, dataset.edges) if s.z]
    statements = [s for s in with_condition if {s.x, s.y, *s.z} <= observed]
    return statements, len(with_condition) - len(statements)


def _seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big") >> 1


def run_real_job(
    dataset: RealDataset, spec: CITestSpec, n_splits: int = N_SPLITS, version: str = ""
) -> pd.DataFrame:
    """Run one CI test on every testable statement of a data set.

    The p-value with all the data is corrected with Holm's method over the statements, and
    the robustness is the fraction of ``n_splits`` random half samples in which the statement
    is rejected. The splits depend only on the data set, so every test sees the same halves.
    A statement whose test raises an error is recorded with the error and without a p-value.

    Parameters
    ----------
    dataset : RealDataset
        Data and reference graph.

    spec : CITestSpec
        CI test to run.

    n_splits : int, optional (default: 50)
        Number of random half samples; 0 does not measure the robustness.

    version : str, optional (default: "")
        Version of the code, written in every row.

    Returns
    -------
    results : DataFrame
        One row per statement: ``dataset``, ``ci_test``, ``x``, ``y``, ``z`` (names joined by
        ``;``), ``n_samples``, ``p_value``, ``p_holm``, ``rejected`` (``p_holm < ALPHA``),
        ``robustness``, ``n_failed_splits`` (halves in which the test raised an error),
        ``seconds``, ``error`` and ``code_version``.
    """
    statements, _ = build_statements(dataset)
    data = spec.post_process(dataset.data)
    rows = []
    for s in statements:
        row: dict[str, object] = {
            "dataset": dataset.name,
            "ci_test": spec.name,
            "x": s.x,
            "y": s.y,
            "z": ";".join(s.z),
            "n_samples": len(data),
            "p_value": float("nan"),
            "error": "",
        }
        start = time.perf_counter()
        try:
            row["p_value"] = spec.test(data, s.x, s.y, list(s.z), **spec.kwargs(0)).p_value
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        row["seconds"] = time.perf_counter() - start
        rows.append(row)
    results = pd.DataFrame(rows)
    valid = results["p_value"].notna()
    results["p_holm"] = float("nan")
    results.loc[valid, "p_holm"] = multipletests(results.loc[valid, "p_value"], method="holm")[1]
    results["rejected"] = results["p_holm"] < ALPHA
    results["robustness"] = float("nan")
    results["n_failed_splits"] = 0
    if n_splits > 0:
        # each half is post-processed again: a half can miss a rare category, which some
        # tests reject (Mixed-CI) unless the unused categories are removed
        results["robustness"], results["n_failed_splits"] = selection_frequency(
            data,
            statements,
            lambda frame, x, y, z: spec.test(spec.post_process(frame), x, y, z, **spec.kwargs(0)),
            np.random.default_rng(_seed(dataset.name)),
            n_splits,
        )
    results["code_version"] = version
    return results


def _run(job: tuple[tuple[str, str], int, str]) -> pd.DataFrame:
    (name, test), n_splits, version = job
    spec = next(s for s in TESTS if s.name == test)
    return run_real_job(LOADERS[name](), spec, n_splits, version)


def rank_cases(results: pd.DataFrame, threshold: float = ROBUST) -> pd.DataFrame:
    """Rank the statements to study, one per pair of nodes.

    A statement is eligible if it is robust (at least ``threshold``) in some test. Eligible
    statements are ordered by the number of tests in which they are robust, then by the mean
    robustness in those tests and then by the smallest Holm-corrected p-value; only the best
    statement of each pair of nodes is kept.

    Parameters
    ----------
    results : DataFrame
        Rows of ``run_real_job`` of one or several data sets and tests.

    threshold : float, optional (default: 0.8)
        Smallest robustness for a statement to count as robust in a test.

    Returns
    -------
    ranking : DataFrame
        One row per eligible pair, best first inside each data set: ``dataset``, ``x``, ``y``,
        ``z``, ``n_robust_tests``, ``robust_tests``, ``mean_robustness``, ``min_p_holm``,
        ``rejected_by`` (tests that reject it after Holm) and ``contradiction`` (robust in a
        test whose p-value with all the data is above ``ALPHA``, a sign that the full-sample
        result depends on a few observations).
    """
    key = ["dataset", "x", "y", "z"]
    rows = []
    for _, group in results.groupby(key, sort=False):
        robust = group[group["robustness"] >= threshold]
        if robust.empty:
            continue
        rows.append(
            {
                **{k: group[k].iloc[0] for k in key},
                "n_robust_tests": len(robust),
                "robust_tests": ";".join(robust["ci_test"]),
                "mean_robustness": robust["robustness"].mean(),
                "min_p_holm": group["p_holm"].min(),
                "rejected_by": ";".join(group.loc[group["rejected"], "ci_test"]),
                "contradiction": bool((robust["p_value"] > ALPHA).any()),
            }
        )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    ranking["pair"] = ["~".join(sorted(p)) for p in zip(ranking["x"], ranking["y"], strict=True)]
    ranking = ranking.sort_values(
        ["dataset", "n_robust_tests", "mean_robustness", "min_p_holm"],
        ascending=[True, False, False, True],
    )
    return ranking.drop_duplicates(["dataset", "pair"]).drop(columns="pair").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", choices=list(LOADERS), default=list(LOADERS))
    parser.add_argument("--tests", nargs="+", choices=TEST_ORDER, default=list(TEST_ORDER))
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    arguments = parser.parse_args()

    done = set()
    if arguments.output.exists() and arguments.output.stat().st_size > 0:
        # a pair counts as done only with one row per statement; a cut write is run again
        previous = pd.read_csv(arguments.output, dtype={"x": str, "y": str, "z": str})
        expected = {name: len(build_statements(load())[0]) for name, load in LOADERS.items()}
        sizes = previous.groupby(["dataset", "ci_test"]).size().reset_index(name="rows")
        done = {
            (name, test)
            for name, test, rows in sizes.itertuples(index=False)
            if rows == expected.get(name)
        }
        keep = [pair in done for pair in zip(previous["dataset"], previous["ci_test"], strict=True)]
        if not all(keep):
            previous[keep].to_csv(arguments.output, index=False)
    jobs = [
        (name, test)
        for test in TEST_ORDER
        for name in LOADERS
        if name in arguments.datasets and test in arguments.tests and (name, test) not in done
    ]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    version = code_version()
    context = multiprocessing.get_context("spawn")
    with context.Pool(min(arguments.workers, max(len(jobs), 1))) as pool:
        work = [(job, arguments.splits, version) for job in jobs]
        for count, frame in enumerate(pool.imap_unordered(_run, work), start=1):
            if arguments.output.exists():
                header = pd.read_csv(arguments.output, nrows=0).columns.tolist()
                if header != frame.columns.tolist():
                    raise ValueError(
                        f"{arguments.output} has a different header; use another file."
                    )
            frame.to_csv(
                arguments.output, mode="a", header=not arguments.output.exists(), index=False
            )
            print(f"[{count}/{len(jobs)}] {frame['dataset'].iloc[0]} {frame['ci_test'].iloc[0]}")

    results = pd.read_csv(arguments.output, dtype={"x": str, "y": str, "z": str})
    summary = results.groupby(["dataset", "ci_test"]).agg(
        n_statements=("x", "size"), n_rejected=("rejected", "sum")
    )
    summary["graph_rejected"] = summary["n_rejected"] > 0
    print(summary.to_string())
    ranking = rank_cases(results)
    ranking.to_csv(arguments.output.with_name(arguments.output.stem + "_ranking.csv"), index=False)
    if ranking.empty:
        print("No statement is robust in any test.")
    else:
        print(ranking.groupby("dataset").size().rename("eligible_pairs").to_string())


if __name__ == "__main__":
    main()
