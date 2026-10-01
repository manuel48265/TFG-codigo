import pandas as pd

from src.Experiments.Experimental_cases.run_real_experiment import rank_cases


def _row(test: str, x: str, y: str, z: str, p: float, holm: float, robustness: float) -> dict:
    return {
        "dataset": "d",
        "ci_test": test,
        "x": x,
        "y": y,
        "z": z,
        "p_value": p,
        "p_holm": holm,
        "rejected": holm < 0.05,
        "robustness": robustness,
    }


def test_rank_cases_orders_by_breadth_and_keeps_one_statement_per_pair() -> None:
    results = pd.DataFrame(
        [
            # pair a~b: two statements, the second one is robust in two tests
            _row("t1", "a", "b", "c", 1e-3, 0.01, 0.9),
            _row("t2", "a", "b", "c", 1e-3, 0.01, 0.5),
            _row("t1", "b", "a", "d", 1e-5, 0.001, 0.95),
            _row("t2", "b", "a", "d", 1e-5, 0.001, 0.85),
            # pair c~d: robust in one test only
            _row("t1", "c", "d", "a", 1e-6, 0.0001, 0.99),
            _row("t2", "c", "d", "a", 1e-6, 0.0001, 0.2),
            # pair e~f: not robust anywhere
            _row("t1", "e", "f", "a", 0.3, 1.0, 0.1),
            _row("t2", "e", "f", "a", 0.3, 1.0, 0.0),
        ]
    )
    ranking = rank_cases(results)
    assert list(zip(ranking["x"], ranking["y"], ranking["z"], strict=True)) == [
        ("b", "a", "d"),
        ("c", "d", "a"),
    ]
    assert ranking["n_robust_tests"].tolist() == [2, 1]
    assert ranking["robust_tests"].tolist() == ["t1;t2", "t1"]
    assert ranking["rejected_by"].tolist() == ["t1;t2", "t1;t2"]  # rejected, but robust in t1 only


def test_rank_cases_flags_robust_statements_that_the_full_sample_does_not_reject() -> None:
    results = pd.DataFrame([_row("t1", "a", "b", "c", 0.77, 1.0, 0.9)])
    assert rank_cases(results)["contradiction"].tolist() == [True]


def test_rank_cases_with_nothing_robust_is_empty() -> None:
    assert rank_cases(pd.DataFrame([_row("t1", "a", "b", "c", 0.3, 1.0, 0.1)])).empty
