import numpy as np
import pandas as pd

from src.CI_tests.Fisher_Z.fisher_z import fisher_z_test
from src.Experiments.Experimental_cases.dagitty_bridge import Independence
from src.Experiments.Experimental_cases.multi_sample_split import multi_split_p_values


def test_violated_statement_is_rejected_and_true_ones_are_not() -> None:
    rng = np.random.default_rng(0)
    n = 600
    a = rng.normal(size=n)
    b = a + rng.normal(size=n)
    c = b + rng.normal(size=n)
    data = pd.DataFrame({"a": a, "b": b, "c": c, "noise": rng.normal(size=n)})
    statements = [
        Independence("a", "c", ("b",)),  # true: chain a -> b -> c
        Independence("noise", "c", ("b",)),  # true
        Independence("a", "c", ("noise",)),  # false: a and c are dependent
    ]
    p, selected = multi_split_p_values(data, statements, fisher_z_test, rng, n_splits=40)
    assert p[2] < 0.01
    assert selected[2] > 0.9
    assert p[0] > 0.05 and p[1] > 0.05
    assert selected[0] < 0.3 and selected[1] < 0.3


def test_nothing_selected_gives_p_value_one() -> None:
    rng = np.random.default_rng(1)
    data = pd.DataFrame(rng.normal(size=(200, 3)), columns=["x", "y", "z"])
    p, selected = multi_split_p_values(
        data, [Independence("x", "y", ("z",))], fisher_z_test, rng, n_splits=20, screen_alpha=0.0
    )
    assert p[0] == 1.0 and selected[0] == 0.0


def test_selection_frequency_is_high_for_a_violation_and_low_for_a_true_statement() -> None:
    from src.Experiments.Experimental_cases.multi_sample_split import selection_frequency

    rng = np.random.default_rng(2)
    n = 600
    a = rng.normal(size=n)
    b = a + rng.normal(size=n)
    data = pd.DataFrame({"a": a, "b": b, "c": b + rng.normal(size=n)})
    statements = [Independence("a", "c", ("b",)), Independence("a", "c", ())]
    freq, failed = selection_frequency(
        data, statements, fisher_z_test, np.random.default_rng(0), 40
    )
    assert (failed == 0).all()
    assert freq[0] < 0.2  # true statement: rejected about alpha of the time
    assert freq[1] == 1.0  # violated statement
    again, _ = selection_frequency(data, statements, fisher_z_test, np.random.default_rng(0), 40)
    assert (freq == again).all()  # same seed, same splits


def test_selection_frequency_counts_failures_as_not_rejected() -> None:
    from src.CI_tests.common import CIResult
    from src.Experiments.Experimental_cases.multi_sample_split import selection_frequency

    def broken(data: pd.DataFrame, x: str, y: str, z: list[str]) -> CIResult:
        raise ValueError("constant column")

    data = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 3)), columns=["x", "y", "z"])
    freq, failed = selection_frequency(
        data, [Independence("x", "y", ("z",))], broken, np.random.default_rng(0), 7
    )
    assert freq.tolist() == [0.0] and failed.tolist() == [7]
