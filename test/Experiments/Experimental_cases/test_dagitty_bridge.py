import subprocess

import pytest

from src.Experiments.Experimental_cases.dagitty_bridge import Independence, implied_independencies

_HAS_DAGITTY = (
    subprocess.run(
        ["Rscript", "-e", 'quit(status = !requireNamespace("dagitty", quietly = TRUE))'],
        capture_output=True,
    ).returncode
    == 0
)

pytestmark = pytest.mark.skipif(not _HAS_DAGITTY, reason="R package dagitty not installed")


def _as_set(independencies: list[Independence]) -> set[tuple[frozenset[str], frozenset[str]]]:
    return {(frozenset((i.x, i.y)), frozenset(i.z)) for i in independencies}


def test_returns_every_minimal_separator_of_a_pair() -> None:
    nodes = ["x", "p", "m1", "m2", "y"]
    edges = [("x", "p"), ("p", "m1"), ("p", "m2"), ("m1", "y"), ("m2", "y")]
    pair = frozenset(("x", "y"))
    found = _as_set(implied_independencies(nodes, edges))
    assert (pair, frozenset(("p",))) in found
    assert (pair, frozenset(("m1", "m2"))) in found
    assert (pair, frozenset(("m1", "p"))) not in found  # not minimal


def test_names_with_spaces_and_numbers_and_isolated_nodes() -> None:
    found = implied_independencies(["model year", "0", "lonely"], [("model year", "0")])
    assert _as_set(found) == {
        (frozenset(("model year", "lonely")), frozenset()),
        (frozenset(("0", "lonely")), frozenset()),
    }
    assert all(isinstance(i.z, tuple) for i in found)
