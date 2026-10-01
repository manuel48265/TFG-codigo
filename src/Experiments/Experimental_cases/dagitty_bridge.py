import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# TODO: port the enumeration of minimal separators to Python (about 130 lines of dagitty's
# GraphAnalyzer.js: listMinimalImplications, listDseparators, listMinimalSeparators and
# connectedComponentAvoiding) to drop the dependency on R.
_R_SCRIPT = """
library(dagitty)
g <- dagitty(paste(readLines(file("stdin")), collapse = " "))
for (s in impliedConditionalIndependencies(g, type = "missing.edge"))
  cat(paste(c(s$X, s$Y, unlist(s$Z)), collapse = "\\t"), "\\n", sep = "")
"""


@dataclass(frozen=True)
class Independence:
    """Conditional independence ``x ⫫ y | z`` implied by a DAG.

    Attributes
    ----------
    x, y : str
        Names of the two variables.

    z : tuple of str
        Names of the conditioning variables; empty for a marginal independence.
    """

    x: str
    y: str
    z: tuple[str, ...]


def implied_independencies(
    nodes: Sequence[str], edges: Iterable[tuple[str, str]]
) -> list[Independence]:
    """List the independencies a DAG implies, one or more per pair of non-adjacent nodes.

    Calls ``dagitty::impliedConditionalIndependencies(type = "missing.edge")``, which for
    every pair of non-adjacent nodes returns each minimal set of variables that d-separates
    them. A pair can therefore appear more than once, with different conditioning sets.
    Requires ``Rscript`` and the R package ``dagitty``.

    Parameters
    ----------
    nodes : sequence of str
        Names of all the nodes, including those without edges.

    edges : iterable of (str, str)
        Directed edges ``(parent, child)``.

    Returns
    -------
    independencies : list of Independence
        The implied independencies, in the order dagitty returns them.

    Raises
    ------
    RuntimeError
        If R or dagitty fail; the message is the error output of R.
    """
    declaration = " ".join(f'"{node}"' for node in nodes)
    arrows = " ".join(f'"{parent}" -> "{child}"' for parent, child in edges)
    result = subprocess.run(
        ["Rscript", "-e", _R_SCRIPT],
        input=f"dag {{ {declaration} {arrows} }}\n",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    rows = (line.split("\t") for line in result.stdout.splitlines() if line.strip())
    return [Independence(row[0], row[1], tuple(row[2:])) for row in rows]
