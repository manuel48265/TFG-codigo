from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.Experiments.Node_Generation.causal_models import Model
from src.Experiments.Node_Generation.node_generator import generate_node, generate_root
from src.Experiments.Node_Generation.utils import VariableType

# Extra rows generated with every dataset and dropped before returning it. The constants
# the generator estimates from the data (medians, robust scales, projections, noise
# levels) are computed on them too, so they approximate population values: each
# mechanism is a fixed function of its parents, the same whatever ``n``, and the
# returned rows are independent draws from it.
REFERENCE_ROWS = 10_000


@dataclass(frozen=True)
class GeneratedDataset:
    """A generated dataset and the role of the parents of each variable.

    Attributes
    ----------
    data : DataFrame
        One column per variable, named ``v0``, ``v1``, ... in topological order.

    roles : dict
        For every variable with parents, the pair ``(mean_parents, scale_parent)`` of
        names. An edge from a scale parent has an effect on the variance and not on the
        mean. Variables without parents do not appear.
    """

    data: pd.DataFrame
    roles: dict[str, tuple[list[str], str | None]]


def assemble_dataset(
    rng: np.random.Generator,
    topology: list[list[int]],
    types: list[VariableType],
    rules: Model | Sequence[Model],
    n: int,
) -> GeneratedDataset:
    """Generate a dataset from a DAG, one variable after the other.

    ``n + REFERENCE_ROWS`` rows are generated and the first ``n`` are returned; the
    rows are independent, so they are a sample from the model, whose constants were
    estimated on all the rows.

    Parameters
    ----------
    rng : Generator
        Source of randomness.

    topology : list of list of int
        Parents of each variable, as indices of variables that come earlier in the list.

    types : list of VariableType
        Type of each variable.

    rules : Model or sequence of Model
        Family of structural models every variable belongs to; a sequence gives the family
        of each variable, in the order of ``topology``.

    n : int
        Number of observations.

    Returns
    -------
    dataset : GeneratedDataset
        The data and the roles of the parents of each variable.
    """
    columns: dict[str, pd.Series] = {}
    roles: dict[str, tuple[list[str], str | None]] = {}
    total = n + REFERENCE_ROWS
    for j, parents in enumerate(topology):
        name = f"v{j}"
        rule = rules if isinstance(rules, Model) else rules[j]
        if not parents:
            columns[name] = generate_root(rng, total, rule, types[j])
            continue
        node = generate_node(rng, pd.DataFrame(columns), [f"v{p}" for p in parents], rule, types[j])
        columns[name] = node.values
        roles[name] = (node.mean_parents, node.scale_parent)
    data = pd.DataFrame(columns).iloc[:n].reset_index(drop=True)
    return GeneratedDataset(data, roles)
