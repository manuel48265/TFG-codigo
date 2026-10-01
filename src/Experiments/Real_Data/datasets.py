from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class RealDataset:
    """A real data set together with the causal graph it is evaluated against.

    The graph is the reference graph of the source and the data are its observations. The
    graph can have nodes without a column in the data (APM) and the data can have columns
    that are not nodes of the graph (Auto MPG).

    Attributes
    ----------
    name : str
        ``'sachs'``, ``'auto_mpg'`` or ``'apm'``.

    data : DataFrame
        One column per variable.

    nodes : tuple of str
        Names of all the nodes of the graph.

    edges : tuple of (str, str)
        Directed edges ``(parent, child)`` of the graph.
    """

    name: str
    data: pd.DataFrame
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]


def _load(name: str, data: pd.DataFrame) -> RealDataset:
    table = pd.read_csv(DATA_DIR / f"{name}_edges.csv", dtype=str)
    edges = tuple(zip(table["parent"], table["child"], strict=True))
    nodes = tuple(dict.fromkeys(node for edge in edges for node in edge))
    return RealDataset(name, data, nodes, edges)


def load_sachs() -> RealDataset:
    """Load the Sachs protein-signalling data and its reference graph.

    Returns
    -------
    dataset : RealDataset
        853 observations of 11 variables and a graph with 11 nodes and 19 edges.
    """
    return _load("sachs", pd.read_csv(DATA_DIR / "sachs_data.csv"))


def load_auto_mpg() -> RealDataset:
    """Load the Auto MPG data, without missing values, and its reference graph.

    ``model year`` and ``origin`` are columns of the data but not nodes of the graph.

    Returns
    -------
    dataset : RealDataset
        392 observations of 8 variables and a graph with 6 nodes and 9 edges.
    """
    return _load("auto_mpg", pd.read_csv(DATA_DIR / "auto_mpg_data.csv"))


def load_apm() -> RealDataset:
    """Load the application performance monitoring (APM) data and its reference graph.

    Every column with a missing value is dropped from the data, but not from the graph, so
    some nodes of the graph have no data. Names are strings (``'0'``, ``'1'``, ...).

    Returns
    -------
    dataset : RealDataset
        432 observations of 26 variables and a graph with 39 nodes and 40 edges.
    """
    return _load("apm", pd.read_csv(DATA_DIR / "apm_data.csv").dropna(axis=1))
