import re
from pathlib import Path

_BNLEARN_DIR = Path(__file__).parent / "bnlearn_networks"

BNLEARN_NETWORKS = {
    "asia": _BNLEARN_DIR / "asia.bif",
    "sachs": _BNLEARN_DIR / "sachs.bif",
    "alarm": _BNLEARN_DIR / "alarm.bif",
    "insurance": _BNLEARN_DIR / "insurance.bif",
}

_PROBABILITY_HEADER = re.compile(r"probability\s*\(\s*([^|)\s]+)\s*(?:\|\s*([^)]+))?\)")


def _parse_bif_structure(text: str) -> dict[str, list[str]]:
    parents_by_name: dict[str, list[str]] = {}
    for child, parent_str in _PROBABILITY_HEADER.findall(text):
        parents_by_name[child] = [p.strip() for p in parent_str.split(",")] if parent_str else []
    return parents_by_name


def _topological_order(parents_by_name: dict[str, list[str]]) -> list[str]:
    order: list[str] = []
    resolved: set[str] = set()
    remaining = dict(parents_by_name)
    while remaining:
        ready = sorted(name for name, parents in remaining.items() if resolved.issuperset(parents))
        if not ready:
            raise ValueError("cycle detected while sorting bnlearn network")
        order.extend(ready)
        resolved.update(ready)
        for name in ready:
            del remaining[name]
    return order


def load_bif_topology(path: Path) -> list[list[int]]:
    parents_by_name = _parse_bif_structure(path.read_text())
    order = _topological_order(parents_by_name)
    index_of = {name: i for i, name in enumerate(order)}
    return [sorted(index_of[p] for p in parents_by_name[name]) for name in order]
