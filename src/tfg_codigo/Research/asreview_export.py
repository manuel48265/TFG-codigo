"""Write a pool of candidates as a CSV that ASReview can load, with the `included` column
carrying prior knowledge: "1" relevant, "0" irrelevant, "" not yet screened.
"""

from pathlib import Path

from .queries_research import write_csv

ASREVIEW_FIELDS = ["title", "abstract", "authors", "date", "doi", "included", "source"]


def to_asreview_row(row: dict, included: str, source: str) -> dict:
    """Map a row from queries_research/snowball's dict shape (with `openalex_id`/
    `scopus_id`, `query`, ...) to the columns ASReview's loader expects."""
    return {
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "authors": row.get("authors", ""),
        "date": row.get("date", ""),
        "doi": row.get("doi", ""),
        "included": included,
        "source": source,
    }


def write_asreview_csv(rows: list[dict], path: Path) -> None:
    """`rows` must already be in ASReview's column shape (see `to_asreview_row`) and
    deduplicated by the caller."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, rows)
