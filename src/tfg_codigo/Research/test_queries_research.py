import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

from tfg_codigo.Research.queries_research import (
    merge_and_dedup,
    reconstruct_abstract,
    search_openalex,
    search_scopus,
    write_csv,
)


def _page(entries, total):
    response = MagicMock()
    response.json.return_value = {
        "search-results": {
            "entry": entries,
            "opensearch:totalResults": str(total),
        }
    }
    return response


@patch("tfg_codigo.Research.queries_research.requests.get")
def test_search_scopus_paginates_until_total_reached(mock_get):
    entry = lambda i: {"dc:identifier": f"SCOPUS_ID:{i}", "dc:title": f"t{i}"}
    mock_get.side_effect = [
        _page([entry(1), entry(2)], total=3),
        _page([entry(3)], total=3),
    ]

    rows = search_scopus("TITLE-ABS-KEY(test)", "q_test")

    assert [r["scopus_id"] for r in rows] == ["SCOPUS_ID:1", "SCOPUS_ID:2", "SCOPUS_ID:3"]
    assert mock_get.call_count == 2


@patch("tfg_codigo.Research.queries_research.requests.get")
def test_search_scopus_stops_on_empty_page(mock_get):
    mock_get.return_value = _page([], total=0)

    rows = search_scopus("TITLE-ABS-KEY(test)", "q_test")

    assert rows == []
    assert mock_get.call_count == 1


def _oa_page(results, next_cursor):
    response = MagicMock()
    response.json.return_value = {
        "results": results,
        "meta": {"next_cursor": next_cursor},
    }
    return response


def _oa_work(i):
    return {
        "id": f"https://openalex.org/W{i}",
        "display_name": f"t{i}",
        "authorships": [{"author": {"display_name": "A. Uthor"}}],
        "doi": "https://doi.org/10.1/x",
        "primary_location": {"source": {"display_name": "Journal X"}},
        "publication_date": "2022-01-01",
        "cited_by_count": 0,
        "abstract_inverted_index": {"hello": [0], "world": [1]},
    }


@patch("tfg_codigo.Research.queries_research.requests.get")
def test_search_openalex_paginates_with_cursor(mock_get):
    mock_get.side_effect = [
        _oa_page([_oa_work(1)], next_cursor="abc"),
        _oa_page([_oa_work(2)], next_cursor=None),
    ]

    rows = search_openalex("dagitty", "q_test")

    assert [r["openalex_id"] for r in rows] == [
        "https://openalex.org/W1",
        "https://openalex.org/W2",
    ]
    assert rows[0]["abstract"] == "hello world"
    assert mock_get.call_count == 2


def test_reconstruct_abstract_orders_by_position():
    assert reconstruct_abstract({"world": [1], "hello": [0]}) == "hello world"
    assert reconstruct_abstract(None) == ""


def test_merge_and_dedup_collapses_same_doi_across_sources():
    scopus_rows = [
        {"query": "q1", "doi": "https://doi.org/10.1/X", "title": "Paper A", "abstract": ""},
    ]
    openalex_rows = [
        {"query": "q3", "doi": "10.1/x", "title": "Paper A", "abstract": "full abstract"},
    ]

    merged = merge_and_dedup(scopus_rows, openalex_rows)

    assert len(merged) == 1
    row = merged[0]
    assert row["sources"] == "scopus;openalex"
    assert row["queries"] == "q1;q3"
    assert row["abstract"] == "full abstract"  # fills scopus's empty field


def test_merge_and_dedup_falls_back_to_title_without_doi():
    scopus_rows = [{"query": "q1", "doi": "", "title": "Same Title!!"}]
    openalex_rows = [{"query": "q2", "doi": "", "title": "same title"}]

    merged = merge_and_dedup(scopus_rows, openalex_rows)

    assert len(merged) == 1


def test_merge_and_dedup_merges_same_title_when_only_one_has_doi():
    openalex_rows = [
        {"query": "q1", "doi": "10.48550/arxiv.1", "title": "Falsifying Causal Graphs With Outlier Events"},
        {"query": "q1", "doi": "", "title": "Falsifying Causal Graphs With Outlier Events"},
    ]

    merged = merge_and_dedup([], openalex_rows)

    assert len(merged) == 1


def test_merge_and_dedup_merges_same_title_with_different_dois():
    # Zenodo issues a new DOI for every version uploaded to the same deposit.
    openalex_rows = [
        {"query": "q1", "doi": "10.5281/zenodo.1", "title": "Operational Falsification Governance"},
        {"query": "q1", "doi": "10.5281/zenodo.2", "title": "Operational Falsification Governance"},
        {"query": "q1", "doi": "10.5281/zenodo.3", "title": "Operational Falsification Governance"},
    ]

    merged = merge_and_dedup([], openalex_rows)

    assert len(merged) == 1


def test_write_csv_handles_rows_with_different_columns(tmp_path=Path("/tmp/test_write_csv.csv")):
    # scopus-only and openalex-only rows have different columns (scopus_id vs
    # openalex_id); it shouldn't matter which one comes first in the list.
    rows = [
        {"title": "A", "scopus_id": "S1"},
        {"title": "B", "openalex_id": "O1"},
    ]

    write_csv(tmp_path, rows)

    with open(tmp_path, newline="", encoding="utf-8") as f:
        read_rows = list(csv.DictReader(f))

    assert read_rows[0]["title"] == "A"
    assert read_rows[1]["openalex_id"] == "O1"
    tmp_path.unlink()


def test_merge_and_dedup_keeps_distinct_papers_separate():
    scopus_rows = [{"query": "q1", "doi": "10.1/a", "title": "A"}]
    openalex_rows = [{"query": "q2", "doi": "10.1/b", "title": "B"}]

    merged = merge_and_dedup(scopus_rows, openalex_rows)

    assert len(merged) == 2


if __name__ == "__main__":
    test_search_scopus_paginates_until_total_reached()
    test_search_scopus_stops_on_empty_page()
    test_search_openalex_paginates_with_cursor()
    test_reconstruct_abstract_orders_by_position()
    test_merge_and_dedup_collapses_same_doi_across_sources()
    test_merge_and_dedup_falls_back_to_title_without_doi()
    test_merge_and_dedup_merges_same_title_when_only_one_has_doi()
    test_merge_and_dedup_merges_same_title_with_different_dois()
    test_write_csv_handles_rows_with_different_columns()
    test_merge_and_dedup_keeps_distinct_papers_separate()
    print("ok")
