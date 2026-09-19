import argparse
import csv
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

SCOPUS_URL = "https://api.elsevier.com/content/search/scopus"
API_KEY = os.environ["SCOPUS_API"]

OPENALEX_URL = "https://api.openalex.org/works"
OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO")  # optional: polite pool, more quota/speed

REPO_ROOT = Path(__file__).resolve().parents[3]
SCOPUS_OUTPUT_DIR = REPO_ROOT / "data" / "scopus"
OPENALEX_OUTPUT_DIR = REPO_ROOT / "data" / "openalex"
MERGED_OUTPUT_DIR = REPO_ROOT / "data" / "merged"


FROM_DATE = "2021-01-01"
TO_DATE = "2026-08-01"

SCOPUS_QUERIES = {
    "q1_falsification": """
        TITLE-ABS-KEY(
            "falsifying causal graph"
            OR "falsifying causal graphs"
            OR "falsifying causal model"
            OR "falsifying causal models"
            OR "falsify causal graph"
            OR "graph falsification"
        )
    """,
    "q2_validating": """
        TITLE-ABS-KEY(
            "validating causal graph"
            OR "validating causal graphs"
            OR "validating causal model"
            OR "validating causal models"
            OR "validate causal graph"
            OR "causal graph validation"
            OR "causal model validation"
        )
    """,
    "q3_testing": """
        TITLE-ABS-KEY(
            "testing causal graph"
            OR "testing causal graphs"
            OR "testing causal model"
            OR "testing causal models"
            OR "test causal graph"
            OR "causal graph testing"
            OR "causal model testing"
            OR "testing directed acyclic graph"
            OR "testing a directed acyclic graph"
        )
    """,
}

OPENALEX_QUERIES = {
    "q1_falsification": """
        "falsifying causal graph"
        OR "falsifying causal graphs"
        OR "falsifying causal model"
        OR "falsifying causal models"
        OR "falsify causal graph"
        OR "graph falsification"
    """,
    "q2_validating": """
        "validating causal graph"
        OR "validating causal graphs"
        OR "validating causal model"
        OR "validating causal models"
        OR "validate causal graph"
        OR "causal graph validation"
        OR "causal model validation"
    """,
    "q3_testing": """
        "testing causal graph"
        OR "testing causal graphs"
        OR "testing causal model"
        OR "testing causal models"
        OR "test causal graph"
        OR "causal graph testing"
        OR "causal model testing"
        OR "testing directed acyclic graph"
        OR "testing a directed acyclic graph"
    """,
}


def search_scopus(query: str, query_name: str) -> list[dict]:
    rows = []
    start = 0
    count = 25

    headers = {
        "X-ELS-APIKey": API_KEY,
        "Accept": "application/json",
    }

    while True:
        params = {
            "query": " ".join(query.split()),
            "start": start,
            "count": count,
            "view": "STANDARD",
        }

        response = requests.get(SCOPUS_URL, headers=headers, params=params, timeout=60)
        response.raise_for_status()

        data = response.json()["search-results"]
        entries = data.get("entry", [])

        if not entries:
            break

        for entry in entries:
            rows.append(
                {
                    "query": query_name,
                    "scopus_id": entry.get("dc:identifier", ""),
                    "title": entry.get("dc:title", ""),
                    "authors": entry.get("dc:creator", ""),
                    "doi": entry.get("prism:doi", ""),
                    "journal": entry.get("prism:publicationName", ""),
                    "date": entry.get("prism:coverDate", ""),
                    "cited_by": entry.get("citedby-count", ""),
                    "abstract": entry.get("dc:description", ""),
                    "url": entry.get("prism:url", ""),
                }
            )

        start += len(entries)

        total = int(data.get("opensearch:totalResults", 0))
        if start >= total:
            break

    return rows


def count_results(query: str) -> int:
    headers = {"X-ELS-APIKey": API_KEY, "Accept": "application/json"}
    params = {"query": " ".join(query.split()), "count": 1, "view": "STANDARD"}

    response = requests.get(SCOPUS_URL, headers=headers, params=params, timeout=60)
    response.raise_for_status()

    return int(response.json()["search-results"]["opensearch:totalResults"])


def reconstruct_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""

    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions[idx] = word

    return " ".join(positions[i] for i in sorted(positions))


def backfill_abstract_by_doi(doi: str) -> str:
    """Look up a single work in OpenAlex by DOI and return its abstract, if any.
    Used for rows whose abstract came back empty from the query APIs (mostly Scopus's
    STANDARD view, which often omits dc:description)."""
    doi = _normalize_doi(doi)
    if not doi:
        return ""
    params = {"mailto": OPENALEX_MAILTO} if OPENALEX_MAILTO else {}
    response = requests.get(f"{OPENALEX_URL}/https://doi.org/{doi}", params=params, timeout=30)
    if response.status_code != 200:
        return ""
    return reconstruct_abstract(response.json().get("abstract_inverted_index"))


def backfill_missing_abstracts(rows: list[dict]) -> list[dict]:
    recovered = 0
    for row in rows:
        if (row.get("abstract") or "").strip() or not (row.get("doi") or "").strip():
            continue
        abstract = backfill_abstract_by_doi(row["doi"])
        if abstract:
            row["abstract"] = abstract
            recovered += 1
    print(f"[backfill] recovered {recovered} missing abstracts via OpenAlex DOI lookup")
    return rows


def _openalex_filter(query: str) -> str:
    return (
        f"title_and_abstract.search:{' '.join(query.split())},"
        f"from_publication_date:{FROM_DATE},"
        f"to_publication_date:{TO_DATE}"
    )


def _openalex_row(work: dict, query_name: str) -> dict:
    source = (work.get("primary_location") or {}).get("source") or {}
    return {
        "query": query_name,
        "openalex_id": work.get("id", ""),
        "title": work.get("display_name", ""),
        "authors": "; ".join(
            a["author"]["display_name"] for a in work.get("authorships", []) if a.get("author")
        ),
        "doi": (work.get("doi") or "").replace("https://doi.org/", ""),
        "journal": source.get("display_name", ""),
        "date": work.get("publication_date", ""),
        "cited_by": work.get("cited_by_count", ""),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "url": work.get("id", ""),
        "referenced_works": ";".join(
            rid.rsplit("/", 1)[-1] for rid in work.get("referenced_works", [])
        ),
    }


def _paginate_openalex(filter_str: str, query_name: str) -> list[dict]:
    rows = []
    cursor = "*"
    params = {"filter": filter_str, "per-page": 200}
    if OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO

    while cursor:
        response = requests.get(OPENALEX_URL, params={**params, "cursor": cursor}, timeout=60)
        response.raise_for_status()
        data = response.json()

        rows.extend(_openalex_row(work, query_name) for work in data.get("results", []))

        if not data.get("results"):
            break
        cursor = data.get("meta", {}).get("next_cursor")

    return rows


def search_openalex(query: str, query_name: str) -> list[dict]:
    return _paginate_openalex(_openalex_filter(query), query_name)


def count_results_openalex(query: str) -> int:
    params = {"filter": _openalex_filter(query), "per-page": 1}
    if OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO

    response = requests.get(OPENALEX_URL, params=params, timeout=60)
    response.raise_for_status()

    return int(response.json()["meta"]["count"])


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _normalize_doi(doi: str) -> str:
    doi = (doi or "").strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi.rstrip("/")


def _normalize_title(title: str) -> str:
    title = re.sub(r"[^a-z0-9]+", " ", (title or "").strip().lower())
    return " ".join(title.split())


def merge_and_dedup(scopus_rows: list[dict], openalex_rows: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    members: dict[str, list[dict]] = {}
    key_to_group: dict[str, str] = {}
    order: list[str] = []

    for source, rows in (("scopus", scopus_rows), ("openalex", openalex_rows)):
        for row in rows:
            doi = _normalize_doi(row.get("doi", ""))
            doi_key = f"doi:{doi}" if doi else None
            title_key = f"title:{_normalize_title(row.get('title', ''))}"

            group_key = next(
                (key_to_group[k] for k in (doi_key, title_key) if k and k in key_to_group),
                None,
            )

            if group_key is None:
                group_key = doi_key or title_key
                unified = dict(row)
                unified["doi"] = doi
                unified["sources"] = source
                unified["queries"] = row.get("query", "")
                order.append(group_key)
                merged[group_key] = unified
                members[group_key] = [dict(row, source=source, doi=doi)]
            else:
                existing = merged[group_key]
                if source not in existing["sources"].split(";"):
                    existing["sources"] += f";{source}"
                query = row.get("query", "")
                if query and query not in existing["queries"].split(";"):
                    existing["queries"] += f";{query}"
                for field, value in row.items():
                    if not existing.get(field) and value:
                        existing[field] = value
                members[group_key].append(dict(row, source=source, doi=doi))

            if doi_key:
                key_to_group[doi_key] = group_key
            key_to_group[title_key] = group_key

    for group_key, group_members in members.items():
        dois = {m["doi"] for m in group_members if m["doi"]}
        title_only = len(group_members) > 1 and len(dois) != 1
        merged[group_key]["needs_manual_review"] = "yes" if title_only else ""

    return [merged[key] for key in order]


def title_only_merge_review(merged_rows: list[dict]) -> list[dict]:
    """Rows merged without all members sharing the same DOI: check by hand
    that they're really the same work before trusting the dedup."""
    return [row for row in merged_rows if row.get("needs_manual_review") == "yes"]


def run(source: str, search_fn, queries: dict[str, str], output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for query_name, query in queries.items():
        rows = search_fn(query, query_name)
        all_rows.extend(rows)
        write_csv(output_dir / f"{query_name}.csv", rows)
        print(f"[{source}] {query_name}: {len(rows)} results")

    write_csv(output_dir / f"{source}_all_queries.csv", all_rows)
    print(f"[{source}] Raw total: {len(all_rows)} results")
    return all_rows


def merge_from_disk() -> None:
    scopus_rows = read_csv(SCOPUS_OUTPUT_DIR / "scopus_all_queries.csv")
    openalex_rows = read_csv(OPENALEX_OUTPUT_DIR / "openalex_all_queries.csv")
    _write_merged(scopus_rows, openalex_rows)


def _write_merged(scopus_rows: list[dict], openalex_rows: list[dict]) -> None:
    merged = merge_and_dedup(scopus_rows, openalex_rows)
    merged = backfill_missing_abstracts(merged)
    MERGED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(MERGED_OUTPUT_DIR / "all_papers_deduped.csv", merged)

    review = title_only_merge_review(merged)
    write_csv(MERGED_OUTPUT_DIR / "title_only_merges_review.csv", review)
    print(f"Merges by title only (no matching DOI) to review by hand: {len(review)}")

    raw_total = len(scopus_rows) + len(openalex_rows)
    print(f"Raw total (scopus+openalex, may include duplicates across queries): {raw_total}")
    print(f"After DOI/title dedup: {len(merged)} ({raw_total - len(merged)} collapsed)")


def main() -> None:
    scopus_rows = run("scopus", search_scopus, SCOPUS_QUERIES, SCOPUS_OUTPUT_DIR)
    openalex_rows = run("openalex", search_openalex, OPENALEX_QUERIES, OPENALEX_OUTPUT_DIR)
    _write_merged(scopus_rows, openalex_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--counts", action="store_true", help="only show totalResults per query, without downloading"
    )
    parser.add_argument(
        "--source", choices=["scopus", "openalex", "all"], default="all"
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="redo the dedup from the already-downloaded CSVs, without calling the APIs",
    )
    args = parser.parse_args()

    if args.merge_only:
        merge_from_disk()
    elif args.counts:
        if args.source in ("scopus", "all"):
            for query_name, query in SCOPUS_QUERIES.items():
                print(f"[scopus] {query_name}: {count_results(query)} results")
        if args.source in ("openalex", "all"):
            for query_name, query in OPENALEX_QUERIES.items():
                print(f"[openalex] {query_name}: {count_results_openalex(query)} results")
    elif args.source == "scopus":
        run("scopus", search_scopus, SCOPUS_QUERIES, SCOPUS_OUTPUT_DIR)
    elif args.source == "openalex":
        run("openalex", search_openalex, OPENALEX_QUERIES, OPENALEX_OUTPUT_DIR)
    else:
        main()
