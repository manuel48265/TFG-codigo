"""Backward/forward snowballing (Wohlin 2014) from a seed set of OpenAlex work IDs.

The seed set doesn't need to be reproducible (see informacion_relevante.md) — the rigor is in
this step: fully reproducible, one hop at a time, re-run with an expanded seeds file for the
next iteration. Screening/inclusion decisions stay manual.
"""

import argparse

import requests

from .queries_research import (
    FROM_DATE,
    OPENALEX_MAILTO,
    OPENALEX_URL,
    REPO_ROOT,
    TO_DATE,
    _openalex_row,
    _paginate_openalex,
    write_csv,
)

SNOWBALL_OUTPUT_DIR = REPO_ROOT / "data" / "snowball"
SNOWBALL_SEEDS_FILE = REPO_ROOT / "data" / "snowball_seeds.txt"


def forward_citations(work_id: str, query_name: str = "snowball_forward") -> list[dict]:
    """Papers citing `work_id` — reproducible equivalent of Google Scholar's "cited by"."""
    work_id = work_id.rsplit("/", 1)[-1]
    filter_str = f"cites:{work_id},from_publication_date:{FROM_DATE},to_publication_date:{TO_DATE}"
    return _paginate_openalex(filter_str, query_name)


def backward_references(work_id: str, query_name: str = "snowball_backward") -> list[dict]:
    """Papers cited by `work_id` (its own bibliography), resolved via OpenAlex IDs."""
    work_id = work_id.rsplit("/", 1)[-1]
    params = {"mailto": OPENALEX_MAILTO} if OPENALEX_MAILTO else {}
    response = requests.get(f"{OPENALEX_URL}/{work_id}", params=params, timeout=60)
    response.raise_for_status()
    ref_ids = [rid.rsplit("/", 1)[-1] for rid in response.json().get("referenced_works", [])]

    rows = []
    for i in range(0, len(ref_ids), 50):
        chunk_params = {"filter": f"openalex_id:{'|'.join(ref_ids[i : i + 50])}", "per-page": 50}
        if OPENALEX_MAILTO:
            chunk_params["mailto"] = OPENALEX_MAILTO
        response = requests.get(OPENALEX_URL, params=chunk_params, timeout=60)
        response.raise_for_status()
        rows.extend(_openalex_row(work, query_name) for work in response.json().get("results", []))

    return rows


def read_seeds(path: str = SNOWBALL_SEEDS_FILE) -> list[str]:
    seeds = []
    for line in open(path):
        seed = line.split("#", 1)[0].strip()
        if seed:
            seeds.append(seed)
    return seeds


def run_snowball(seeds_file=SNOWBALL_SEEDS_FILE, output_dir=SNOWBALL_OUTPUT_DIR) -> None:
    seeds = read_seeds(seeds_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    backward_rows, forward_rows = [], []
    for seed in seeds:
        print(f"[snowball] backward from {seed}...")
        backward_rows.extend(backward_references(seed, query_name=f"backward_{seed}"))
        print(f"[snowball] forward from {seed}...")
        forward_rows.extend(forward_citations(seed, query_name=f"forward_{seed}"))

    write_csv(output_dir / "backward.csv", backward_rows)
    write_csv(output_dir / "forward.csv", forward_rows)
    print(
        f"[snowball] backward: {len(backward_rows)} rows, forward: {len(forward_rows)} rows"
        f" (from {len(seeds)} seeds, before dedup)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds-file",
        default=str(SNOWBALL_SEEDS_FILE),
        help="one OpenAlex work ID per line, '#' starts a comment (default: data/snowball_seeds.txt)",
    )
    args = parser.parse_args()
    run_snowball(seeds_file=args.seeds_file)
