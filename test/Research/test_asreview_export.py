from pathlib import Path

import asreview

from tfg_codigo.Research.asreview_export import to_asreview_row, write_asreview_csv


def test_to_asreview_row_maps_query_research_shape():
    row = {
        "title": "Toy Paper",
        "abstract": "Falsifies a candidate DAG against i.i.d. data.",
        "authors": "A. Uthor",
        "date": "2024-01-01",
        "doi": "10.1/toy",
        "openalex_id": "https://openalex.org/W1",  # dropped, not an ASReview column
    }

    out = to_asreview_row(row, included="1", source="q1-q3")

    assert out == {
        "title": "Toy Paper",
        "abstract": "Falsifies a candidate DAG against i.i.d. data.",
        "authors": "A. Uthor",
        "date": "2024-01-01",
        "doi": "10.1/toy",
        "included": "1",
        "source": "q1-q3",
    }


def test_written_csv_round_trips_through_asreview_load_dataset(tmp_path):
    # The one thing worth an integration test here: does ASReview's own loader
    # accept our column names and preserve "included" as 1 / 0 / blank (unscreened)?
    rows = [
        to_asreview_row(
            {"title": "Relevant paper", "abstract": "Tests a candidate DAG.", "doi": "10.1/a"},
            included="1",
            source="q1-q3",
        ),
        to_asreview_row(
            {"title": "Irrelevant paper", "abstract": "Tourism in Bangkok.", "doi": "10.1/b"},
            included="0",
            source="q1-q3",
        ),
        to_asreview_row(
            {"title": "Unscreened paper", "abstract": "Not yet labeled.", "doi": "10.1/c"},
            included="",
            source="snowball",
        ),
    ]
    path = tmp_path / "pool_asreview.csv"

    write_asreview_csv(rows, path)
    # file-backed db: asreview's default ":memory:" db can open a fresh empty connection
    # per pooled checkout, losing what load_dataset just wrote.
    db = asreview.Database(str(tmp_path / "asreview.db"))
    df = asreview.load_dataset(str(path), db=db).input.get_df()

    assert path.exists()
    by_title = df.set_index("title")
    assert by_title.loc["Relevant paper", "included"] == 1
    assert by_title.loc["Irrelevant paper", "included"] == 0
    assert pd_isna(by_title.loc["Unscreened paper", "included"])
    assert by_title.loc["Relevant paper", "abstract"] == "Tests a candidate DAG."


def pd_isna(value) -> bool:
    import pandas as pd

    return pd.isna(value)


if __name__ == "__main__":
    test_to_asreview_row_maps_query_research_shape()
    test_written_csv_round_trips_through_asreview_load_dataset(Path("/tmp/test_asreview_export"))
    print("ok")
