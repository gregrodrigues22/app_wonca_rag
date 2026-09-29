import json
from pathlib import Path

import pandas as pd

from wonca_rag.rag.corpus import (
    filter_temporal_history,
    resolve_text_source,
)


def test_ocr_is_preferred_over_native(tmp_path: Path):
    processed = tmp_path / "processed"
    ocr_dir = processed / "ocr"
    processed.mkdir()
    ocr_dir.mkdir()

    native = processed / "ART_A.json"
    ocr = ocr_dir / "ART_A.json"

    native.write_text(json.dumps({"pages": []}), encoding="utf-8")
    ocr.write_text(json.dumps({"pages": []}), encoding="utf-8")

    path, source = resolve_text_source("ART_A", processed)

    assert path == ocr
    assert source == "ocr_tesseract"


def test_temporal_history_excludes_future_and_same_year_without_dates():
    df = pd.DataFrame(
        [
            {
                "article_id": "A",
                "publication_year": 1989,
                "publication_date": None,
                "chunk_id": "A1",
            },
            {
                "article_id": "B",
                "publication_year": 1990,
                "publication_date": None,
                "chunk_id": "B1",
            },
            {
                "article_id": "C",
                "publication_year": 1991,
                "publication_date": None,
                "chunk_id": "C1",
            },
        ]
    )

    eligible = filter_temporal_history(
        df,
        current_year=1990,
        current_article_id="B",
    )

    assert eligible["article_id"].tolist() == ["A"]


def test_temporal_history_allows_earlier_same_year_only_with_dates():
    df = pd.DataFrame(
        [
            {
                "article_id": "A",
                "publication_year": 1990,
                "publication_date": "1990-01-10",
                "chunk_id": "A1",
            },
            {
                "article_id": "B",
                "publication_year": 1990,
                "publication_date": "1990-05-01",
                "chunk_id": "B1",
            },
            {
                "article_id": "C",
                "publication_year": 1990,
                "publication_date": None,
                "chunk_id": "C1",
            },
        ]
    )

    eligible = filter_temporal_history(
        df,
        current_year=1990,
        current_publication_date="1990-03-01",
        current_article_id="B",
    )

    assert eligible["article_id"].tolist() == ["A"]
