from pathlib import Path

import pandas as pd

from wonca_rag.ingestion.manifest import (
    build_manifest,
    infer_first_author_from_filename,
    infer_year_from_filename,
    validate_manifest,
)


def test_filename_metadata_inference():
    assert infer_year_from_filename("Abramson2005.pdf") == 2005
    assert infer_year_from_filename("Berra2011b.pdf") == 2011
    assert infer_year_from_filename("sem_ano.pdf") is None

    assert infer_first_author_from_filename("Abramson2005.pdf") == "Abramson"
    assert infer_first_author_from_filename("Starfield_1998.pdf") == "Starfield"


def test_manifest_detects_duplicate_content(tmp_path: Path):
    pdf_root = tmp_path / "pdf"
    folder_a = pdf_root / "folder_a"
    folder_b = pdf_root / "folder_b"
    folder_a.mkdir(parents=True)
    folder_b.mkdir(parents=True)

    same_content = b"%PDF-1.4 same article content"

    (folder_a / "Abramson2005.pdf").write_bytes(same_content)
    (folder_b / "Abramson_copy2005.pdf").write_bytes(same_content)
    (pdf_root / "Allen1998.pdf").write_bytes(b"%PDF-1.4 different content")

    df = build_manifest(pdf_root)

    assert len(df) == 3
    assert df["sha256"].nunique() == 2
    assert int(df["is_duplicate_content"].sum()) == 2

    duplicate_ids = df.loc[df["is_duplicate_content"], "article_id"].unique()
    assert len(duplicate_ids) == 1


def test_manifest_marks_missing_metadata(tmp_path: Path):
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()

    (pdf_root / "arquivo_sem_metadata.pdf").write_bytes(b"%PDF-1.4 test")

    df = build_manifest(pdf_root)

    assert len(df) == 1
    assert pd.isna(df.loc[0, "publication_year"])
    assert bool(df.loc[0, "needs_metadata_review"]) is True


def test_manifest_validates(tmp_path: Path):
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()

    (pdf_root / "Bice1973.pdf").write_bytes(b"%PDF-1.4 test")

    df = build_manifest(pdf_root)
    problems = validate_manifest(df)

    assert problems == []
