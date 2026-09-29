from pathlib import Path

import fitz
import pandas as pd

from wonca_rag.ingestion.pdf_extractor import (
    choose_unique_articles,
    extract_pdf_to_dict,
    extract_unique_articles,
)


def _create_text_pdf(path: Path, texts: list[str]) -> None:
    doc = fitz.open()
    for text in texts:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_extract_pdf_page_by_page(tmp_path: Path):
    pdf_path = tmp_path / "Example2001.pdf"
    _create_text_pdf(pdf_path, ["Page one text", "Page two text"])

    result = extract_pdf_to_dict(
        pdf_path=pdf_path,
        article_id="ART_TEST",
        sha256="abc123",
        filename=pdf_path.name,
        relative_path=pdf_path.name,
    )

    assert result["quality"]["n_pages"] == 2
    assert result["quality"]["total_characters"] > 0
    assert len(result["pages"]) == 2
    assert result["pages"][0]["page_number"] == 1
    assert "Page one text" in result["pages"][0]["text"]


def test_choose_unique_articles_uses_sha256():
    df = pd.DataFrame(
        [
            {
                "article_id": "ART_A",
                "filename": "A.pdf",
                "relative_path": "folder/A.pdf",
                "sha256": "same",
            },
            {
                "article_id": "ART_A",
                "filename": "A_copy.pdf",
                "relative_path": "folder2/A_copy.pdf",
                "sha256": "same",
            },
            {
                "article_id": "ART_B",
                "filename": "B.pdf",
                "relative_path": "B.pdf",
                "sha256": "different",
            },
        ]
    )

    unique_df = choose_unique_articles(df)

    assert len(unique_df) == 2
    assert unique_df["sha256"].nunique() == 2


def test_extraction_resume_skips_current(tmp_path: Path):
    pdf_root = tmp_path / "pdf"
    processed_root = tmp_path / "processed"
    checkpoints_root = tmp_path / "checkpoints"
    pdf_root.mkdir()

    pdf_path = pdf_root / "Bice1973.pdf"
    _create_text_pdf(pdf_path, ["Primary care article"])

    import hashlib

    sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    manifest_df = pd.DataFrame(
        [
            {
                "article_id": f"ART_{sha256[:12].upper()}",
                "filename": pdf_path.name,
                "relative_path": pdf_path.name,
                "sha256": sha256,
            }
        ]
    )

    first = extract_unique_articles(
        manifest_df=manifest_df,
        pdf_root=pdf_root,
        processed_root=processed_root,
        checkpoints_root=checkpoints_root,
        resume=True,
    )

    second = extract_unique_articles(
        manifest_df=manifest_df,
        pdf_root=pdf_root,
        processed_root=processed_root,
        checkpoints_root=checkpoints_root,
        resume=True,
    )

    assert first.loc[0, "status"] == "DONE"
    assert second.loc[0, "status"] == "SKIPPED_CURRENT"
