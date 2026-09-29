from pathlib import Path

import fitz
import pandas as pd
from PIL import Image

from wonca_rag.ingestion import ocr as ocr_module
from wonca_rag.ingestion.ocr import (
    ocr_pdf_to_dict,
    select_ocr_candidates,
)


def _create_image_only_pdf(path: Path) -> None:
    image = Image.new("RGB", (600, 300), "white")
    image_path = path.with_suffix(".png")
    image.save(image_path)

    doc = fitz.open()
    page = doc.new_page(width=600, height=300)
    page.insert_image(page.rect, filename=str(image_path))
    doc.save(path)
    doc.close()


def test_select_zero_text_candidates():
    df = pd.DataFrame(
        [
            {
                "article_id": "A",
                "status": "DONE",
                "total_characters": 0,
                "likely_scanned": True,
            },
            {
                "article_id": "B",
                "status": "DONE",
                "total_characters": 1000,
                "likely_scanned": False,
            },
        ]
    )

    selected = select_ocr_candidates(
        df,
        trigger="zero_text_only",
    )

    assert selected["article_id"].tolist() == ["A"]


def test_ocr_pdf_structure_without_real_tesseract(
    tmp_path: Path,
    monkeypatch,
):
    pdf_path = tmp_path / "scan.pdf"
    _create_image_only_pdf(pdf_path)

    monkeypatch.setattr(
        ocr_module,
        "configure_tesseract",
        lambda: Path("fake_tesseract"),
    )
    monkeypatch.setattr(
        ocr_module,
        "available_languages",
        lambda: ["eng"],
    )
    monkeypatch.setattr(
        ocr_module,
        "_ocr_page_image",
        lambda image, language, psm: "Simulated OCR text",
    )

    result = ocr_pdf_to_dict(
        pdf_path=pdf_path,
        article_id="ART_TEST",
        sha256="abc",
        filename="scan.pdf",
        relative_path="scan.pdf",
        language="eng",
        dpi=150,
        psm=3,
    )

    assert result["text_source"] == "ocr_tesseract"
    assert result["quality"]["n_pages"] == 1
    assert result["quality"]["total_characters"] > 0
    assert result["pages"][0]["text"] == "Simulated OCR text"
