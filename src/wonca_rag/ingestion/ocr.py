from __future__ import annotations

import json
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path

import fitz
import pandas as pd
import pytesseract
from PIL import Image
from tqdm.auto import tqdm


OCR_VERSION = "1.0.0"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_tesseract() -> Path | None:
    """
    Find Tesseract using:
    1. TESSERACT_CMD environment variable
    2. system PATH
    3. common Windows installation locations
    """
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).exists():
        return Path(env_cmd)

    path_cmd = shutil.which("tesseract")
    if path_cmd:
        return Path(path_cmd)

    candidates = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def configure_tesseract() -> Path:
    tesseract_path = find_tesseract()

    if tesseract_path is None:
        raise RuntimeError(
            "Tesseract was not found. Install Tesseract OCR and ensure that "
            "tesseract.exe is on PATH, or define TESSERACT_CMD."
        )

    pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)
    return tesseract_path


def available_languages() -> list[str]:
    configure_tesseract()
    return sorted(pytesseract.get_languages(config=""))


def _pixmap_to_pil(pix: fitz.Pixmap) -> Image.Image:
    if pix.alpha:
        mode = "RGBA"
    elif pix.n == 1:
        mode = "L"
    else:
        mode = "RGB"

    return Image.frombytes(mode, [pix.width, pix.height], pix.samples)


def _ocr_page_image(
    image: Image.Image,
    language: str,
    psm: int,
) -> str:
    text = pytesseract.image_to_string(
        image,
        lang=language,
        config=f"--psm {psm}",
    )
    return text.replace("\x00", " ").strip()


def ocr_pdf_to_dict(
    pdf_path: Path,
    article_id: str,
    sha256: str,
    filename: str,
    relative_path: str,
    language: str = "eng",
    dpi: int = 300,
    psm: int = 3,
) -> dict:
    configure_tesseract()

    requested_languages = {
        item.strip()
        for item in language.split("+")
        if item.strip()
    }
    installed_languages = set(available_languages())
    missing_languages = requested_languages - installed_languages

    if missing_languages:
        raise RuntimeError(
            "Missing Tesseract language packs: "
            + ", ".join(sorted(missing_languages))
        )

    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)

    with fitz.open(pdf_path) as doc:
        pages: list[dict] = []
        total_characters = 0
        empty_pages = 0

        for page_index, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = _pixmap_to_pil(pix)

            text = _ocr_page_image(
                image=image,
                language=language,
                psm=psm,
            )

            n_characters = len(text)
            total_characters += n_characters

            if n_characters == 0:
                empty_pages += 1

            pages.append(
                {
                    "page_number": page_index + 1,
                    "n_characters": n_characters,
                    "has_text": n_characters > 0,
                    "text_source": "ocr_tesseract",
                    "text": text,
                }
            )

        n_pages = len(pages)
        mean_characters_per_page = (
            total_characters / n_pages if n_pages else 0.0
        )
        empty_page_ratio = (
            empty_pages / n_pages if n_pages else 1.0
        )

        return {
            "schema_version": "1.0",
            "ocr_version": OCR_VERSION,
            "article_id": article_id,
            "sha256": sha256,
            "filename": filename,
            "relative_path": relative_path,
            "source_pdf": str(pdf_path),
            "text_source": "ocr_tesseract",
            "ocr": {
                "engine": "tesseract",
                "language": language,
                "dpi": dpi,
                "psm": psm,
                "completed_at_utc": _utc_now_iso(),
            },
            "quality": {
                "n_pages": n_pages,
                "total_characters": total_characters,
                "mean_characters_per_page": mean_characters_per_page,
                "empty_pages": empty_pages,
                "empty_page_ratio": empty_page_ratio,
            },
            "pages": pages,
        }


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def select_ocr_candidates(
    extraction_report_df: pd.DataFrame,
    trigger: str = "zero_text_only",
) -> pd.DataFrame:
    df = extraction_report_df.copy()

    if trigger == "zero_text_only":
        total_chars = pd.to_numeric(
            df["total_characters"],
            errors="coerce",
        )
        mask = (
            df["status"].isin(["DONE", "SKIPPED_CURRENT"])
            & total_chars.fillna(-1).eq(0)
        )
    elif trigger == "likely_scanned":
        mask = (
            df["status"].isin(["DONE", "SKIPPED_CURRENT"])
            & df["likely_scanned"].fillna(False).astype(bool)
        )
    else:
        raise ValueError(
            "trigger must be 'zero_text_only' or 'likely_scanned'"
        )

    return df.loc[mask].reset_index(drop=True)


def _is_current_ocr(
    output_json: Path,
    expected_sha256: str,
    language: str,
    dpi: int,
    psm: int,
) -> bool:
    if not output_json.exists():
        return False

    try:
        payload = _load_json(output_json)
    except Exception:
        return False

    ocr_meta = payload.get("ocr", {})

    return (
        payload.get("sha256") == expected_sha256
        and payload.get("ocr_version") == OCR_VERSION
        and ocr_meta.get("language") == language
        and ocr_meta.get("dpi") == dpi
        and ocr_meta.get("psm") == psm
    )


def run_selective_ocr(
    candidates_df: pd.DataFrame,
    pdf_root: Path,
    ocr_output_root: Path,
    checkpoints_root: Path,
    language: str = "eng",
    dpi: int = 300,
    psm: int = 3,
    min_characters_after_ocr: int = 100,
    resume: bool = True,
) -> pd.DataFrame:
    pdf_root = Path(pdf_root).resolve()
    ocr_output_root = Path(ocr_output_root).resolve()
    checkpoints_root = Path(checkpoints_root).resolve()

    ocr_output_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    configure_tesseract()

    rows: list[dict] = []

    for row in tqdm(
        candidates_df.to_dict(orient="records"),
        desc="OCR PDFs",
    ):
        article_id = row["article_id"]
        sha256 = row["sha256"]
        filename = row["filename"]
        relative_path = row["relative_path"]

        pdf_path = pdf_root / relative_path
        output_json = ocr_output_root / f"{article_id}.json"
        checkpoint_json = checkpoints_root / f"{article_id}.json"

        base = {
            "article_id": article_id,
            "sha256": sha256,
            "filename": filename,
            "relative_path": relative_path,
            "output_json": str(output_json),
        }

        if (
            resume
            and _is_current_ocr(
                output_json=output_json,
                expected_sha256=sha256,
                language=language,
                dpi=dpi,
                psm=psm,
            )
        ):
            payload = _load_json(output_json)
            quality = payload.get("quality", {})
            rows.append(
                {
                    **base,
                    "status": "SKIPPED_CURRENT",
                    "passes_min_characters": (
                        quality.get("total_characters", 0)
                        >= min_characters_after_ocr
                    ),
                    "error_type": None,
                    "error_message": None,
                    **quality,
                }
            )
            continue

        started_at = _utc_now_iso()
        _save_json(
            {
                "article_id": article_id,
                "sha256": sha256,
                "status": "RUNNING",
                "started_at_utc": started_at,
                "ocr_version": OCR_VERSION,
            },
            checkpoint_json,
        )

        try:
            payload = ocr_pdf_to_dict(
                pdf_path=pdf_path,
                article_id=article_id,
                sha256=sha256,
                filename=filename,
                relative_path=relative_path,
                language=language,
                dpi=dpi,
                psm=psm,
            )

            _save_json(payload, output_json)

            quality = payload["quality"]
            passes = (
                quality["total_characters"]
                >= min_characters_after_ocr
            )

            _save_json(
                {
                    "article_id": article_id,
                    "sha256": sha256,
                    "status": "DONE",
                    "started_at_utc": started_at,
                    "finished_at_utc": _utc_now_iso(),
                    "ocr_version": OCR_VERSION,
                    "passes_min_characters": passes,
                    "output_json": str(output_json),
                },
                checkpoint_json,
            )

            rows.append(
                {
                    **base,
                    "status": "DONE",
                    "passes_min_characters": passes,
                    "error_type": None,
                    "error_message": None,
                    **quality,
                }
            )

        except Exception as exc:
            error_payload = {
                "article_id": article_id,
                "sha256": sha256,
                "status": "ERROR",
                "started_at_utc": started_at,
                "finished_at_utc": _utc_now_iso(),
                "ocr_version": OCR_VERSION,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            _save_json(error_payload, checkpoint_json)

            rows.append(
                {
                    **base,
                    "status": "ERROR",
                    "passes_min_characters": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "n_pages": None,
                    "total_characters": None,
                    "mean_characters_per_page": None,
                    "empty_pages": None,
                    "empty_page_ratio": None,
                }
            )

    return pd.DataFrame(rows)


def save_ocr_report(
    report_df: pd.DataFrame,
    output_dir: Path,
    stem: str = "ocr_report",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{stem}.parquet"
    xlsx_path = output_dir / f"{stem}.xlsx"
    csv_path = output_dir / f"{stem}.csv"

    report_df.to_parquet(parquet_path, index=False)
    report_df.to_excel(xlsx_path, index=False)
    report_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return {
        "parquet": parquet_path,
        "xlsx": xlsx_path,
        "csv": csv_path,
    }


def summarize_ocr(report_df: pd.DataFrame) -> dict:
    if report_df.empty:
        return {
            "n_candidates": 0,
            "n_done": 0,
            "n_skipped_current": 0,
            "n_error": 0,
            "n_pass_min_characters": 0,
            "n_still_zero_text": 0,
        }

    total_chars = pd.to_numeric(
        report_df["total_characters"],
        errors="coerce",
    )

    return {
        "n_candidates": int(len(report_df)),
        "n_done": int((report_df["status"] == "DONE").sum()),
        "n_skipped_current": int(
            (report_df["status"] == "SKIPPED_CURRENT").sum()
        ),
        "n_error": int((report_df["status"] == "ERROR").sum()),
        "n_pass_min_characters": int(
            report_df["passes_min_characters"].fillna(False).astype(bool).sum()
        ),
        "n_still_zero_text": int(
            (total_chars.fillna(-1) == 0).sum()
        ),
    }
