from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import fitz
import pandas as pd
from tqdm.auto import tqdm


EXTRACTION_VERSION = "1.0.0"


@dataclass(frozen=True)
class ExtractionPaths:
    article_json: Path
    checkpoint_json: Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = "\n".join(lines).strip()
    return cleaned


def extract_pdf_to_dict(
    pdf_path: Path,
    article_id: str,
    sha256: str,
    filename: str,
    relative_path: str,
) -> dict:
    """
    Extract one PDF page by page using PyMuPDF.

    This function does not perform OCR. It only records whether OCR may be
    necessary based on the amount of text extracted.
    """
    pdf_path = Path(pdf_path)

    with fitz.open(pdf_path) as doc:
        metadata = doc.metadata or {}

        pages: list[dict] = []
        total_characters = 0
        empty_pages = 0

        for page_index, page in enumerate(doc):
            text = _clean_text(page.get_text("text"))
            n_characters = len(text)

            if n_characters == 0:
                empty_pages += 1

            total_characters += n_characters

            pages.append(
                {
                    "page_number": page_index + 1,
                    "n_characters": n_characters,
                    "has_text": n_characters > 0,
                    "text": text,
                }
            )

        n_pages = len(pages)
        mean_characters_per_page = (
            total_characters / n_pages if n_pages else 0.0
        )
        empty_page_ratio = empty_pages / n_pages if n_pages else 1.0

        # Conservative flags only. OCR is not performed automatically.
        likely_scanned = (
            n_pages > 0
            and (
                mean_characters_per_page < 80
                or empty_page_ratio >= 0.80
            )
        )

        return {
            "schema_version": "1.0",
            "extraction_version": EXTRACTION_VERSION,
            "article_id": article_id,
            "sha256": sha256,
            "filename": filename,
            "relative_path": relative_path,
            "source_pdf": str(pdf_path),
            "extracted_at_utc": _utc_now_iso(),
            "pdf_metadata": {
                "title": metadata.get("title") or None,
                "author": metadata.get("author") or None,
                "subject": metadata.get("subject") or None,
                "keywords": metadata.get("keywords") or None,
                "creator": metadata.get("creator") or None,
                "producer": metadata.get("producer") or None,
                "creationDate": metadata.get("creationDate") or None,
                "modDate": metadata.get("modDate") or None,
            },
            "quality": {
                "n_pages": n_pages,
                "total_characters": total_characters,
                "mean_characters_per_page": mean_characters_per_page,
                "empty_pages": empty_pages,
                "empty_page_ratio": empty_page_ratio,
                "likely_scanned": likely_scanned,
            },
            "pages": pages,
        }


def _article_paths(
    processed_root: Path,
    checkpoints_root: Path,
    article_id: str,
) -> ExtractionPaths:
    return ExtractionPaths(
        article_json=processed_root / f"{article_id}.json",
        checkpoint_json=checkpoints_root / f"{article_id}.json",
    )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _is_current_extraction(
    article_json: Path,
    expected_sha256: str,
) -> bool:
    if not article_json.exists():
        return False

    try:
        payload = _load_json(article_json)
    except Exception:
        return False

    return (
        payload.get("sha256") == expected_sha256
        and payload.get("extraction_version") == EXTRACTION_VERSION
    )


def choose_unique_articles(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Choose one physical PDF per unique SHA-256 content.

    Duplicate physical files remain represented in the manifest, but extraction
    and later model inference should happen once per unique content.
    """
    if manifest_df.empty:
        return manifest_df.copy()

    required = {
        "article_id",
        "filename",
        "relative_path",
        "sha256",
    }
    missing = required.difference(manifest_df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    unique_df = (
        manifest_df
        .sort_values("relative_path", kind="stable")
        .drop_duplicates(subset=["sha256"], keep="first")
        .reset_index(drop=True)
    )

    if unique_df["article_id"].duplicated().any():
        raise ValueError("Unique-content manifest contains duplicate article_id values.")

    return unique_df


def extract_unique_articles(
    manifest_df: pd.DataFrame,
    pdf_root: Path,
    processed_root: Path,
    checkpoints_root: Path,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Extract all unique PDF contents.

    Returns one quality/status row per unique article content.
    """
    pdf_root = Path(pdf_root).resolve()
    processed_root = Path(processed_root).resolve()
    checkpoints_root = Path(checkpoints_root).resolve()

    processed_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    unique_df = choose_unique_articles(manifest_df)
    status_rows: list[dict] = []

    for row in tqdm(
        unique_df.to_dict(orient="records"),
        desc="Extracting PDFs",
    ):
        article_id = row["article_id"]
        sha256 = row["sha256"]
        relative_path = row["relative_path"]
        filename = row["filename"]
        pdf_path = pdf_root / relative_path
        paths = _article_paths(
            processed_root=processed_root,
            checkpoints_root=checkpoints_root,
            article_id=article_id,
        )

        base_status = {
            "article_id": article_id,
            "sha256": sha256,
            "filename": filename,
            "relative_path": relative_path,
            "pdf_path": str(pdf_path),
            "article_json": str(paths.article_json),
            "checkpoint_json": str(paths.checkpoint_json),
        }

        if (
            resume
            and _is_current_extraction(
                paths.article_json,
                expected_sha256=sha256,
            )
        ):
            payload = _load_json(paths.article_json)
            quality = payload.get("quality", {})
            status_rows.append(
                {
                    **base_status,
                    "status": "SKIPPED_CURRENT",
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
                "extraction_version": EXTRACTION_VERSION,
            },
            paths.checkpoint_json,
        )

        try:
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            payload = extract_pdf_to_dict(
                pdf_path=pdf_path,
                article_id=article_id,
                sha256=sha256,
                filename=filename,
                relative_path=relative_path,
            )

            _save_json(payload, paths.article_json)

            finished_at = _utc_now_iso()
            _save_json(
                {
                    "article_id": article_id,
                    "sha256": sha256,
                    "status": "DONE",
                    "started_at_utc": started_at,
                    "finished_at_utc": finished_at,
                    "extraction_version": EXTRACTION_VERSION,
                    "article_json": str(paths.article_json),
                },
                paths.checkpoint_json,
            )

            status_rows.append(
                {
                    **base_status,
                    "status": "DONE",
                    "error_type": None,
                    "error_message": None,
                    **payload["quality"],
                }
            )

        except Exception as exc:
            finished_at = _utc_now_iso()

            error_payload = {
                "article_id": article_id,
                "sha256": sha256,
                "status": "ERROR",
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
                "extraction_version": EXTRACTION_VERSION,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            _save_json(error_payload, paths.checkpoint_json)

            status_rows.append(
                {
                    **base_status,
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "n_pages": None,
                    "total_characters": None,
                    "mean_characters_per_page": None,
                    "empty_pages": None,
                    "empty_page_ratio": None,
                    "likely_scanned": None,
                }
            )

    return pd.DataFrame(status_rows)


def save_extraction_report(
    report_df: pd.DataFrame,
    output_dir: Path,
    stem: str = "pdf_extraction_report",
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


def summarize_extraction(report_df: pd.DataFrame) -> dict:
    if report_df.empty:
        return {
            "n_unique_articles": 0,
            "n_done": 0,
            "n_skipped_current": 0,
            "n_error": 0,
            "n_likely_scanned": 0,
            "n_zero_text": 0,
        }

    likely_scanned = report_df["likely_scanned"].fillna(False).astype(bool)
    total_chars = pd.to_numeric(
        report_df["total_characters"],
        errors="coerce",
    )

    return {
        "n_unique_articles": int(len(report_df)),
        "n_done": int((report_df["status"] == "DONE").sum()),
        "n_skipped_current": int(
            (report_df["status"] == "SKIPPED_CURRENT").sum()
        ),
        "n_error": int((report_df["status"] == "ERROR").sum()),
        "n_likely_scanned": int(likely_scanned.sum()),
        "n_zero_text": int((total_chars.fillna(-1) == 0).sum()),
    }
