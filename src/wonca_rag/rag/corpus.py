from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _load_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_text_source(
    article_id: str,
    processed_root: Path,
) -> tuple[Path, str]:
    """
    Prefer OCR text when an OCR JSON exists, otherwise use native extraction.

    Returns:
        (json_path, text_source)
    """
    processed_root = Path(processed_root)

    ocr_path = processed_root / "ocr" / f"{article_id}.json"
    native_path = processed_root / f"{article_id}.json"

    if ocr_path.exists():
        return ocr_path, "ocr_tesseract"

    if native_path.exists():
        return native_path, "native_pdf_text"

    raise FileNotFoundError(
        f"No processed text found for article_id={article_id}. "
        f"Checked: {ocr_path} and {native_path}"
    )


def load_article_text(
    article_id: str,
    processed_root: Path,
) -> dict:
    """
    Load the preferred processed text for one article.
    """
    path, resolved_source = resolve_text_source(
        article_id=article_id,
        processed_root=processed_root,
    )

    payload = _load_json(path)

    pages = payload.get("pages", [])
    if not isinstance(pages, list):
        raise ValueError(f"Invalid pages structure in {path}")

    return {
        "article_id": article_id,
        "resolved_text_source": resolved_source,
        "source_json": str(path),
        "payload": payload,
        "pages": pages,
    }


def build_article_source_table(
    manifest_df: pd.DataFrame,
    processed_root: Path,
) -> pd.DataFrame:
    """
    Build one row per unique article content and identify the preferred text source.
    """
    if manifest_df.empty:
        return pd.DataFrame(
            columns=[
                "article_id",
                "sha256",
                "filename",
                "relative_path",
                "publication_year",
                "publication_date",
                "resolved_text_source",
                "source_json",
            ]
        )

    unique_df = (
        manifest_df
        .sort_values("relative_path", kind="stable")
        .drop_duplicates(subset=["sha256"], keep="first")
        .reset_index(drop=True)
    )

    rows: list[dict] = []

    for row in unique_df.to_dict(orient="records"):
        article_id = row["article_id"]

        try:
            source_json, text_source = resolve_text_source(
                article_id=article_id,
                processed_root=processed_root,
            )
            error = None
        except Exception as exc:
            source_json = None
            text_source = None
            error = str(exc)

        rows.append(
            {
                "article_id": article_id,
                "sha256": row.get("sha256"),
                "filename": row.get("filename"),
                "relative_path": row.get("relative_path"),
                "publication_year": row.get("publication_year"),
                "publication_date": row.get("publication_date"),
                "resolved_text_source": text_source,
                "source_json": str(source_json) if source_json else None,
                "source_error": error,
            }
        )

    df = pd.DataFrame(rows)

    if "publication_year" in df.columns:
        df["publication_year"] = pd.to_numeric(
            df["publication_year"],
            errors="coerce",
        ).astype("Int64")

    if "publication_date" in df.columns:
        df["publication_date"] = pd.to_datetime(
            df["publication_date"],
            errors="coerce",
        )

    return df


def filter_temporal_history(
    chunks_df: pd.DataFrame,
    current_year: int,
    current_publication_date: str | pd.Timestamp | None = None,
    current_article_id: str | None = None,
) -> pd.DataFrame:
    """
    Return only chunks that are temporally eligible as historical context.

    Rules:
    1. Articles from earlier years are eligible.
    2. Same-year articles are excluded by default.
    3. Same-year articles become eligible only when BOTH the current article and
       historical article have exact publication dates and the historical date is earlier.
    4. The current article itself is always excluded.

    This prevents arbitrary within-year alphabetical/file ordering from becoming
    a false chronology.
    """
    if chunks_df.empty:
        return chunks_df.copy()

    df = chunks_df.copy()

    years = pd.to_numeric(df["publication_year"], errors="coerce")
    eligible = years < int(current_year)

    if current_publication_date is not None:
        current_date = pd.to_datetime(
            current_publication_date,
            errors="coerce",
        )

        if pd.notna(current_date) and "publication_date" in df.columns:
            dates = pd.to_datetime(
                df["publication_date"],
                errors="coerce",
            )

            same_year_earlier = (
                years.eq(int(current_year))
                & dates.notna()
                & dates.lt(current_date)
            )

            eligible = eligible | same_year_earlier

    if current_article_id is not None:
        eligible = eligible & df["article_id"].ne(current_article_id)

    return df.loc[eligible].copy().reset_index(drop=True)
