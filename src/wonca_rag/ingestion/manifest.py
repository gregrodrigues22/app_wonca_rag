from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


YEAR_PATTERN = re.compile(r"(?<!\d)((?:18|19|20)\d{2})(?!\d)")
AUTHOR_BEFORE_YEAR_PATTERN = re.compile(
    r"^\s*([A-Za-zÀ-ÖØ-öø-ÿ'`´._ -]+?)\s*((?:18|19|20)\d{2})",
    flags=re.IGNORECASE,
)


MANIFEST_COLUMNS = [
    "article_id",
    "filename",
    "relative_path",
    "source_folder",
    "file_size_bytes",
    "sha256",
    "is_duplicate_content",
    "duplicate_group",
    "inferred_first_author",
    "publication_year",
    "publication_date",
    "temporal_group",
    "year_inferred_from_filename",
    "needs_metadata_review",
]


@dataclass(frozen=True)
class ManifestOutput:
    dataframe: pd.DataFrame
    parquet_path: Path
    xlsx_path: Path
    csv_path: Path
    summary_path: Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return SHA-256 for a file without loading it all into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def infer_year_from_filename(filename: str) -> int | None:
    """Infer publication year from the first plausible 4-digit year."""
    match = YEAR_PATTERN.search(Path(filename).stem)
    return int(match.group(1)) if match else None


def infer_first_author_from_filename(filename: str) -> str | None:
    """
    Infer a first-author token/name from filenames such as:
    Abramson2005.pdf
    Starfield_1998.pdf
    Berra2011b.pdf

    This is intentionally conservative. It is only provisional metadata.
    """
    stem = Path(filename).stem.strip()

    match = AUTHOR_BEFORE_YEAR_PATTERN.search(stem)
    if match:
        author = match.group(1)
    else:
        year_match = YEAR_PATTERN.search(stem)
        if not year_match:
            return None
        author = stem[: year_match.start()]

    author = re.sub(r"[_\-]+", " ", author)
    author = re.sub(r"\s+", " ", author).strip(" ._-")

    return author or None


def stable_article_id(sha256: str) -> str:
    """
    Stable internal identifier derived from file content.

    Renaming the PDF will not change the article_id.
    """
    return f"ART_{sha256[:12].upper()}"


def _iter_pdfs(pdf_root: Path) -> list[Path]:
    if not pdf_root.exists():
        return []

    return sorted(
        (
            path
            for path in pdf_root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: str(path.relative_to(pdf_root)).lower(),
    )


def build_manifest(pdf_root: Path) -> pd.DataFrame:
    """
    Inventory all PDFs recursively and build the raw article manifest.

    Important:
    - This stage does not read article text.
    - Year and author inferred from filename are provisional.
    - `publication_date` remains empty until metadata enrichment.
    - Articles sharing the same year must not be treated as temporally ordered
      unless an exact publication date is later available.
    """
    pdf_root = Path(pdf_root).resolve()
    pdf_paths = _iter_pdfs(pdf_root)

    records: list[dict] = []

    for pdf_path in pdf_paths:
        relative_path = pdf_path.relative_to(pdf_root)
        file_hash = sha256_file(pdf_path)
        year = infer_year_from_filename(pdf_path.name)
        author = infer_first_author_from_filename(pdf_path.name)

        records.append(
            {
                "article_id": stable_article_id(file_hash),
                "filename": pdf_path.name,
                "relative_path": relative_path.as_posix(),
                "source_folder": (
                    relative_path.parent.as_posix()
                    if relative_path.parent.as_posix() != "."
                    else ""
                ),
                "file_size_bytes": pdf_path.stat().st_size,
                "sha256": file_hash,
                "inferred_first_author": author,
                "publication_year": year,
                "publication_date": pd.NaT,
                "temporal_group": year,
                "year_inferred_from_filename": year is not None,
                "needs_metadata_review": year is None or author is None,
            }
        )

    if not records:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    df = pd.DataFrame(records)

    duplicate_counts = df.groupby("sha256")["sha256"].transform("size")
    df["is_duplicate_content"] = duplicate_counts.gt(1)
    df["duplicate_group"] = df["sha256"].where(
        df["is_duplicate_content"],
        other=pd.NA,
    )

    # Keep one row per physical file. Duplicate content is marked, not removed.
    # Sort is only for readability. It is not a temporal ordering within a year.
    df = df.sort_values(
        by=[
            "publication_year",
            "inferred_first_author",
            "relative_path",
        ],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    df["publication_year"] = df["publication_year"].astype("Int64")
    df["temporal_group"] = df["temporal_group"].astype("Int64")

    return df[MANIFEST_COLUMNS]


def validate_manifest(df: pd.DataFrame) -> list[str]:
    """Return human-readable validation problems. Empty list means valid."""
    problems: list[str] = []

    missing_columns = [column for column in MANIFEST_COLUMNS if column not in df.columns]
    if missing_columns:
        problems.append(f"Missing columns: {missing_columns}")
        return problems

    if df.empty:
        problems.append("No PDF files were found.")
        return problems

    if df["relative_path"].duplicated().any():
        duplicated_paths = (
            df.loc[df["relative_path"].duplicated(keep=False), "relative_path"]
            .astype(str)
            .tolist()
        )
        problems.append(f"Duplicate relative paths: {duplicated_paths}")

    invalid_ids = ~df["article_id"].astype(str).str.match(r"^ART_[A-F0-9]{12}$")
    if invalid_ids.any():
        problems.append("One or more article_id values are invalid.")

    invalid_years = df["publication_year"].dropna().map(
        lambda year: not (1800 <= int(year) <= 2100)
    )
    if invalid_years.any():
        problems.append("One or more publication years are outside 1800-2100.")

    return problems


def summarize_manifest(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "n_files": 0,
            "n_unique_contents": 0,
            "n_duplicate_files": 0,
            "n_missing_year": 0,
            "n_needs_metadata_review": 0,
            "year_min": None,
            "year_max": None,
            "source_folders": {},
        }

    years = df["publication_year"].dropna()

    source_counts = (
        df["source_folder"]
        .fillna("")
        .replace("", "(root)")
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "n_files": int(len(df)),
        "n_unique_contents": int(df["sha256"].nunique()),
        "n_duplicate_files": int(df["is_duplicate_content"].sum()),
        "n_missing_year": int(df["publication_year"].isna().sum()),
        "n_needs_metadata_review": int(df["needs_metadata_review"].sum()),
        "year_min": int(years.min()) if not years.empty else None,
        "year_max": int(years.max()) if not years.empty else None,
        "source_folders": source_counts,
    }


def save_manifest(
    df: pd.DataFrame,
    output_dir: Path,
    stem: str = "articles_manifest",
) -> ManifestOutput:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{stem}.parquet"
    xlsx_path = output_dir / f"{stem}.xlsx"
    csv_path = output_dir / f"{stem}.csv"
    summary_path = output_dir / f"{stem}_summary.json"

    df.to_parquet(parquet_path, index=False)
    df.to_excel(xlsx_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = summarize_manifest(df)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    return ManifestOutput(
        dataframe=df,
        parquet_path=parquet_path,
        xlsx_path=xlsx_path,
        csv_path=csv_path,
        summary_path=summary_path,
    )


def build_and_save_manifest(
    pdf_root: Path,
    output_dir: Path,
    stem: str = "articles_manifest",
) -> ManifestOutput:
    df = build_manifest(pdf_root)
    problems = validate_manifest(df)

    fatal_problems = [
        problem for problem in problems if problem != "No PDF files were found."
    ]

    if fatal_problems:
        raise ValueError(
            "Manifest validation failed:\n- " + "\n- ".join(fatal_problems)
        )

    return save_manifest(df=df, output_dir=output_dir, stem=stem)
