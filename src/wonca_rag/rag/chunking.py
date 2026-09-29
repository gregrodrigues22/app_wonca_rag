from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tiktoken
from tqdm.auto import tqdm

from wonca_rag.rag.corpus import load_article_text


CHUNK_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ChunkingSettings:
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    encoding_name: str = "cl100k_base"

    def validate(self) -> None:
        if self.chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be > 0")
        if self.chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens must be >= 0")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                "chunk_overlap_tokens must be smaller than chunk_size_tokens"
            )


def _clean_page_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\x00", " ")
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def _stable_chunk_id(
    article_id: str,
    chunk_index: int,
    text: str,
) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12].upper()
    return f"{article_id}_C{chunk_index:04d}_{digest}"


def _tokenize_pages(
    pages: list[dict],
    encoding_name: str,
) -> tuple[list[int], list[int]]:
    """
    Flatten pages into a token stream while preserving token-to-page mapping.
    """
    encoding = tiktoken.get_encoding(encoding_name)

    all_tokens: list[int] = []
    token_pages: list[int] = []

    for page in pages:
        page_number = int(page.get("page_number"))
        text = _clean_page_text(page.get("text", ""))

        if not text:
            continue

        # Separator improves readability when adjacent pages are joined.
        if all_tokens:
            separator = encoding.encode("\n\n")
            all_tokens.extend(separator)
            token_pages.extend([page_number] * len(separator))

        page_tokens = encoding.encode(text)
        all_tokens.extend(page_tokens)
        token_pages.extend([page_number] * len(page_tokens))

    return all_tokens, token_pages


def chunk_article(
    article_id: str,
    sha256: str,
    publication_year: int | None,
    publication_date,
    filename: str,
    source_json: str,
    text_source: str,
    pages: list[dict],
    settings: ChunkingSettings,
) -> list[dict]:
    settings.validate()

    encoding = tiktoken.get_encoding(settings.encoding_name)
    all_tokens, token_pages = _tokenize_pages(
        pages=pages,
        encoding_name=settings.encoding_name,
    )

    if not all_tokens:
        return []

    step = settings.chunk_size_tokens - settings.chunk_overlap_tokens
    chunks: list[dict] = []

    chunk_index = 0

    for start in range(0, len(all_tokens), step):
        end = min(start + settings.chunk_size_tokens, len(all_tokens))

        token_slice = all_tokens[start:end]
        page_slice = token_pages[start:end]

        if not token_slice:
            break

        text = encoding.decode(token_slice).strip()

        if not text:
            if end >= len(all_tokens):
                break
            continue

        page_start = min(page_slice)
        page_end = max(page_slice)

        chunks.append(
            {
                "chunk_schema_version": CHUNK_SCHEMA_VERSION,
                "article_id": article_id,
                "chunk_index": chunk_index,
                "chunk_id": _stable_chunk_id(
                    article_id=article_id,
                    chunk_index=chunk_index,
                    text=text,
                ),
                "sha256": sha256,
                "filename": filename,
                "publication_year": publication_year,
                "publication_date": publication_date,
                "text_source": text_source,
                "source_json": source_json,
                "page_start": int(page_start),
                "page_end": int(page_end),
                "token_start": int(start),
                "token_end": int(end),
                "n_tokens": int(len(token_slice)),
                "text": text,
            }
        )

        chunk_index += 1

        if end >= len(all_tokens):
            break

    return chunks


def build_chunk_corpus(
    manifest_df: pd.DataFrame,
    processed_root: Path,
    settings: ChunkingSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the chunk corpus for all unique article contents.

    Returns:
        chunks_df
        article_quality_df
    """
    settings.validate()

    unique_df = (
        manifest_df
        .sort_values("relative_path", kind="stable")
        .drop_duplicates(subset=["sha256"], keep="first")
        .reset_index(drop=True)
    )

    all_chunks: list[dict] = []
    quality_rows: list[dict] = []

    for row in tqdm(
        unique_df.to_dict(orient="records"),
        desc="Building chunks",
    ):
        article_id = row["article_id"]

        try:
            article = load_article_text(
                article_id=article_id,
                processed_root=processed_root,
            )

            pages = article["pages"]
            payload = article["payload"]
            text_source = article["resolved_text_source"]
            source_json = article["source_json"]

            chunks = chunk_article(
                article_id=article_id,
                sha256=row["sha256"],
                publication_year=(
                    int(row["publication_year"])
                    if pd.notna(row.get("publication_year"))
                    else None
                ),
                publication_date=row.get("publication_date"),
                filename=row.get("filename"),
                source_json=source_json,
                text_source=text_source,
                pages=pages,
                settings=settings,
            )

            all_chunks.extend(chunks)

            total_chars = sum(
                len(_clean_page_text(page.get("text", "")))
                for page in pages
            )

            quality_rows.append(
                {
                    "article_id": article_id,
                    "filename": row.get("filename"),
                    "publication_year": row.get("publication_year"),
                    "text_source": text_source,
                    "source_json": source_json,
                    "n_pages": len(pages),
                    "total_characters": total_chars,
                    "n_chunks": len(chunks),
                    "status": "DONE" if chunks else "NO_CHUNKS",
                    "error_message": None,
                }
            )

        except Exception as exc:
            quality_rows.append(
                {
                    "article_id": article_id,
                    "filename": row.get("filename"),
                    "publication_year": row.get("publication_year"),
                    "text_source": None,
                    "source_json": None,
                    "n_pages": None,
                    "total_characters": None,
                    "n_chunks": 0,
                    "status": "ERROR",
                    "error_message": str(exc),
                }
            )

    chunks_df = pd.DataFrame(all_chunks)
    quality_df = pd.DataFrame(quality_rows)

    if not chunks_df.empty:
        chunks_df["publication_year"] = pd.to_numeric(
            chunks_df["publication_year"],
            errors="coerce",
        ).astype("Int64")

        chunks_df["publication_date"] = pd.to_datetime(
            chunks_df["publication_date"],
            errors="coerce",
        )

        chunks_df = chunks_df.sort_values(
            by=[
                "publication_year",
                "article_id",
                "chunk_index",
            ],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)

    return chunks_df, quality_df


def save_chunk_corpus(
    chunks_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    chunks_output_dir: Path,
    logs_output_dir: Path,
) -> dict[str, Path]:
    chunks_output_dir = Path(chunks_output_dir)
    logs_output_dir = Path(logs_output_dir)

    chunks_output_dir.mkdir(parents=True, exist_ok=True)
    logs_output_dir.mkdir(parents=True, exist_ok=True)

    chunks_parquet = chunks_output_dir / "article_chunks.parquet"
    chunks_csv = chunks_output_dir / "article_chunks.csv"

    quality_parquet = logs_output_dir / "chunking_report.parquet"
    quality_xlsx = logs_output_dir / "chunking_report.xlsx"

    chunks_df.to_parquet(chunks_parquet, index=False)
    chunks_df.to_csv(chunks_csv, index=False, encoding="utf-8-sig")

    quality_df.to_parquet(quality_parquet, index=False)
    quality_df.to_excel(quality_xlsx, index=False)

    return {
        "chunks_parquet": chunks_parquet,
        "chunks_csv": chunks_csv,
        "quality_parquet": quality_parquet,
        "quality_xlsx": quality_xlsx,
    }


def summarize_chunk_corpus(
    chunks_df: pd.DataFrame,
    quality_df: pd.DataFrame,
) -> dict:
    if quality_df.empty:
        return {
            "n_articles": 0,
            "n_articles_done": 0,
            "n_articles_no_chunks": 0,
            "n_articles_error": 0,
            "n_articles_ocr": 0,
            "n_chunks": 0,
            "mean_chunks_per_article": 0.0,
        }

    n_articles = len(quality_df)
    n_chunks = len(chunks_df)

    return {
        "n_articles": int(n_articles),
        "n_articles_done": int((quality_df["status"] == "DONE").sum()),
        "n_articles_no_chunks": int(
            (quality_df["status"] == "NO_CHUNKS").sum()
        ),
        "n_articles_error": int(
            (quality_df["status"] == "ERROR").sum()
        ),
        "n_articles_ocr": int(
            (quality_df["text_source"] == "ocr_tesseract").sum()
        ),
        "n_chunks": int(n_chunks),
        "mean_chunks_per_article": (
            float(n_chunks / n_articles) if n_articles else 0.0
        ),
    }
