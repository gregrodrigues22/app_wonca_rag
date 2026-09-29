from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", flags=re.UNICODE)
BM25_INDEX_VERSION = "1.0.0"


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def build_bm25_token_corpus(
    chunks_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """
    Persist tokenized corpus. BM25 itself is rebuilt quickly in memory.
    """
    if chunks_df.empty:
        raise ValueError("Cannot build BM25 corpus from empty chunks.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_rows = []
    for row_idx, row in enumerate(chunks_df.itertuples(index=False)):
        token_rows.append(
            {
                "bm25_row": row_idx,
                "chunk_id": row.chunk_id,
                "tokens": tokenize(row.text),
            }
        )

    token_df = pd.DataFrame(token_rows)

    parquet_path = output_dir / "bm25_tokens.parquet"
    metadata_path = output_dir / "bm25_index_metadata.json"

    token_df.to_parquet(parquet_path, index=False)

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "index_version": BM25_INDEX_VERSION,
                "n_chunks": int(len(token_df)),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "tokens_parquet": parquet_path,
        "metadata_json": metadata_path,
    }


def load_bm25(
    output_dir: Path,
) -> tuple[BM25Okapi, pd.DataFrame]:
    output_dir = Path(output_dir)
    token_df = pd.read_parquet(output_dir / "bm25_tokens.parquet")

    tokenized_corpus = [
        list(tokens)
        for tokens in token_df["tokens"].tolist()
    ]

    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, token_df


def bm25_rank(
    query: str,
    bm25: BM25Okapi,
    candidate_rows: np.ndarray | list[int] | None = None,
    top_k: int = 12,
) -> pd.DataFrame:
    scores = np.asarray(
        bm25.get_scores(tokenize(query)),
        dtype=float,
    )

    if candidate_rows is None:
        candidate_rows = np.arange(len(scores), dtype=int)
    else:
        candidate_rows = np.asarray(candidate_rows, dtype=int)

    if candidate_rows.size == 0:
        return pd.DataFrame(
            columns=["bm25_row", "bm25_score", "bm25_rank"]
        )

    candidate_scores = scores[candidate_rows]
    order = np.argsort(-candidate_scores)[:top_k]

    selected_rows = candidate_rows[order]
    selected_scores = candidate_scores[order]

    return pd.DataFrame(
        {
            "bm25_row": selected_rows.astype(int),
            "bm25_score": selected_scores.astype(float),
            "bm25_rank": range(1, len(selected_rows) + 1),
        }
    )
