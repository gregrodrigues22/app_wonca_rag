from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


EMBEDDING_INDEX_VERSION = "1.0.0"


@dataclass(frozen=True)
class SemanticIndexPaths:
    embeddings_npy: Path
    metadata_parquet: Path
    index_metadata_json: Path


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def corpus_fingerprint(chunks_df: pd.DataFrame) -> str:
    """
    Stable fingerprint of chunk IDs + texts in their stored order.
    """
    digest = hashlib.sha256()

    for row in chunks_df[["chunk_id", "text"]].itertuples(index=False):
        digest.update(str(row.chunk_id).encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(str(row.text).encode("utf-8"))
        digest.update(b"\x1e")

    return digest.hexdigest()


def semantic_index_paths(output_dir: Path) -> SemanticIndexPaths:
    output_dir = Path(output_dir)
    return SemanticIndexPaths(
        embeddings_npy=output_dir / "chunk_embeddings.npy",
        metadata_parquet=output_dir / "chunk_embedding_metadata.parquet",
        index_metadata_json=output_dir / "semantic_index_metadata.json",
    )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def semantic_index_is_current(
    chunks_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    normalize_embeddings: bool,
) -> bool:
    paths = semantic_index_paths(output_dir)

    if not all(
        [
            paths.embeddings_npy.exists(),
            paths.metadata_parquet.exists(),
            paths.index_metadata_json.exists(),
        ]
    ):
        return False

    try:
        metadata = _load_json(paths.index_metadata_json)
    except Exception:
        return False

    expected_fingerprint = corpus_fingerprint(chunks_df)

    return (
        metadata.get("index_version") == EMBEDDING_INDEX_VERSION
        and metadata.get("model_name") == model_name
        and metadata.get("normalize_embeddings") == normalize_embeddings
        and metadata.get("corpus_fingerprint") == expected_fingerprint
        and metadata.get("n_chunks") == len(chunks_df)
    )


def build_semantic_index(
    chunks_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    normalize_embeddings: bool = True,
    batch_size: int = 32,
    force: bool = False,
) -> SemanticIndexPaths:
    """
    Encode every chunk once and persist the matrix + aligned metadata.

    With only a few thousand chunks, exact cosine/dot-product retrieval is
    simpler and fully reproducible; FAISS is not necessary here.
    """
    if chunks_df.empty:
        raise ValueError("Cannot build semantic index from an empty chunk corpus.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = semantic_index_paths(output_dir)

    if (
        not force
        and semantic_index_is_current(
            chunks_df=chunks_df,
            output_dir=output_dir,
            model_name=model_name,
            normalize_embeddings=normalize_embeddings,
        )
    ):
        return paths

    model = SentenceTransformer(model_name)

    texts = chunks_df["text"].astype(str).tolist()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    ).astype(np.float32)

    np.save(paths.embeddings_npy, embeddings)

    metadata_columns = [
        column
        for column in [
            "chunk_id",
            "article_id",
            "chunk_index",
            "publication_year",
            "publication_date",
            "page_start",
            "page_end",
            "n_tokens",
            "text_source",
            "filename",
        ]
        if column in chunks_df.columns
    ]

    metadata_df = chunks_df[metadata_columns].copy()
    metadata_df.insert(0, "embedding_row", range(len(metadata_df)))
    metadata_df.to_parquet(paths.metadata_parquet, index=False)

    _save_json(
        {
            "index_version": EMBEDDING_INDEX_VERSION,
            "model_name": model_name,
            "normalize_embeddings": normalize_embeddings,
            "embedding_dimension": int(embeddings.shape[1]),
            "n_chunks": int(embeddings.shape[0]),
            "corpus_fingerprint": corpus_fingerprint(chunks_df),
        },
        paths.index_metadata_json,
    )

    return paths


def load_semantic_index(
    output_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    paths = semantic_index_paths(output_dir)

    embeddings = np.load(paths.embeddings_npy)
    metadata_df = pd.read_parquet(paths.metadata_parquet)
    metadata = _load_json(paths.index_metadata_json)

    if len(embeddings) != len(metadata_df):
        raise ValueError(
            "Semantic index matrix and metadata have different lengths."
        )

    return embeddings, metadata_df, metadata


def encode_query(
    query: str,
    model_name: str,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    model = SentenceTransformer(model_name)
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )[0].astype(np.float32)
    return vector


def semantic_rank_from_vector(
    query_vector: np.ndarray,
    embeddings: np.ndarray,
    candidate_rows: np.ndarray | list[int] | None = None,
    top_k: int = 12,
) -> pd.DataFrame:
    """
    Exact dot-product ranking.

    If embeddings are normalized, dot product equals cosine similarity.
    """
    if candidate_rows is None:
        candidate_rows = np.arange(len(embeddings), dtype=int)
    else:
        candidate_rows = np.asarray(candidate_rows, dtype=int)

    if candidate_rows.size == 0:
        return pd.DataFrame(
            columns=["embedding_row", "semantic_score", "semantic_rank"]
        )

    candidate_matrix = embeddings[candidate_rows]
    scores = candidate_matrix @ query_vector

    order = np.argsort(-scores)[:top_k]
    selected_rows = candidate_rows[order]
    selected_scores = scores[order]

    return pd.DataFrame(
        {
            "embedding_row": selected_rows.astype(int),
            "semantic_score": selected_scores.astype(float),
            "semantic_rank": range(1, len(selected_rows) + 1),
        }
    )
