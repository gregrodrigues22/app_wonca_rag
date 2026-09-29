from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wonca_rag.rag.bm25 import bm25_rank, load_bm25
from wonca_rag.rag.corpus import filter_temporal_history
from wonca_rag.rag.embeddings import (
    encode_query,
    load_semantic_index,
    semantic_rank_from_vector,
)
from wonca_rag.rag.hybrid import reciprocal_rank_fusion


@dataclass(frozen=True)
class RetrievalSettings:
    semantic_top_k: int = 12
    bm25_top_k: int = 12
    final_top_k: int = 8
    rrf_k: int = 60


class HybridRetriever:
    def __init__(
        self,
        chunks_df: pd.DataFrame,
        embeddings_dir: Path,
        bm25_dir: Path,
        model_name: str,
        normalize_embeddings: bool = True,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self.chunks_df = chunks_df.reset_index(drop=True).copy()
        self.embeddings, self.embedding_metadata, self.embedding_index_meta = (
            load_semantic_index(embeddings_dir)
        )
        self.bm25, self.bm25_token_df = load_bm25(bm25_dir)
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.settings = settings or RetrievalSettings()

        if len(self.chunks_df) != len(self.embeddings):
            raise ValueError(
                "Chunk corpus and semantic index do not have the same number of rows."
            )

        if len(self.chunks_df) != len(self.bm25_token_df):
            raise ValueError(
                "Chunk corpus and BM25 token corpus do not have the same number of rows."
            )

        if not (
            self.chunks_df["chunk_id"].astype(str).tolist()
            == self.embedding_metadata["chunk_id"].astype(str).tolist()
        ):
            raise ValueError("Semantic index row alignment does not match chunks.")

        if not (
            self.chunks_df["chunk_id"].astype(str).tolist()
            == self.bm25_token_df["chunk_id"].astype(str).tolist()
        ):
            raise ValueError("BM25 row alignment does not match chunks.")

    def _rank(
        self,
        query: str,
        candidate_rows: np.ndarray | list[int],
    ) -> pd.DataFrame:
        candidate_rows = np.asarray(candidate_rows, dtype=int)

        query_vector = encode_query(
            query=query,
            model_name=self.model_name,
            normalize_embeddings=self.normalize_embeddings,
        )

        semantic_df = semantic_rank_from_vector(
            query_vector=query_vector,
            embeddings=self.embeddings,
            candidate_rows=candidate_rows,
            top_k=self.settings.semantic_top_k,
        )

        bm25_df = bm25_rank(
            query=query,
            bm25=self.bm25,
            candidate_rows=candidate_rows,
            top_k=self.settings.bm25_top_k,
        )

        fused = reciprocal_rank_fusion(
            semantic_df=semantic_df,
            bm25_df=bm25_df,
            rrf_k=self.settings.rrf_k,
        ).head(self.settings.final_top_k)

        if fused.empty:
            return fused

        result = fused.merge(
            self.chunks_df.reset_index(names="row_index"),
            on="row_index",
            how="left",
            validate="one_to_one",
        )

        return result.sort_values(
            "hybrid_rank",
            kind="stable",
        ).reset_index(drop=True)

    def retrieve_current_article(
        self,
        query: str,
        article_id: str,
    ) -> pd.DataFrame:
        """
        Retrieve only from the article currently being screened.
        """
        candidate_rows = self.chunks_df.index[
            self.chunks_df["article_id"].eq(article_id)
        ].to_numpy(dtype=int)

        return self._rank(
            query=query,
            candidate_rows=candidate_rows,
        )

    def retrieve_history(
        self,
        query: str,
        current_year: int,
        current_publication_date=None,
        current_article_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Retrieve only from temporally eligible historical articles.

        Temporal filtering happens BEFORE ranking. This is deliberate: filtering
        after top-k retrieval could hide eligible older chunks behind future ones.
        """
        eligible_df = filter_temporal_history(
            chunks_df=self.chunks_df.reset_index(names="row_index"),
            current_year=current_year,
            current_publication_date=current_publication_date,
            current_article_id=current_article_id,
        )

        candidate_rows = eligible_df["row_index"].to_numpy(dtype=int)

        return self._rank(
            query=query,
            candidate_rows=candidate_rows,
        )
