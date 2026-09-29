import numpy as np
import pandas as pd

from wonca_rag.rag.bm25 import bm25_rank
from wonca_rag.rag.embeddings import semantic_rank_from_vector
from wonca_rag.rag.hybrid import reciprocal_rank_fusion


class FakeBM25:
    def get_scores(self, tokens):
        return np.array([0.1, 3.0, 1.0, 2.0], dtype=float)


def test_semantic_rank_respects_candidate_rows():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    query = np.array([1.0, 0.0], dtype=np.float32)

    result = semantic_rank_from_vector(
        query_vector=query,
        embeddings=embeddings,
        candidate_rows=[1, 2],
        top_k=2,
    )

    assert result["embedding_row"].tolist() == [2, 1]


def test_bm25_rank_respects_candidate_rows():
    result = bm25_rank(
        query="anything",
        bm25=FakeBM25(),
        candidate_rows=[0, 2, 3],
        top_k=2,
    )

    assert result["bm25_row"].tolist() == [3, 2]


def test_rrf_combines_rankings():
    semantic_df = pd.DataFrame(
        [
            {
                "embedding_row": 0,
                "semantic_score": 0.9,
                "semantic_rank": 1,
            },
            {
                "embedding_row": 1,
                "semantic_score": 0.8,
                "semantic_rank": 2,
            },
        ]
    )

    bm25_df = pd.DataFrame(
        [
            {
                "bm25_row": 1,
                "bm25_score": 5.0,
                "bm25_rank": 1,
            },
            {
                "bm25_row": 2,
                "bm25_score": 4.0,
                "bm25_rank": 2,
            },
        ]
    )

    result = reciprocal_rank_fusion(
        semantic_df=semantic_df,
        bm25_df=bm25_df,
        rrf_k=60,
    )

    assert result.iloc[0]["row_index"] == 1
    assert result.iloc[0]["hybrid_rank"] == 1
