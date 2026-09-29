from __future__ import annotations

import pandas as pd


def reciprocal_rank_fusion(
    semantic_df: pd.DataFrame,
    bm25_df: pd.DataFrame,
    semantic_row_col: str = "embedding_row",
    bm25_row_col: str = "bm25_row",
    rrf_k: int = 60,
) -> pd.DataFrame:
    """
    Fuse semantic and lexical rankings using Reciprocal Rank Fusion.

    RRF avoids combining raw cosine and BM25 scores, which are on different scales.
    """
    scores: dict[int, dict] = {}

    for row in semantic_df.to_dict(orient="records"):
        idx = int(row[semantic_row_col])
        item = scores.setdefault(
            idx,
            {
                "row_index": idx,
                "rrf_score": 0.0,
                "semantic_rank": None,
                "semantic_score": None,
                "bm25_rank": None,
                "bm25_score": None,
            },
        )
        rank = int(row["semantic_rank"])
        item["semantic_rank"] = rank
        item["semantic_score"] = float(row["semantic_score"])
        item["rrf_score"] += 1.0 / (rrf_k + rank)

    for row in bm25_df.to_dict(orient="records"):
        idx = int(row[bm25_row_col])
        item = scores.setdefault(
            idx,
            {
                "row_index": idx,
                "rrf_score": 0.0,
                "semantic_rank": None,
                "semantic_score": None,
                "bm25_rank": None,
                "bm25_score": None,
            },
        )
        rank = int(row["bm25_rank"])
        item["bm25_rank"] = rank
        item["bm25_score"] = float(row["bm25_score"])
        item["rrf_score"] += 1.0 / (rrf_k + rank)

    if not scores:
        return pd.DataFrame(
            columns=[
                "row_index",
                "rrf_score",
                "hybrid_rank",
                "semantic_rank",
                "semantic_score",
                "bm25_rank",
                "bm25_score",
            ]
        )

    result = pd.DataFrame(scores.values())
    result = result.sort_values(
        by=["rrf_score", "row_index"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    result.insert(
        2,
        "hybrid_rank",
        range(1, len(result) + 1),
    )

    return result
