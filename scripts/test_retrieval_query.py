import pandas as pd

from wonca_rag.config import load_experiment_config
from wonca_rag.paths import PATHS
from wonca_rag.rag.retrieval import (
    HybridRetriever,
    RetrievalSettings,
)


def main() -> None:
    config = load_experiment_config()

    chunks_df = pd.read_parquet(
        PATHS.artifacts_root
        / "chunks"
        / "article_chunks.parquet"
    )

    retriever = HybridRetriever(
        chunks_df=chunks_df,
        embeddings_dir=PATHS.artifacts_root / "embeddings",
        bm25_dir=PATHS.artifacts_root / "indexes" / "bm25",
        model_name=config.embeddings.model_name,
        normalize_embeddings=config.embeddings.normalize_embeddings,
        settings=RetrievalSettings(
            semantic_top_k=config.retrieval.semantic_top_k,
            bm25_top_k=config.retrieval.bm25_top_k,
            final_top_k=config.retrieval.final_top_k,
            rrf_k=config.retrieval.rrf_k,
        ),
    )

    sample = (
        chunks_df[["article_id", "publication_year"]]
        .dropna()
        .drop_duplicates()
        .sort_values("publication_year")
        .iloc[len(chunks_df[["article_id", "publication_year"]].drop_duplicates()) // 2]
    )

    article_id = sample["article_id"]
    year = int(sample["publication_year"])

    query = (
        "primary care continuity coordination first contact "
        "comprehensiveness measurement assessment"
    )

    print("Current article:", article_id, "year:", year)

    current = retriever.retrieve_current_article(
        query=query,
        article_id=article_id,
    )

    history = retriever.retrieve_history(
        query=query,
        current_year=year,
        current_article_id=article_id,
    )

    print("\nCurrent article results:")
    print(
        current[
            [
                "hybrid_rank",
                "article_id",
                "page_start",
                "page_end",
                "rrf_score",
                "text",
            ]
        ].head(5).to_string(index=False)
    )

    print("\nHistorical results:")
    print(
        history[
            [
                "hybrid_rank",
                "article_id",
                "publication_year",
                "page_start",
                "page_end",
                "rrf_score",
                "text",
            ]
        ].head(5).to_string(index=False)
    )

    if not history.empty:
        assert (
            history["publication_year"]
            .dropna()
            .astype(int)
            .lt(year)
            .all()
        )

    print("\nTemporal retrieval smoke test: OK")


if __name__ == "__main__":
    main()
