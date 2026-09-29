import pandas as pd

from wonca_rag.config import load_experiment_config
from wonca_rag.paths import PATHS
from wonca_rag.rag.bm25 import build_bm25_token_corpus
from wonca_rag.rag.embeddings import build_semantic_index


def main() -> None:
    config = load_experiment_config()

    chunks_path = (
        PATHS.artifacts_root
        / "chunks"
        / "article_chunks.parquet"
    )

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunk corpus not found: {chunks_path}"
        )

    chunks_df = pd.read_parquet(chunks_path)

    print(f"Chunks: {len(chunks_df)}")
    print(
        "Embedding model:",
        config.embeddings.model_name,
    )

    semantic_paths = build_semantic_index(
        chunks_df=chunks_df,
        output_dir=PATHS.artifacts_root / "embeddings",
        model_name=config.embeddings.model_name,
        normalize_embeddings=config.embeddings.normalize_embeddings,
        batch_size=config.embeddings.batch_size,
        force=False,
    )

    bm25_paths = build_bm25_token_corpus(
        chunks_df=chunks_df,
        output_dir=PATHS.artifacts_root / "indexes" / "bm25",
    )

    print("\nSemantic index:")
    print("embeddings:", semantic_paths.embeddings_npy)
    print("metadata:", semantic_paths.metadata_parquet)
    print("index metadata:", semantic_paths.index_metadata_json)

    print("\nBM25:")
    for key, path in bm25_paths.items():
        print(f"{key}: {path}")

    print("\nRetrieval indexes: OK")


if __name__ == "__main__":
    main()
