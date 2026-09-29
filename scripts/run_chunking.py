import pandas as pd

from wonca_rag.config import load_experiment_config
from wonca_rag.paths import PATHS
from wonca_rag.rag.chunking import (
    ChunkingSettings,
    build_chunk_corpus,
    save_chunk_corpus,
    summarize_chunk_corpus,
)


def main() -> None:
    config = load_experiment_config()

    manifest_path = (
        PATHS.data_root
        / "metadata"
        / "articles_manifest.parquet"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    manifest_df = pd.read_parquet(manifest_path)

    settings = ChunkingSettings(
        chunk_size_tokens=config.chunking.chunk_size_tokens,
        chunk_overlap_tokens=config.chunking.chunk_overlap_tokens,
    )

    chunks_df, quality_df = build_chunk_corpus(
        manifest_df=manifest_df,
        processed_root=PATHS.data_root / "processed",
        settings=settings,
    )

    output_paths = save_chunk_corpus(
        chunks_df=chunks_df,
        quality_df=quality_df,
        chunks_output_dir=PATHS.artifacts_root / "chunks",
        logs_output_dir=PATHS.outputs_root / "logs",
    )

    summary = summarize_chunk_corpus(
        chunks_df=chunks_df,
        quality_df=quality_df,
    )

    print("\nChunking completed.")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nOutputs:")
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
