import pandas as pd

from wonca_rag.ingestion.pdf_extractor import (
    extract_unique_articles,
    save_extraction_report,
    summarize_extraction,
)
from wonca_rag.paths import PATHS


def main() -> None:
    manifest_path = PATHS.data_root / "metadata" / "articles_manifest.parquet"

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Manifest not found. Run Stage 1 first: "
            f"{manifest_path}"
        )

    manifest_df = pd.read_parquet(manifest_path)

    pdf_root = PATHS.data_root / "raw" / "pdf"
    processed_root = PATHS.data_root / "processed"
    checkpoints_root = PATHS.outputs_root / "checkpoints" / "pdf_extraction"
    report_output_dir = PATHS.outputs_root / "logs"

    report_df = extract_unique_articles(
        manifest_df=manifest_df,
        pdf_root=pdf_root,
        processed_root=processed_root,
        checkpoints_root=checkpoints_root,
        resume=True,
    )

    paths = save_extraction_report(
        report_df=report_df,
        output_dir=report_output_dir,
    )

    summary = summarize_extraction(report_df)

    print("\nPDF extraction completed.")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nReports:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
