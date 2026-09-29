from wonca_rag.ingestion.manifest import (
    build_and_save_manifest,
    summarize_manifest,
)
from wonca_rag.paths import PATHS


def main() -> None:
    pdf_root = PATHS.data_root / "raw" / "pdf"
    output_dir = PATHS.data_root / "metadata"

    result = build_and_save_manifest(
        pdf_root=pdf_root,
        output_dir=output_dir,
    )

    summary = summarize_manifest(result.dataframe)

    print("Manifest created successfully.")
    print(f"PDF root: {pdf_root}")
    print(f"Files found: {summary['n_files']}")
    print(f"Unique contents: {summary['n_unique_contents']}")
    print(f"Duplicate files: {summary['n_duplicate_files']}")
    print(f"Missing year: {summary['n_missing_year']}")
    print(f"Needs metadata review: {summary['n_needs_metadata_review']}")
    print(f"Parquet: {result.parquet_path}")
    print(f"Excel: {result.xlsx_path}")
    print(f"CSV: {result.csv_path}")
    print(f"Summary: {result.summary_path}")


if __name__ == "__main__":
    main()
