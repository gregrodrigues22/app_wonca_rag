import pandas as pd

from wonca_rag.config import load_experiment_config
from wonca_rag.ingestion.ocr import (
    run_selective_ocr,
    save_ocr_report,
    select_ocr_candidates,
    summarize_ocr,
)
from wonca_rag.paths import PATHS


def main() -> None:
    config = load_experiment_config()

    extraction_report_path = (
        PATHS.outputs_root
        / "logs"
        / "pdf_extraction_report.parquet"
    )

    if not extraction_report_path.exists():
        raise FileNotFoundError(
            "PDF extraction report not found. Run Stage 2 first: "
            f"{extraction_report_path}"
        )

    extraction_report_df = pd.read_parquet(
        extraction_report_path
    )

    candidates_df = select_ocr_candidates(
        extraction_report_df=extraction_report_df,
        trigger=config.ocr.trigger,
    )

    print(f"OCR candidates: {len(candidates_df)}")

    if candidates_df.empty:
        print("No OCR candidates. Nothing to do.")
        return

    report_df = run_selective_ocr(
        candidates_df=candidates_df,
        pdf_root=PATHS.data_root / "raw" / "pdf",
        ocr_output_root=PATHS.data_root / "processed" / "ocr",
        checkpoints_root=(
            PATHS.outputs_root / "checkpoints" / "ocr"
        ),
        language=config.ocr.language,
        dpi=config.ocr.dpi,
        psm=config.ocr.psm,
        min_characters_after_ocr=(
            config.ocr.min_characters_after_ocr
        ),
        resume=True,
    )

    paths = save_ocr_report(
        report_df=report_df,
        output_dir=PATHS.outputs_root / "logs",
    )

    summary = summarize_ocr(report_df)

    print("\nSelective OCR completed.")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nReports:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
