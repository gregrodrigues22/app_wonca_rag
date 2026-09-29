from wonca_rag.ingestion.ocr import (
    available_languages,
    configure_tesseract,
)


def main() -> None:
    path = configure_tesseract()
    languages = available_languages()

    print("Tesseract OCR: OK")
    print("Executable:", path)
    print("Available languages:", ", ".join(languages))


if __name__ == "__main__":
    main()
