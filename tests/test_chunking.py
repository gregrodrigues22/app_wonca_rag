from wonca_rag.rag.chunking import (
    ChunkingSettings,
    chunk_article,
)


def test_chunk_article_preserves_page_range():
    pages = [
        {
            "page_number": 1,
            "text": "Primary care " * 500,
        },
        {
            "page_number": 2,
            "text": "Continuity coordination " * 500,
        },
    ]

    settings = ChunkingSettings(
        chunk_size_tokens=200,
        chunk_overlap_tokens=40,
    )

    chunks = chunk_article(
        article_id="ART_TEST",
        sha256="abc",
        publication_year=1990,
        publication_date=None,
        filename="Test1990.pdf",
        source_json="test.json",
        text_source="native_pdf_text",
        pages=pages,
        settings=settings,
    )

    assert len(chunks) > 1
    assert chunks[0]["page_start"] >= 1
    assert chunks[0]["page_end"] >= chunks[0]["page_start"]
    assert all(chunk["n_tokens"] <= 200 for chunk in chunks)
    assert all(chunk["article_id"] == "ART_TEST" for chunk in chunks)


def test_chunk_ids_are_stable():
    pages = [{"page_number": 1, "text": "hello world " * 200}]
    settings = ChunkingSettings(
        chunk_size_tokens=100,
        chunk_overlap_tokens=20,
    )

    kwargs = dict(
        article_id="ART_TEST",
        sha256="abc",
        publication_year=2000,
        publication_date=None,
        filename="A.pdf",
        source_json="A.json",
        text_source="native_pdf_text",
        pages=pages,
        settings=settings,
    )

    chunks_1 = chunk_article(**kwargs)
    chunks_2 = chunk_article(**kwargs)

    assert [x["chunk_id"] for x in chunks_1] == [
        x["chunk_id"] for x in chunks_2
    ]
