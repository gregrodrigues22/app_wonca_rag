import numpy as np
import pandas as pd

from wonca_rag.rag.corpus import filter_temporal_history


def test_future_rows_never_enter_temporal_candidates():
    chunks_df = pd.DataFrame(
        [
            {
                "article_id": "OLD",
                "publication_year": 1980,
                "publication_date": None,
                "chunk_id": "OLD_1",
            },
            {
                "article_id": "CURRENT",
                "publication_year": 1990,
                "publication_date": None,
                "chunk_id": "CURRENT_1",
            },
            {
                "article_id": "FUTURE",
                "publication_year": 2000,
                "publication_date": None,
                "chunk_id": "FUTURE_1",
            },
        ]
    )

    result = filter_temporal_history(
        chunks_df=chunks_df,
        current_year=1990,
        current_article_id="CURRENT",
    )

    assert result["article_id"].tolist() == ["OLD"]
    assert "FUTURE" not in result["article_id"].tolist()
    assert "CURRENT" not in result["article_id"].tolist()
