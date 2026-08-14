"""B2: TF-IDF (1-2 grams) + Ridge regression."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from sp500vol.models.classical_text.bow_ridge import BoWRidge


class TfidfRidge(BoWRidge):
    """TF-IDF over 1-2 grams + Ridge — a stronger BoW variant."""

    name = "B2_tfidf_ridge"

    def _build_vectorizer(self) -> TfidfVectorizer:  # type: ignore[override]
        return TfidfVectorizer(
            lowercase=True,
            max_features=self.max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b[a-z]{2,}\b",
        )
