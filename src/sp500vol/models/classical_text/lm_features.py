"""B4: L-M proportions + engineered length / readability features.

Augments B3 with:
  - log(token_count)
  - mean_word_length (chars per token)
  - mean_sentence_length (tokens per sentence; sentences are heuristic)
  - punctuation_density (non-alpha non-space chars / total chars)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from sp500vol.features.text_preprocess import normalize_text, tokenize
from sp500vol.models.classical_text.lm_linear import LMLinear

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")

ENGINEERED_FEATURE_NAMES: tuple[str, ...] = (
    "log_token_count",
    "mean_word_length",
    "mean_sentence_length",
    "punctuation_density",
)


class LMFeatures(LMLinear):
    """L-M proportions + light readability/length features."""

    name = "B4_lm_features"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.feature_names = tuple(self.feature_names) + ENGINEERED_FEATURE_NAMES

    def _row_features(self, text: str) -> list[float]:
        lm = super()._row_features(text)
        normalised = normalize_text(text)
        toks = tokenize(normalised)
        n_tokens = len(toks)
        log_n = float(np.log1p(n_tokens))
        mean_word = float(np.mean([len(t) for t in toks])) if toks else 0.0

        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if sentences:
            sentence_lengths = [len(tokenize(s)) for s in sentences]
            mean_sent = float(np.mean(sentence_lengths))
        else:
            mean_sent = float(n_tokens)

        n_chars = max(len(text), 1)
        punctuation = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
        punctuation_density = punctuation / n_chars

        return [*lm, log_n, mean_word, mean_sent, punctuation_density]


def _peek(_: pd.DataFrame) -> None:
    """Type-only marker so editors that strip imports keep pandas around."""
