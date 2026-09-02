"""Text preprocessing utilities for classical text baselines.

These helpers are intentionally simple and dependency-free so they can run on
CPU during dataset construction and inside the Block B classical baselines
(BoW / TF-IDF / Loughran-McDonald).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

# Loughran-McDonald and BoW conventions: lowercase, ASCII-folded, tokens of
# alphabetic characters (length >= 2) split by whitespace.
_TOKEN_RE = re.compile(r"[a-z]{2,}")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, fold accents, collapse whitespace.

    Idempotent. Preserves only printable characters; useful as the canonical
    representation before vectorisation or dictionary lookup.
    """
    if text is None:
        return ""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    folded = folded.lower()
    return _WHITESPACE_RE.sub(" ", folded).strip()


def tokenize(text: str) -> list[str]:
    """Whitespace tokeniser that keeps only alphabetic tokens of length >= 2."""
    return _TOKEN_RE.findall(normalize_text(text))


def token_count(text: str) -> int:
    """Return the count produced by `tokenize`, without materialising the list."""
    return sum(1 for _ in _TOKEN_RE.finditer(normalize_text(text)))


def count_lexicon_matches(text: str, lexicon: Iterable[str]) -> int:
    """Count tokens in `text` that appear in `lexicon`.

    `lexicon` is expected to contain already-normalised lowercase words.
    """
    lex_set = lexicon if isinstance(lexicon, set | frozenset) else set(lexicon)
    return sum(1 for tok in tokenize(text) if tok in lex_set)
