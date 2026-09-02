"""Loughran-McDonald sentiment dictionary (2011) loader.

Source: https://sraf.nd.edu/loughranmcdonald-master-dictionary/

Citation:
  Loughran, T. and McDonald, B. (2011). When is a liability not a liability?
  Textual analysis, dictionaries, and 10-Ks. Journal of Finance 66(1):35-65.

The official master CSV ships with columns like Negative / Positive / Uncertainty /
Litigious / Constraining / Modal (1=strong, 2=moderate, 3=weak). A word belongs
to a category when the column is non-zero.

For reproducibility we expect the user to download the CSV separately (it is
several MB) and point the loader at the local file. A small mock dictionary
is exposed for tests and CI runs that should not touch the filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sp500vol.features.text_preprocess import token_count, tokenize

# Categories we report. We collapse the three modal-strength columns into one
# "modal" bucket — the modelling layer can split if needed.
LM_CATEGORIES: tuple[str, ...] = (
    "negative",
    "positive",
    "uncertainty",
    "litigious",
    "constraining",
    "superfluous",
    "interesting",
    "modal",
)


@dataclass(frozen=True, slots=True)
class LoughranMcDonaldDictionary:
    """In-memory L-M lexicon partitioned by sentiment category."""

    by_category: dict[str, frozenset[str]]

    @classmethod
    def from_csv(cls, path: Path) -> LoughranMcDonaldDictionary:
        """Load the official L-M master CSV.

        Args:
            path: local copy of `Loughran-McDonald_MasterDictionary_*.csv`.
        """
        df = pd.read_csv(path)
        # Normalise column case for resilience across release years.
        df.columns = [c.strip().lower() for c in df.columns]
        if "word" not in df.columns:
            raise ValueError(f"L-M CSV at {path} missing 'word' column")

        words = df["word"].astype(str).str.lower().str.strip()
        valid = words.str.match(r"^[a-z]{2,}$")
        df = df.loc[valid].assign(word=words[valid]).reset_index(drop=True)

        columns = set(df.columns)
        by_category: dict[str, frozenset[str]] = {}
        for category in LM_CATEGORIES:
            modal_cols = [
                c
                for c in ("modal1", "modal2", "modal3", "strong_modal", "weak_modal")
                if c in columns
            ]
            if category in columns:
                mask = _nonzero(df[category])
            elif category == "modal" and modal_cols:
                # Real L-M master CSV stores Strong_Modal / Weak_Modal as
                # separate non-zero-coded columns; older releases used
                # Modal1/2/3. Union them all into one "modal" bucket.
                mask = pd.concat([_nonzero(df[c]) for c in modal_cols], axis=1).any(axis=1)
            else:
                by_category[category] = frozenset()
                continue
            by_category[category] = frozenset(df.loc[mask, "word"].unique().tolist())

        return cls(by_category=by_category)

    @classmethod
    def mock(cls) -> LoughranMcDonaldDictionary:
        """A tiny in-memory L-M-like dictionary for tests and CI smoke runs."""
        return cls(
            by_category={
                "negative": frozenset({"loss", "decline", "adverse", "litigation"}),
                "positive": frozenset({"gain", "growth", "improve", "innovation"}),
                "uncertainty": frozenset({"may", "could", "approximate", "uncertain"}),
                "litigious": frozenset({"litigation", "court", "settlement"}),
                "constraining": frozenset({"required", "must", "restrict"}),
                "superfluous": frozenset(),
                "interesting": frozenset(),
                "modal": frozenset({"may", "might", "could", "should", "must"}),
            }
        )

    def counts(self, text: str) -> dict[str, int]:
        """Raw token counts per category."""
        toks = tokenize(text)
        out: dict[str, int] = {}
        for category, lex in self.by_category.items():
            out[category] = sum(1 for t in toks if t in lex)
        return out

    def proportions(self, text: str) -> dict[str, float]:
        """Token counts per category divided by total document tokens.

        Returns 0.0 for empty documents to keep downstream features finite.
        """
        n = token_count(text)
        if n == 0:
            return {c: 0.0 for c in self.by_category}
        c = self.counts(text)
        return {category: count / n for category, count in c.items()}


def _nonzero(series: pd.Series) -> pd.Series:
    """Boolean mask of rows where a (possibly year-coded) L-M column is non-zero.

    The master dictionary codes category membership as the year a word entered
    the category (e.g. 2009, 2011), or 0 for non-members. Coerce to numeric and
    treat any non-zero value as membership.
    """
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return numeric != 0
