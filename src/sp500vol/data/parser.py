"""HTML / iXBRL parser for EDGAR filings.

10-K and 10-Q come as HTML or iXBRL (inline XBRL). We extract:
  - Item 1A (Risk Factors)
  - Item 7 / 7A (MD&A)
  - Full body text minus financial statement boilerplate

8-K comes as HTML with optional Item codes. We extract:
  - Item code (1.01, 2.02, 5.02, 7.01, 8.01, etc.)
  - Item body text
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

MIN_TEXT_TOKENS = 100


@dataclass(frozen=True)
class ParsedFiling:
    form: str
    sections: dict[str, str]  # e.g. {"item_1a": "...", "item_7": "..."}
    full_text: str  # concatenated body
    token_count: int  # estimated (whitespace-split)
    parse_warnings: list[str]


def parse_10k(html_path: Path) -> ParsedFiling:
    text, warnings = _extract_visible_text(html_path)
    sections = {
        "item_1a": _extract_section(text, "item 1a", ["item 1b", "item 2"]),
        "item_7": _extract_section(text, "item 7", ["item 7a", "item 8"]),
        "item_7a": _extract_section(text, "item 7a", ["item 8"]),
    }
    return _parsed("10-K", sections, text, warnings)


def parse_10q(html_path: Path) -> ParsedFiling:
    text, warnings = _extract_visible_text(html_path)
    sections = {
        "part_ii_item_1a": _extract_section(text, "item 1a", ["item 2", "item 3"]),
        "part_i_item_2": _extract_section(text, "item 2", ["item 3", "item 4"]),
    }
    return _parsed("10-Q", sections, text, warnings)


def parse_8k(html_path: Path) -> ParsedFiling:
    text, warnings = _extract_visible_text(html_path)
    item_matches = list(re.finditer(r"\bitem\s+(\d\.\d{2})\b", text, flags=re.IGNORECASE))
    sections: dict[str, str] = {}
    for idx, match in enumerate(item_matches):
        item_code = match.group(1)
        end = item_matches[idx + 1].start() if idx + 1 < len(item_matches) else len(text)
        sections[f"item_{item_code.replace('.', '_')}"] = text[match.start() : end].strip()
    return _parsed("8-K", sections, text, warnings)


def parse_filing(html_path: Path, form: str) -> ParsedFiling:
    """Dispatch to the appropriate parser based on form type."""
    if form == "10-K":
        return parse_10k(html_path)
    if form == "10-Q":
        return parse_10q(html_path)
    if form == "8-K":
        return parse_8k(html_path)
    raise ValueError(f"Unsupported form: {form}")


def _extract_visible_text(html_path: Path) -> tuple[str, list[str]]:
    parse_warnings: list[str] = []
    raw = html_path.read_bytes()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript", "ix:header"]):
        tag.decompose()

    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        parse_warnings.append("no visible text extracted")
    return text, parse_warnings


def _extract_section(text: str, start_label: str, end_labels: list[str]) -> str:
    normalised_text = text.lower()
    start_match = re.search(_label_pattern(start_label), normalised_text)
    if not start_match:
        return ""

    end_pos = len(text)
    for label in end_labels:
        end_match = re.search(_label_pattern(label), normalised_text[start_match.end() :])
        if end_match:
            end_pos = min(end_pos, start_match.end() + end_match.start())

    return text[start_match.start() : end_pos].strip()


def _label_pattern(label: str) -> str:
    return r"\b" + re.escape(label).replace(r"\ ", r"\s+") + r"\b"


def _parsed(
    form: str,
    sections: dict[str, str],
    full_text: str,
    warnings: list[str],
) -> ParsedFiling:
    non_empty_sections = {key: value for key, value in sections.items() if value}
    token_count = len(full_text.split())
    if token_count < MIN_TEXT_TOKENS:
        warnings = [*warnings, f"short extracted text: {token_count} tokens"]
    return ParsedFiling(
        form=form,
        sections=non_empty_sections,
        full_text=full_text,
        token_count=token_count,
        parse_warnings=warnings,
    )
