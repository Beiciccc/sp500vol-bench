"""E1 / C6+D4 — prompt templates for the generative-LLM volatility forecaster.

Two variants (one LLM call per filing covers all three horizons at once):
  * c6_text  (run-dir model_id C6_llmtext) — TEXT ONLY: filing excerpt + form/type
    + filing date. NO price data. Comparable to the C-block text models.
  * d4_fused (run-dir model_id D4_llmfused) — c6_text + the three HAR lags
    feature_rv_1d / feature_rv_5d / feature_rv_22d as context. Comparable to D-block.

The model must answer with ONLY a JSON object:
    {"vol_5d": x, "vol_10d": x, "vol_20d": x}
where each value is an ANNUALIZED realised-vol forecast as a decimal (0.25 = 25%).

Excerpt policy (cap total prompt ~6000 tokens):
  * 8-K   : full text (median ~930 tokens), truncated to the budget if needed.
  * 10-K  : sections item_1a + item_7 + item_7a from sections_json IF substantive
            (combined length >= MIN_SECTIONS_CHARS — many sections_json entries are
            only table-of-contents stubs, ~70 chars); else head-truncate full text.
  * 10-Q  : sections part_i_item_2 + part_ii_item_1a, same substantive check,
            else head-truncate.

Token accounting uses a chars/4 heuristic (no tokenizer dependency off-box); the
box-side runner re-checks real token counts with the model tokenizer and hard-caps
at max_model_len.
"""
from __future__ import annotations

import json
import re

# ---------------------------------------------------------------- budgets
MAX_PROMPT_TOKENS = 6000          # total prompt cap (task spec)
CHARS_PER_TOKEN = 4.0             # heuristic
SCAFFOLD_TOKENS = 700             # system + instructions + metadata + chat template
EXCERPT_CHAR_BUDGET = int((MAX_PROMPT_TOKENS - SCAFFOLD_TOKENS) * CHARS_PER_TOKEN)  # ~21200
MIN_SECTIONS_CHARS = 2000         # below this, sections_json is TOC stubs -> fallback

SECTION_KEYS = {
    "10-K": ["item_1a", "item_7", "item_7a"],
    "10-Q": ["part_i_item_2", "part_ii_item_1a"],
}

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "vol_5d": {"type": "number"},
        "vol_10d": {"type": "number"},
        "vol_20d": {"type": "number"},
    },
    "required": ["vol_5d", "vol_10d", "vol_20d"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a quantitative analyst who forecasts equity realised volatility from "
    "corporate SEC disclosures. You always answer with ONLY a single JSON object and "
    "no other text, no markdown, no explanation."
)

_TASK_TEXT_ONLY = (
    "Task: based ONLY on the disclosure text above, forecast this company's ANNUALIZED "
    "realised stock-return volatility over the next 5, 10 and 20 trading days after the "
    "filing. Express each forecast as a decimal (e.g. 0.25 means 25% annualized; typical "
    "values lie between 0.10 and 1.00; extreme stress can exceed 1.00).\n"
    'Respond with ONLY this JSON object: {"vol_5d": <number>, "vol_10d": <number>, '
    '"vol_20d": <number>}'
)

_TASK_FUSED = (
    "Task: based on the disclosure text above AND the recent realised-volatility levels, "
    "forecast this company's ANNUALIZED realised stock-return volatility over the next 5, "
    "10 and 20 trading days after the filing. Express each forecast as a decimal (e.g. "
    "0.25 means 25% annualized). Use the recent volatility as your anchor and adjust it "
    "up or down according to the disclosure content.\n"
    'Respond with ONLY this JSON object: {"vol_5d": <number>, "vol_10d": <number>, '
    '"vol_20d": <number>}'
)


# --- elicitation-sensitivity paraphrases (round-2 P1): same semantics, different wording
_TASK_PARA1 = (
    "You are given a company's SEC disclosure above. Using nothing but that text, "
    "estimate how volatile the company's stock returns will be over the 5, 10 and 20 "
    "trading days following the filing, stated as ANNUALIZED realised volatility in "
    "decimal form (0.25 = 25% a year; most large-caps fall between 0.10 and 1.00, and "
    "crisis names can exceed 1.00).\n"
    'Answer with ONLY the JSON object {"vol_5d": <number>, "vol_10d": <number>, '
    '"vol_20d": <number>} and nothing else.'
)
_TASK_PARA2 = (
    "Read the filing excerpt above carefully. Your job: predict this firm's forward "
    "realised return volatility (ANNUALIZED, decimal units, e.g. 0.25 for 25%) over "
    "three horizons - the next 5, 10 and 20 trading days. Rely solely on the disclosure "
    "content; typical values 0.10-1.00.\n"
    'Reply with exactly one JSON object of the form {"vol_5d": <number>, '
    '"vol_10d": <number>, "vol_20d": <number>}.'
)

RETRY_SUFFIX = (
    "\n\nIMPORTANT: your previous answer could not be parsed. Output ONLY the JSON object "
    'in exactly this form, nothing else: {"vol_5d": 0.25, "vol_10d": 0.25, "vol_20d": 0.25}'
)


# ---------------------------------------------------------------- excerpting
def build_excerpt(form: str, sections_json: str | None, full_text: str) -> tuple[str, str]:
    """Return (excerpt, source_tag). source_tag in {'full', 'sections', 'head'}."""
    full_text = full_text or ""
    if form == "8-K":
        if len(full_text) <= EXCERPT_CHAR_BUDGET:
            return full_text, "full"
        return full_text[:EXCERPT_CHAR_BUDGET], "head"

    keys = SECTION_KEYS.get(form, [])
    sections = {}
    if sections_json:
        try:
            sections = json.loads(sections_json)
        except (json.JSONDecodeError, TypeError):
            sections = {}
    parts = [(k, sections[k]) for k in keys if isinstance(sections.get(k), str)]
    combined_len = sum(len(v) for _, v in parts)
    if combined_len >= MIN_SECTIONS_CHARS:
        # sequential fill: earlier (risk factors / MD&A) sections have priority
        out, remaining = [], EXCERPT_CHAR_BUDGET
        for key, val in parts:
            if remaining <= 200:
                break
            take = val[:remaining]
            out.append(f"[Section {key}]\n{take}")
            remaining -= len(take) + 20
        return "\n\n".join(out), "sections"
    # fallback: head-truncate the full document
    return full_text[:EXCERPT_CHAR_BUDGET], "head"


# ---------------------------------------------------------------- messages
def build_messages(row: dict, full_text: str, variant: str,
                   retry: bool = False) -> list[dict]:
    """Build chat messages for one filing.

    row needs: form, item_subtype, filing_date, sections_json, and for d4_fused
    additionally feature_rv_1d, feature_rv_5d, feature_rv_22d.
    """
    if variant not in ("c6_text", "d4_fused", "c6_dateonly", "c6_datefirm",
                       "c6_para1", "c6_para2"):
        raise ValueError(f"unknown variant {variant!r}")
    item = row.get("item_subtype") or ""
    item_str = f" (items: {item})" if item else ""
    header = (
        f"Company SEC filing.\n"
        f"- Form type: {row['form']}{item_str}\n"
        f"- Filing date: {row['filing_date']}\n"
    )
    if variant in ("c6_dateonly", "c6_datefirm"):
        # Contamination controls (mock-review P0-2): NO filing text. c6_dateonly
        # tests pure regime-timing / pretraining-era knowledge from form+date;
        # c6_datefirm adds the ticker to also expose memorized firm identity.
        ident = f"- Ticker: {row.get('ticker', '')}\n" if variant == "c6_datefirm" else ""
        user = header + ident + "\n(No filing text is provided.)\n\n" + _TASK_TEXT_ONLY
        if retry:
            user += RETRY_SUFFIX
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
    excerpt, _src = build_excerpt(row["form"], row.get("sections_json"), full_text)
    body = f"\nFiling excerpt:\n<<<\n{excerpt}\n>>>\n\n"
    if variant == "c6_para1":
        user = header + body + _TASK_PARA1
        if retry:
            user += RETRY_SUFFIX
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
    if variant == "c6_para2":
        user = header + body + _TASK_PARA2
        if retry:
            user += RETRY_SUFFIX
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
    if variant == "d4_fused":
        ctx = (
            "Recent ANNUALIZED realised volatility of this stock, computed from prices "
            "strictly before the filing:\n"
            f"- over the last 1 trading day:  {float(row['feature_rv_1d']):.4f}\n"
            f"- over the last 5 trading days: {float(row['feature_rv_5d']):.4f}\n"
            f"- over the last 22 trading days: {float(row['feature_rv_22d']):.4f}\n\n"
        )
        user = header + body + ctx + _TASK_FUSED
    else:
        user = header + body + _TASK_TEXT_ONLY
    if retry:
        user += RETRY_SUFFIX
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------- parsing
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_KEY_RES = {
    k: re.compile(rf'"?{k}"?\s*[:=]\s*"?\s*([0-9]*\.?[0-9]+)\s*(%?)', re.IGNORECASE)
    for k in ("vol_5d", "vol_10d", "vol_20d")
}


def _coerce(val: float, pct: bool) -> float | None:
    v = float(val)
    if pct or v > 3.0:          # "25%" or bare 25 -> 0.25 (annualized vol >300% = garbage)
        if v > 100.0 * 3.0:
            return None
        v = v / 100.0
    if not (0.0 < v < 10.0):
        return None
    return v


def parse_output(text: str) -> dict | None:
    """Extract {"vol_5d","vol_10d","vol_20d"} from raw model output.

    Returns dict of floats (annualized decimals) or None if unparseable.
    Handles: proper JSON, JSON embedded in prose/markdown fences, percent values,
    bare percents like 25 meaning 25%.
    """
    if not text:
        return None
    # 1) strict-ish: last JSON-looking block
    for m in reversed(_JSON_RE.findall(text)):
        try:
            d = json.loads(m)
        except json.JSONDecodeError:
            continue
        out = {}
        for k in ("vol_5d", "vol_10d", "vol_20d"):
            if k not in d:
                break
            try:
                raw = d[k]
                pct = isinstance(raw, str) and raw.strip().endswith("%")
                num = float(str(raw).strip().rstrip("%"))
            except (TypeError, ValueError):
                break
            v = _coerce(num, pct)
            if v is None:
                break
            out[k] = v
        if len(out) == 3:
            return out
    # 2) regex fallback per key over the whole text
    out = {}
    for k, rx in _KEY_RES.items():
        m = rx.search(text)
        if not m:
            return None
        v = _coerce(float(m.group(1)), m.group(2) == "%")
        if v is None:
            return None
        out[k] = v
    return out
