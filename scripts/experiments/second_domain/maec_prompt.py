"""MAEC audit — FROZEN prompt templates for the prompted arm and the identity
probe (prereg configs/prereg_maec_audit.md §5-2/§5-3, tag prereg-maec-v1.0).

Mirrors scripts/experiments/e1_llm_forecast/prompt.py so the run_inference-style
vLLM machinery (guided JSON, one retry pass, budget re-fit) can consume it
unchanged in structure. One LLM call per CALL covers all four horizons at once.

Variants:
  * maec_text      — full transcript arm (§5-2): ticker + CRSP company name +
                     call date + transcript (head-only truncation, OPEN-12).
  * maec_identity  — zero-content identity probe (§5-3, OPEN-7): ticker + CRSP
                     company name (comnam) + call date, NO transcript; every
                     other token identical to maec_text.

Ask (§5-2): future n-trading-day ANNUALISED volatility in PERCENT for
n in {3, 7, 15, 30}, as ONE JSON object:
    {"vol_3d": x, "vol_7d": x, "vol_15d": x, "vol_30d": x}
Postprocessing (frozen): clip to [3, 300] %, then
    sigma_daily = (ann_pct / 100) / sqrt(252)     ->     v_hat = ln(sigma_daily).

Truncation (OPEN-12, head-only): 12,000 prompt tokens for the transcript under
the chars/4 heuristic (box runner re-checks with the real tokenizer and re-fits
to max_model_len = 16,384); truncation trigger counts must be disclosed.

Company name source: 700-side ticker_permno_map.parquet `name`; 513-side CRSP
IssuerNm point-in-time — both materialised as the panel's `company_name` column
by maec_build_panel.py; the runner passes it in the row dict.

Runner sampling contract (§5-2, frozen): Qwen3-32B-AWQ, single GPU, vLLM
offline batch, temperature 0, single seed (C6-primary discipline), thinking
mode OFF, max_tokens ~160.

Prediction parquet contract for maec_protocol.py (post-processing target):
    [permno, call_date, horizon, split, label, prediction, arm]
with prediction = v_hat, one row per (call x horizon), arm in
{"prompted_qwen", "identity_probe"}.
"""
from __future__ import annotations

import json
import math
import re

# ---------------------------------------------------------------- budgets
MAX_MODEL_LEN = 16_384            # §5-2
TRANSCRIPT_TOKENS = 12_000        # OPEN-12 head-only budget (transcript only)
CHARS_PER_TOKEN = 4.0             # heuristic; box runner re-checks real tokens
TRANSCRIPT_CHAR_BUDGET = int(TRANSCRIPT_TOKENS * CHARS_PER_TOKEN)  # 48,000
CLIP_PCT_LO, CLIP_PCT_HI = 3.0, 300.0     # §5-2 clip range, annualised %
HORIZONS = (3, 7, 15, 30)
VOL_KEYS = ("vol_3d", "vol_7d", "vol_15d", "vol_30d")

JSON_SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "number"} for k in VOL_KEYS},
    "required": list(VOL_KEYS),
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a quantitative analyst who forecasts equity realised volatility from "
    "corporate earnings conference calls. You always answer with ONLY a single JSON "
    "object and no other text, no markdown, no explanation."
)

_TASK = (
    "Task: forecast this company's ANNUALISED realised stock-return volatility, in "
    "PERCENT, over the next 3, 7, 15 and 30 trading days after the call. Express each "
    "forecast as a percent number (e.g. 25 means 25% annualised; typical values lie "
    "between 10 and 100; extreme stress can exceed 100).\n"
    'Respond with ONLY this JSON object: {"vol_3d": <number>, "vol_7d": <number>, '
    '"vol_15d": <number>, "vol_30d": <number>}'
)

_TASK_TEXT = (
    "Task: based ONLY on the earnings-call transcript above, forecast this company's "
    "ANNUALISED realised stock-return volatility, in PERCENT, over the next 3, 7, 15 "
    "and 30 trading days after the call. Express each forecast as a percent number "
    "(e.g. 25 means 25% annualised; typical values lie between 10 and 100; extreme "
    "stress can exceed 100).\n"
    'Respond with ONLY this JSON object: {"vol_3d": <number>, "vol_7d": <number>, '
    '"vol_15d": <number>, "vol_30d": <number>}'
)

RETRY_SUFFIX = (
    "\n\nIMPORTANT: your previous answer could not be parsed. Output ONLY the JSON "
    'object in exactly this form, nothing else: {"vol_3d": 25, "vol_7d": 25, '
    '"vol_15d": 25, "vol_30d": 25}'
)


# ---------------------------------------------------------------- excerpting
def build_excerpt(full_text: str) -> tuple[str, str, bool]:
    """Head-only truncation to the 12k-token (chars/4) budget (OPEN-12).
    Returns (excerpt, source_tag, truncated). Truncation counts are a disclosed
    quantity — the runner must aggregate the `truncated` flag."""
    full_text = full_text or ""
    if len(full_text) <= TRANSCRIPT_CHAR_BUDGET:
        return full_text, "full", False
    return full_text[:TRANSCRIPT_CHAR_BUDGET], "head", True


# ---------------------------------------------------------------- messages
def build_messages(row: dict, full_text: str, variant: str,
                   retry: bool = False) -> list[dict]:
    """Build chat messages for one call.

    row needs: ticker, company_name, call_date (str or date-like).
    full_text: the transcript (ignored for maec_identity).
    """
    if variant not in ("maec_text", "maec_identity"):
        raise ValueError(f"unknown variant {variant!r}")
    call_date = str(row["call_date"])[:10]
    header = (
        f"Company earnings conference call.\n"
        f"- Ticker: {row['ticker']}\n"
        f"- Company: {row['company_name']}\n"
        f"- Call date: {call_date}\n"
    )
    if variant == "maec_identity":
        # §5-3: ticker + CRSP comnam + call date, NO transcript; the probe must
        # maximally elicit the identity prior (OPEN-7).
        user = header + "\n(No transcript is provided.)\n\n" + _TASK
    else:
        excerpt, _src, _trunc = build_excerpt(full_text)
        user = header + f"\nCall transcript:\n<<<\n{excerpt}\n>>>\n\n" + _TASK_TEXT
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
    for k in VOL_KEYS
}


def _coerce(val: float) -> float | None:
    """Percent-unit coercion: values in (0, 3) are read as DECIMAL annualised
    vol (0.25 -> 25%) — any true percent below 3 clips to 3 anyway, so the rule
    is loss-free w.r.t. the frozen [3, 300] clip. Non-positive or absurd
    (> 3000) values are unparseable."""
    v = float(val)
    if v <= 0.0:
        return None
    if v < CLIP_PCT_LO:
        v = v * 100.0
    if v > 3000.0:
        return None
    return v


def parse_output(text: str) -> dict | None:
    """Extract the four annualised-percent forecasts from raw model output.
    Returns {vol_3d, vol_7d, vol_15d, vol_30d} (floats, percent units, NOT yet
    clipped) or None. Handles proper JSON, JSON in prose/markdown, % suffixes,
    and decimal-form answers."""
    if not text:
        return None
    for m in reversed(_JSON_RE.findall(text)):
        try:
            d = json.loads(m)
        except json.JSONDecodeError:
            continue
        out = {}
        for k in VOL_KEYS:
            if k not in d:
                break
            try:
                raw = d[k]
                num = float(str(raw).strip().rstrip("%"))
            except (TypeError, ValueError):
                break
            v = _coerce(num)
            if v is None:
                break
            out[k] = v
        if len(out) == len(VOL_KEYS):
            return out
    out = {}
    for k, rx in _KEY_RES.items():
        m = rx.search(text)
        if not m:
            return None
        v = _coerce(float(m.group(1)))
        if v is None:
            return None
        out[k] = v
    return out


def to_v(ann_pct: float) -> float:
    """Frozen unit conversion: clip [3, 300]% -> sigma_daily -> v = ln sigma."""
    p = min(max(float(ann_pct), CLIP_PCT_LO), CLIP_PCT_HI)
    return math.log((p / 100.0) / math.sqrt(252.0))


def horizon_of(key: str) -> int:
    return int(key.split("_")[1].rstrip("d"))
