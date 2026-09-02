"""C-anon step 1 (prereg-ea-v1.0, configs/prereg_swap_lf_and_anon.md §C-anon) —
NER + rule-table entity masking of a document panel:
  --panel ed  (default) the EVENT-DRIVEN (8-K) panel — the registered primary;
  --panel lf  the LONG-FORM (10-K/10-Q) panel — the registered stretch
              ("long-form as stretch, only after ED completes"), same rule table, same NER model,
              same placeholders, same audit protocol; only the doc-id set and
              the output names change (…_lf).

Registered design (binding): "NER masking (company names/ticker/executive person names/product names/CIK;
spaCy en_core_web_lg + rule table, masking rate and sample disclosure) applied to the event-driven panel".
Deterministic placeholders: [FIRM] [TICKER] [PERSON] [PRODUCT] [CIK].

What is masked (span sources, priority CIK > TICKER > FIRM > PERSON > PRODUCT
when overlapping spans merge):
  1. Rule — company names: CRSP IssuerNm (raw/wrds/crsp_names_csv.zip, panel
     permnos via processed/full/universe.parquet) + Compustat conm/conml
     (raw/wrds/comp_company_csv.zip, panel CIKs). Token-sequence matching,
     punctuation/case tolerant; hyphen/& variants generated. NON-own firms
     need >=2 distinctive tokens, or 1 distinctive token + a corporate suffix
     (Inc/Corp/Co/...), to avoid masking common words ("Target" alone).
     The filing's OWN firm is masked aggressively: any single distinctive
     name token with initial capital (unless a common English word) -> [FIRM].
  2. Rule — tickers: the filing's own ticker (len>=2, case-sensitive \b);
     any panel-universe ticker len>=2 as an ALL-CAPS token (minus a fixed
     common-word blocklist); any symbol in exchange context
     ("NYSE: XYZ", "under the symbol 'XYZ'") incl. 1-letter tickers.
  3. Rule — CIK strings: any digit run equal to a panel CIK (guarding the
     1900-2100 year range), SEC accession numbers dddddddddd-dd-dddddd, and
     Commission File Number values. -> [CIK]
  4. Rule — product heuristic: capitalised name immediately followed by a
     (R)/(TM)/(SM) trademark sign -> [PRODUCT].
  5. spaCy en_core_web_lg NER: ORG -> [FIRM], PERSON -> [PERSON] (this is the
     registered executive-name channel), PRODUCT -> [PRODUCT]. Long documents
     are chunked (<=90k chars, split at whitespace) with offset bookkeeping;
     NER output is deterministic per doc for a fixed model version.

Gates served here:
  G2 — 100-doc audit sample (seed 2026) written side-by-side for the human
       rule check, plus quantitative leak rates (own ticker / own name token
       still present after masking) in mask_stats.json.

Outputs (…_lf names for --panel lf):
  masked store   <data_root>/processed/_text_cache/filing_texts_anonmask_ed.parquet
                 (lf: filing_texts_anonmask_lf.parquet)
                 schema (text_path, text) — a drop-in replacement for the
                 shared text cache, consumed by anon_run_arms.py /
                 anon_run_arms_lf.py.
  per-doc stats  <store>.stats.parquet
  aggregate      results/anon/mask_stats[_lf].json          (smoke: *_smoke.json)
  audit sample   results/anon/mask_audit_sample[_lf].md     (smoke: *_smoke.md)

CPU only. Local smoke:   .venv/bin/python scripts/analysis/anon_mask_build.py --limit 200
LF local smoke:          .venv/bin/python scripts/analysis/anon_mask_build.py --panel lf --limit 200
Box full ED build:       python scripts/analysis/anon_mask_build.py --threads 16
Box full LF build:       python scripts/analysis/anon_mask_build.py --panel lf --threads 16 --batch-docs 2000
(box: export SP500VOL_DATA_ROOT=/data/sp500vol-data; spaCy + en_core_web_lg
wheels must be installed in the venv — the box is offline, ship the wheels.)

--batch-docs N (recommended for lf: long-form docs average ~250k chars, the
full build runs for hours): masks N docs at a time, writing each batch as an
atomic part parquet under <store>.parts/; re-invoking SKIPS complete parts, so
a crash costs at most one batch. The final store/stats/audit are assembled
from the parts and the parts dir is removed. With --batch-docs 0 (default)
the original single-shot flow (with --resume append semantics) is unchanged.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # spaCy parallelism = processes, not BLAS
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from sp500vol.utils.paths import data_path  # noqa: E402

PLACEHOLDER = {"CIK": "[CIK]", "TICKER": "[TICKER]", "FIRM": "[FIRM]",
               "PERSON": "[PERSON]", "PRODUCT": "[PRODUCT]"}
PRIORITY = {"CIK": 0, "TICKER": 1, "FIRM": 2, "PERSON": 3, "PRODUCT": 4}
NER_LABELS = {"ORG": "FIRM", "PERSON": "PERSON", "PRODUCT": "PRODUCT"}
CHUNK_CHARS = 90_000
AUDIT_N, AUDIT_SEED, AUDIT_EXCERPT = 100, 2026, 1200

# Corporate suffix / generic name tokens (dropped from the "distinctive core",
# consumed optionally as span extensions).
GENERIC = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "COMPANIES",
    "LTD", "LIMITED", "PLC", "LP", "LLP", "LLC", "LLLP", "TRUST", "HOLDINGS",
    "HOLDING", "HLDGS", "GROUP", "GRP", "INTL", "THE", "NEW", "DEL", "CL",
    "COM", "ADR", "ADS", "SER", "SERIES", "AND", "OF",
}
# Common-English guard for single-token own-firm masking and the ticker rule.
COMMON_WORDS = {
    "GENERAL", "AMERICAN", "NATIONAL", "INTERNATIONAL", "UNITED", "FIRST",
    "GLOBAL", "STANDARD", "FEDERAL", "CONSOLIDATED", "CONTINENTAL", "PUBLIC",
    "SERVICE", "SERVICES", "ENERGY", "POWER", "ELECTRIC", "GAS", "OIL",
    "WATER", "AIR", "HEALTH", "CAPITAL", "FINANCIAL", "INSURANCE", "BANK",
    "TRUST", "PROPERTIES", "REALTY", "STORES", "BRANDS", "FOODS", "MOTORS",
    "AIRLINES", "SYSTEMS", "TECHNOLOGIES", "TECHNOLOGY", "COMMUNICATIONS",
    "RESOURCES", "INDUSTRIES", "MATERIALS", "PRODUCTS", "SOLUTIONS",
    "PARTNERS", "ENTERPRISES", "WESTERN", "EASTERN", "SOUTHERN", "NORTHERN",
    "PACIFIC", "ATLANTIC", "MID", "ONE", "PLUS", "BEST", "KEY", "CROWN",
    "MARATHON", "DISCOVER", "PROGRESSIVE", "GAP", "TARGET", "APACHE",
    "VISA", "CATERPILLAR",  # last 4 kept ONLY out of single-token own rule
}
# Tickers that read as common words/abbreviations: excluded from the ALL-CAPS
# panel-ticker rule (still masked in exchange context and by the own-firm rule).
TICKER_BLOCKLIST = {
    "A", "ALL", "AN", "AND", "ANY", "ARE", "AT", "BE", "BEN", "BIG", "BY",
    "CAN", "DAY", "DD", "DE", "DO", "EAT", "FAST", "FOR", "GAS", "HAS", "HE",
    "IT", "ITS", "KEY", "LOW", "MA", "MAN", "MAY", "MET", "MO", "NOW", "ON",
    "ONE", "OR", "PEG", "PM", "RE", "SEE", "SO", "TAP", "TEL", "WELL", "ARM",
    "CO", "AM", "PS", "AWAY", "BAND", "BILL", "BRO", "CAR", "CARS", "CASH",
    "COST", "EDIT", "EYE", "FIT", "FIVE", "FLOW", "FUN", "GO", "GOOD",
    "HELP", "HI", "HOG", "HOPE", "INFO", "JOB", "LAND", "LIFE", "LOVE",
    "MAIN", "MASS", "NET", "NICE", "OPEN", "OUT", "PAY", "PLAY", "PRO",
    "RACE", "RIDE", "ROAD", "ROCK", "SAFE", "SAVE", "SELF", "SHIP", "SITE",
    "SKIN", "STAY", "STEP", "SUM", "TALK", "TEAM", "TECH", "TEN", "TWO",
    "USA", "WOOD", "YOU",
}  # NB: CAT/ICE/KEY-like real panel firms stay maskable unless listed here;
#    KEY is listed (KeyCorp caught by name/NER rules instead).
YEAR_LO, YEAR_HI = 1900, 2100

# NER ORG spans that are regulators / statutes / boilerplate — carry zero firm
# identity; masking them only destroys content symmetrically. Normalised
# (uppercase, alnum+space) exact matches are left unmasked.
NER_ALLOWLIST = {
    "SECURITIES AND EXCHANGE COMMISSION",
    "UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
    "THE UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
    "SEC", "IRS", "CFR", "FASB", "GAAP", "FDIC", "FERC", "FDA", "EPA", "DOJ",
    "FTC", "OSHA", "CFTC", "FINRA", "PCAOB", "INTERNAL REVENUE SERVICE",
    "SECURITIES ACT", "THE SECURITIES ACT", "SECURITIES ACT OF 1933",
    "THE SECURITIES ACT OF 1933", "EXCHANGE ACT", "THE EXCHANGE ACT",
    "SECURITIES EXCHANGE ACT", "THE SECURITIES EXCHANGE ACT",
    "SECURITIES EXCHANGE ACT OF 1934", "THE SECURITIES EXCHANGE ACT OF 1934",
    "COMMISSION", "THE COMMISSION", "CONGRESS", "TREASURY",
    "FEDERAL RESERVE", "THE FEDERAL RESERVE", "US TREASURY", "U S TREASURY",
    "STATE OR OTHER JURISDICTION", "COMMISSION FILE NUMBER",
}


def norm_phrase(s: str) -> str:
    return " ".join(re.findall(r"[A-Z0-9]+", s.upper()))

TOKEN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9'\.\-&]*[A-Za-z0-9])?")
DIGIT_RE = re.compile(r"\b\d{4,10}\b")
ACCESSION_RE = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")
FILENUM_RE = re.compile(
    r"(?:Commission\s+File\s+(?:No\.?|Number)[:\s]*)(\d{1,3}-\d{2,6})", re.IGNORECASE)
EXCHANGE_RE = re.compile(
    r"(?:NYSE|NASDAQ|Nasdaq|AMEX|New\s+York\s+Stock\s+Exchange|"
    r"Nasdaq\s+(?:Global\s+Select\s+|Global\s+|Stock\s+)?Market(?:\s+LLC)?)"
    r"\s*[:\-–]\s*([A-Z]{1,5}(?:\.[A-Z])?)")
SYMBOL_RE = re.compile(
    r"(?:under\s+the\s+(?:trading\s+|ticker\s+)?symbol|trading\s+symbol|"
    r"ticker\s+symbol)[\s:]*[\"'“‘]?([A-Z]{1,5}(?:\.[A-Z])?)")
TRADEMARK_RE = re.compile(
    r"\b((?:[A-Z][\w\-]*)(?:\s+[A-Z][\w\-]*){0,3})\s*(?:[®™℠]|\(R\)|\(TM\)|\(SM\))")
MAX_GAP = 4  # max chars between adjacent name tokens (", ", " & ", ". ")


def path_key(p: str) -> str:
    """Root-invariant text_path key. The processed parquets and the text cache
    may carry DIFFERENT absolute roots for the same file (Mac /path/to/data-root/...
    vs box /data/...; the box has no interim/ tree at all), so matching is on
    the stable suffix from 'interim/' onwards (interim/<ds>/<form>/<cik>/
    <accession>.txt); fallback: last 4 path components."""
    s = str(p).replace("\\", "/")
    i = s.find("interim/")
    if i >= 0:
        return s[i:]
    parts = [x for x in s.split("/") if x]
    return "/".join(parts[-4:])


def norm_token(tok: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", tok.upper())


def name_token_variants(name: str) -> list[list[tuple[str, str]]]:
    """Tokenise a company name; return 1-2 token-sequence variants, each a list
    of (norm, raw). Variant 2 splits hyphen/& compounds (Coca-Cola -> COCA COLA)."""
    raw = [m.group(0) for m in TOKEN_RE.finditer(name)]
    if not raw:
        return []
    v1 = [(norm_token(t), t) for t in raw if norm_token(t)]
    out = [v1] if v1 else []
    if any(("-" in t or "&" in t) for t in raw):
        split_raw = []
        for t in raw:
            split_raw.extend(p for p in re.split(r"[\-&]", t) if p)
        v2 = [(norm_token(t), t) for t in split_raw if norm_token(t)]
        if v2 and v2 != v1:
            out.append(v2)
    return out


def strip_generic(tokens: list[tuple[str, str]]):
    """Split (leading generics, core, trailing generics) on the norm side."""
    i, j = 0, len(tokens)
    while i < j and tokens[i][0] in GENERIC:
        i += 1
    while j > i and tokens[j - 1][0] in GENERIC:
        j -= 1
    return tokens[i:j], tokens[j:]


class RuleTable:
    """Panel-derived deterministic masking rules (names / tickers / CIKs)."""

    def __init__(self, universe: pd.DataFrame, crsp_names: pd.DataFrame,
                 comp: pd.DataFrame):
        self.panel_tickers = set()
        for t in universe["ticker"].astype(str):
            nt = norm_token(t)
            if len(nt) >= 2 and nt not in TICKER_BLOCKLIST:
                self.panel_tickers.add(nt)
        self.panel_ciks = {int(c) for c in universe["cik"].dropna().astype(str)
                           .str.lstrip("0").replace("", "0").astype(int) if int(c) > 0}

        # permno -> cik (via universe) so CRSP names attach to a filer cik
        perm2cik: dict[int, set[int]] = {}
        for _, r in universe.iterrows():
            try:
                perm2cik.setdefault(int(r["permno"]), set()).add(int(str(r["cik"]).lstrip("0")))
            except (TypeError, ValueError):
                continue

        # entries: first_norm -> list of (token_seq(core), trailing_generics, cik_set)
        self.index: dict[str, list[tuple[tuple[str, ...], int, frozenset]]] = {}
        self.names_by_cik: dict[int, set[str]] = {}

        def add_name(name: str, ciks: set[int]):
            if not isinstance(name, str) or not name.strip():
                return
            for cik in ciks:
                self.names_by_cik.setdefault(cik, set()).add(name)
            for variant in name_token_variants(name):
                core, trail = strip_generic(variant)
                if not core:
                    continue
                seq = tuple(n for n, _ in core)
                key = seq[0]
                entry = (seq, len(trail), frozenset(ciks))
                bucket = self.index.setdefault(key, [])
                if entry not in bucket:
                    bucket.append(entry)

        panel_permnos = {int(p) for p in universe["permno"].dropna().astype(int)}
        cn = crsp_names[crsp_names["PERMNO"].isin(panel_permnos)]
        for _, r in cn.drop_duplicates(subset=["PERMNO", "IssuerNm"]).iterrows():
            add_name(str(r["IssuerNm"]), perm2cik.get(int(r["PERMNO"]), set()))
        comp = comp[comp["cik"].notna()].copy()
        comp["cik_int"] = comp["cik"].astype(float).astype(int)
        cc = comp[comp["cik_int"].isin(self.panel_ciks)]
        for _, r in cc.iterrows():
            for col in ("conm", "conml"):
                add_name(str(r.get(col, "") or ""), {int(r["cik_int"])})
        # sort each bucket longest-first (greedy longest match)
        for k in self.index:
            self.index[k].sort(key=lambda e: -len(e[0]))

        # own-firm single-token sets: cik -> {norm tokens}
        self.own_tokens: dict[int, set[str]] = {}
        for key, entries in self.index.items():
            for seq, _tr, ciks in entries:
                for cik in ciks:
                    s = self.own_tokens.setdefault(cik, set())
                    for tok in seq:
                        if (tok not in GENERIC and tok not in COMMON_WORDS
                                and (len(tok) >= 3 or any(ch.isdigit() for ch in tok))):
                            s.add(tok)

    def n_names(self) -> int:
        return sum(len(v) for v in self.index.values())


def rule_spans(text: str, own_cik: int, own_ticker: str, rt: RuleTable):
    """Deterministic rule-table spans on the ORIGINAL text."""
    spans: list[tuple[int, int, str]] = []
    toks = [(m.start(), m.end(), m.group(0)) for m in TOKEN_RE.finditer(text)]
    norms = [norm_token(t) for _, _, t in toks]
    own_tok_set = rt.own_tokens.get(own_cik, set())
    own_tick = norm_token(own_ticker or "")

    for i, (s, e, raw) in enumerate(toks):
        n = norms[i]
        if not n:
            continue
        # --- company-name sequences ---
        for seq, n_trail, ciks in rt.index.get(n, ()):
            j, pos_end = i, e
            ok = True
            for k in range(1, len(seq)):
                j += 1
                if (j >= len(toks) or norms[j] != seq[k]
                        or toks[j][0] - pos_end > MAX_GAP):
                    ok = False
                    break
                pos_end = toks[j][1]
            if not ok:
                continue
            is_own = own_cik in ciks
            # consume optional trailing generic suffix tokens (Inc., Corp, ...)
            jj, suffix_used = j, 0
            while (jj + 1 < len(toks) and norms[jj + 1] in GENERIC
                   and toks[jj + 1][0] - toks[jj][1] <= MAX_GAP):
                jj += 1
                suffix_used += 1
            distinct = sum(1 for t in seq if t not in GENERIC)
            accept = (is_own and distinct >= 1) or distinct >= 2 or (
                distinct == 1 and suffix_used >= 1)
            if distinct == 1 and not raw[0].isupper() and not raw[0].isdigit():
                accept = False  # lone lowercase word never a firm name
            if accept:
                spans.append((s, toks[jj][1] if suffix_used else pos_end, "FIRM"))
                break  # longest-first: take the first (longest) accepted entry
        # --- own-firm single distinctive token ---
        if n in own_tok_set and (raw[0].isupper() or raw[0].isdigit()):
            spans.append((s, e, "FIRM"))
        # --- ALL-CAPS panel ticker ---
        if raw.isupper() and len(n) >= 2 and (
                n == own_tick or n in rt.panel_tickers):
            spans.append((s, e, "TICKER"))

    # --- own ticker as a raw regex pass (case-sensitive, \b boundaries) ---
    # catches compound-token occurrences the token scan glues together:
    # filename headers ("AWK-8K-20160630"), dotted class variants ("DHR.PRA").
    if own_ticker and len(own_ticker) >= 2 and own_ticker.upper() == own_ticker:
        for m in re.finditer(rf"\b{re.escape(own_ticker)}\b", text):
            spans.append((m.start(), m.end(), "TICKER"))
    # --- exchange-context ticker symbols (any length, incl. blocklisted) ---
    for rx in (EXCHANGE_RE, SYMBOL_RE):
        for m in rx.finditer(text):
            spans.append((m.start(1), m.end(1), "TICKER"))
    # --- CIK digit runs / accession numbers / file numbers ---
    for m in DIGIT_RE.finditer(text):
        v = int(m.group(0))
        if YEAR_LO <= v <= YEAR_HI:
            continue
        if v in rt.panel_ciks:
            spans.append((m.start(), m.end(), "CIK"))
    for m in ACCESSION_RE.finditer(text):
        spans.append((m.start(), m.end(), "CIK"))
    for m in FILENUM_RE.finditer(text):
        spans.append((m.start(1), m.end(1), "CIK"))
    # --- product heuristic: Name(R)/(TM) ---
    for m in TRADEMARK_RE.finditer(text):
        spans.append((m.start(1), m.end(1), "PRODUCT"))
    return spans


def chunk_text(text: str, limit: int = CHUNK_CHARS):
    """Yield (offset, chunk) pieces split at whitespace, each <= limit chars."""
    if len(text) <= limit:
        yield 0, text
        return
    pos = 0
    while pos < len(text):
        end = min(pos + limit, len(text))
        if end < len(text):
            cut = text.rfind("\n", pos, end)
            if cut <= pos:
                cut = text.rfind(" ", pos, end)
            if cut > pos:
                end = cut
        yield pos, text[pos:end]
        pos = end


def merge_spans(spans):
    """Merge overlapping spans; on overlap keep the higher-priority label."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))
    out = [list(spans[0])]
    for s, e, lab in spans[1:]:
        cur = out[-1]
        if s < cur[1]:  # overlap -> union
            cur[1] = max(cur[1], e)
            if PRIORITY[lab] < PRIORITY[cur[2]]:
                cur[2] = lab
        else:
            out.append([s, e, lab])
    return [(s, e, lab) for s, e, lab in out]


def apply_spans(text: str, spans) -> str:
    if not spans:
        return text
    parts, pos = [], 0
    for s, e, lab in spans:
        parts.append(text[pos:s])
        parts.append(PLACEHOLDER[lab])
        pos = e
    parts.append(text[pos:])
    return "".join(parts)


def leak_flags(masked: str, own_cik: int, own_ticker: str, rt: RuleTable):
    """G2 quantitative leak check on the MASKED text."""
    tick = norm_token(own_ticker or "")
    leak_tick = bool(tick and len(tick) >= 2
                     and re.search(rf"\b{re.escape(own_ticker)}\b", masked))
    leak_name = False
    own_toks = rt.own_tokens.get(own_cik, set())
    if own_toks:
        for m in TOKEN_RE.finditer(masked):
            raw = m.group(0)
            if (raw[0].isupper() or raw[0].isdigit()) and norm_token(raw) in own_toks:
                leak_name = True
                break
    return leak_tick, leak_name


def mask_docs(nlp, rt: "RuleTable", docs: list[dict], texts: dict[str, str],
              threads: int) -> tuple[list[dict], list[dict]]:
    """NER + rule spans + merge + replace for `docs` (records with text_path/
    ticker/cik_int). Returns (store rows, per-doc stat rows) in `docs` order.
    Deterministic per doc for a fixed model version (chunked NER, offset-kept)."""
    pieces, owners = [], []
    for di, r in enumerate(docs):
        for off, chunk in chunk_text(texts[r["text_path"]] or ""):
            pieces.append(chunk)
            owners.append((di, off))
    print(f"[mask] NER over {len(pieces)} chunks "
          f"({sum(len(p) for p in pieces)/1e6:.1f}M chars), "
          f"n_process={threads}")
    ner_spans: dict[int, list] = {i: [] for i in range(len(docs))}
    t0 = time.time()
    n_proc = threads if len(pieces) > 64 else 1
    for k, doc in enumerate(nlp.pipe(pieces, batch_size=32, n_process=n_proc)):
        di, off = owners[k]
        for ent in doc.ents:
            lab = NER_LABELS.get(ent.label_)
            if lab and not (lab == "FIRM" and norm_phrase(ent.text) in NER_ALLOWLIST):
                ner_spans[di].append((off + ent.start_char, off + ent.end_char, lab))
        if (k + 1) % 2000 == 0:
            rate = (k + 1) / (time.time() - t0)
            print(f"  ner {k+1}/{len(pieces)} chunks, {rate:.1f} chunks/s, "
                  f"ETA {(len(pieces)-k-1)/rate/60:.0f} min")
    print(f"[mask] NER done in {(time.time()-t0)/60:.1f} min")

    rows, stat_rows = [], []
    t0 = time.time()
    for di, r in enumerate(docs):
        tp, text = r["text_path"], texts[r["text_path"]] or ""
        cik, tick = int(r["cik_int"]), str(r["ticker"])
        spans = merge_spans(rule_spans(text, cik, tick, rt) + ner_spans[di])
        masked = apply_spans(text, spans)
        n_by = {lab: 0 for lab in PLACEHOLDER}
        chars = 0
        for s, e, lab in spans:
            n_by[lab] += 1
            chars += e - s
        lt, ln = leak_flags(masked, cik, tick, rt)
        rows.append({"text_path": tp, "text": masked})
        stat_rows.append({
            "text_path": tp, "ticker": tick, "cik": cik, "n_chars": len(text),
            "n_spans": len(spans), "chars_masked": chars,
            "mask_char_frac": chars / max(len(text), 1),
            **{f"n_{lab.lower()}": n_by[lab] for lab in PLACEHOLDER},
            "leak_own_ticker": lt, "leak_own_name_token": ln,
        })
        if (di + 1) % 5000 == 0:
            print(f"  rules {di+1}/{len(docs)} docs "
                  f"({(di+1)/(time.time()-t0):.0f} docs/s)")
    return rows, stat_rows


def aggregate_stats(st: pd.DataFrame, rt: "RuleTable", ner_meta: dict,
                    text_prov: dict, smoke: bool, out_path: Path,
                    panel: str) -> dict:
    return {
        "prereg": "prereg-ea-v1.0 §C-anon"
                  + (" (long_form stretch)" if panel == "lf" else ""),
        "panel": panel, "smoke": smoke, "ner": ner_meta,
        "text_provenance": text_prov,
        "n_docs": int(len(st)),
        "n_name_entries": rt.n_names(),
        "docs_with_any_mask_pct": float(100 * (st.n_spans > 0).mean()),
        "mean_spans_per_doc": float(st.n_spans.mean()),
        "median_spans_per_doc": float(st.n_spans.median()),
        "mean_mask_char_frac": float(st.mask_char_frac.mean()),
        "median_mask_char_frac": float(st.mask_char_frac.median()),
        "mean_chars_per_doc": float(st.n_chars.mean()),
        "median_chars_per_doc": float(st.n_chars.median()),
        "spans_by_type_mean": {lab: float(st[f"n_{lab.lower()}"].mean())
                               for lab in PLACEHOLDER},
        "spans_by_type_total": {lab: int(st[f"n_{lab.lower()}"].sum())
                                for lab in PLACEHOLDER},
        "leak_own_ticker_pct": float(100 * st.leak_own_ticker.mean()),
        "leak_own_name_token_pct": float(100 * st.leak_own_name_token.mean()),
        "store": str(out_path),
    }


def write_audit(audit_path: Path, st: pd.DataFrame, masked_by_tp: dict[str, str],
                orig_by_tp: dict[str, str], agg: dict, ner_meta: dict,
                rt: "RuleTable") -> None:
    """G2 audit sample (seed 2026 over the stats row order), original vs masked."""
    rng = np.random.default_rng(AUDIT_SEED)
    pick = rng.choice(len(st), size=min(AUDIT_N, len(st)), replace=False)
    md = [f"# C-anon G2 audit sample — {len(pick)} docs (seed {AUDIT_SEED})",
          "", f"NER: {ner_meta}. Rule table: {rt.n_names()} names. "
          f"Leak rates: own-ticker {agg['leak_own_ticker_pct']:.2f}%, "
          f"own-name-token {agg['leak_own_name_token_pct']:.2f}%.", ""]
    for i in sorted(pick.tolist()):
        srow = st.iloc[i]
        tp = srow["text_path"]
        orig = (orig_by_tp.get(tp) or "")[:AUDIT_EXCERPT]
        msk = (masked_by_tp.get(tp) or "")[:AUDIT_EXCERPT + 200]
        md += [f"## {srow['ticker']} — {tp}",
               f"spans={srow['n_spans']} frac={srow['mask_char_frac']:.3f} "
               f"leak_ticker={srow['leak_own_ticker']} "
               f"leak_name={srow['leak_own_name_token']}",
               "", "### original", "```", orig, "```",
               "### masked", "```", msk, "```", ""]
    audit_path.write_text("\n".join(md))


# --------------------------------------------------------------------- driver
def load_rule_table() -> RuleTable:
    universe = pd.read_parquet(data_path("processed", "full", "universe.parquet"))
    nz = data_path("raw", "wrds", "crsp_names_csv.zip")
    cz = data_path("raw", "wrds", "comp_company_csv.zip")
    for p in (nz, cz):
        if not Path(p).exists():
            raise SystemExit(f"FATAL: {p} missing — ship raw/wrds/*.zip with the "
                             "data root (CRSP names + Compustat company map).")
    with zipfile.ZipFile(nz) as z:
        with z.open(z.namelist()[0]) as fh:
            crsp = pd.read_csv(io.BytesIO(fh.read()),
                               usecols=["PERMNO", "IssuerNm"], dtype=str)
    crsp["PERMNO"] = crsp["PERMNO"].astype(int)
    with zipfile.ZipFile(cz) as z:
        with z.open(z.namelist()[0]) as fh:
            comp = pd.read_csv(io.BytesIO(fh.read()),
                               usecols=["cik", "conm", "conml"], low_memory=False)
    return RuleTable(universe, crsp, comp)


# panel -> (forms, output-name suffix). "ed" keeps the original suffix-free
# names so the committed ED path/gates are untouched; "lf" is the registered
# long-form stretch (the train.py long_form disclosure = 10-K + 10-Q).
PANELS = {"ed": (("8-K",), ""), "lf": (("10-K", "10-Q"), "_lf")}


def load_doc_meta(limit: int, panel: str = "ed") -> pd.DataFrame:
    forms, _ = PANELS[panel]
    a = pd.read_parquet(data_path("processed", "full", "aligned_filings.parquet"),
                        columns=["form", "text_path", "ticker", "cik"])
    ed = a[a["form"].isin(forms)].drop_duplicates("text_path").reset_index(drop=True)
    ed["cik_int"] = ed["cik"].astype(str).str.lstrip("0").replace("", "0").astype(int)
    if limit:
        rng = np.random.default_rng(AUDIT_SEED)
        ed = ed.iloc[np.sort(rng.choice(len(ed), size=min(limit, len(ed)),
                                        replace=False))].reset_index(drop=True)
    return ed


def stream_texts(text_paths: set[str]) -> tuple[dict[str, str], dict]:
    """Stream the needed rows from the shared text cache (never load it whole).

    The cache parquet is the SELF-CONTAINED primary source (its `text` column
    is the parsed filing text); rows are matched on the root-invariant
    path_key so a cache built under a different data root (the box) still
    serves a panel whose text_path strings carry another root. Only rows
    absent from the cache or with empty cache text fall back to reading the
    .txt file from disk (impossible on the box — counted and reported).

    Returns (dict ORIGINAL panel text_path -> text, provenance counters).
    """
    from sp500vol.utils.paths import resolve_data_path

    cache = data_path("processed", "_text_cache", "filing_texts.parquet")
    want: dict[str, str] = {}
    for p in text_paths:
        k = path_key(p)
        if k in want:
            raise SystemExit(f"FATAL: panel text_paths collide on path_key "
                             f"{k!r} ({want[k]!r} vs {p!r})")
        want[k] = p
    got: dict[str, str] = {}
    n_dup_cache = 0
    pf = pq.ParquetFile(cache)
    for batch in pf.iter_batches(batch_size=2048, columns=["text_path", "text"]):
        tp = batch.column("text_path").to_pylist()
        tx = None
        for i, cp in enumerate(tp):
            k = path_key(cp)
            orig = want.get(k)
            if orig is None:
                continue
            if orig in got:
                n_dup_cache += 1  # deterministic: first occurrence wins
                continue
            if tx is None:
                tx = batch.column("text").to_pylist()
            got[orig] = tx[i]
        if len(got) == len(want):
            break
    # disk fallback ONLY for cache-missing or empty-text rows
    n_empty_cache, n_disk, still_missing = 0, 0, []
    for orig in text_paths:
        t = got.get(orig)
        if t:  # non-empty cache hit — the normal path
            continue
        if orig in got:
            n_empty_cache += 1
        try:
            got[orig] = resolve_data_path(orig).read_text(
                encoding="utf-8", errors="replace")
            n_disk += 1
        except OSError:
            if orig not in got:
                still_missing.append(orig)
    prov = {"n_from_cache": len(text_paths) - n_disk - len(still_missing),
            "n_disk_fallback": n_disk, "n_empty_cache_text": n_empty_cache,
            "n_dup_cache_keys": n_dup_cache, "n_missing": len(still_missing)}
    if still_missing:
        raise SystemExit(
            f"FATAL: {len(still_missing)} text_paths absent from the text "
            f"cache AND unreadable from disk (first: {still_missing[0]}) — "
            f"refuse to build a partial store. provenance={prov}")
    return got, prov


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", choices=sorted(PANELS), default="ed",
                    help="ed = event-driven 8-K panel (registered primary, "
                         "default); lf = long-form 10-K/10-Q panel (registered "
                         "stretch; outputs get _lf names)")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: mask only N (seed-2026 sampled) docs; outputs "
                         "get a _smoke suffix and the store is NOT the real one")
    ap.add_argument("--threads", type=int, default=4,
                    help="spaCy n_process (keep <=4 locally; raise on the box)")
    ap.add_argument("--out", default=None,
                    help="masked store parquet (default <data_root>/processed/"
                         "_text_cache/filing_texts_anonmask_<panel>.parquet)")
    ap.add_argument("--resume", action="store_true",
                    help="single-shot mode: skip text_paths already present in "
                         "--out (batch mode resumes by complete parts instead)")
    ap.add_argument("--batch-docs", type=int, default=0,
                    help="mask N docs per atomic part (crash costs <=1 batch; "
                         "recommended 2000 for --panel lf). 0 = original "
                         "single-shot flow.")
    args = ap.parse_args()
    smoke = bool(args.limit)
    _, sfx = PANELS[args.panel]

    try:
        import spacy
    except ImportError:
        raise SystemExit("FATAL: spaCy not importable — prereg requires "
                         "en_core_web_lg; install the shipped wheels first.")
    try:
        nlp = spacy.load("en_core_web_lg",
                         disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    except OSError:
        raise SystemExit("FATAL: en_core_web_lg model not installed — install "
                         "the staged en_core_web_lg-3.8.0-py3-none-any.whl.")
    ner_meta = {"spacy": spacy.__version__,
                "model": f"en_core_web_lg-{nlp.meta['version']}"}
    print(f"[mask] {ner_meta}")

    default_out = data_path("processed", "_text_cache",
                            f"filing_texts_anonmask_{args.panel}.parquet")
    if smoke and args.out is None:
        out_path = Path(str(default_out).replace(".parquet", "_smoke.parquet"))
    else:
        out_path = Path(args.out) if args.out else Path(default_out)
    tag = "_smoke" if smoke else ""
    anon_dir = REPO / "results" / "anon"
    anon_dir.mkdir(parents=True, exist_ok=True)
    stats_path = anon_dir / f"mask_stats{sfx}{tag}.json"
    audit_path = anon_dir / f"mask_audit_sample{sfx}{tag}.md"

    rt = load_rule_table()
    print(f"[mask] rule table: {rt.n_names()} name entries, "
          f"{len(rt.panel_tickers)} maskable panel tickers, "
          f"{len(rt.panel_ciks)} panel CIKs")

    meta = load_doc_meta(args.limit, args.panel)
    forms, _ = PANELS[args.panel]
    print(f"[mask] panel={args.panel} forms={forms}: {len(meta)} unique docs")

    if args.batch_docs:
        run_batched(args, meta, rt, nlp, ner_meta, out_path, stats_path,
                    audit_path, smoke)
        return

    # ---------------- original single-shot flow (ED default) ----------------
    done: set[str] = set()
    if args.resume and out_path.exists():
        done = set(pd.read_parquet(out_path, columns=["text_path"])["text_path"])
        print(f"[mask] resume: {len(done)} docs already masked")
    todo = meta[~meta["text_path"].isin(done)].reset_index(drop=True)
    print(f"[mask] docs to mask: {len(todo)} (of {len(meta)} unique docs)")
    if not len(todo):
        print("[mask] nothing to do")
        return

    texts, text_prov = stream_texts(set(todo["text_path"]))
    print(f"[mask] text provenance: {text_prov}")

    docs = todo.to_dict("records")
    rows, stat_rows = mask_docs(nlp, rt, docs, texts, args.threads)

    out_df = pd.DataFrame(rows)
    st = pd.DataFrame(stat_rows)
    if done:  # resume: append previously masked docs unchanged
        prev = pd.read_parquet(out_path)
        out_df = pd.concat([prev, out_df], ignore_index=True)
        prev_st_path = Path(str(out_path).replace(".parquet", ".stats.parquet"))
        if prev_st_path.exists():
            st = pd.concat([pd.read_parquet(prev_st_path), st], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(out_df, preserve_index=False), tmp)
    tmp.replace(out_path)
    st.to_parquet(str(out_path).replace(".parquet", ".stats.parquet"), index=False)

    agg = aggregate_stats(st, rt, ner_meta, text_prov, smoke, out_path,
                          args.panel)
    stats_path.write_text(json.dumps(agg, indent=2))

    # ---- G2: 100-doc audit sample (seed 2026), original vs masked ----
    masked_by_tp = dict(zip(out_df["text_path"], out_df["text"]))
    orig_by_tp = texts if not done else {
        **{tp: "" for tp in masked_by_tp}, **texts}
    if done:  # resume path: re-fetch originals for the audit picks
        rng = np.random.default_rng(AUDIT_SEED)
        pick = rng.choice(len(st), size=min(AUDIT_N, len(st)), replace=False)
        need = set(st.iloc[sorted(pick.tolist())]["text_path"]) - set(texts)
        if need:
            fetched, _ = stream_texts(need)
            orig_by_tp.update(fetched)
    write_audit(audit_path, st, masked_by_tp, orig_by_tp, agg, ner_meta, rt)

    print(f"[mask] wrote store   -> {out_path} ({len(out_df)} docs)")
    print(f"[mask] wrote stats   -> {stats_path}")
    print(f"[mask] wrote audit   -> {audit_path}")
    print(json.dumps(agg, indent=2))


def run_batched(args, meta, rt, nlp, ner_meta, out_path: Path,
                stats_path: Path, audit_path: Path, smoke: bool) -> None:
    """Part-per-batch build: each batch is written atomically under
    <store>.parts/; re-invocation skips complete parts (crash costs <=1 batch);
    the final store/stats/audit are assembled once every part exists."""
    if out_path.exists():
        raise SystemExit(f"[mask] batch mode: {out_path} already exists — the "
                         "build is complete (delete the store to rebuild).")
    parts_dir = Path(str(out_path) + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    n_batches = (len(meta) + args.batch_docs - 1) // args.batch_docs
    provs = []
    for bi in range(n_batches):
        part = parts_dir / f"part-{bi:04d}.parquet"
        stat_part = parts_dir / f"part-{bi:04d}.stats.parquet"
        if part.exists() and stat_part.exists():
            print(f"[mask] batch {bi+1}/{n_batches}: part exists — skipped")
            continue
        chunk = meta.iloc[bi * args.batch_docs:(bi + 1) * args.batch_docs] \
            .reset_index(drop=True)
        t0 = time.time()
        texts, prov = stream_texts(set(chunk["text_path"]))
        provs.append(prov)
        rows, stat_rows = mask_docs(nlp, rt, chunk.to_dict("records"), texts,
                                    args.threads)
        tmp = part.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows),
                                            preserve_index=False), tmp)
        tmp.replace(part)
        pd.DataFrame(stat_rows).to_parquet(stat_part, index=False)
        n_ch = sum(s["n_chars"] for s in stat_rows)
        dt = time.time() - t0
        print(f"[mask] batch {bi+1}/{n_batches}: {len(rows)} docs "
              f"({n_ch/1e6:.1f}M chars) in {dt/60:.1f} min "
              f"({n_ch/max(dt,1e-9)/1e3:.0f}k chars/s) -> {part.name}")

    # ---- assemble ----
    st = pd.concat([pd.read_parquet(parts_dir / f"part-{bi:04d}.stats.parquet")
                    for bi in range(n_batches)], ignore_index=True)
    out_df = pd.concat([pd.read_parquet(parts_dir / f"part-{bi:04d}.parquet")
                        for bi in range(n_batches)], ignore_index=True)
    if list(out_df["text_path"]) != list(meta["text_path"]):
        raise SystemExit("[mask] FATAL: assembled parts do not cover the panel "
                         "in order — remove the parts dir and rebuild.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(out_df, preserve_index=False), tmp)
    tmp.replace(out_path)
    st.to_parquet(str(out_path).replace(".parquet", ".stats.parquet"), index=False)

    text_prov = {k: sum(p.get(k, 0) for p in provs)
                 for k in ("n_from_cache", "n_disk_fallback",
                           "n_empty_cache_text", "n_dup_cache_keys", "n_missing")}
    text_prov["note"] = f"summed over {len(provs)} freshly-built batches " \
                        f"({n_batches} total; skipped parts not re-counted)"
    agg = aggregate_stats(st, rt, ner_meta, text_prov, smoke, out_path,
                          args.panel)
    stats_path.write_text(json.dumps(agg, indent=2))

    # audit: originals re-fetched only for the sampled docs
    rng = np.random.default_rng(AUDIT_SEED)
    pick = rng.choice(len(st), size=min(AUDIT_N, len(st)), replace=False)
    audit_tps = set(st.iloc[sorted(pick.tolist())]["text_path"])
    orig_by_tp, _ = stream_texts(audit_tps)
    masked_by_tp = {tp: tx for tp, tx in zip(out_df["text_path"], out_df["text"])
                    if tp in audit_tps}
    write_audit(audit_path, st, masked_by_tp, orig_by_tp, agg, ner_meta, rt)

    for bi in range(n_batches):  # store verified in order — parts now redundant
        (parts_dir / f"part-{bi:04d}.parquet").unlink(missing_ok=True)
        (parts_dir / f"part-{bi:04d}.stats.parquet").unlink(missing_ok=True)
    try:
        parts_dir.rmdir()
    except OSError:
        pass
    print(f"[mask] wrote store   -> {out_path} ({len(out_df)} docs)")
    print(f"[mask] wrote stats   -> {stats_path}")
    print(f"[mask] wrote audit   -> {audit_path}")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
