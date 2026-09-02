"""Prereg M2 — the elicitation-fairness test (configs/prereg_elicitation_fairness.md,
tag prereg-ef-v1.0). Give every dead family a FAIR prompt-adaptation shot; select on val
health ONLY; report all four registered branches.

Internal adversarial dry-run (the confound this experiment tests): "If 4 of 5
instruction-tuned families collapse under this elicitation, the most economical explanation
is that the prompt protocol is fragile and was developed on Qwen3-32B -- the screen may be

MECHANISM = the committed machinery, IMPORTED, never re-typed:
  * scripts/analysis/crossfamily_gemma27.py (prereg-rfa v1.3, the committed pilot-gate
    template) supplies PILOT_N, the canonical sort, the health formula
    (standalone_stats / pilot_health), the health gate text, the float comparator
    (close), the Gemma system-fold (fold_system / FoldChatGenerator) and resolve_model.
  * scripts/experiments/e1_llm_forecast/prompt.py supplies the committed C6 prompt.
    V0 IS that prompt, obtained BY IMPORT (build_messages(row, text, "c6_text")) — it is
    never re-typed here, so an edit to prompt.py changes V0's G-E3 hash and trips the gate.
  * scripts/experiments/e1_llm_forecast/run_inference.py supplies the sampling stack and
    the committed ONE-re-ask-on-parse-failure retry (_flush), used identically by V0/V1/V2.

THE THREE FROZEN VARIANTS (prereg SS Design; text frozen in this tag, hash-pinned by G-E3):
  V0  the committed C6 prompt, verbatim (the baseline that produced the committed
      instrument-dead readings).
  V1  few-shot: V0 + two validation examples — deterministically the FIRST TWO val
      documents by the panel's canonical sort, with their true RV labels. The rendered
      example text is byte-identical for every family.
  V2  format-hardened: V0 + an explicit numeric-range hint ("annualised RV, typically
      0.05--1.50") + a restatement of the committed JSON schema + one re-ask on parse
      failure (the committed retry mechanism, prereg: "same mechanism as committed retry").
  Families differ only by the chat-template system-fold (Gemma has no system role) —
  the delta already disclosed under prereg-rfa v1.3.

SELECTION (val only, never test): each family x variant runs ONE pass over the SAME 2,000
val documents (the v1.3 pilot set); the committed health formula is applied verbatim
(healthy <=> max variance-unit QLIKE < 4 AND max modal share of round(pred,2) < 60%).
Each family keeps the variant with the LOWEST val modal share (ties: V0 -> V1 -> V2). If
that variant is healthy the family is released to the full pass; if all three variants are
unhealthy the family is registered elicitation-robust instrument-dead.

============================ HARDWARE AMENDMENT (prereg v1.1, PENDING) ================
The prereg text was written for a 2xA100-40GB box and therefore says TP=2. The run box is
now a SINGLE A100-80GB, so the pilot runs at TP=1 (both 24B and 27B bf16 fit in 80GB).
  * TP is a CLI parameter (--tp, default 1) and the EFFECTIVE TP is recorded in every
    output (shards, pilot json, md, logs).
  * G-E1 is therefore NOT run as the registered bit-exact reproduction gate. It cannot
    be: the committed readings were produced at TP=2 and TP changes the reduction order,
    so bit-exactness is unavailable by construction. G-E1 is instead run as a
    TP-INVARIANCE DIAGNOSTIC (--tp != 2):
      - V0@TP1's val health columns are reported BESIDE the committed TP=2 readings, with
        the deltas printed and recorded;
      - the script CONTINUES (no abort);
      - it LOUDLY flags any family whose healthy/dead VERDICT flips between the
        committed TP=2 reading and V0@TP1, because that would mean TP confounds the
        committed instrument-dead judgement.
    If --tp 2 is passed the registered bit-exact gate is available again (--strict-ge1).
  The amendment is the author's to file; this script only makes TP explicit and logged.

GATES
  G-E1  TP-invariance diagnostic (see above; bit-exact only under --tp 2 --strict-ge1).
  G-E2  val-only: reading ANY test row in pilot mode is a fatal error (assertion + log).
  G-E3  SHA-256 of every rendered variant template is recorded in each shard and in the
        pilot json, and re-checked across every (family, variant) subprocess.
  G-E4  the full-run health formula is re-applied to the 3-seed ensemble (--full).

============================ THE PILOT SIDECAR (why it exists) ========================
The run box carries ONLY the pilot slice:
    $SP500VOL_DATA_ROOT/processed/full/aligned_ed_val.parquet      (42,724 val 8-K rows)
    $SP500VOL_DATA_ROOT/processed/_text_cache/filing_texts_ed_val.parquet (14,266 docs)
Two facts make those two files insufficient on their own, and neither is negotiable:
  1. The committed C6 prompt renders "- Filing date: {filing_date}". filing_date comes
     from the EDGAR metadata JSON, NOT from filing_time_utc: across the committed
     manifest, filing_date - utc_date takes values -633..+5 days (2,473/51,229 rows
     disagree). It is NOT reconstructible from the box slice. Guessing it would change
     V0's bytes and silently void the whole comparison.
  2. The box's aligned_ed_val.parquet holds 14,266 ED val docs; the committed manifest's
     ED val panel holds 14,213 (A2 drops filings without sufficient HAR history). A naive
     head(2,000) on the box would therefore select a DIFFERENT set from the registered
     v1.3 pilot -- violating "the same set as the v1.3 pilot".
So `--emit-sidecar` runs LOCALLY (where the committed manifest + A2 live) and freezes the
registered pilot into one small, auditable, hash-pinned JSON: the exact 2,000 v1.3 pilot
docs in canonical order, their prompt fields (incl. filing_date), their A2 val labels,
feature_rv_22d for the committed parse-failure fill, and the committed TP=2 val-side
anchors. It contains NO test row and NO new statistic — every field is copied from a
committed artifact. Ship it with the scripts; the pilot then needs the sidecar + the two
box files and nothing else.

OUTPUTS (write-once, single shot -- prereg SS Artifacts and boundaries):
  results/e1_llm_forecast/ef_pilot_sidecar.json    (--emit-sidecar, local)
  results/e1_llm_forecast/ef_pilot_shards/*.json   (--pilot --family F --variant V)
  results/tables/elicitation_fairness_pilot.json   (--pilot --assemble)
  results/tables/elicitation_fairness.{csv,md}     (--full)

BOX (1xA100-80GB, from /root/gpu-data/repo, system python3, no venv):
  python3 scripts/analysis/elicitation_fairness.py --pilot --family mistral24 --variant V0
  ...                                              (one subprocess per family x variant)
  python3 scripts/analysis/elicitation_fairness.py --pilot --assemble
  bash scripts/box/_ef_pilot.sh          # the whole chain -> DONE_ef_pilot / FAIL_ef_pilot
LOCAL (no GPU):
  .venv/bin/python scripts/analysis/elicitation_fairness.py --selftest
  .venv/bin/python scripts/analysis/elicitation_fairness.py --emit-sidecar
  .venv/bin/python scripts/analysis/elicitation_fairness.py --show-variants
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
if os.path.isdir("/root/gpu-data"):  # box defaults (launch.sh convention)
    os.environ.setdefault("SP500VOL_DATA_ROOT", "/root/gpu-data/sp500vol-data")
    os.environ.setdefault("HF_HOME", "/root/gpu-data/hf")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
sys.path.insert(0, str(_REPO / "scripts" / "experiments" / "e1_llm_forecast"))
# forecast_combination (imported transitively by the committed template) does
# `sys.path.insert(0, "src")` — a RELATIVE path, so the committed stack only imports when
# the cwd happens to be the repo root. Insert the ABSOLUTE path first so this script
# imports from any cwd; `src/sp500vol` must therefore be staged on the box, not just
# scripts/.
sys.path.insert(0, str(_REPO / "src"))

import crossfamily_gemma27 as cf  # noqa: E402  the committed v1.3 template (machinery)
import prompt as prompt_mod  # noqa: E402  the committed C6 prompt — V0 comes from here
import run_inference as ri  # noqa: E402
import postprocess as pp  # noqa: E402

# ---- everything scientific is the committed constant, IMPORTED (single source of truth)
HORIZONS = cf.HORIZONS
DISC = cf.DISC
PILOT_N = cf.PILOT_N                      # 2,000 val docs (the v1.3 pilot set)
PILOT_SEED = cf.PILOT_SEED
HEALTH_QLIKE_MAX = cf.HEALTH_QLIKE_MAX    # < 4
HEALTH_MODE_MAX = cf.HEALTH_MODE_MAX      # < 60%
HEALTH_GATE_TEXT = cf.HEALTH_GATE_TEXT
TP_COMMITTED = cf.TP                      # 2 — the TP the committed readings were made at
MAX_MODEL_LEN = cf.MAX_MODEL_LEN
MAX_TOKENS = cf.MAX_TOKENS
CHECKPOINT_EVERY = cf.CHECKPOINT_EVERY
RTOL = cf.RTOL                            # 1e-12
C6_VARIANT = cf.VARIANT                   # "c6_text" — the committed C6 prompt variant
CANONICAL_SORT = ("filing_time_utc", "ticker", "accession")

VARIANTS = ("V0", "V1", "V2")
PREREG = "configs/prereg_elicitation_fairness.md (tag prereg-ef-v1.0)"
PREREG_AMEND = ("prereg-ef v1.1 amendment PENDING: single A100-80GB -> TP=1 (the v1.0 "
                "text says TP=2, written for 2xA100-40GB); G-E1 runs as a TP-invariance "
                "diagnostic, not a bit-exact gate")

# health columns carried by the committed pilot json / compared by the TP diagnostic
HEALTH_COLS = ("qlike_vol", "qlike_var", "r2", "pred_sd", "n_unique_2dp",
               "mode_val_2dp", "mode_share_pct", "n")

# ---------------------------------------------------------------- the two arms
# prereg SS Design: both bf16 (same precision as the primary -- removes llama70's int4
# confound), both 24-27B (matched class vs 32B), both already in the box HF cache.
FAMILIES = {
    "mistral24": {
        "model_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "desc": "Mistral-Small-24B-Instruct-2501 (bf16)",
        # launch.sh (row16) override: the snapshot minus tekken.json/params.json so
        # transformers resolves the FAST tokenizer (the mistral-common backend lacks
        # .is_fast and crashes vLLM's check). The committed run's config.json records
        # exactly this path as `llm`, so "same code path, same weights" wants it here too.
        "path_override": "/root/gpu-data/hf/mistral24_hfview",
        "committed_pilot_json": None,
        "committed_run": "C6_llmtext_mistral24_full_event_driven_seed2026",
        "committed_table": "results/tables/crossfamily_mistral24.csv",
        "committed_table_family": "mistral24_bf16",
    },
    "gemma27": {
        "model_id": "unsloth/gemma-3-27b-it",
        "desc": "Gemma-3-27B-it (bf16)",
        "path_override": None,
        "committed_pilot_json": "results/tables/crossfamily_gemma27_pilot.json",
        "committed_run": "C6_llmtext_gemma27_full_event_driven_seed2026",
        "committed_table": "results/tables/crossfamily_gemma27.csv",
        "committed_table_family": "gemma27_bf16",
    },
}

SIDECAR = "results/e1_llm_forecast/ef_pilot_sidecar.json"
SHARD_DIR = "results/e1_llm_forecast/ef_pilot_shards"
PILOT_JSON = "results/tables/elicitation_fairness_pilot.json"
OUT_CSV = "results/tables/elicitation_fairness.csv"
OUT_MD = "results/tables/elicitation_fairness.md"

MANIFEST = cf.MANIFEST                                   # local only (133MB, untracked)
A2_RUN = f"results/runs/A2_har_rv_full_{DISC}_seed2026/predictions.parquet"


# =============================================================== frozen variant text
# FROZEN AT tag prereg-ef-v1.0. Not one byte may change for any run result (prereg:
# "V1/V2 text is fixed in this tag; not modifiable for any run outcome"). G-E3 hashes every rendered
# template; V0's share of the hash comes from the IMPORTED prompt.py text, so editing
# the committed prompt also trips the gate.

# V1 few-shot: the exemplar excerpt is produced by the committed build_excerpt and then
# head-capped at FEWSHOT_EXCERPT_CHARS so the few-shot block cannot crowd the target
# filing out of the 8,192-token window. Frozen here; part of the V1 hash.
FEWSHOT_EXCERPT_CHARS = 3000

V1_HEADER = (
    "Two worked examples from the validation period are given first. Each shows a "
    "company SEC filing and the CORRECT annualized realised-volatility answer for that "
    "filing. Study how the filing content maps to the three numbers, then answer the "
    "new filing in exactly the same JSON format."
)

V1_BRIDGE = (
    "[END OF EXAMPLES]\n\n"
    "Now forecast the following filing, in exactly the same JSON format as the examples "
    "above."
)

# V2 format-hardening. The numeric-range string is the prereg's, verbatim
# ("annualised RV, typically 0.05--1.50"); the schema is the COMMITTED prompt.JSON_SCHEMA,
# rendered from the import (never re-typed), so a schema edit trips the V2 hash too.
V2_RANGE_HINT = (
    "Numeric range: the three values are annualised RV, typically 0.05--1.50. A value "
    "outside that range is almost always a formatting mistake (for example writing 25 "
    "or \"25%\" instead of 0.25)."
)
V2_SCHEMA_LEAD = "Required JSON schema (restated):"
V2_CLOSER = (
    "Respond with ONLY a single JSON object matching this schema - no prose, no "
    "markdown, no code fences. A well-formed answer looks exactly like this: "
    '{"vol_5d": 0.25, "vol_10d": 0.27, "vol_20d": 0.30}'
)

# canonical probe used ONLY to render a hashable template (never scored, never a filing)
CANON_ROW = {"form": "8-K", "item_subtype": "2.02", "filing_date": "2024-01-05",
             "ticker": "ZZZ", "sections_json": None}
CANON_TEXT = "<<canonical G-E3 probe text; not a real filing>>"

VARIANT_DOC = {
    "V0": "committed C6 prompt, verbatim (imported from prompt.build_messages, c6_text)",
    "V1": (f"V0 + two few-shot validation examples (first two val docs by canonical "
           f"sort, with true RV labels; exemplar excerpt head-capped at "
           f"{FEWSHOT_EXCERPT_CHARS} chars)"),
    "V2": ("V0 + numeric-range hint (\"annualised RV, typically 0.05--1.50\") + restated "
           "committed JSON schema + one re-ask on parse failure (committed retry)"),
}

BRANCH_ZH = {
    "a": ("**(a) adapted-healthy and Holm-robust replication** (>=2/3 horizons, vs firm-identity, clustered "
          "DM<0 and Holm(3)<.05) -> **the confound is confirmed**."),
    "b": ("**(b) adapted-healthy, directional replication** (3/3 DM<0, Holm<2) -> tabled at the same grade as llama70; "
          "the confound partly holds (health can be restored by prompting)."),
    "c": ("**(c) adapted-healthy but no replication** -> **the residual is downgraded to Qwen-conditional**; the confound holds and is "
          "unfavourable to this paper -- reported truthfully."),
    "d": ("**(d) all three variants unhealthy (both families)** -> **the confound is falsified**: fair prompt adaptation cannot "
          "revive them, so the health screen is not selecting elicitation-protocol fit but measuring a capability floor."),
    "mixed": "mixed (one family (a)/(b)/(c), the other (d)) -> per-family truthful reporting, wording takes the **more conservative** branch.",
}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _abort(gate, lines):
    print(f"\n{'=' * 78}\n{gate} FAIL — {PREREG}\n{'=' * 78}")
    for ln in lines:
        print(ln)
    print("=" * 78)
    sys.exit(1)


def _banner(title, lines, char="!"):
    print(f"\n{char * 78}\n{title}\n{char * 78}")
    for ln in lines:
        print(ln)
    print(char * 78 + "\n")


def assert_writeonce(*paths):
    hit = [p for p in paths if Path(p).exists()]
    if hit:
        print(f"WRITE-ONCE guard (prereg-ef v1.0 'single-shot guard'): output(s) already exist -- "
              f"refusing to overwrite: {hit}\n"
              "If a rerun is genuinely intended, inspect and move the existing file(s) "
              "manually first; this script never overwrites its registered outputs.")
        sys.exit(3)


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(_REPO),
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def ef_tag(fam, variant):
    return f"ef_{fam}_{variant.lower()}"


def pilot_rawdir(fam, variant):
    return Path(f"results/e1_llm_forecast/raw_{ef_tag(fam, variant)}_pilot")


def shard_of(fam, variant):
    return Path(SHARD_DIR) / f"{fam}_{variant}.json"


def doc_key(text_path):
    """Join key that survives a data-root rewrite. text_path is an ABSOLUTE path recorded
    at ingest time (/path/to/data-root/...); the box's copies may carry a different prefix, so
    every join in this script keys on the basename. Proven unique across the committed
    manifest (51,229 rows, 0 duplicate basenames) and equal to '<accession>.txt'."""
    return Path(str(text_path)).name


# ============================================================ data resolution
def data_root():
    return Path(os.environ.get("SP500VOL_DATA_ROOT", "/path/to/data-root/sp500vol-data"))


def resolve_text_cache(override=None):
    """The ED-val text cache on the box; the full cache locally."""
    if override:
        return Path(override)
    root = data_root()
    for name in ("filing_texts_ed_val.parquet", "filing_texts.parquet"):
        p = root / "processed" / "_text_cache" / name
        if p.exists():
            return p
    return root / "processed" / "_text_cache" / "filing_texts_ed_val.parquet"


def resolve_aligned_val(override=None):
    if override:
        return Path(override)
    return data_root() / "processed" / "full" / "aligned_ed_val.parquet"


def stream_texts_by_key(keys, cache_path):
    """The committed ri.stream_texts pattern (stream the cache, never load it whole),
    keyed on the basename instead of the raw absolute text_path so that a data-root
    rewrite between the ingest box and the run box cannot silently empty the join."""
    keys = set(keys)
    got = {}
    if not Path(cache_path).exists():
        _abort("PILOT DATA", [
            f"text cache not found: {cache_path}",
            "Expected $SP500VOL_DATA_ROOT/processed/_text_cache/filing_texts_ed_val.parquet"
            " on the box (or filing_texts.parquet locally).",
            f"SP500VOL_DATA_ROOT={data_root()}"])
    pf = pq.ParquetFile(str(cache_path))
    for batch in pf.iter_batches(batch_size=2048, columns=["text_path", "text"]):
        tp = batch.column("text_path").to_pylist()
        hit = [i for i, p in enumerate(tp) if doc_key(p) in keys]
        if hit:
            tx = batch.column("text").take(hit).to_pylist()
            for i, j in enumerate(hit):
                got[doc_key(tp[j])] = tx[i]
        if len(got) == len(keys):
            break
    missing = keys - got.keys()
    if missing:
        print(f"[warn] {len(missing)} pilot text_paths not found in {cache_path.name}")
    return got


# ============================================================ the pilot sidecar
def _health_from_long(df, pred_col="prediction_realised_vol"):
    """The committed health formula (cf.standalone_stats + cf.pilot_health), verbatim."""
    per_h = {}
    for h in HORIZONS:
        d = df[df.horizon_days == h]
        st = cf.standalone_stats(d.label_realised_vol.to_numpy(), d[pred_col].to_numpy())
        st["n"] = int(len(d))
        per_h[str(h)] = st
    mq = max(v["qlike_var"] for v in per_h.values())
    mm = max(v["mode_share_pct"] for v in per_h.values())
    return {"healthy": bool(mq < HEALTH_QLIKE_MAX and mm < HEALTH_MODE_MAX),
            "per_h": per_h, "max_qlike_var": float(mq), "max_mode_share_pct": float(mm)}


def _committed_val_anchor(fam, panel_keys):
    """The committed TP=2 VAL-side reading on the same 2,000 pilot docs.

    tier A: the committed val-side pilot json (gemma27) — the same docs, same V0, TP=2.
    tier B: the val block of the committed seed-2026 run dir restricted to the pilot docs
            (mistral24 — its committed table is TEST-side, so the run dir's val rows are
            the comparable val-side object).
    Returns None when neither is on disk."""
    cfg = FAMILIES[fam]
    pj = cfg.get("committed_pilot_json")
    if pj and Path(pj).exists():
        d = json.loads(Path(pj).read_text())
        return {"tier": "A", "source": pj, "tp": d.get("sampling", {}).get("tp"),
                "healthy": d["healthy"], "per_h": d["per_h"],
                "max_qlike_var": d["max_qlike_var"],
                "max_mode_share_pct": d["max_mode_share_pct"],
                "comparable": "strict",
                "note": (f"committed val-side pilot json ({d.get('n_docs')} val docs, "
                         f"V0/c6_text, TP={d.get('sampling', {}).get('tp')})")}
    run = cfg.get("committed_run")
    p = Path(f"results/runs/{run}/predictions.parquet") if run else None
    if p is not None and p.exists():
        pred = pd.read_parquet(p)
        d = pred[(pred["split"] == "val")
                 & (pred["text_path"].map(doc_key).isin(panel_keys))]
        h = _health_from_long(d)
        h.update({"tier": "B", "source": str(p), "tp": TP_COMMITTED,
                  "comparable": "strict",
                  "note": ("no committed val-side pilot json for this family; the "
                           "comparable committed val-side reading is the VAL block of "
                           "the committed seed-2026 run dir (identical V0/c6_text "
                           "protocol, TP=2), restricted to the same 2,000 pilot docs")})
        return h
    return None


def _committed_test_reading(fam):
    """The committed TEST-side table reading (context only — NOT the val-side object the
    TP diagnostic compares to; a val-vs-test delta is a split delta, not a TP delta)."""
    cfg = FAMILIES[fam]
    p = Path(cfg["committed_table"])
    if not p.exists():
        return None
    df = pd.read_csv(p)
    d = df[(df.family == cfg["committed_table_family"]) & (df.disc == DISC)]
    if not len(d) or d["mode_share_pct"].isna().all():
        return None
    per_h = {str(int(r.h)): {"qlike_var": float(r.qlike_var),
                             "mode_share_pct": float(r.mode_share_pct),
                             "mode_val_2dp": float(r.mode_val_2dp),
                             "n": int(r.n_test)} for r in d.itertuples()}
    mq = max(v["qlike_var"] for v in per_h.values())
    mm = max(v["mode_share_pct"] for v in per_h.values())
    return {"tier": "T", "source": str(p), "tp": TP_COMMITTED, "split": "test",
            "comparable": "cross-split",
            "healthy": bool(mq < HEALTH_QLIKE_MAX and mm < HEALTH_MODE_MAX),
            "per_h": per_h, "max_qlike_var": float(mq), "max_mode_share_pct": float(mm),
            "note": ("committed TEST-side full-run table (the published "
                     "instrument-dead verdict). Reported for context: the pilot is "
                     "val-side, so a delta against this object mixes split with TP.")}


def emit_sidecar(args):
    """LOCAL: freeze the registered pilot into one small hash-pinned JSON. Reads only
    committed artifacts (the manifest + A2 + the committed readings). No test row."""
    out = Path(SIDECAR)
    assert_writeonce(out)
    if not Path(MANIFEST).exists():
        _abort("SIDECAR", [
            f"the committed manifest is not on this machine: {MANIFEST}",
            "--emit-sidecar must run where the committed manifest + A2 live (locally),",
            "not on the run box. Ship the emitted JSON to the box with the scripts."])
    if not Path(A2_RUN).exists():
        _abort("SIDECAR", [f"the committed A2 run is not on this machine: {A2_RUN}"])

    panel = cf.pilot_panel(PILOT_N)          # the registered v1.3 selection, imported
    if not (panel["split"] == "val").all():
        _abort("G-E2 (val-only)", ["cf.pilot_panel() returned a non-val row"])
    a2 = pd.read_parquet(A2_RUN)
    a2 = a2[a2["split"] == "val"]
    keys = set(panel.text_path.map(doc_key))
    a2 = a2[a2["text_path"].map(doc_key).isin(keys)]

    docs = []
    for rank, r in enumerate(panel.to_dict("records")):
        k = doc_key(r["text_path"])
        d = a2[a2["text_path"].map(doc_key) == k]
        labels = {str(int(x.horizon_days)): float(x.label_realised_vol)
                  for x in d.itertuples()}
        if not labels:
            _abort("SIDECAR", [f"pilot doc {k} has no A2 val label rows"])
        rv22 = float(d.feature_rv_22d.iloc[0])
        fd = r["filing_date"]
        if not isinstance(fd, str):
            _abort("SIDECAR", [
                f"filing_date for {k} is {type(fd).__name__}, not str: {fd!r}",
                "The committed prompt renders it with an f-string; a non-str dtype would "
                "change V0's bytes (e.g. '2020-01-02 00:00:00'). Refusing to freeze."])
        docs.append({
            "rank": rank, "doc_key": k, "text_path": r["text_path"],
            "ticker": r["ticker"], "accession": r["accession"], "form": r["form"],
            "item_subtype": r["item_subtype"], "filing_date": fd,
            "filing_time_utc": str(r["filing_time_utc"]),
            "feature_rv_22d": rv22, "labels": labels,
        })

    anchors = {}
    for fam in FAMILIES:
        a = _committed_val_anchor(fam, keys)
        t = _committed_test_reading(fam)
        anchors[fam] = {"val": a, "test": t}
        if a:
            print(f"[sidecar] {fam} committed val anchor: tier {a['tier']} "
                  f"max_qlike_var={a['max_qlike_var']:.6f} "
                  f"max_mode_share={a['max_mode_share_pct']:.4f}% healthy={a['healthy']}")
        else:
            print(f"[sidecar] {fam} committed val anchor: NONE on this machine")

    payload = {
        "schema": "ef_pilot_sidecar/v1",
        "prereg": PREREG,
        "built_utc": _now(),
        "git_commit": _git_commit(),
        "built_from": {"manifest": MANIFEST, "a2": A2_RUN},
        "pilot_n": PILOT_N,
        "canonical_sort": list(CANONICAL_SORT),
        "selection": ("deterministic: FIRST 2,000 event_driven VALIDATION filings by the "
                      "panel's canonical sort (filing_time_utc, ticker, accession) — the "
                      "registered v1.3 pilot set, imported via cf.pilot_panel()"),
        "split": "val (no test row is read or recorded)",
        "tp_committed": TP_COMMITTED,
        "committed_anchors": anchors,
        "docs": docs,
    }
    payload["docs_sha256"] = sha(json.dumps(docs, sort_keys=True, ensure_ascii=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    n_lab = sum(len(d["labels"]) for d in docs)
    print(f"wrote {out}  ({len(docs)} docs, {n_lab} label rows, "
          f"{out.stat().st_size / 1e6:.2f} MB)")
    print(f"docs_sha256 = {payload['docs_sha256']}")
    print("\nShip this file to the box alongside the scripts:")
    print(f"  scp {SIDECAR} box:/root/gpu-data/repo/{SIDECAR}")
    return payload


def load_sidecar(path=SIDECAR):
    p = Path(path)
    if not p.exists():
        _abort("PILOT DATA", [
            f"the pilot sidecar is missing: {path}",
            "",
            "The box slice (aligned_ed_val.parquet + filing_texts_ed_val.parquet) cannot "
            "reconstruct the registered pilot on its own:",
            "  * the committed C6 prompt renders filing_date, which comes from the EDGAR "
            "metadata JSON and is NOT derivable from filing_time_utc (-633..+5 day "
            "deltas across the committed manifest);",
            "  * the box panel holds 14,266 ED val docs vs the committed manifest's "
            "14,213, so head(2,000) on the box would select a DIFFERENT set from the "
            "registered v1.3 pilot.",
            "",
            "Emit it LOCALLY (where the committed manifest + A2 live) and scp it here:",
            "  .venv/bin/python scripts/analysis/elicitation_fairness.py --emit-sidecar",
            f"  scp {SIDECAR} box:/root/gpu-data/repo/{SIDECAR}"])
    d = json.loads(p.read_text())
    if d.get("schema") != "ef_pilot_sidecar/v1":
        _abort("PILOT DATA", [f"unexpected sidecar schema: {d.get('schema')!r}"])
    want = sha(json.dumps(d["docs"], sort_keys=True, ensure_ascii=False))
    if want != d.get("docs_sha256"):
        _abort("PILOT DATA", [
            f"sidecar integrity check failed: {path}",
            f"  recorded docs_sha256 : {d.get('docs_sha256')}",
            f"  recomputed           : {want}",
            "The sidecar was edited after it was emitted. Re-emit it locally."])
    if len(d["docs"]) != d["pilot_n"]:
        _abort("PILOT DATA", [f"sidecar holds {len(d['docs'])} docs, expected "
                              f"{d['pilot_n']}"])
    return d


def sidecar_panel(sc):
    """The pilot panel as the committed inference stack expects it (one row per doc)."""
    rows = []
    for d in sorted(sc["docs"], key=lambda x: x["rank"]):
        rows.append({
            "text_path": d["text_path"], "doc_key": d["doc_key"], "rank": d["rank"],
            "ticker": d["ticker"], "accession": d["accession"], "form": d["form"],
            "item_subtype": d["item_subtype"], "filing_date": d["filing_date"],
            "filing_time_utc": d["filing_time_utc"],
            "feature_rv_22d": d["feature_rv_22d"], "sections_json": None,
            "split": "val",
        })
    return pd.DataFrame(rows)


def sidecar_labels(sc):
    """Long label frame: one row per (doc, horizon) — the A2 val labels, ragged exactly
    as A2 is (the committed pilot's n = 2000 / 1997 / 1991)."""
    rows = []
    for d in sc["docs"]:
        for h, y in d["labels"].items():
            rows.append({"doc_key": d["doc_key"], "horizon_days": int(h),
                         "label_realised_vol": float(y),
                         "feature_rv_22d": d["feature_rv_22d"], "split": "val"})
    return pd.DataFrame(rows)


# ============================================================ G-E2 (val-only)
def assert_val_only(panel, labels, aligned_path, where="pilot"):
    """G-E2 (prereg): 'any read of a test row during the pilot phase is a fatal error (code-level assertion + logging)'.

    On the box there is no test data at all, so the assertion is made positively: every
    pilot doc must be present in aligned_ed_val.parquet, which IS the val slice — a
    membership proof that the pilot reads val rows only. Where the committed manifest is
    present (locally) the stronger negative check also runs: the panel shares no document
    with the ED test split."""
    for name, fr in (("panel", panel), ("labels", labels)):
        if "split" in fr.columns and len(fr) and not (fr["split"] == "val").all():
            _abort("G-E2 (val-only)", [
                f"{where}: frame '{name}' carries non-val rows: "
                f"{fr['split'].value_counts().to_dict()}"])

    proofs = []
    ap = Path(aligned_path)
    if ap.exists():
        av = pq.read_table(str(ap), columns=["text_path"]).column("text_path").to_pylist()
        val_keys = {doc_key(p) for p in av}
        missing = set(panel.doc_key) - val_keys
        if missing:
            _abort("G-E2 (val-only)", [
                f"{where}: {len(missing)} pilot document(s) are NOT in the val slice "
                f"{ap.name} — the pilot must read val rows only.",
                f"Examples: {sorted(missing)[:5]}",
                "Either the sidecar is stale or the box's val slice is a different "
                "vintage. Refusing to run."])
        proofs.append(f"all {len(panel)} pilot docs ∈ {ap.name} ({len(val_keys)} val docs)")
    else:
        proofs.append(f"[warn] {ap.name} absent — membership proof skipped")

    if Path(MANIFEST).exists():
        m = pd.read_parquet(MANIFEST, columns=["text_path", "split", "disclosure"])
        test_keys = {doc_key(p) for p in m[(m.split == "test")].text_path}
        leak = set(panel.doc_key) & test_keys
        if leak:
            _abort("G-E2 (val-only)", [
                f"{where}: {len(leak)} pilot document(s) are TEST filings: "
                f"{sorted(leak)[:5]}"])
        proofs.append(f"0 of {len(test_keys)} committed test filings touched")

    print(f"G-E2 PASS ({where}): {len(panel)} documents, all val | " + " | ".join(proofs))


def check_label_parity(labels, aligned_path):
    """Guard against a stale sidecar: the A2 val labels frozen in the sidecar must equal
    the box's own aligned_ed_val labels on the same (doc, horizon) cells."""
    ap = Path(aligned_path)
    if not ap.exists():
        return {"checked": False, "reason": f"{ap.name} absent"}
    cols = ["text_path", "horizon_days", "label_realised_vol"]
    av = pq.read_table(str(ap), columns=cols).to_pandas()
    av["doc_key"] = av.text_path.map(doc_key)
    m = labels.merge(av[["doc_key", "horizon_days", "label_realised_vol"]],
                     on=["doc_key", "horizon_days"], how="inner",
                     suffixes=("_sidecar", "_box"))
    if not len(m):
        _abort("PILOT DATA", [
            "label parity: the sidecar and the box's aligned_ed_val.parquet share no "
            "(doc, horizon) cell. The sidecar is for a different dataset vintage."])
    d = (m.label_realised_vol_sidecar - m.label_realised_vol_box).abs()
    worst = float(d.max())
    n_cov = len(m)
    if worst > 1e-12:
        _abort("PILOT DATA", [
            f"label parity FAILED: the sidecar's committed A2 val labels differ from "
            f"{ap.name} on {int((d > 1e-12).sum())}/{n_cov} cells (max |delta| {worst:.3e}).",
            "The sidecar is stale relative to the box's data. Re-emit it locally."])
    print(f"label parity PASS: {n_cov} (doc, horizon) cells match {ap.name} "
          f"to {worst:.1e}")
    return {"checked": True, "cells": int(n_cov), "max_abs_delta": worst}


# ============================================================ variant rendering
def exemplar_docs(sc):
    """The two registered V1 exemplars: the FIRST TWO val documents by the panel's
    canonical sort — i.e. ranks 0 and 1 of the very same 2,000-doc pilot set, frozen in
    the sidecar. Deterministic, no RNG, identical for every family."""
    return sorted(sc["docs"], key=lambda x: x["rank"])[:2]


def exemplar_block(sc, texts):
    """The V1 few-shot block: frozen framing + the FIRST TWO val docs by canonical sort
    with their true RV labels. Rendered ONCE and reused for every family and document, so
    the block is byte-identical across families (prereg: 'exemplars verbatim identical across families')."""
    head = exemplar_docs(sc)
    if len(head) < 2:
        _abort("V1 EXEMPLAR", ["fewer than 2 pilot docs — cannot render the frozen V1"])
    parts = [V1_HEADER]
    for i, r in enumerate(head, 1):
        text = texts.get(r["doc_key"])
        if text is None:
            _abort("V1 EXEMPLAR", [
                f"exemplar {i} ({r['ticker']}/{r['accession']}) has no text in the "
                f"cache. The registered V1 text cannot be rendered without it; do NOT "
                f"substitute another document -- the exemplars are frozen as 'the 1st and "
                f"2nd documents in the val panel canonical sort'."])
        excerpt, _src = prompt_mod.build_excerpt(r["form"], r.get("sections_json"), text)
        excerpt = _sanitize(excerpt[:FEWSHOT_EXCERPT_CHARS])
        item = r.get("item_subtype") or ""
        # (the exemplar header mirrors the committed prompt's own header format)
        item_str = f" (items: {item})" if item else ""
        lab = r["labels"]
        ans = json.dumps({f"vol_{h}d": round(float(lab[str(h)]), 4) for h in HORIZONS
                          if str(h) in lab})
        parts.append(
            f"[EXAMPLE {i}]\n"
            f"Company SEC filing.\n"
            f"- Form type: {r['form']}{item_str}\n"
            f"- Filing date: {r['filing_date']}\n"
            f"Filing excerpt:\n[[[\n{excerpt}\n]]]\n"
            f"Correct answer: {ans}\n"
            f"[END EXAMPLE {i}]")
    parts.append(V1_BRIDGE)
    return "\n\n".join(parts)


def _sanitize(excerpt):
    """The committed _fit_budget shrinks the text between the FIRST '<<<' and the LAST
    '>>>' of the user turn. V1 prepends exemplars to V0's user turn, so those markers
    must occur ONLY around the target filing; an exemplar body that happened to contain
    them would make the committed budget-fitter truncate across the whole prompt.
    Deterministic de-fanging of the EXEMPLAR excerpts only (V0's own target excerpt is
    never touched, so V0 stays byte-identical to the committed prompt)."""
    return excerpt.replace("<<<", "< < <").replace(">>>", "> > >")


def build_messages_variant(row, full_text, variant, fewshot_block):
    """V0 is the committed prompt, BY IMPORT. V1/V2 wrap it without touching a byte of it.

    Every variant stays a [system, user] pair, which is load-bearing: the committed retry
    in run_inference._flush rebuilds the re-ask as [messages[0], messages[1] +
    RETRY_SUFFIX], and the committed _fit_budget shrinks messages[-1]. A multi-turn
    few-shot rendering would silently drop the exemplars (or the target) on re-ask; the
    single-user-turn rendering keeps the committed retry byte-compatible for all three
    variants -- exactly what V2's registered 'same mechanism as committed retry' requires."""
    msgs = prompt_mod.build_messages(row, full_text, C6_VARIANT)  # V0, verbatim
    if variant == "V0":
        return msgs
    sys_m, user_m = msgs[0], msgs[1]
    if variant == "V1":
        return [sys_m, {"role": "user",
                        "content": fewshot_block + "\n\n" + user_m["content"]}]
    if variant == "V2":
        return [sys_m, {"role": "user", "content": user_m["content"] + v2_suffix()}]
    raise ValueError(f"unknown EF variant {variant!r}")


def v2_suffix():
    """V2's addition, built from the COMMITTED schema object (imported, not re-typed)."""
    schema = json.dumps(prompt_mod.JSON_SCHEMA, indent=2, sort_keys=True)
    return ("\n\n" + V2_RANGE_HINT + "\n" + V2_SCHEMA_LEAD + "\n" + schema + "\n"
            + V2_CLOSER)


def static_text(variant):
    """The module-constant text a variant owns. V0's 'constants' are the COMMITTED
    prompt's, imported — so an edit to prompt.py trips G-E3 for all three variants."""
    committed = "\x00".join([prompt_mod.SYSTEM_PROMPT, prompt_mod._TASK_TEXT_ONLY,
                             prompt_mod.RETRY_SUFFIX,
                             json.dumps(prompt_mod.JSON_SCHEMA, sort_keys=True)])
    if variant == "V0":
        return committed
    if variant == "V1":
        return "\x00".join([committed, V1_HEADER, V1_BRIDGE, str(FEWSHOT_EXCERPT_CHARS)])
    if variant == "V2":
        return "\x00".join([committed, V2_RANGE_HINT, V2_SCHEMA_LEAD, V2_CLOSER])
    raise ValueError(variant)


def variant_hashes(fewshot_block):
    """G-E3 payload: for each variant, the SHA-256 of (a) the fully rendered template on a
    canonical probe row — which covers the imported V0 text AND the frozen V1/V2 additions
    AND the rendered exemplars — and (b) the static module text alone."""
    out = {}
    for v in VARIANTS:
        msgs = build_messages_variant(CANON_ROW, CANON_TEXT, v, fewshot_block)
        rendered = json.dumps(msgs, ensure_ascii=False, sort_keys=True)
        out[v] = {"render_sha256": sha(rendered), "static_sha256": sha(static_text(v)),
                  "doc": VARIANT_DOC[v]}
    out["_fewshot_block"] = {"sha256": sha(fewshot_block), "chars": len(fewshot_block)}
    return out


def check_hashes(recorded, current, where):
    """G-E3 on every run after the first: the recorded hashes are the law."""
    bad = []
    for v in VARIANTS:
        for k in ("render_sha256", "static_sha256"):
            r, c = recorded.get(v, {}).get(k), current.get(v, {}).get(k)
            if r != c:
                bad.append(f"  {v}.{k}\n    recorded : {r}\n    now      : {c}")
    r = recorded.get("_fewshot_block", {}).get("sha256")
    c = current.get("_fewshot_block", {}).get("sha256")
    if r != c:
        bad.append(f"  _fewshot_block.sha256\n    recorded : {r}\n    now      : {c}")
    if bad:
        _abort("G-E3 (variant-text hash)", [
            f"The frozen variant text changed between runs ({where}).",
            "prereg-ef v1.0: \"V1/V2 text is fixed in this tag; not modifiable for any run outcome\" and",
            "G-E3: \"variant text hashes match this tag's fixed values (prevents mid-run prompt edits)\".",
            "", *bad, "",
            "Nothing is written. Restore the tagged text (git checkout the tag) and "
            "re-run; if the change is intended it is a NEW prereg tag and a NEW "
            "experiment, not an edit to this one."])
    print(f"G-E3 PASS: all variant template hashes match ({where})")


# ============================================================ inference
def resolve_model_ef(fam):
    cfg = FAMILIES[fam]
    ov = cfg.get("path_override")
    if ov and Path(ov).is_dir():
        return ov, "hfview-override (row16 launch.sh; the committed run's `llm` path)"
    return cf.resolve_model(cfg["model_id"])


def make_generator(fam, tp, mock=False):
    """The committed sampling stack at the EFFECTIVE TP; Gemma keeps the disclosed
    prereg-rfa v1.3 system->user fold (it has no system role)."""
    if mock:
        return ri.MockGenerator(seed=PILOT_SEED)
    cfg = FAMILIES[fam]
    path, how = resolve_model_ef(fam)
    cls = cf.FoldChatGenerator if cf.needs_fold(cfg["model_id"]) else ri.VllmGenerator
    print(f"[{fam}] model {cfg['model_id']} -> {path} ({how}); "
          f"generator={cls.__name__} TP={tp} max_model_len={MAX_MODEL_LEN}")
    return cls(path, MAX_MODEL_LEN, tp, MAX_TOKENS)


def do_infer_variant(panel, texts, out_dir, gen, fam, variant, fewshot_block,
                     checkpoint_every=CHECKPOINT_EVERY):
    """cf.do_infer, with the one delta this experiment needs: the message builder and the
    raw `variant` tag are EF-variant-aware. Everything else — resume via part-*.parquet,
    the committed _flush (generate + ONE re-ask on parse failure) — is committed code."""
    vtag = ef_tag(fam, variant)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = ri.load_done(out_dir)
    pending = [r for r in panel.to_dict("records")
               if (r["text_path"], vtag) not in done]
    print(f"[infer] {out_dir}: {len(panel)} filings, {len(pending)} pending "
          f"({len(done)} resumed)")
    if not pending:
        return
    part_idx = len(list(out_dir.glob("part-*.parquet")))
    chunk, n_done = [], 0
    for row in pending:
        text = texts.get(row["doc_key"])
        if text is None:
            continue
        _, src = prompt_mod.build_excerpt(row["form"], row.get("sections_json"), text)
        chunk.append({"row": row, "variant": vtag, "excerpt_source": src,
                      "messages": build_messages_variant(row, text, variant,
                                                         fewshot_block)})
        if len(chunk) >= checkpoint_every:
            part_idx = ri._flush(gen, chunk, out_dir, part_idx)
            n_done += len(chunk)
            print(f"  [infer] {n_done}/{len(pending)} done")
            chunk = []
    if chunk:
        ri._flush(gen, chunk, out_dir, part_idx)
    print(f"[infer] {out_dir}: complete")


def score_variant(panel, labels, raw_dir, fam, variant):
    """The committed pilot scoring path (cf.run_pilot's body): join the model's per-doc
    vols onto the A2 val label rows, fill parse failures with rv22, clip [0.03, 3.0],
    then apply the committed health formula."""
    vtag = ef_tag(fam, variant)
    raw = pp.load_raw(str(raw_dir))
    rv = raw[raw.variant == vtag]
    if not len(rv):
        _abort("PILOT", [f"no rows with variant=={vtag!r} under {raw_dir}"])
    rv = rv.copy()
    rv["doc_key"] = rv.text_path.map(doc_key)
    base = labels.merge(rv[["doc_key", "vol_5d", "vol_10d", "vol_20d"]], on="doc_key",
                        how="inner")
    volmap = {5: "vol_5d", 10: "vol_10d", 20: "vol_20d"}
    pred = np.full(len(base), np.nan)
    for h, col in volmap.items():
        ix = base["horizon_days"] == h
        pred[ix.to_numpy()] = base.loc[ix, col].to_numpy()
    n_all, n_miss = len(base), int(np.isnan(pred).sum())
    pred = np.where(np.isnan(pred), base["feature_rv_22d"].to_numpy(), pred)  # rv22
    n_clip = int(((pred < pp.CLIP_LO) | (pred > pp.CLIP_HI)).sum())
    base = base.copy()
    base["prediction_realised_vol"] = np.clip(pred, pp.CLIP_LO, pp.CLIP_HI)
    h = _health_from_long(base)
    h.update({
        "parse_fail_rate": round(n_miss / n_all, 4) if n_all else float("nan"),
        "clipped_rate": round(n_clip / max(n_all - n_miss, 1), 4),
        "n_docs": int(base.doc_key.nunique()),
    })
    return h


# ============================================================ G-E1 / TP-invariance
def tp_invariance(committed, v0_health, tp_effective):
    """G-E1 AS A DIAGNOSTIC (hardware amendment): report V0@TP-effective BESIDE the
    committed TP=2 reading, print the deltas, and CONTINUE. The one thing that must be
    impossible to miss is a VERDICT FLIP: if healthy/dead disagrees between the committed
    TP=2 reading and V0 at the effective TP, then TP confounds the committed
    instrument-dead judgement and the committed reading cannot be quoted as if it were
    TP-free. This function never exits; it returns the finding."""
    if committed is None:
        return {"status": "no_committed_anchor", "tp_effective": tp_effective,
                "tp_committed": TP_COMMITTED, "verdict_flip": None,
                "note": ("no committed val-side reading exists for this family — V0's "
                         "val health is RECORDED as the new committed-basis anchor "
                         "(nothing to compare; no flip can be detected)")}
    deltas = {}
    for h in HORIZONS:
        hh = str(h)
        if hh not in committed.get("per_h", {}) or hh not in v0_health["per_h"]:
            continue
        row = {}
        for c in HEALTH_COLS:
            a = committed["per_h"][hh].get(c)
            b = v0_health["per_h"][hh].get(c)
            if a is None or b is None:
                continue
            row[c] = {"committed_tp2": float(a), "v0_tp_eff": float(b),
                      "delta": float(b) - float(a)}
        deltas[hh] = row
    flip = bool(committed["healthy"] != v0_health["healthy"])
    return {
        "status": "compared",
        "tp_effective": tp_effective,
        "tp_committed": committed.get("tp", TP_COMMITTED),
        "anchor_tier": committed.get("tier"),
        "anchor_source": committed.get("source"),
        "anchor_comparability": committed.get("comparable"),
        "anchor_note": committed.get("note"),
        "committed": {"healthy": committed["healthy"],
                      "max_qlike_var": committed["max_qlike_var"],
                      "max_mode_share_pct": committed["max_mode_share_pct"]},
        "v0_at_tp_effective": {"healthy": v0_health["healthy"],
                               "max_qlike_var": v0_health["max_qlike_var"],
                               "max_mode_share_pct": v0_health["max_mode_share_pct"]},
        "delta_max_qlike_var": (v0_health["max_qlike_var"]
                                - committed["max_qlike_var"]),
        "delta_max_mode_share_pct": (v0_health["max_mode_share_pct"]
                                     - committed["max_mode_share_pct"]),
        "per_h_deltas": deltas,
        "verdict_flip": flip,
        "bitexact": bool(not flip and all(
            cf.close(v0_health["per_h"][h][c], committed["per_h"][h][c])
            for h in v0_health["per_h"] if h in committed.get("per_h", {})
            for c in HEALTH_COLS
            if c in committed["per_h"][h] and c in v0_health["per_h"][h])),
    }


def report_tp_invariance(fam, ti):
    """Print the diagnostic. A verdict flip gets a banner that cannot be scrolled past."""
    if ti["status"] == "no_committed_anchor":
        print(f"\n[{fam}] G-E1 / TP-invariance: {ti['note']}")
        return
    print(f"\n[{fam}] G-E1 as TP-INVARIANCE DIAGNOSTIC "
          f"(V0@TP{ti['tp_effective']} vs committed TP{ti['tp_committed']}) — "
          f"anchor tier {ti['anchor_tier']} [{ti['anchor_comparability']}]: "
          f"{ti['anchor_source']}")
    print(f"  {ti['anchor_note']}")
    print(f"  {'statistic':<22} {'committed TP2':>16} {'V0@TP' + str(ti['tp_effective']):>16} "
          f"{'delta':>14}")
    c, v = ti["committed"], ti["v0_at_tp_effective"]
    print(f"  {'max QLIKE(var)  (<4)':<22} {c['max_qlike_var']:>16.6f} "
          f"{v['max_qlike_var']:>16.6f} {ti['delta_max_qlike_var']:>+14.6f}")
    print(f"  {'max modal share (<60)':<22} {c['max_mode_share_pct']:>16.4f} "
          f"{v['max_mode_share_pct']:>16.4f} {ti['delta_max_mode_share_pct']:>+14.4f}")
    print(f"  {'VERDICT':<22} {('healthy' if c['healthy'] else 'DEAD'):>16} "
          f"{('healthy' if v['healthy'] else 'DEAD'):>16}")
    for h, row in ti["per_h_deltas"].items():
        bits = ", ".join(f"{k} {d['delta']:+.4g}" for k, d in row.items()
                         if k in ("qlike_var", "mode_share_pct", "n"))
        print(f"    h={h:<3} {bits}")
    if ti["bitexact"]:
        print(f"  [{fam}] every health cell is bit-identical to the committed reading "
              f"(rtol {RTOL:g}).")
    if ti["verdict_flip"]:
        _banner(
            f"!! TP VERDICT FLIP — {fam} !!",
            [f"The committed TP={ti['tp_committed']} reading says "
             f"{'HEALTHY' if c['healthy'] else 'DEAD'}; "
             f"V0 at TP={ti['tp_effective']} says "
             f"{'HEALTHY' if v['healthy'] else 'DEAD'}.",
             "",
             "The committed instrument-dead judgement for this family is CONFOUNDED BY "
             "TENSOR PARALLELISM: the same prompt, weights and documents give a different",
             "health verdict purely by changing the TP split. Any M2 conclusion that "
             "leans on the committed reading must say so.",
             "",
             f"anchor: tier {ti['anchor_tier']} [{ti['anchor_comparability']}] "
             f"{ti['anchor_source']}",
             ("NOTE: this anchor is CROSS-SPLIT (committed test-side vs pilot val-side), "
              "so the flip mixes split with TP and is NOT clean evidence of a TP "
              "confound." if ti["anchor_comparability"] == "cross-split" else
              "This anchor is val-side on the same 2,000 docs, so the flip is attributable "
              "to TP (up to the sampler's own run-to-run jitter)."),
             "",
             "The run CONTINUES (hardware amendment: G-E1 is a diagnostic, not a gate).",
             "This finding is recorded in the pilot json under tp_invariance."])
    else:
        print(f"  [{fam}] no verdict flip: both readings say "
              f"{'healthy' if v['healthy'] else 'DEAD'}.")


def strict_ge1(fam, committed, v0_health):
    """The registered bit-exact G-E1, available only when --tp equals the committed TP
    (otherwise it is unsatisfiable by construction and must not masquerade as a gate)."""
    if committed is None:
        print(f"[{fam}] G-E1 strict: no committed val-side anchor — nothing to "
              f"reproduce; V0 is RECORDED as the new committed-basis anchor.")
        return
    bad = []
    for h in HORIZONS:
        hh = str(h)
        if hh not in committed["per_h"]:
            continue
        for c in HEALTH_COLS:
            if c not in committed["per_h"][hh] or c not in v0_health["per_h"][hh]:
                continue
            a, b = committed["per_h"][hh][c], v0_health["per_h"][hh][c]
            if not cf.close(b, a):
                bad.append((h, c, b, a, abs(float(b) - float(a))))
    if bad:
        _abort("G-E1 (strict, as registered)", [
            f"family {fam}: the V0 re-run does NOT reproduce the committed val-side "
            f"reading to machine precision (rtol {RTOL:g}).",
            f"  anchor tier {committed['tier']}: {committed['source']}",
            "",
            "  horizon | column          | V0 re-run            | committed            "
            "| |delta|",
            *[f"  {h:7d} | {c:<15s} | {b!r:<20.20} | {a!r:<20.20} | {d:.3e}"
              for h, c, b, a, d in bad[:24]],
            "",
            "REGISTERED CONSEQUENCE (prereg-ef v1.0 G-E1): the experiment aborts and pipeline drift is reported"
            " (old readings must not be overwritten with new numbers). Nothing is written.",
            "",
            "NOTE: the committed pipeline is not run-to-run deterministic (`--seed` is "
            "not plumbed into vLLM's sampler), so this gate can fire on sampler jitter "
            "alone. Relaxing it is a prereg amendment — the author's call, to be made "
            "before any M2 statistic exists, not this script's."])
    print(f"[{fam}] G-E1 strict PASS: V0 reproduces {committed['source']} to rtol "
          f"{RTOL:g}")


# ============================================================ selection rule
def apply_selection(per_variant):
    """prereg SS Selection rule: 'per family select the variant with the lowest val modal share (ties V0 -> V1 -> V2)'.

    The modal share compared is the MAX over horizons — the same statistic the registered
    health gate thresholds ('max modal share(round(pred,2)) < 60%'), so selection and
    health read the identical number. Ties break in the registered order V0 > V1 > V2,
    which is also the stable order of VARIANTS, so a plain min() over
    (mode_share, variant_rank) implements the rule exactly."""
    ranked = sorted(
        [(per_variant[v]["max_mode_share_pct"], VARIANTS.index(v), v)
         for v in VARIANTS if v in per_variant])
    if not ranked:
        _abort("SELECTION", ["no variant results to select from"])
    best = ranked[0][2]
    tied = [r[2] for r in ranked if r[0] == ranked[0][0]]
    return {
        "selected_variant": best,
        "selected_max_mode_share_pct": ranked[0][0],
        "healthy": bool(per_variant[best]["healthy"]),
        "rule": ("lowest val modal share (max over horizons); ties V0 -> V1 -> V2 "
                 "(prereg SS Selection rule)"),
        "ranking": [{"variant": v, "max_mode_share_pct": s,
                     "healthy": bool(per_variant[v]["healthy"])}
                    for s, _, v in ranked],
        "tie": len(tied) > 1,
        "tied_variants": tied if len(tied) > 1 else [],
        "released_to_full": bool(per_variant[best]["healthy"]),
        "registered_consequence": (
            "if that variant is healthy -> cleared for the full run" if per_variant[best]["healthy"]
            else "all three variants unhealthy -> the family is registered as elicitation-robust "
                 "instrument-dead, no full run"),
    }


# ============================================================ pilot
def _load_pilot_inputs(args):
    sc = load_sidecar(args.sidecar)
    panel = sidecar_panel(sc)
    labels = sidecar_labels(sc)
    aligned = resolve_aligned_val(args.aligned)
    assert_val_only(panel, labels, aligned, where="pilot")
    parity = check_label_parity(labels, aligned)
    cache = resolve_text_cache(args.text_cache)
    print(f"[pilot] sidecar {args.sidecar} ({len(panel)} docs, "
          f"docs_sha256 {sc['docs_sha256'][:16]}...)")
    print(f"[pilot] text cache {cache}")
    return sc, panel, labels, cache, parity


def run_pilot_cell(args):
    """ONE (family, variant) cell — the unit the box runs as its own subprocess so vLLM
    frees the GPU between cells. Writes a shard json; assembles nothing."""
    fam, variant = args.family, args.variant
    shard = shard_of(fam, variant)
    assert_writeonce(shard)
    sc, panel, labels, cache, parity = _load_pilot_inputs(args)

    texts = stream_texts_by_key(set(panel.doc_key), cache)
    fewshot = exemplar_block(sc, texts)
    hashes = variant_hashes(fewshot)
    print(f"[{fam}/{variant}] G-E3 render_sha256 = {hashes[variant]['render_sha256']}")

    raw_dir = pilot_rawdir(fam, variant)
    gen = make_generator(fam, args.tp, mock=args.mock)
    do_infer_variant(panel, texts, raw_dir, gen, fam, variant, fewshot,
                     checkpoint_every=args.checkpoint_every)
    health = score_variant(panel, labels, raw_dir, fam, variant)

    out = {
        "schema": "ef_pilot_shard/v1",
        "prereg": PREREG,
        "prereg_amendment": PREREG_AMEND,
        "family": fam,
        "model_id": FAMILIES[fam]["model_id"],
        "model_desc": FAMILIES[fam]["desc"],
        "variant": variant,
        "variant_doc": VARIANT_DOC[variant],
        "tp_effective": args.tp,
        "tp_committed": TP_COMMITTED,
        "tp_matches_committed": bool(args.tp == TP_COMMITTED),
        "hashes": hashes,
        "sidecar_docs_sha256": sc["docs_sha256"],
        "label_parity": parity,
        "split": "val (no test row read)",
        "n_docs": health["n_docs"],
        "healthy": health["healthy"],
        "gate": HEALTH_GATE_TEXT,
        "max_qlike_var": health["max_qlike_var"],
        "max_mode_share_pct": health["max_mode_share_pct"],
        "per_h": health["per_h"],
        "parse_fail_rate": health["parse_fail_rate"],
        "clipped_rate": health["clipped_rate"],
        "sampling": {"temperature": 0.0, "retry_temperature": 0.2,
                     "max_tokens": MAX_TOKENS, "max_model_len": MAX_MODEL_LEN,
                     "tp": args.tp, "guided_json": True,
                     "prompt_variant": C6_VARIANT, "ef_variant": variant,
                     "system_fold": bool(getattr(gen, "system_fold", False)),
                     "mock": bool(args.mock)},
        "seed": PILOT_SEED,
        "raw_dir": str(raw_dir),
        "git_commit": _git_commit(),
        "timestamp_utc": _now(),
    }
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {shard}")
    print(f"[{fam}/{variant}] TP={args.tp} health: max QLIKE(var)="
          f"{health['max_qlike_var']:.4f} (<{HEALTH_QLIKE_MAX:g}), max modal share="
          f"{health['max_mode_share_pct']:.2f}% (<{HEALTH_MODE_MAX:g}%) -> "
          f"{'HEALTH_PASS' if health['healthy'] else 'HEALTH_FAIL'}")
    return out


def assemble_pilot(args):
    """Combine the six shards -> the registered pilot json: per-variant health, the
    selection, the healthy bool, the hashes, the effective TP and the TP-invariance
    deltas against the committed TP=2 readings."""
    assert_writeonce(PILOT_JSON)
    shards = {}
    missing = []
    for fam in FAMILIES:
        for v in VARIANTS:
            p = shard_of(fam, v)
            if not p.exists():
                missing.append(str(p))
            else:
                shards[(fam, v)] = json.loads(p.read_text())
    if missing:
        _abort("ASSEMBLE", [
            "the pilot is incomplete — the registered design is 2 families x 3 variants "
            "over the SAME 2,000 val docs, and the selection rule cannot be applied to a "
            "partial grid.",
            "", "missing shards:", *[f"  {m}" for m in missing]])

    # G-E3 across subprocesses: every shard must carry the same frozen text.
    ref = shards[("mistral24", "V0")]["hashes"]
    for (fam, v), s in shards.items():
        check_hashes(ref, s["hashes"], f"shard {fam}/{v}")

    tps = {s["tp_effective"] for s in shards.values()}
    if len(tps) > 1:
        _abort("TP", [f"shards disagree on the effective TP: {sorted(tps)}",
                      "Every cell of one pilot must run at one TP."])
    tp_eff = tps.pop()

    sc = load_sidecar(args.sidecar)
    panel_keys = {d["doc_key"] for d in sc["docs"]}

    per_family, families = {}, {}
    for fam in FAMILIES:
        per_variant = {v: {k: shards[(fam, v)][k] for k in
                           ("healthy", "max_qlike_var", "max_mode_share_pct", "per_h",
                            "parse_fail_rate", "clipped_rate", "n_docs")}
                       for v in VARIANTS}
        sel = apply_selection(per_variant)

        # TP-invariance: V0 at the effective TP vs the committed TP=2 readings.
        committed = _committed_val_anchor(fam, panel_keys)
        if committed is None:
            committed = (sc.get("committed_anchors", {}).get(fam) or {}).get("val")
            if committed:
                committed = dict(committed)
                committed["source"] = (str(committed.get("source"))
                                       + f" [frozen in {Path(args.sidecar).name}]")
        ti = tp_invariance(committed, per_variant["V0"], tp_eff)
        report_tp_invariance(fam, ti)

        test_ref = (_committed_test_reading(fam)
                    or (sc.get("committed_anchors", {}).get(fam) or {}).get("test"))
        ti_test = None
        if test_ref is not None:
            ti_test = tp_invariance(test_ref, per_variant["V0"], tp_eff)
            report_tp_invariance(fam + " [cross-split context]", ti_test)

        if args.strict_ge1:
            if tp_eff != TP_COMMITTED:
                _abort("G-E1 (strict)", [
                    f"--strict-ge1 demands the registered bit-exact reproduction, but "
                    f"the effective TP is {tp_eff} and the committed readings were made "
                    f"at TP={TP_COMMITTED}.",
                    "TP changes the reduction order, so bit-exactness is unavailable by "
                    "construction. Run at --tp 2, or drop --strict-ge1 and read the "
                    "TP-invariance diagnostic instead."])
            strict_ge1(fam, committed, per_variant["V0"])

        families[fam] = {
            "model_id": FAMILIES[fam]["model_id"],
            "model_desc": FAMILIES[fam]["desc"],
            "per_variant": per_variant,
            "selection": sel,
            "healthy": sel["healthy"],
            "tp_invariance": ti,
            "tp_invariance_vs_committed_test_side": ti_test,
            "shards": {v: str(shard_of(fam, v)) for v in VARIANTS},
        }
        per_family[fam] = sel

    all_dead = all(not f["healthy"] for f in families.values())
    released = [f for f, d in families.items() if d["healthy"]]
    flips = [f for f, d in families.items()
             if (d["tp_invariance"] or {}).get("verdict_flip")]

    out = {
        "schema": "ef_pilot/v1",
        "prereg": PREREG,
        "prereg_amendment": PREREG_AMEND,
        "stage": "pilot (val only; no increment statistic; no test row read)",
        "gate": HEALTH_GATE_TEXT,
        "selection_rule": ("per family select the variant with the lowest val modal share (ties V0 -> V1 -> V2); "
                           "if that variant is healthy -> cleared for the full run; if all three are unhealthy -> the family is registered as "
                           "elicitation-robust instrument-dead"),
        "pilot_n": PILOT_N,
        "pilot_docs": ("the registered v1.3 pilot set: FIRST 2,000 event_driven "
                       "VALIDATION filings by the canonical sort "
                       "(filing_time_utc, ticker, accession)"),
        "sidecar": args.sidecar,
        "sidecar_docs_sha256": sc["docs_sha256"],
        "tp_effective": tp_eff,
        "tp_committed": TP_COMMITTED,
        "tp_matches_committed": bool(tp_eff == TP_COMMITTED),
        "tp_note": PREREG_AMEND,
        "hashes": ref,
        "families": families,
        "released_to_full": released,
        "all_families_dead": all_dead,
        "tp_verdict_flips": flips,
        "branch_pending": ("(d) -- both families, all three variants unhealthy: the confound is falsified" if all_dead
                           else f"(a)/(b)/(c) pending the full pass for: {released}"),
        "branch_text": BRANCH_ZH,
        "seed": PILOT_SEED,
        "git_commit": _git_commit(),
        "timestamp_utc": _now(),
    }
    Path(PILOT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(PILOT_JSON).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {PILOT_JSON}")

    print(f"\n{'=' * 78}\nEF PILOT SUMMARY (TP={tp_eff}, committed TP={TP_COMMITTED})"
          f"\n{'=' * 78}")
    print(f"{'family':<11} {'variant':<8} {'max QLIKE(var)':>15} {'max modal %':>12} "
          f"{'healthy':>8}")
    for fam, d in families.items():
        for v in VARIANTS:
            pv = d["per_variant"][v]
            mark = " <- selected" if v == d["selection"]["selected_variant"] else ""
            print(f"{fam:<11} {v:<8} {pv['max_qlike_var']:>15.4f} "
                  f"{pv['max_mode_share_pct']:>12.2f} "
                  f"{str(pv['healthy']):>8}{mark}")
    print()
    for fam, d in families.items():
        s = d["selection"]
        print(f"{fam}: selected {s['selected_variant']} "
              f"(modal {s['selected_max_mode_share_pct']:.2f}%) -> "
              f"{'RELEASED to full' if s['healthy'] else 'elicitation-robust INSTRUMENT-DEAD'}")
    if flips:
        _banner("!! TP VERDICT FLIP(S) RECORDED !!",
                [f"families whose healthy/dead verdict flips between committed TP="
                 f"{TP_COMMITTED} and V0@TP{tp_eff}: {flips}",
                 "The committed instrument-dead judgement for these families is "
                 "TP-confounded. See tp_invariance in the pilot json."])
    if all_dead:
        print("\nREGISTERED BRANCH (d): both families, all three variants unhealthy -> the confound is falsified "
              "(fair prompt adaptation cannot revive them). No family advances to --full.")
    else:
        print(f"\nReleased to the full pass: {released} — the branch among (a)/(b)/(c) "
              f"is decided by the full run, which touches test ONCE.")
    return out


def run_pilot(args):
    """--pilot with no --family runs the whole grid in-process (local/mock convenience);
    the box runs one cell per subprocess so vLLM releases the GPU between cells."""
    if args.assemble:
        return assemble_pilot(args)
    if args.family and args.variant:
        return run_pilot_cell(args)
    if args.family or args.variant:
        _abort("CLI", ["--family and --variant must be given together "
                       "(one cell per subprocess), or neither (run the whole grid)"])
    for fam in FAMILIES:
        for v in VARIANTS:
            args.family, args.variant = fam, v
            run_pilot_cell(args)
    args.family = args.variant = None
    return assemble_pilot(args)


# ============================================================ full
def full_data_staged():
    """Which of the full-panel inputs are on this machine. The box currently carries the
    val pilot slice ONLY, so --full stops here with a clear message instead of crashing
    halfway through a 1.5-box-day run."""
    missing = []
    if not Path(MANIFEST).exists():
        missing.append(("committed manifest (val+test panel)", MANIFEST))
    if not Path(A2_RUN).exists():
        missing.append(("A2 HAR reference run", A2_RUN))
    full_cache = data_root() / "processed" / "_text_cache" / "filing_texts.parquet"
    if not full_cache.exists():
        missing.append(("full text cache (val+test)", str(full_cache)))
    return missing


def run_full(args):
    """The registered full pass. Guard chain first: pilot -> health -> hashes -> data.

    The full panel is NOT staged on the run box, so this stops at the staging gate by
    design. The registered protocol beyond it (3 jitter seeds x 39,322 ED docs +
    arithmetic ensemble, protocol verbatim identical to B1/B2, G-E4 re-applied on the ensemble base, test
    touched ONCE) runs only once the author stages the data."""
    if not Path(PILOT_JSON).exists():
        _abort("FULL", [
            f"--full refuses to run without the registered pilot: {PILOT_JSON} is absent.",
            "prereg-ef v1.0: the full pass is gated on the val pilot — "
            "'if that variant is healthy -> cleared for the full run'. Run --pilot first."])
    pilot = json.loads(Path(PILOT_JSON).read_text())

    released = pilot.get("released_to_full", [])
    if not released:
        print(f"\n{'=' * 78}\nFULL REFUSED — no family passed the registered health gate"
              f"\n{'=' * 78}")
        print("prereg-ef v1.0: 'if all three variants of a family are unhealthy -> that family is registered as "
              "elicitation-robust instrument-dead, no full run.'")
        print(f"pilot verdict: {pilot.get('branch_pending')}")
        print("Registered branch (d) is already decided by the pilot; the full pass "
              "would touch test for no registered reason. Nothing to run.")
        sys.exit(2)
    if args.family:
        if args.family not in released:
            _abort("FULL", [
                f"family {args.family!r} did not pass the pilot health gate "
                f"(released: {released}).",
                "prereg-ef v1.0: only health-passed families advance; test is touched "
                "ONLY by a healthy released family."])
        released = [args.family]
    print(f"[full] health-passed families released by the pilot: {released}")
    for fam in released:
        v = pilot["families"][fam]["selection"]["selected_variant"]
        print(f"[full] {fam}: selected variant {v} "
              f"(modal {pilot['families'][fam]['selection']['selected_max_mode_share_pct']:.2f}%)")

    if pilot["tp_effective"] != args.tp:
        print(f"[full] WARNING: the pilot selected at TP={pilot['tp_effective']} but "
              f"--tp {args.tp} was passed. The full run's effective TP is recorded as "
              f"{args.tp}.")

    missing = full_data_staged()
    if missing:
        print(f"\n{'=' * 78}\nFULL-PANEL DATA NOT STAGED — nothing was run, nothing "
              f"written\n{'=' * 78}")
        print("The run box carries the val pilot slice only. The registered full pass "
              "(3 jitter seeds x 39,322 event_driven documents + arithmetic ensemble, "
              "protocol verbatim identical to B1/B2) needs the val+test panel, which is not here.\n")
        print("missing:")
        for what, where in missing:
            print(f"  - {what}\n      {where}")
        print("\npresent and ready:")
        print(f"  - pilot json      {PILOT_JSON}")
        print(f"  - released family {released}")
        print(f"  - effective TP    {args.tp} (committed TP={TP_COMMITTED})")
        print(f"\nStage the full panel under SP500VOL_DATA_ROOT={data_root()} and the "
              f"committed manifest + A2 run under results/, then re-run:")
        print(f"  python3 scripts/analysis/elicitation_fairness.py --full --tp {args.tp}")
        print("\nThis is a clean refusal, not a failure: exit 4.")
        sys.exit(4)

    _abort("FULL", [
        "the full-panel data IS staged, but the registered full pass is deliberately "
        "not wired up in this build.",
        "",
        "Scope note (author): the M2 full pass reuses the committed B1/B2 machinery "
        "verbatim (3 jitter seeds x 39,322 ED docs -> build_run_dir -> arithmetic "
        "ensemble -> G-E4 health re-check -> clustered DM + Holm(3) -> branch ruling). "
        "It was left unimplemented because the data was not stageable at build time and "
        "untested full-run code that touches test ONCE is worse than no code.",
        "",
        f"The pilot has released: {released}. Wire the full pass against the staged "
        f"panel, re-run the selftests, and only then spend the test touch."])


# ============================================================ diagnostics (no writes)
def show_variants(args):
    """Print the EXACT rendered text of V0/V1/V2 on the registered exemplars. Writes
    nothing. Needs the text cache (the V1 exemplar excerpts are real filing text)."""
    sc = load_sidecar(args.sidecar)
    cache = resolve_text_cache(args.text_cache)
    texts = stream_texts_by_key({d["doc_key"] for d in exemplar_docs(sc)}, cache)
    fewshot = exemplar_block(sc, texts)
    hashes = variant_hashes(fewshot)

    print("=" * 78)
    print(f"EF FROZEN VARIANT TEXT — {PREREG}")
    print("=" * 78)
    for i, d in enumerate(exemplar_docs(sc), 1):
        print(f"V1 exemplar {i}: {d['ticker']} {d['accession']} {d['form']} "
              f"items={d['item_subtype']} filing_date={d['filing_date']} "
              f"labels={d['labels']}")
    print()
    print("-" * 78)
    print("V1 FEW-SHOT BLOCK (prepended to V0's user turn; excerpts head-capped at "
          f"{FEWSHOT_EXCERPT_CHARS} chars)")
    print("-" * 78)
    if args.truncate:
        print(_elide(fewshot, args.truncate))
    else:
        print(fewshot)
    print()
    print("-" * 78)
    print("V2 SUFFIX (appended to V0's user turn)")
    print("-" * 78)
    print(v2_suffix())
    print()
    print("-" * 78)
    print("G-E3 HASHES")
    print("-" * 78)
    for v in VARIANTS:
        print(f"  {v}  render {hashes[v]['render_sha256']}")
        print(f"      static {hashes[v]['static_sha256']}")
    print(f"  fewshot block sha256 {hashes['_fewshot_block']['sha256']} "
          f"({hashes['_fewshot_block']['chars']} chars)")
    return hashes


def _elide(s, keep):
    if len(s) <= 2 * keep:
        return s
    return (s[:keep] + f"\n\n... [{len(s) - 2 * keep} chars elided] ...\n\n" + s[-keep:])


# ============================================================ selftest
def _selftest():  # noqa: C901
    """Synthetic fixtures in a sandbox tmpdir. Never touches the real results/ tree."""
    import shutil
    import tempfile

    ok, fail = [], []

    def check(name, cond, detail=""):
        (ok if cond else fail).append(name)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail
                                                            and not cond else ""))

    real_cwd = Path.cwd()
    tmp = Path(tempfile.mkdtemp(prefix="ef_selftest_"))
    print(f"\nselftest sandbox: {tmp}\n(the real results/ tree is never touched)\n")

    # ---- synthetic fixtures -------------------------------------------------
    n_docs = 40
    rng = np.random.default_rng(7)
    docs, texts = [], []
    for i in range(n_docs):
        k = f"000000{i:04d}-20-000001.txt"
        rv22 = float(np.clip(0.18 + 0.05 * rng.standard_normal(), 0.05, 1.2))
        labels = {"5": rv22 * 1.02, "10": rv22 * 0.98, "20": rv22 * 1.05}
        if i == 3:
            labels.pop("20")          # ragged horizons, exactly like A2
        docs.append({
            "rank": i, "doc_key": k,
            "text_path": f"/path/to/data-root/sp500vol-data/interim/full/8-K/000{i:04d}/{k}",
            "ticker": f"T{i:03d}", "accession": k[:-4], "form": "8-K",
            "item_subtype": "2.02", "filing_date": f"2020-01-{(i % 28) + 1:02d}",
            "filing_time_utc": f"2020-01-{(i % 28) + 1:02d} 12:00:00+00:00",
            "feature_rv_22d": rv22, "labels": labels,
        })
        texts.append({
            # a BOX-style prefix: proves the basename join survives a data-root rewrite
            "text_path": f"/root/gpu-data/sp500vol-data/interim/full/8-K/000{i:04d}/{k}",
            "text": f"Item 2.02 Results of Operations. Synthetic filing {i}. " * 12,
        })

    def write_fixtures(root, docs_):
        root.mkdir(parents=True, exist_ok=True)
        (root / "results/e1_llm_forecast").mkdir(parents=True, exist_ok=True)
        (root / "results/tables").mkdir(parents=True, exist_ok=True)
        (root / "data/processed/_text_cache").mkdir(parents=True, exist_ok=True)
        (root / "data/processed/full").mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "ef_pilot_sidecar/v1", "prereg": PREREG, "built_utc": _now(),
            "git_commit": "selftest", "built_from": {"manifest": "synthetic"},
            "pilot_n": len(docs_), "canonical_sort": list(CANONICAL_SORT),
            "selection": "synthetic", "split": "val", "tp_committed": TP_COMMITTED,
            "committed_anchors": {}, "docs": docs_,
        }
        payload["docs_sha256"] = sha(json.dumps(docs_, sort_keys=True,
                                                ensure_ascii=False))
        (root / SIDECAR).write_text(json.dumps(payload, indent=2))
        pd.DataFrame(texts).to_parquet(
            root / "data/processed/_text_cache/filing_texts_ed_val.parquet", index=False)
        al = []
        for d in docs_:
            for h, y in d["labels"].items():
                al.append({"text_path": d["text_path"], "horizon_days": int(h),
                           "label_realised_vol": float(y)})
        pd.DataFrame(al).to_parquet(
            root / "data/processed/full/aligned_ed_val.parquet", index=False)
        return payload

    class A:  # argparse stand-in
        def __init__(self, **kw):
            self.sidecar = SIDECAR
            self.aligned = None
            self.text_cache = None
            self.tp = 1
            self.mock = True
            self.checkpoint_every = 20
            self.assemble = False
            self.family = None
            self.variant = None
            self.strict_ge1 = False
            self.truncate = 0
            for k, v in kw.items():
                setattr(self, k, v)

    try:
        sandbox = tmp / "box"
        write_fixtures(sandbox, docs)
        os.chdir(sandbox)
        os.environ["SP500VOL_DATA_ROOT"] = str(sandbox / "data")

        # ---- 1. G-E3: hash gate fires on a prompt edit ----------------------
        print("1. G-E3 hash gate")
        sc = load_sidecar()
        panel = sidecar_panel(sc)
        cache = resolve_text_cache()
        tx = stream_texts_by_key(set(panel.doc_key), cache)
        check("text cache joins on basename across a data-root rewrite",
              len(tx) == n_docs, f"got {len(tx)}/{n_docs}")
        fs = exemplar_block(sc, tx)
        h0 = variant_hashes(fs)
        check("hashes are stable across identical renders",
              variant_hashes(exemplar_block(sc, tx)) == h0)
        check("the V1 exemplars are ranks 0 and 1 of the canonical sort",
              [d["rank"] for d in exemplar_docs(sc)] == [0, 1])
        check("the few-shot block renders both exemplars' true labels",
              "[EXAMPLE 1]" in fs and "[EXAMPLE 2]" in fs
              and f'"vol_5d": {round(docs[0]["labels"]["5"], 4)}' in fs)

        saved = prompt_mod._TASK_TEXT_ONLY
        try:
            prompt_mod._TASK_TEXT_ONLY = saved + " (edited)"
            h1 = variant_hashes(fs)
            edited_v0 = h1["V0"]["static_sha256"] != h0["V0"]["static_sha256"]
            fired = False
            try:
                check_hashes(h0, h1, "selftest/prompt-edit")
            except SystemExit:
                fired = True
        finally:
            prompt_mod._TASK_TEXT_ONLY = saved
        check("editing the committed prompt changes V0's static hash", edited_v0)
        check("G-E3 aborts when the frozen text changes", fired)

        saved_v2 = globals()["V2_RANGE_HINT"]
        try:
            globals()["V2_RANGE_HINT"] = saved_v2 + " tweak"
            fired2 = False
            try:
                check_hashes(h0, variant_hashes(fs), "selftest/v2-edit")
            except SystemExit:
                fired2 = True
        finally:
            globals()["V2_RANGE_HINT"] = saved_v2
        check("G-E3 aborts when V2's frozen text changes", fired2)
        check("G-E3 passes when nothing changed",
              check_hashes(h0, variant_hashes(fs), "selftest/clean") is None)

        # ---- 2. G-E2: val-only assertion fires on a test row ----------------
        print("2. G-E2 val-only")
        labels = sidecar_labels(sc)
        aligned = resolve_aligned_val()
        check("G-E2 passes on an all-val panel",
              assert_val_only(panel, labels, aligned, "selftest") is None)
        bad_panel = panel.copy()
        bad_panel.loc[0, "split"] = "test"
        fired = False
        try:
            assert_val_only(bad_panel, labels, aligned, "selftest")
        except SystemExit:
            fired = True
        check("G-E2 aborts on a test-split row in the panel", fired)
        bad_labels = labels.copy()
        bad_labels.loc[0, "split"] = "test"
        fired = False
        try:
            assert_val_only(panel, bad_labels, aligned, "selftest")
        except SystemExit:
            fired = True
        check("G-E2 aborts on a test-split row in the label frame", fired)
        # a doc that is NOT in the val slice = not provably val
        ghost = panel.copy()
        ghost.loc[0, "doc_key"] = "9999999999-99-999999.txt"
        fired = False
        try:
            assert_val_only(ghost, labels, aligned, "selftest")
        except SystemExit:
            fired = True
        check("G-E2 aborts when a pilot doc is absent from the val slice", fired)

        # ---- 3. selection rule incl. tie order ------------------------------
        print("3. selection rule")

        def pv(*shares, healthy=(True, True, True)):
            return {v: {"max_mode_share_pct": s, "healthy": h,
                        "max_qlike_var": 1.0}
                    for v, s, h in zip(VARIANTS, shares, healthy)}

        s = apply_selection(pv(50.0, 30.0, 40.0))
        check("selects the lowest val modal share", s["selected_variant"] == "V1",
              s["selected_variant"])
        s = apply_selection(pv(30.0, 30.0, 40.0))
        check("tie V0/V1 -> V0", s["selected_variant"] == "V0" and s["tie"])
        s = apply_selection(pv(40.0, 30.0, 30.0))
        check("tie V1/V2 -> V1", s["selected_variant"] == "V1" and s["tie"])
        s = apply_selection(pv(30.0, 30.0, 30.0))
        check("three-way tie -> V0", s["selected_variant"] == "V0"
              and s["tied_variants"] == ["V0", "V1", "V2"])
        s = apply_selection(pv(50.0, 30.0, 40.0, healthy=(True, False, True)))
        check("selection is on modal share ALONE — an unhealthy winner is not "
              "swapped for a healthy loser",
              s["selected_variant"] == "V1" and not s["healthy"]
              and not s["released_to_full"])
        check("healthy winner is released", apply_selection(
            pv(50.0, 30.0, 40.0))["released_to_full"])

        # ---- 4. TP-invariance flag ------------------------------------------
        print("4. TP-invariance diagnostic")

        def health(q, m, healthy=None):
            per = {str(h): {c: 1.0 for c in HEALTH_COLS} for h in HORIZONS}
            for h in per:
                per[h]["qlike_var"] = q
                per[h]["mode_share_pct"] = m
                per[h]["n"] = 2000
            return {"per_h": per, "max_qlike_var": q, "max_mode_share_pct": m,
                    "healthy": (bool(q < HEALTH_QLIKE_MAX and m < HEALTH_MODE_MAX)
                                if healthy is None else healthy)}

        comm = dict(health(3.66, 45.2), tier="A", source="committed_pilot.json", tp=2,
                    comparable="strict", note="synthetic committed")
        ti = tp_invariance(comm, health(3.70, 45.9), 1)
        check("no flag when both readings are healthy", ti["verdict_flip"] is False)
        check("TP recorded in the diagnostic",
              ti["tp_effective"] == 1 and ti["tp_committed"] == 2)
        check("deltas are computed", abs(ti["delta_max_qlike_var"] - 0.04) < 1e-9
              and abs(ti["delta_max_mode_share_pct"] - 0.7) < 1e-9)
        # healthy at TP2 -> DEAD at TP1 (modal crosses 60)
        ti = tp_invariance(comm, health(3.70, 61.2), 1)
        check("FLIP fires when V0@TP1 crosses the modal gate", ti["verdict_flip"] is True)
        # healthy at TP2 -> DEAD at TP1 (qlike crosses 4)
        ti = tp_invariance(comm, health(4.10, 45.0), 1)
        check("FLIP fires when V0@TP1 crosses the QLIKE gate", ti["verdict_flip"] is True)
        # dead at TP2 -> healthy at TP1
        comm_dead = dict(health(4.33, 57.6), tier="B", source="run_dir", tp=2,
                         comparable="strict", note="synthetic committed dead")
        ti = tp_invariance(comm_dead, health(3.2, 50.0), 1)
        check("FLIP fires when a committed-DEAD family is healthy at TP1",
              ti["verdict_flip"] is True)
        report_tp_invariance("selftest_flip", ti)   # must not raise, must not exit
        check("a flip prints a banner and CONTINUES (no abort)", True)
        ti = tp_invariance(None, health(3.2, 50.0), 1)
        check("no committed anchor -> recorded, no flip claimed",
              ti["status"] == "no_committed_anchor" and ti["verdict_flip"] is None)
        ti = tp_invariance(comm, health(3.66, 45.2), 2)
        check("bit-exact detection when the readings agree", ti["bitexact"] is True)

        # ---- 5. end-to-end mock pilot: json shape + TP recorded -------------
        print("5. end-to-end mock pilot (json shape, TP recorded)")
        run_pilot(A(tp=1))
        pj = json.loads(Path(PILOT_JSON).read_text())
        for k in ("schema", "prereg", "prereg_amendment", "gate", "selection_rule",
                  "tp_effective", "tp_committed", "tp_matches_committed", "hashes",
                  "families", "released_to_full", "all_families_dead",
                  "tp_verdict_flips", "branch_pending", "sidecar_docs_sha256",
                  "timestamp_utc"):
            check(f"pilot json carries {k!r}", k in pj)
        check("pilot json records the effective TP", pj["tp_effective"] == 1)
        check("pilot json records the committed TP", pj["tp_committed"] == 2)
        check("pilot json flags the TP mismatch",
              pj["tp_matches_committed"] is False)
        check("pilot json covers both families",
              set(pj["families"]) == set(FAMILIES))
        for fam in FAMILIES:
            d = pj["families"][fam]
            check(f"{fam}: all three variants scored",
                  set(d["per_variant"]) == set(VARIANTS))
            check(f"{fam}: selection recorded",
                  d["selection"]["selected_variant"] in VARIANTS)
            check(f"{fam}: healthy bool recorded",
                  isinstance(d["healthy"], bool))
            check(f"{fam}: tp_invariance recorded", "tp_invariance" in d)
            for v in VARIANTS:
                pvv = d["per_variant"][v]
                check(f"{fam}/{v}: health columns present",
                      all(c in pvv["per_h"]["5"] for c in HEALTH_COLS))
        for fam in FAMILIES:
            for v in VARIANTS:
                sh = json.loads(shard_of(fam, v).read_text())
                check(f"shard {fam}/{v} records the effective TP",
                      sh["tp_effective"] == 1 and sh["sampling"]["tp"] == 1)
        check("ragged horizons survive (h=20 has one doc fewer)",
              pj["families"]["mistral24"]["per_variant"]["V0"]["per_h"]["20"]["n"]
              == n_docs - 1)

        # G-E3 ACROSS subprocesses: a shard rendered from edited prompt text must not
        # be assembled with the others (the box runs one cell per subprocess, so this
        # is the only place a mid-run prompt edit can be caught).
        victim = shard_of("gemma27", "V2")
        good = victim.read_text()
        tam = json.loads(good)
        tam["hashes"]["V2"]["render_sha256"] = "0" * 64
        victim.write_text(json.dumps(tam))
        Path(PILOT_JSON).unlink()
        code = None
        try:
            assemble_pilot(A(tp=1, assemble=True))
        except SystemExit as e:
            code = e.code
        check("G-E3 aborts when one shard's frozen text disagrees (exit 1)", code == 1)
        victim.write_text(good)

        # every cell of one pilot must run at one TP
        tam = json.loads(good)
        tam["tp_effective"] = 2
        victim.write_text(json.dumps(tam))
        code = None
        try:
            assemble_pilot(A(tp=1, assemble=True))
        except SystemExit as e:
            code = e.code
        check("assemble aborts when shards disagree on the effective TP (exit 1)",
              code == 1)
        victim.write_text(good)

        # an incomplete grid cannot be selected over
        victim.unlink()
        code = None
        try:
            assemble_pilot(A(tp=1, assemble=True))
        except SystemExit as e:
            code = e.code
        check("assemble aborts on a partial 2x3 grid (exit 1)", code == 1)
        victim.write_text(good)
        assemble_pilot(A(tp=1, assemble=True))   # restore the pilot json for §6-7

        # ---- 6. single-shot guard ------------------------------------------
        print("6. single-shot / write-once guard")
        code = None
        try:
            assemble_pilot(A(tp=1, assemble=True))
        except SystemExit as e:
            code = e.code
        check("re-assembling refuses to overwrite the pilot json (exit 3)", code == 3)
        code = None
        try:
            run_pilot_cell(A(tp=1, family="mistral24", variant="V0"))
        except SystemExit as e:
            code = e.code
        check("re-running a cell refuses to overwrite its shard (exit 3)", code == 3)

        # ---- 7. --full graceful exits ---------------------------------------
        print("7. --full guards")
        code = None
        try:
            run_full(A(tp=1))
        except SystemExit as e:
            code = e.code
        # every synthetic family is healthy or dead depending on the mock; both paths are legal
        check("--full exits cleanly (staging refusal 4 / no-release 2), never crashes",
              code in (2, 4), f"exit={code}")
        if pj["released_to_full"]:
            check("--full's refusal is the staging gate (exit 4)", code == 4)

        # a pilot json with no released family -> exit 2
        moved = Path(PILOT_JSON + ".bak")
        shutil.copy(PILOT_JSON, moved)
        dead = json.loads(Path(PILOT_JSON).read_text())
        dead["released_to_full"] = []
        dead["all_families_dead"] = True
        Path(PILOT_JSON).write_text(json.dumps(dead))
        code = None
        try:
            run_full(A(tp=1))
        except SystemExit as e:
            code = e.code
        check("--full refuses when no family passed the gate (exit 2)", code == 2)
        shutil.copy(moved, PILOT_JSON)

        # --full without a pilot at all
        Path(PILOT_JSON).unlink()
        code = None
        try:
            run_full(A(tp=1))
        except SystemExit as e:
            code = e.code
        check("--full refuses without a pilot (exit 1)", code == 1)
        shutil.copy(moved, PILOT_JSON)

        # ---- 8. sidecar integrity ------------------------------------------
        print("8. sidecar integrity + strict-ge1 guard")
        d = json.loads(Path(SIDECAR).read_text())
        d["docs"][0]["filing_date"] = "1999-01-01"      # tamper
        Path(SIDECAR).write_text(json.dumps(d))
        code = None
        try:
            load_sidecar()
        except SystemExit as e:
            code = e.code
        check("a tampered sidecar fails its hash check (exit 1)", code == 1)
        write_fixtures(sandbox, docs)                   # restore

        code = None
        try:
            args = A(tp=1, assemble=True, strict_ge1=True)
            Path(PILOT_JSON).unlink()
            assemble_pilot(args)
        except SystemExit as e:
            code = e.code
        check("--strict-ge1 at TP!=committed refuses (bit-exactness is unavailable)",
              code == 1, f"exit={code}")

    finally:
        os.chdir(real_cwd)
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 78}")
    print(f"SELFTEST: {len(ok)} passed, {len(fail)} failed")
    if fail:
        for f in fail:
            print(f"  FAILED: {f}")
        print("=" * 78)
        sys.exit(1)
    print("=" * 78)
    return True


# ============================================================ cli
def _build_parser():
    p = argparse.ArgumentParser(
        description=f"M2 elicitation-fairness ({PREREG}). {PREREG_AMEND}")
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--emit-sidecar", action="store_true",
                   help="LOCAL: freeze the registered 2,000-doc pilot + committed "
                        "anchors into a small json to ship to the box")
    m.add_argument("--pilot", action="store_true",
                   help="val-only pilot: family x variant health (no test row)")
    m.add_argument("--full", action="store_true",
                   help="registered full pass (health-passed families only)")
    m.add_argument("--selftest", action="store_true", help="synthetic fixtures, sandboxed")
    m.add_argument("--show-variants", action="store_true",
                   help="print the exact rendered V1/V2 text + G-E3 hashes; writes nothing")
    p.add_argument("--tp", type=int, default=1,
                   help="tensor-parallel size (default 1: single A100-80GB). The "
                        "committed readings were made at TP=2; the effective TP is "
                        "recorded in every output.")
    p.add_argument("--family", choices=sorted(FAMILIES))
    p.add_argument("--variant", choices=VARIANTS)
    p.add_argument("--assemble", action="store_true",
                   help="--pilot: combine the six shards -> the registered pilot json")
    p.add_argument("--strict-ge1", action="store_true",
                   help="run G-E1 as the registered bit-exact gate (requires --tp 2)")
    p.add_argument("--sidecar", default=SIDECAR)
    p.add_argument("--text-cache", default=None)
    p.add_argument("--aligned", default=None)
    p.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    p.add_argument("--mock", action="store_true",
                   help="no-GPU mock generator (selftest/plumbing only)")
    p.add_argument("--truncate", type=int, default=0,
                   help="--show-variants: elide the middle of the few-shot block")
    return p


def main():
    args = _build_parser().parse_args()
    if args.selftest:
        return _selftest()
    if args.tp < 1:
        _abort("CLI", [f"--tp must be >= 1 (got {args.tp})"])
    if args.tp != TP_COMMITTED and not args.emit_sidecar and not args.show_variants:
        print(f"[TP] effective TP={args.tp}; the committed readings were made at "
              f"TP={TP_COMMITTED}. {PREREG_AMEND}")
    if args.emit_sidecar:
        return emit_sidecar(args)
    if args.show_variants:
        return show_variants(args)
    if args.pilot:
        return run_pilot(args)
    if args.full:
        return run_full(args)


if __name__ == "__main__":
    main()
