"""Prereg B2 (configs/prereg_residual_family_audit.md v1.3, tag prereg-rfa-v1.3) — FOURTH
cross-family probe: Gemma-3-27B-it (bf16, NO quantization), pilot health gate + full 3-seed
reproducibility-jitter ensemble on the 8-K (event_driven) channel.

Mechanism is the committed B1 machinery (scripts/analysis/crossfamily_mistral24.py — the
template this file adapts): same ED panel/excerpts, VERBATIM C6 prompt
(scripts/experiments/e1_llm_forecast/prompt.py, variant c6_text), same vLLM sampling stack
as the committed B1 runs (temperature=0 first pass / 0.2 retry, max_tokens 120, guided
JSON, clip [0.03,3.0], on_missing=rv22, max_model_len 8192, TP=2, checkpoint-every 500 —
scripts/experiments/row16_mistral24_ensemble/launch.sh), arithmetic 3-seed ensemble, the
same health columns and the same M1 log-space increment block, day-clustered DM.

ONLY the model changes: unsloth/gemma-3-27b-it (bf16 mirror of google/gemma-3-27b-it).
The model NAME is parameterised (--model-id) because the prereg registers an ordered
fallback (GLM-4-32B) that is advanced ONLY on hard failure (download unavailable or pilot
health fail) — never on results. When the fallback is used, pass --family-tag so outputs
are named after the actual model (prereg v1.3: "if downgraded, name after the actual model").

REGISTERED PILOT HEALTH GATE (prereg v1.3, must run BEFORE the full pass):
  exactly 2,000 event_driven VALIDATION-split documents — deterministic selection: the
  FIRST 2,000 by the panel's canonical sort (filing_time_utc, ticker, accession), which is
  the manifest's own storage order; single inference pass, seed 2026. The pilot computes
  ONLY the health columns and applies the registered formula VERBATIM (the Yi/Phi criteria
  of crossfamily_mistral24.py): healthy <=> max variance-unit QLIKE < 4 AND max modal
  share of round(pred,2) < 60%. It touches NO test rows (labels come from A2 VAL rows
  only) and computes NO increment statistics. Output:
  results/tables/crossfamily_<tag>_pilot.json + prints HEALTH_PASS / HEALTH_FAIL.

FULL MODE REFUSES to run unless the pilot json exists with healthy=true and a matching
model_id. There is deliberately NO --force-unhealthy / --skip-pilot override, and none
must ever be added: prereg v1.3 rules a pilot health failure "instrument-dead-at-pilot"
(branch (d)) — the full pass must not happen, the fallback model is the only next step.

MULTIPLICITY (registered delta from B1): prereg v1.3 registers "Holm(3) per reference" — Holm
within {3 horizons} PER REFERENCE (one Holm(3) family vs HAR, one vs HAR+firmID), and the
branch rules below consume these. B1-comparable pooled Holm(6) values are reported as
info-only columns (p_*_holm6), disclosed in the md; they are NOT decision-bearing.

REPLICATION DECISION RULE (prereg v1.3, quantified; ALL branches go into the paper);
health (same formula, on the FULL-run ensemble rows) is the precondition:
  (a) Holm-robust  <=> vs HAR+firmID, >=2/3 horizons clustered DM<0 AND Holm(3)<.05;
  (b) directional  <=> 3/3 horizons DM<0 vs HAR+firmID but <2 Holm(3)-significant;
  (c) no replication <=> healthy but neither (a) nor (b);
  (d) instrument-dead <=> health formula fails (pilot or full) — tabled per the Mistral
      precedent, NO inference drawn.

SANITY GATES (HARD RULE — any failure aborts before writing tables), verbatim from the
B1 template: G1'' (committed llama70 single+ens rows reproduce crossfamily_llama70_ens.csv
to machine precision on this code path), G1q (committed qwen ED rows reproduce
crossfamily_llm.csv), G5 (ensemble == row-wise ARITHMETIC mean of the three seed
predictions, rtol 1e-6, 1:1 merge; no "approximately holds"), G3'' (recomputed
variance-unit QLIKE matches each gemma run's stored metrics.json within 1e-3 relative).

OUTPUTS are WRITE-ONCE, single shot (prereg v1.3 "write-once single-shot"): the script refuses
to overwrite results/tables/crossfamily_<tag>_pilot.json and crossfamily_<tag>.{csv,md}.

GEMMA CHAT-TEMPLATE ADAPTATION (disclosed protocol delta, mirroring B1's disclosure of
the Mistral tokenizer caveat): Gemma has no system role. The committed C6 prompt is a
[system, user] pair; for Gemma the system message is folded VERBATIM into the user turn
as prefix + "\n\n" (the same convention Gemma's own chat template uses for system
content). The fold is recorded in each run's config.json and disclosed in the md.

BOX ENV (2xA100-40G; documented, auto-defaulted when /root/rivermind-data exists):
  export SP500VOL_DATA_ROOT=/root/rivermind-data/sp500vol-data
  export HF_HOME=/root/rivermind-data/hf
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # box has no HF egress: cache only
  (weights pre-fetched via mirror into $HF_HOME/hub/models--unsloth--gemma-3-27b-it)
  VLLM_WORKER_MULTIPROC_METHOD=spawn is set in-script (fork deadlocks with TP2).

RUN (on the box, from the repo root /root/rivermind-data/repo):
  /root/rivermind-data/venvs/main/bin/python scripts/analysis/crossfamily_gemma27.py --pilot
  /root/rivermind-data/venvs/main/bin/python scripts/analysis/crossfamily_gemma27.py   # full
LOCAL (no GPU):
  .venv/bin/python scripts/analysis/crossfamily_gemma27.py --dry-run
  .venv/bin/python scripts/analysis/crossfamily_gemma27.py --selftest
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # fork deadlocks w/ TP2
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
if os.path.isdir("/root/rivermind-data"):  # box defaults (launch.sh convention)
    os.environ.setdefault("SP500VOL_DATA_ROOT", "/root/rivermind-data/sp500vol-data")
    os.environ.setdefault("HF_HOME", "/root/rivermind-data/hf")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
sys.path.insert(0, str(_REPO / "scripts" / "experiments" / "e1_llm_forecast"))
import clustered_dm as cdm
import forecast_combination as fc
import postprocess as pp
import prompt as prompt_mod
import run_inference as ri

KEY = ["ticker", "accession", "horizon_days"]
EPS = 1e-8
HORIZONS = (5, 10, 20)
RTOL = 1e-12    # machine-precision gate for CSV float round-trip (G1''/G1q)
RTOL_G5 = 1e-6  # prereg row-wise ensemble identity tolerance (G5)
DISC = "event_driven"
VARIANT = "c6_text"          # the committed C6 prompt, verbatim (prompt.py)
SEEDS = ("2026", "2027", "2028")
PILOT_N = 2000               # registered pilot size (prereg v1.3)
PILOT_SEED = 2026
HEALTH_QLIKE_MAX = 4.0       # registered gate: max variance-unit QLIKE < 4
HEALTH_MODE_MAX = 60.0       # registered gate: max modal share (round(pred,2)) < 60%
# committed B1 sampling stack (row16 launch.sh / run_inference.py defaults) — FIXED,
# deliberately not CLI-exposed so the protocol cannot drift:
TP = 2
MAX_MODEL_LEN = 8192
MAX_TOKENS = 120
CHECKPOINT_EVERY = 500

DEFAULT_MODEL_ID = "unsloth/gemma-3-27b-it"  # bf16 mirror of google/gemma-3-27b-it
DEFAULT_TAG = "gemma27"
MANIFEST = "results/e1_llm_forecast/manifest_valtest.parquet"

LLA_RUNS = {  # committed llama70 anchors, recomputed for G1''
    "llama70_awq": "C6_llmtext_llama70_full_event_driven_seed2026",
    "llama70_awq_ens3": "C6_llmtext_llama70ens_full_event_driven_seed2026",
}
QWEN_RUN = "C6_llmtext_full_event_driven_seed2026"

M1_COLS = ["n_test", "n_days", "rel_har", "dm_har", "p_har",
           "rel_firm", "dm_firm", "p_firm", "g_text"]
DIAG_COLS = ["qlike_vol", "qlike_var", "r2", "pred_sd",
             "n_unique_2dp", "mode_val_2dp", "mode_share_pct"]

HEALTH_GATE_TEXT = ("healthy <=> max variance-unit QLIKE < 4 AND max modal share of "
                    "round(pred,2) < 60% (Yi/Phi criteria, verbatim the "
                    "crossfamily_mistral24.py / crossfamily_llama70.py health formula)")


# --------------------------------------------------------------- naming helpers
def rundir_of(tag, seed):
    sfx = f"_{tag}" if seed == "2026" else f"_{tag}_s{seed}"
    return f"results/runs/C6_llmtext{sfx}_full_{DISC}_seed2026"


def ensdir_of(tag):
    return f"results/runs/C6_llmtext_{tag}ens_full_{DISC}_seed2026"


def rawdir_of(tag, seed):
    sfx = f"_{tag}" if seed == "2026" else f"_{tag}_s{seed}"
    return Path(f"results/e1_llm_forecast/raw{sfx}")


def pilot_json_of(tag):
    return f"results/tables/crossfamily_{tag}_pilot.json"


def table_of(tag, ext):
    return f"results/tables/crossfamily_{tag}.{ext}"


def assert_writeonce(*paths):
    hit = [p for p in paths if Path(p).exists()]
    if hit:
        print(f"WRITE-ONCE guard (prereg v1.3 'single-shot'): output(s) already exist — "
              f"refusing to overwrite: {hit}\n"
              "If a rerun is genuinely intended, inspect and move the existing file(s) "
              "manually first; this script never overwrites its registered outputs.")
        sys.exit(3)


# --------------------------------------------------- verbatim B1 statistics block
def ols(y, X):  # verbatim from crossfamily_mistral24.py
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def holm(ps):  # verbatim from crossfamily_mistral24.py
    ps = np.asarray(ps, float)
    n = len(ps)
    order = np.argsort(ps)
    out = np.empty(n)
    for rank, idx in enumerate(order):
        out[idx] = ps[idx] * (n - rank)
    run = 0.0
    for idx in order:
        run = max(run, out[idx])
        out[idx] = min(run, 1.0)
    return out


def close(a, b):  # verbatim from crossfamily_mistral24.py
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) <= RTOL * max(abs(a), abs(b), 1.0)


def standalone_stats(y, f):  # verbatim from crossfamily_standalone.py
    y = np.asarray(y, float)
    f = np.asarray(f, float)
    vals, counts = np.unique(np.round(f, 2), return_counts=True)
    i = int(np.argmax(counts))
    return {
        "qlike_vol": float(fc.qlike(y, f).mean()),
        "qlike_var": float(fc.qlike(y ** 2, f ** 2).mean()),
        "r2": float(1.0 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "pred_sd": float(f.std()),
        "n_unique_2dp": len(vals),
        "mode_val_2dp": float(vals[i]),
        "mode_share_pct": float(100.0 * counts[i] / len(f)),
    }


def m1_rows(fam, preds, a2):
    """M1 block — verbatim from crossfamily_mistral24.py / crossfamily_llama70.py."""
    t = preds[KEY + ["prediction_realised_vol"]].rename(
        columns={"prediction_realised_vol": "ft"})
    rows = []
    for h in HORIZONS:
        m = a2[a2.horizon_days == h].merge(t[t.horizon_days == h], on=KEY).dropna()
        v, te = m[m.split == "val"], m[m.split == "test"]
        y = te.label_realised_vol.values
        fR, fU, g = fc.log_combo(v.label_realised_vol.values, v.fh.values,
                                 v.ft.values, te.fh.values, te.ft.values)
        qR, qU = fc.qlike(y, fR), fc.qlike(y, fU)
        rel = 100 * np.mean(qR - qU) / np.mean(qR)
        dm, pv, nd = cdm.dm_test_clustered(qU, qR, te.effective_trading_day.values, h)
        # firm-identity-augmented reference (val-window firm mean spec)
        fm = v.groupby("ticker").label_realised_vol.mean()
        gmean = v.label_realised_vol.mean()
        fid_v = v.ticker.map(fm).fillna(gmean).values
        fid_t = te.ticker.map(fm).fillna(gmean).values
        L = lambda x: np.log(np.clip(x, EPS, None))
        ly = L(v.label_realised_vol.values)
        bR = ols(ly, np.column_stack([np.ones(len(v)), L(v.fh.values), L(fid_v)]))
        bU = ols(ly, np.column_stack([np.ones(len(v)), L(v.fh.values), L(fid_v),
                                      L(v.ft.values)]))
        fRf = np.exp(bR[0] + bR[1] * L(te.fh.values) + bR[2] * L(fid_t))
        fUf = np.exp(bU[0] + bU[1] * L(te.fh.values) + bU[2] * L(fid_t)
                     + bU[3] * L(te.ft.values))
        qRf, qUf = fc.qlike(y, fRf), fc.qlike(y, fUf)
        relf = 100 * np.mean(qRf - qUf) / np.mean(qRf)
        dmf, pf, _ = cdm.dm_test_clustered(qUf, qRf, te.effective_trading_day.values, h)
        rows.append(dict(disc=DISC, family=fam, h=h, n_test=len(te), n_days=nd,
                         rel_har=rel, dm_har=dm, p_har=pv,
                         rel_firm=relf, dm_firm=dmf, p_firm=pf, g_text=g))
    return rows


def add_holm3(df, tag):
    """Registered B2 multiplicity (prereg v1.3: 'Holm(3) per reference'): Holm within the 3
    horizons PER REFERENCE. Pooled Holm(6) is added as B1-comparable INFO ONLY."""
    df = df.copy().sort_values("h").reset_index(drop=True)
    df["p_har_holm"] = holm(df.p_har.values)     # Holm(3), vs single recalibrated HAR
    df["p_firm_holm"] = holm(df.p_firm.values)   # Holm(3), vs HAR+firmID
    ps6 = holm(np.concatenate([df.p_har.values, df.p_firm.values]))
    df["p_har_holm6"] = ps6[:len(df)]
    df["p_firm_holm6"] = ps6[len(df):]
    df["holm_family"] = (f"{tag} Holm(3) per reference (prereg-rfa v1.3 B2); "
                         "p_*_holm6 = pooled Holm(6), B1-comparable info only")
    return df


def add_holm6(df):
    """B1 convention (used for the qwen anchor exactly as crossfamily_mistral24.py)."""
    ps6 = np.concatenate([df.p_har.values, df.p_firm.values])
    adj = holm(ps6)
    df = df.copy()
    df["p_har_holm"] = adj[:len(df)]
    df["p_firm_holm"] = adj[len(df):]
    return df


def diag_rows(fam, run, preds):
    """Standalone health diagnostics (TEST split) + metrics.json cross-check (G3'').
    Verbatim from crossfamily_mistral24.py."""
    mj = {(r["split"], r["horizon_days"]): r for r in json.load(
        open(f"results/runs/{run}/metrics.json"))}
    cfg = json.load(open(f"results/runs/{run}/config.json"))
    te_all = preds[preds.split == "test"]
    out = {}
    for h in HORIZONS:
        d = te_all[te_all.horizon_days == h]
        st = standalone_stats(d.label_realised_vol.to_numpy(),
                              d.prediction_realised_vol.to_numpy())
        stored = mj[("test", h)]["qlike"]
        st["qlike_var_metricsjson"] = float(stored)
        st["metrics_sanity"] = ("PASS" if abs(st["qlike_var"] - stored)
                                <= 1e-3 * max(abs(stored), 1.0) else "FAIL")
        st["parse_fail_rate"] = float(cfg["stats"].get("parse_fail_rate", np.nan))
        st["clipped_rate"] = float(cfg["stats"].get("clipped_rate", np.nan))
        out[(fam, h)] = st
    return out


def ladder(la, model_desc, qf_str):
    """Pre-registered verdict ladder — thresholds verbatim from the committed B1
    template (crossfamily_mistral24.py). For B2 the Holm columns it reads are the
    REGISTERED Holm(3)-per-reference values (prereg v1.3)."""
    la = la.sort_values("h")
    n_rep_firm = int(((la.dm_firm < 0) & (la.p_firm_holm < .05)).sum())
    n_rep_har = int(((la.dm_har < 0) & (la.p_har_holm < .05)).sum())
    n_pos_firm = int((la.rel_firm > 0).sum())
    n_neg_dm_firm = int((la.dm_firm < 0).sum())
    n_sig_raw_firm = int(((la.dm_firm < 0) & (la.p_firm < .05)).sum())
    _rf = "/".join(f"{x:+.2f}" for x in la.rel_firm)
    if n_rep_firm == 3:
        tier = "REPLICATES"
        verdict = (f"**REPLICATES.** The {model_desc} reproduces the "
                   "Qwen event-driven residual over the firm-identity-augmented "
                   "reference in 3/3 horizons (Holm<.05, day-clustered).")
    elif n_pos_firm == 3 and (n_sig_raw_firm >= 2 or n_rep_har >= 1):
        tier = "DIRECTIONALLY REPLICATES"
        verdict = (f"**DIRECTIONALLY REPLICATES, significance attenuated.** The "
                   f"{model_desc} reproduces the SIGN of the Qwen "
                   f"8-K residual vs HAR+firmID in 3/3 horizons ({_rf}% vs Qwen's "
                   f"{qf_str}%), clustered DM<0 in {n_neg_dm_firm}/3 and raw p<.05 in "
                   f"{n_sig_raw_firm}/3; but after the registered Holm only "
                   f"{n_rep_har}/3 vs-single-HAR cells survive (min firmID Holm "
                   f"p={la.p_firm_holm.min():.5f}, {n_rep_firm}/3 firmID cells <.05).")
    elif n_pos_firm == 0 and n_sig_raw_firm == 0:
        tier = "DOES NOT REPLICATE"
        verdict = (f"**Does NOT replicate.** The {model_desc} shows "
                   "no positive increment over the firm-identity-augmented reference "
                   "in any horizon.")
    else:
        tier = "PARTIAL/MIXED"
        verdict = (f"**PARTIAL/MIXED replication** ({n_rep_firm}/3 firm-ID cells "
                   f"Holm<.05, {n_sig_raw_firm}/3 raw p<.05, {n_pos_firm}/3 positive; "
                   f"{n_rep_har}/3 vs single recalibrated HAR after Holm).")
    counts = dict(n_rep_firm=n_rep_firm, n_rep_har=n_rep_har, n_pos_firm=n_pos_firm,
                  n_neg_dm_firm=n_neg_dm_firm, n_sig_raw_firm=n_sig_raw_firm)
    return tier, verdict, counts


def decide_branch(la, healthy):
    """Prereg v1.3 quantified replication branches (a)-(d). `la` = ensemble rows with
    the REGISTERED Holm(3)-per-reference columns; `healthy` = full-run health formula
    on those rows. Returns (branch_letter, fired_line, consequence_line)."""
    la = la.sort_values("h")
    n_a = int(((la.dm_firm < 0) & (la.p_firm_holm < .05)).sum())
    n_neg = int((la.dm_firm < 0).sum())
    if not healthy:
        return ("d",
                f"BRANCH (d) — instrument-dead: the full-run health formula FAILED "
                f"({HEALTH_GATE_TEXT}); max QLIKE(var)="
                f"{la.qlike_var.max():.2f}, max modal share={la.mode_share_pct.max():.1f}%.",
                "Consequence (registered): tabled per the Mistral precedent, NO "
                "inference drawn; the probe-denominator sentence updates and the "
                "pilot/full health failure is reported as-is in Stress Tests.")
    if n_a >= 2:
        return ("a",
                f"BRANCH (a) — Holm-robust replication: vs HAR+firmID, {n_a}/3 horizons "
                f"have clustered DM<0 AND Holm(3)<.05 (registered threshold >=2/3).",
                "Consequence (registered): residual wording upgrades to "
                "\"family-robust (two healthy families Holm-significant)\"; the abstract drops "
                "\"only partly family-robust\".")
    if n_neg == 3:
        return ("b",
                f"BRANCH (b) — directional replication: 3/3 horizons DM<0 vs HAR+firmID "
                f"but only {n_a}/3 Holm(3)-significant (<2).",
                "Consequence (registered): same tier as llama70 — \"three healthy probes same-sign\" "
                "enters the main text; the Holm-robustness wording is unchanged.")
    return ("c",
            f"BRANCH (c) — does not replicate (healthy instrument): neither (a) nor (b) "
            f"({n_a}/3 Holm(3)-sig, {n_neg}/3 DM<0 vs HAR+firmID).",
            "Consequence (registered): the residual is DOWNGRADED to "
            "\"Qwen-conditional\" in the abstract + sections 06 and 07 "
            "(a fix means a downgrade, committed to execute).")


# ------------------------------------------------------------ prompt / inference
def fold_system(messages):
    """Gemma chat-template adaptation (disclosed protocol delta): Gemma has no system
    role, so the committed C6 [system, user] pair is folded into a single user turn:
    system_content + "\\n\\n" + user_content — the identical convention Gemma's own
    template applies to system content. Byte-identical text otherwise."""
    if messages and messages[0].get("role") == "system":
        sysmsg = messages[0]["content"]
        rest = [dict(m) for m in messages[1:]]
        if rest and rest[0].get("role") == "user":
            rest[0]["content"] = sysmsg + "\n\n" + rest[0]["content"]
            return rest
    return messages


class FoldChatGenerator(ri.VllmGenerator):
    """VllmGenerator with the system->user fold applied at template time (Gemma).
    Everything else (sampling params, guided JSON, budget fitting) is inherited
    verbatim from the committed run_inference.VllmGenerator."""

    system_fold = True

    def _template(self, messages):
        return super()._template(fold_system(messages))


def needs_fold(model_id):
    return "gemma" in str(model_id).lower()


def resolve_model(model_id):
    """Resolve --model-id to a local path: (1) an existing directory wins; (2) the HF
    hub cache snapshot under $HF_HOME/hub (box convention: weights pre-fetched via
    mirror, HF_HUB_OFFLINE=1); (3) fall back to the raw id (only usable online)."""
    p = Path(model_id)
    if p.is_dir():
        return str(p), "local-dir"
    hub = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"
    cache = hub / ("models--" + model_id.replace("/", "--")) / "snapshots"
    if cache.is_dir():
        snaps = sorted([d for d in cache.iterdir()
                        if d.is_dir() and (d / "config.json").exists()])
        if snaps:
            return str(snaps[-1]), "hf-cache-snapshot"
    return model_id, "hub-id-unresolved"


def load_panel(split=None):
    m = pd.read_parquet(MANIFEST)
    m = m[m.disclosure == DISC]
    if split is not None:
        m = m[m.split == split]
    # canonical sort (the manifest's own storage order; re-applied deterministically)
    return m.sort_values(["filing_time_utc", "ticker", "accession"],
                         kind="mergesort").reset_index(drop=True)


def pilot_panel(n=PILOT_N):
    """Registered deterministic pilot selection: FIRST n event_driven VALIDATION
    filings by the canonical sort. No sampling, no RNG."""
    v = load_panel("val")
    if len(v) < n:
        print(f"FATAL: only {len(v)} ED validation filings < registered pilot n={n}")
        sys.exit(1)
    return v.head(n).copy()


def do_infer(panel, out_dir, gen, checkpoint_every=CHECKPOINT_EVERY):
    """Batch inference over `panel`, resumable via part-*.parquet (the committed
    run_inference.py machinery: stream_texts + build_messages + _flush retry pass)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = ri.load_done(out_dir)
    pending = [r for r in panel.to_dict("records")
               if (r["text_path"], VARIANT) not in done]
    print(f"[infer] {out_dir}: {len(panel)} filings, {len(pending)} pending "
          f"({len(done)} resumed)")
    if not pending:
        return
    texts = ri.stream_texts({r["text_path"] for r in pending})
    part_idx = len(list(out_dir.glob("part-*.parquet")))
    chunk, n_done = [], 0
    for row in pending:
        text = texts.get(row["text_path"])
        if text is None:
            continue
        _, src = prompt_mod.build_excerpt(row["form"], row.get("sections_json"), text)
        chunk.append({"row": row, "variant": VARIANT, "excerpt_source": src,
                      "messages": prompt_mod.build_messages(row, text, VARIANT)})
        if len(chunk) >= checkpoint_every:
            part_idx = ri._flush(gen, chunk, out_dir, part_idx)
            n_done += len(chunk)
            print(f"  [infer] {n_done}/{len(pending)} done")
            chunk = []
    if chunk:
        ri._flush(gen, chunk, out_dir, part_idx)
    print(f"[infer] {out_dir}: complete")


def build_run_dir(raw_dir, run_dir, model_label, extra_cfg):
    """Targeted event_driven postprocess for the c6_text raw outputs — the exact
    build-runs logic of postprocess.py (fill rv22, clip [0.03,3.0], stats, metrics),
    restricted to the ED disclosure so no junk combined/long_form dirs are created."""
    raw = pp.load_raw(str(raw_dir))
    rv = raw[raw.variant == VARIANT]
    a2 = pd.read_parquet(f"results/runs/A2_har_rv_full_{DISC}_seed2026/predictions.parquet")
    base = a2[a2["split"].isin(["val", "test"])].copy()
    base = base.merge(rv[["text_path", "vol_5d", "vol_10d", "vol_20d", "parse_ok"]],
                      on="text_path", how="inner")
    volmap = {5: "vol_5d", 10: "vol_10d", 20: "vol_20d"}
    pred = np.full(len(base), np.nan)
    for h, col in volmap.items():
        ix = base["horizon_days"] == h
        pred[ix.to_numpy()] = base.loc[ix, col].to_numpy()
    n_all, n_miss = len(base), int(np.isnan(pred).sum())
    pred = np.where(np.isnan(pred), base["feature_rv_22d"].to_numpy(), pred)  # rv22
    n_clip = int(((pred < pp.CLIP_LO) | (pred > pp.CLIP_HI)).sum())
    base["prediction_realised_vol"] = np.clip(pred, pp.CLIP_LO, pp.CLIP_HI)
    run_id = Path(run_dir).name
    base["run_id"], base["model_id"] = run_id, model_label
    base["dataset"], base["seed"] = "full", 2026
    base["disclosure_subset"] = DISC
    out = base[pp.PRED_COLS].reset_index(drop=True)
    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    out.to_parquet(rd / "predictions.parquet", index=False)
    (rd / "metrics.json").write_text(json.dumps(pp.metrics_rows(out, DISC), indent=2))
    cfg = {
        "model_id": model_label,
        "note": ("Prereg B2 fourth-family probe (prereg-rfa v1.3). Zero-shot C6 prompt "
                 "verbatim; protocol byte-identical to C6/llama70/mistral24 "
                 "(prompt cap 6000, guided JSON, clip [0.03,3.0], on_missing=rv22)."),
        "variant": VARIANT,
        "llm": str(rv["model_name"].iloc[0]),
        "prompt_cap_tokens": 6000,
        "clip_range": [pp.CLIP_LO, pp.CLIP_HI],
        "on_missing": "rv22",
        "stats": {
            "n_rows": len(out),
            "n_filings": int(out["text_path"].nunique()),
            "parse_fail_rows": n_miss,
            "parse_fail_rate": round(n_miss / n_all, 4) if n_all else float("nan"),
            "clipped_rows": n_clip,
            "clipped_rate": round(n_clip / max(n_all - n_miss, 1), 4),
        },
    }
    cfg.update(extra_cfg)
    (rd / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[build-run] wrote {rd}  rows={len(out)} "
          f"parse_fail={n_miss} clip={cfg['stats']['clipped_rate']:.4f}")
    return out


def build_ensemble(tag, model_id):
    """3-seed ensemble run dir — the row16 convention verbatim: per-observation
    ARITHMETIC mean of prediction_realised_vol, inner join 1:1 on KEY, clip.
    Skips if the ensemble run dir already exists (resumable orchestration)."""
    rd = Path(ensdir_of(tag))
    if (rd / "predictions.parquet").exists():
        print(f"[ensemble] {rd} already exists — skipping rebuild")
        return
    frames, base_df = [], None
    for s in SEEDS:
        p = Path(rundir_of(tag, s)) / "predictions.parquet"
        if not p.exists():
            print(f"FATAL: ensemble needs ALL 3 seed run dirs; missing {p}")
            sys.exit(1)
        d = pd.read_parquet(p)
        if s == "2026":
            base_df = d.copy()
        frames.append(d[KEY + ["prediction_realised_vol"]].rename(
            columns={"prediction_realised_vol": f"f{s}"}))
    ens = frames[0]
    for f in frames[1:]:
        ens = ens.merge(f, on=KEY, how="inner")
    ens["prediction_realised_vol"] = ens[[f"f{s}" for s in SEEDS]].mean(axis=1)
    ens["prediction_realised_vol"] = ens["prediction_realised_vol"].clip(
        pp.CLIP_LO, pp.CLIP_HI)
    out = base_df.drop(columns=["prediction_realised_vol"]).merge(
        ens[KEY + ["prediction_realised_vol"]], on=KEY, how="inner")
    model_id_ens = f"C6_llmtext_{tag}ens"
    out["model_id"], out["run_id"] = model_id_ens, rd.name
    out = out[pp.PRED_COLS].reset_index(drop=True)
    rd.mkdir(parents=True, exist_ok=True)
    out.to_parquet(rd / "predictions.parquet", index=False)
    (rd / "metrics.json").write_text(json.dumps(pp.metrics_rows(out, DISC), indent=2))
    (rd / "config.json").write_text(json.dumps({
        "model_id": model_id_ens,
        "note": ("3-seed ensemble of the prereg-B2 fourth family (bf16, no "
                 "quantization) 8-K forecaster. Per-observation ARITHMETIC MEAN of "
                 "prediction_realised_vol across vLLM seeds 2026+2027+2028, "
                 "inner-joined on (ticker,accession,horizon_days) — identical "
                 "convention to m1_ensemble_primary.ensemble_text, "
                 "C6_llmtext_llama70ens and C6_llmtext_mistral24ens. VAL+TEST only."),
        "llm": model_id,
        "seeds_used": list(SEEDS),
        "clip_range": [pp.CLIP_LO, pp.CLIP_HI],
        "stats": {"n_rows": len(out),
                  "n_filings": int(out["text_path"].nunique()),
                  "n_seeds": len(SEEDS)},
    }, indent=2))
    print(f"[ensemble] wrote {rd} rows={len(out)}")


# --------------------------------------------------------------------- pilot mode
def pilot_health(preds_long):
    """Health columns per horizon on VALIDATION rows only. `preds_long` columns:
    horizon_days, label_realised_vol, prediction_realised_vol (already filled+clipped).
    Returns (healthy, per_h dict)."""
    per_h = {}
    for h in HORIZONS:
        d = preds_long[preds_long.horizon_days == h]
        st = standalone_stats(d.label_realised_vol.to_numpy(),
                              d.prediction_realised_vol.to_numpy())
        st["n"] = len(d)
        per_h[str(h)] = st
    mx_q = max(v["qlike_var"] for v in per_h.values())
    mx_m = max(v["mode_share_pct"] for v in per_h.values())
    healthy = bool(mx_q < HEALTH_QLIKE_MAX and mx_m < HEALTH_MODE_MAX)
    return healthy, per_h, mx_q, mx_m


def run_pilot(args, gen=None, n=PILOT_N):
    """Registered pilot: 2,000 val docs, single pass seed 2026, health columns ONLY.
    Touches NO test rows; computes NO increment statistics."""
    tag = args.family_tag
    pj = pilot_json_of(tag)
    assert_writeonce(pj)
    panel = pilot_panel(n)
    raw_dir = Path(f"results/e1_llm_forecast/raw_{tag}_pilot")
    model_path, how = resolve_model(args.model_id)
    if gen is None:
        cls = FoldChatGenerator if needs_fold(args.model_id) else ri.VllmGenerator
        print(f"[pilot] model {args.model_id} -> {model_path} ({how}); "
              f"generator={cls.__name__} TP={TP}")
        gen = cls(model_path, MAX_MODEL_LEN, TP, MAX_TOKENS)
    do_infer(panel, raw_dir, gen)

    raw = pp.load_raw(str(raw_dir))
    rv = raw[raw.variant == VARIANT]
    # labels from A2 VALIDATION rows ONLY — the pilot never touches a test row
    a2 = pd.read_parquet(f"results/runs/A2_har_rv_full_{DISC}_seed2026/predictions.parquet")
    a2 = a2[a2["split"] == "val"]
    base = a2[a2["text_path"].isin(panel.text_path)].merge(
        rv[["text_path", "vol_5d", "vol_10d", "vol_20d"]], on="text_path", how="inner")
    volmap = {5: "vol_5d", 10: "vol_10d", 20: "vol_20d"}
    pred = np.full(len(base), np.nan)
    for h, col in volmap.items():
        ix = base["horizon_days"] == h
        pred[ix.to_numpy()] = base.loc[ix, col].to_numpy()
    n_all, n_miss = len(base), int(np.isnan(pred).sum())
    pred = np.where(np.isnan(pred), base["feature_rv_22d"].to_numpy(), pred)
    n_clip = int(((pred < pp.CLIP_LO) | (pred > pp.CLIP_HI)).sum())
    base["prediction_realised_vol"] = np.clip(pred, pp.CLIP_LO, pp.CLIP_HI)
    healthy, per_h, mx_q, mx_m = pilot_health(base)

    out = {
        "prereg": ("configs/prereg_residual_family_audit.md v1.3 "
                   "(tag prereg-rfa-v1.3), B2 pilot health gate"),
        "family_tag": tag,
        "model_id": args.model_id,
        "model_path_resolved": model_path,
        "healthy": healthy,
        "gate": HEALTH_GATE_TEXT,
        "max_qlike_var": mx_q,
        "max_mode_share_pct": mx_m,
        "n_docs": int(base.text_path.nunique()),
        "selection": ("deterministic: FIRST 2,000 event_driven VALIDATION filings by "
                      "the panel's canonical sort (filing_time_utc, ticker, accession);"
                      " single inference pass, seed 2026 (temperature-0 protocol; the "
                      "seed is not plumbed into vLLM's sampler)"),
        "seed": PILOT_SEED,
        "split": "val (no test rows read)",
        "no_increment_statistics": True,
        "per_h": per_h,
        "parse_fail_rate": round(n_miss / n_all, 4) if n_all else float("nan"),
        "clipped_rate": round(n_clip / max(n_all - n_miss, 1), 4),
        "sampling": {"temperature": 0.0, "retry_temperature": 0.2,
                     "max_tokens": MAX_TOKENS, "max_model_len": MAX_MODEL_LEN,
                     "tp": TP, "guided_json": True, "variant": VARIANT,
                     "system_fold": bool(getattr(gen, "system_fold", False))},
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    Path(pj).parent.mkdir(parents=True, exist_ok=True)
    Path(pj).write_text(json.dumps(out, indent=2))
    print(f"wrote {pj}")
    print(f"pilot health: max QLIKE(var)={mx_q:.3f} (<{HEALTH_QLIKE_MAX:g}), "
          f"max modal share={mx_m:.1f}% (<{HEALTH_MODE_MAX:g}%)")
    print("HEALTH_PASS" if healthy else "HEALTH_FAIL")
    if not healthy:
        print("Registered consequence: this model is instrument-dead-at-pilot "
              "(branch (d)); do NOT run full mode — advance the registered fallback "
              "(GLM-4-32B) with a fresh --family-tag and a fresh pilot.")
    return healthy


def check_pilot_gate(tag, model_id):
    """Full mode precondition. NO override exists (and none may be added)."""
    pj = Path(pilot_json_of(tag))
    if not pj.exists():
        print(f"REFUSING full run: pilot gate file {pj} not found. Run --pilot first "
              "(prereg v1.3: the registered pilot health gate precedes the full pass).")
        sys.exit(2)
    d = json.loads(pj.read_text())
    if d.get("model_id") != model_id:
        print(f"REFUSING full run: pilot json model_id={d.get('model_id')!r} does not "
              f"match --model-id {model_id!r} — pilot and full pass must be the same "
              "model (rerun --pilot for this model under its own --family-tag).")
        sys.exit(2)
    if d.get("healthy") is not True:
        print("REFUSING full run: pilot gate healthy=false — the model is "
              "instrument-dead-at-pilot (prereg v1.3 branch (d)). There is no "
              "override; advance the registered fallback model instead.")
        sys.exit(2)
    print(f"pilot gate OK: {pj} healthy=true "
          f"(max QLIKE(var)={d.get('max_qlike_var'):.3f}, "
          f"max modal share={d.get('max_mode_share_pct'):.1f}%)")
    return d


# ---------------------------------------------------------------------- scoring
def score_full(tag, model_id, pilot_info, anchor_pin=True, model_desc=None):
    """Verbatim B1 scoring protocol (adapted from crossfamily_mistral24.main), with the
    v1.3-registered Holm(3)-per-reference multiplicity and branches (a)-(d)."""
    csv_path, md_path = table_of(tag, "csv"), table_of(tag, "md")
    assert_writeonce(csv_path, md_path)
    model_desc = model_desc or f"3-seed {model_id} (bf16) ensemble"

    a2 = fc.load("A2_har_rv", DISC)[KEY + ["split", "label_realised_vol",
                                           "prediction_realised_vol",
                                           "effective_trading_day"]] \
        .rename(columns={"prediction_realised_vol": "fh"})

    # ---- G1'': reproduce committed crossfamily_llama70_ens.csv (single + ens3) ----
    ref_l = pd.read_csv("results/tables/crossfamily_llama70_ens.csv")
    recomp_l = {}
    for fam, run in LLA_RUNS.items():
        p = pd.read_parquet(f"results/runs/{run}/predictions.parquet")
        recomp_l[fam] = pd.DataFrame(m1_rows(fam, p, a2))
    g1_bad = []
    for _, r in ref_l.iterrows():
        mine = recomp_l[r.family]
        mine = mine[(mine.disc == r.disc) & (mine.h == r.h)]
        if len(mine) != 1:
            g1_bad.append((r.disc, r.family, int(r.h), "row missing"))
            continue
        for c in M1_COLS:
            if not close(mine[c].iloc[0], r[c]):
                g1_bad.append((r.disc, r.family, int(r.h), c,
                               float(mine[c].iloc[0]), float(r[c])))
    if g1_bad:
        print(f"SANITY G1'' FAIL ({len(g1_bad)} mismatches vs committed "
              f"crossfamily_llama70_ens.csv):")
        for b in g1_bad[:20]:
            print("  ", b)
        sys.exit(1)
    print(f"SANITY G1'' PASS: all {len(ref_l)} committed llama70 rows (single-seed AND "
          f"ens3) reproduced to machine precision (rtol {RTOL:g}) on columns {M1_COLS}")

    # ---- G1q: anchor the primary family's event_driven cells (crossfamily_llm.csv) ----
    ref_q = pd.read_csv("results/tables/crossfamily_llm.csv")
    ref_q = ref_q[(ref_q.family == "qwen3_32b") & (ref_q.disc == DISC)]
    pq = pd.read_parquet(f"results/runs/{QWEN_RUN}/predictions.parquet")
    qwen = pd.DataFrame(m1_rows("qwen3_32b", pq, a2))
    g1q_bad = []
    for _, r in ref_q.iterrows():
        mine = qwen[qwen.h == r.h]
        if len(mine) != 1:
            g1q_bad.append((r.disc, r.family, int(r.h), "row missing"))
            continue
        for c in M1_COLS:
            if not close(mine[c].iloc[0], r[c]):
                g1q_bad.append((r.disc, r.family, int(r.h), c,
                                float(mine[c].iloc[0]), float(r[c])))
    if g1q_bad:
        print(f"SANITY G1q FAIL ({len(g1q_bad)} mismatches vs committed "
              f"crossfamily_llm.csv qwen event_driven rows):")
        for b in g1q_bad[:20]:
            print("  ", b)
        sys.exit(1)
    print(f"SANITY G1q PASS: {len(ref_q)} committed qwen3_32b event_driven rows "
          f"reproduced to machine precision (rtol {RTOL:g})")

    # ---- G5: ensemble == arithmetic mean(seed predictions) row-wise, rtol 1e-6 ----
    pens = pd.read_parquet(f"{ensdir_of(tag)}/predictions.parquet")
    m = pens[KEY + ["prediction_realised_vol"]].rename(
        columns={"prediction_realised_vol": "f_ens"})
    n_in = {"ens": len(m)}
    seed_preds = {}
    for s in SEEDS:
        ps = pd.read_parquet(f"{rundir_of(tag, s)}/predictions.parquet")
        seed_preds[s] = ps
        ps = ps[KEY + ["prediction_realised_vol"]].rename(
            columns={"prediction_realised_vol": f"f_{s}"})
        n_in[s] = len(ps)
        m = m.merge(ps, on=KEY, validate="one_to_one")
    if not all(v == len(m) for v in n_in.values()):
        print(f"SANITY G5 FAIL: merge on {KEY} is not 1:1 across files "
              f"(input sizes {n_in}, merged {len(m)})")
        sys.exit(1)
    F = m[[f"f_{s}" for s in SEEDS]].to_numpy(float)
    am = F.mean(axis=1)
    fe = m.f_ens.to_numpy(float)
    reldev = np.abs(fe - am) / np.maximum(np.abs(am), 1e-300)
    bad = reldev > RTOL_G5
    if bad.any():
        m["arith_mean"], m["rel_dev"] = am, reldev
        worst = m[bad].sort_values("rel_dev", ascending=False)
        print("SANITY G5 FAIL: ensemble prediction != mean(seed predictions) "
              f"at rtol {RTOL_G5:g}")
        print(f"  rows failing: {int(bad.sum())} ({100.0 * bad.mean():.3f}%), "
              f"max relative deviation: {reldev.max():.6e}")
        cols = KEY + [f"f_{s}" for s in SEEDS] + ["f_ens", "arith_mean", "rel_dev"]
        print(worst[cols].head(8).to_string(index=False))
        print("\n  Per the prereg (G5), NOT proceeding on 'approximately holds'. "
              "Inspect the ensemble build step of THIS script and the seed run dirs.")
        sys.exit(1)
    print(f"SANITY G5 PASS: {tag} ensemble == arithmetic mean(seed preds) on all "
          f"{len(m)} rows (rtol {RTOL_G5:g}; max reldev {reldev.max():.3e}; "
          f"merge 1:1 verified)")

    # ---- M1 rows for the two new bases, REGISTERED Holm(3) per reference ----
    fam_single, fam_ens = f"{tag}_bf16", f"{tag}_ens3"
    p26 = seed_preds["2026"]
    single = add_holm3(pd.DataFrame(m1_rows(fam_single, p26, a2)), fam_single)
    ens = add_holm3(pd.DataFrame(m1_rows(fam_ens, pens, a2)), fam_ens)

    # ---- health diagnostics + G3'' ----
    diags = {}
    diags.update(diag_rows(fam_single, Path(rundir_of(tag, "2026")).name, p26))
    diags.update(diag_rows(fam_ens, Path(ensdir_of(tag)).name, pens))
    g3_bad = [(k, v["qlike_var"], v["qlike_var_metricsjson"])
              for k, v in diags.items() if v["metrics_sanity"] != "PASS"]
    if g3_bad:
        print("SANITY G3'' FAIL (variance-unit QLIKE vs metrics.json):")
        for b in g3_bad:
            print("  ", b)
        sys.exit(1)
    print(f"SANITY G3'' PASS: recomputed variance-unit QLIKE matches stored "
          f"metrics.json within 1e-3 relative in {len(diags)}/{len(diags)} {tag} cells")
    seed_pc = {}
    for s in SEEDS:
        st = json.load(open(f"{rundir_of(tag, s)}/config.json"))["stats"]
        seed_pc[s] = (float(st["parse_fail_rate"]), float(st["clipped_rate"]))
    for df_ in (single, ens):
        for c in DIAG_COLS + ["qlike_var_metricsjson", "metrics_sanity",
                              "parse_fail_rate", "clipped_rate"]:
            df_[c] = [diags[(df_.family.iloc[0], h)][c] for h in df_.h]
    single["flag"] = "bf16"
    ens["flag"] = "bf16-ens3"

    # ---- anchor rows: qwen primary + committed llama70 + committed mistral24 ----
    qwen_anchor = add_holm6(ref_q.copy())  # committed M1, verified == recompute (G1q)
    qwen_anchor["holm_family"] = ("qwen3_32b Holm(6): 3 horizons x {HAR, HAR+firmID}, "
                                  "event_driven (computed here on committed raw p's; "
                                  "B1 convention, unchanged)")
    std = pd.read_csv("results/tables/crossfamily_standalone.csv")
    std = std[(std.family == "qwen3_32b") & (std.disc == DISC)]
    qwen_anchor = qwen_anchor.merge(
        std[["h"] + DIAG_COLS + ["qlike_var_metricsjson", "metrics_sanity"]], on="h")
    qcfg = json.load(open(f"results/runs/{QWEN_RUN}/config.json"))["stats"]
    qwen_anchor["parse_fail_rate"] = float(qcfg["parse_fail_rate"])
    qwen_anchor["clipped_rate"] = float(qcfg["clipped_rate"])
    qwen_anchor["flag"] = "-"

    l70 = pd.read_csv("results/tables/crossfamily_llama70.csv")
    l70 = l70[(l70.family == "llama70_awq") & (l70.disc == DISC)]
    lla_anchor = ref_l.copy()  # committed M1 + committed Holm(6), carried unchanged
    lla_anchor = lla_anchor.merge(
        l70[["h"] + DIAG_COLS + ["qlike_var_metricsjson", "metrics_sanity",
                                 "parse_fail_rate", "clipped_rate"]],
        on="h", how="left")
    for c in DIAG_COLS + ["qlike_var_metricsjson", "parse_fail_rate", "clipped_rate"]:
        lla_anchor.loc[lla_anchor.family == "llama70_awq_ens3", c] = np.nan
    lla_anchor.loc[lla_anchor.family == "llama70_awq_ens3", "metrics_sanity"] = "-"
    lla_anchor["flag"] = np.where(lla_anchor.family == "llama70_awq_ens3",
                                  "AWQ-INT4-ens3", "AWQ-INT4")

    mis_anchor = pd.read_csv("results/tables/crossfamily_mistral24.csv")
    mis_anchor = mis_anchor[mis_anchor.family.isin(
        ["mistral24_bf16", "mistral24_ens3"])].copy()  # committed values, unchanged

    order_cols = ["disc", "family", "h", "n_test", "n_days",
                  "rel_har", "dm_har", "p_har", "rel_firm", "dm_firm", "p_firm",
                  "g_text", "p_har_holm", "p_firm_holm", "p_har_holm6",
                  "p_firm_holm6", "holm_family"] + DIAG_COLS + \
                 ["qlike_var_metricsjson", "metrics_sanity",
                  "parse_fail_rate", "clipped_rate", "flag"]
    df = pd.concat([qwen_anchor, lla_anchor, mis_anchor, single, ens],
                   ignore_index=True)
    for c in order_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[order_cols]

    # ---- verdict ladder + registered branches (a)-(d) ----
    la = ens.sort_values("h")
    qe = qwen_anchor.sort_values("h")
    _qf = "/".join(f"{x:+.2f}" for x in qe.rel_firm)
    tier_e, verdict, cm = ladder(la, model_desc, _qf)
    strong_e = cm["n_rep_har"] >= 2
    strong_line = (f"Family STRONG readout (B1 formula on the registered Holm(3) vs "
                   f"HAR: >=2/3 horizons Holm<.05 & DM<0): "
                   f"{'PASS' if strong_e else 'FAIL'} ({cm['n_rep_har']}/3)")
    tier_s, _, cs = ladder(single.sort_values("h"),
                           f"single-seed (2026) {model_id} bf16", _qf)
    healthy_full = bool(la.qlike_var.max() < HEALTH_QLIKE_MAX
                        and la.mode_share_pct.max() < HEALTH_MODE_MAX)
    branch, fired, consequence = decide_branch(la, healthy_full)

    if anchor_pin:  # committed-anchor pin (as in crossfamily_mistral24.py):
        # ladder on the committed ens rows, with their committed Holm(6) columns
        ref_le = ref_l[ref_l.family == "llama70_awq_ens3"].sort_values("h")
        tier_l, _, cl = ladder(ref_le, "3-seed Llama-3.1-70B (AWQ-INT4) ensemble", _qf)
        assert cl["n_rep_har"] == 1 and tier_l == "DIRECTIONALLY REPLICATES", \
            f"llama70_ens3 anchor drifted: n_rep_har={cl['n_rep_har']}, tier={tier_l}"

    def m1cell(r):
        s1 = "**" if (r.dm_har < 0 and r.p_har < .05) else ""
        s2 = "**" if (r.dm_firm < 0 and r.p_firm < .05) else ""
        return (f"{r.rel_har:+.2f}%{s1} | {r.dm_har:+.2f} | "
                f"{r.rel_firm:+.2f}%{s2} | {r.dm_firm:+.2f}")

    md = [
        f"# Prereg B2 (prereg-rfa v1.3) — fourth cross-family probe: {model_id} "
        "(bf16), 3-seed ensemble, 8-K channel",
        "",
        "## Disclosures",
        "",
        f"- **Model / precision**: {model_id}, **bf16, NO quantization** (matched-class "
        "to Qwen3-32B; also removes the 70B arm's int4 confound), vLLM TP=2 on "
        "2xA100-40G. Weights resolved OFFLINE from the local HF cache (mirror-"
        "prefetched; box has no HF egress). Protocol otherwise byte-identical to "
        "C6/llama70/mistral24: same manifest/prompt/guided-JSON/clip[0.03,3.0]/"
        "retry stack (scripts/experiments/e1_llm_forecast).",
        "- **Chat-template adaptation (disclosed protocol delta, cf. B1's Mistral "
        "tokenizer caveat)**: Gemma's chat template has NO system role. The committed "
        "C6 prompt is a [system, user] pair; here the system message is folded "
        "VERBATIM into the user turn as prefix + a blank line (the same convention "
        "Gemma's own template applies to system content). Text byte-identical "
        "otherwise; the fold is recorded in each run's config.json "
        f"(system_fold={pilot_info.get('sampling', {}).get('system_fold', 'n/a')}) "
        "and is internally consistent across pilot, all three seeds and the ensemble, "
        "so it cannot differentiate seeds or the cross-family comparison.",
        "- **Seed semantics**: temperature-0 decoding; `--seed` is NOT plumbed into "
        "vLLM's sampler — seeds 2026/2027/2028 differ only through vLLM/TP2 kernel "
        "non-determinism. This is a **reproducibility-jitter ensemble**, not a "
        "stochastic-decoding one (identical to the llama70/mistral24 arms).",
        "- **Registered pilot gate (prereg v1.3, ran BEFORE the full pass)**: "
        f"{pilot_info.get('n_docs', 'n/a')} validation documents "
        "(deterministic FIRST-2,000 by canonical sort, seed 2026, single pass), "
        f"health formula verbatim; result healthy={pilot_info.get('healthy')}, "
        f"max QLIKE(var)={pilot_info.get('max_qlike_var', float('nan')):.3f}, "
        f"max modal share={pilot_info.get('max_mode_share_pct', float('nan')):.1f}% "
        f"({pilot_json_of(tag)}). The pilot touched NO test rows and computed NO "
        "increment statistics.",
        "- **Multiplicity (registered v1.3 delta from B1)**: Holm is applied within "
        "the 3 horizons PER REFERENCE — one Holm(3) family vs the single recalibrated "
        "HAR and one vs HAR+firmID (prereg v1.3: \"Holm(3) per reference\"); the branch rules "
        "below consume these. Pooled Holm(6) values (B1's convention) are reported as "
        "info-only columns p_*_holm6 and are NOT decision-bearing. Anchor rows carry "
        "their own committed Holm(6) values unchanged; families are parallel, never "
        "pooled.",
        "- **Ensemble construction**: per-observation ARITHMETIC mean of "
        "prediction_realised_vol across the three seeds, inner-joined 1:1 on "
        "(ticker, accession, horizon_days), VAL+TEST only — identical convention to "
        "C6_llmtext_llama70ens / C6_llmtext_mistral24ens; verified row-wise by "
        f"sanity gate G5 (rtol {RTOL_G5:g}).",
        f"- parse/clip stats per seed config.json (parse_fail_rate, clipped_rate): "
        f"{seed_pc}; the ens run dir carries no parse stats (its predictions are "
        "means of already-parsed seed forecasts).",
        "- The single-seed rows are retained side by side; the ensemble rows are the "
        "primary basis (prereg v1.3: ensemble as primary basis, single-seed as robustness).",
        "",
        "## Table — M1 increment (log-space, combiner val-fit test-frozen, "
        "day-clustered DM) + standalone health",
        "",
        "rel% is on **volatility-unit** QLIKE (committed-anchor convention); the "
        "QLIKE(var) health column is **variance-unit** (Patton-robust). rel% > 0 = "
        "text lowers QLIKE vs the reference; `**` = clustered DM<0, raw p<.05. "
        f"For the {tag} rows 'Holm p' = the REGISTERED Holm(3) per reference; anchor "
        "rows carry their committed Holm(6).",
        "",
        "| family | h | n_test | rel% vs HAR | DM(clu) | rel% vs HAR+firmID | DM(clu) "
        "| Holm p (HAR) | Holm p (firmID) | QLIKE(var) | R^2 | pred sd | n_uniq(2dp) "
        "| mode share% | parse_ok% | flag |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|",
    ]
    for _, r in df.iterrows():
        qv = f"{r.qlike_var:.3f}" if pd.notna(r.qlike_var) else "-"
        r2 = f"{r.r2:+.3f}" if pd.notna(r.r2) else "-"
        sd = f"{r.pred_sd:.4f}" if pd.notna(r.pred_sd) else "-"
        nu = f"{int(r.n_unique_2dp)}" if pd.notna(r.n_unique_2dp) else "-"
        ms = f"{r.mode_share_pct:.1f}" if pd.notna(r.mode_share_pct) else "-"
        po = (f"{100 * (1 - r.parse_fail_rate):.1f}"
              if pd.notna(r.parse_fail_rate) else "-")
        md.append(f"| {r.family} | {int(r.h)} | {int(r.n_test)} | {m1cell(r)} | "
                  f"{r.p_har_holm:.4g} | {r.p_firm_holm:.4g} | {qv} | {r2} | {sd} | "
                  f"{nu} | {ms} | {po} | {r.flag} |")
    health_line = (
        f"Health check (full run, same registered formula): {fam_ens} is "
        f"{'HEALTHY' if healthy_full else 'NOT healthy — instrument-dead (branch d)'}: "
        f"variance-unit QLIKE {la.qlike_var.min():.2f}-{la.qlike_var.max():.2f} "
        f"(gate <{HEALTH_QLIKE_MAX:g}; Qwen {qe.qlike_var.min():.2f}-"
        f"{qe.qlike_var.max():.2f}, Yi 7.60-8.19 = capability floor), max modal share "
        f"{la.mode_share_pct.max():.1f}% (gate <{HEALTH_MODE_MAX:g}%; Yi 73.6%, "
        f"Mistral-24B 89.5-92.9% = the instrument-dead precedent), pred sd "
        f"{la.pred_sd.min():.3f}-{la.pred_sd.max():.3f}, R^2 "
        f"{la.r2.min():+.2f}-{la.r2.max():+.2f}.")
    md += [
        "",
        "## VERDICT (pre-registered ladder, ensemble rows)" if healthy_full else
        "## VERDICT (ladder shown for completeness — NON-INFERENTIAL, health failed)",
        "",
        verdict,
        "",
        f"- {strong_line}",
        f"- Info (not a branch input) — single-seed {fam_single} ladder: {tier_s} "
        f"({cs['n_rep_firm']}/3 firmID Holm<.05, {cs['n_sig_raw_firm']}/3 raw firmID "
        f"p<.05, {cs['n_pos_firm']}/3 rel_firm>0, {cs['n_rep_har']}/3 vs HAR after "
        "Holm).",
        f"- {health_line}",
        "",
        "## REPLICATION DECISION (prereg v1.3 §B2, quoted verbatim)",
        "",
        "> **Replication decision rule (pre-declared, per ml's verbatim request; ALL branches go into the paper)**: health (full run, same formula) is the precondition;",
        "> - **(a) Holm-robust replication** ⇔ vs the firm-identity reference, ≥2/3 horizons satisfy "
        "clustered DM<0 AND Holm(3)<.05 → residual wording upgrades to \"family-robust (two healthy "
        "families Holm-significant)\", the abstract drops \"only partly family-robust\";",
        "> - **(b) directional replication** ⇔ 3/3 DM<0 but <2 Holm-passing → same tier as llama70, "
        "\"three healthy probes same-sign\" enters the main text, Holm-robustness wording unchanged;",
        "> - **(c) no replication** ⇔ healthy but neither (a) nor (b) → residual downgraded in abstract+06+07 to "
        "\"Qwen-conditional\" (a fix means a downgrade, committed to execute);",
        "> - **(d) instrument-dead** ⇔ health formula fails (pilot or full) → tabled per the Mistral precedent, "
        "no inference drawn.",
        "",
        f"**FIRED: {fired}**",
        "",
        consequence,
        "",
        "## SANITY",
        "",
        f"- G1'' PASS: all {len(ref_l)} committed crossfamily_llama70_ens.csv rows "
        f"reproduced to machine precision (rtol {RTOL:g}) on columns {M1_COLS}.",
        f"- G1q PASS: the {len(ref_q)} committed qwen3_32b event_driven M1 rows "
        f"(crossfamily_llm.csv) reproduced to machine precision (rtol {RTOL:g}).",
        f"- G5 PASS: {tag} ensemble == row-wise ARITHMETIC mean of the three seed "
        f"predictions on all {len(m)} rows (rtol {RTOL_G5:g}; max relative deviation "
        f"{reldev.max():.3e}); merge on {KEY} verified 1:1.",
        f"- G3'' PASS: recomputed variance-unit QLIKE matches stored metrics.json "
        f"within 1e-3 relative in {len(diags)}/{len(diags)} {tag} cells "
        "(single + ens).",
        "- Committed anchors (qwen / llama70 / mistral24) carried unchanged; the "
        "llama70_ens3 anchor pin (n_rep_har=1, DIRECTIONALLY REPLICATES) verified "
        f"in-script: {'yes' if anchor_pin else 'SKIPPED (selftest fixtures)'}.",
        "",
    ]

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    Path(md_path).write_text("\n".join(md))
    print(f"wrote {csv_path} / {md_path}")
    print(df[["family", "h", "n_test", "rel_har", "dm_har", "p_har", "p_har_holm",
              "rel_firm", "dm_firm", "p_firm", "p_firm_holm"]].to_string(index=False))
    print("\nVERDICT:", verdict.replace("**", ""))
    print(strong_line)
    print(health_line)
    print("FIRED:", fired)
    print(consequence)
    return branch, tier_e


# ----------------------------------------------------------------- orchestration
def run_full(args):
    tag = args.family_tag
    assert_writeonce(table_of(tag, "csv"), table_of(tag, "md"))
    pilot_info = check_pilot_gate(tag, args.model_id)
    # anchor inputs must exist BEFORE any GPU time is spent
    needed = ["results/tables/crossfamily_llama70_ens.csv",
              "results/tables/crossfamily_llm.csv",
              "results/tables/crossfamily_standalone.csv",
              "results/tables/crossfamily_llama70.csv",
              "results/tables/crossfamily_mistral24.csv",
              MANIFEST,
              f"results/runs/A2_har_rv_full_{DISC}_seed2026/predictions.parquet",
              f"results/runs/{QWEN_RUN}/predictions.parquet"] + \
             [f"results/runs/{r}/predictions.parquet" for r in LLA_RUNS.values()]
    missing = [p for p in needed if not Path(p).exists()]
    if missing:
        print("FATAL: required committed inputs missing:", missing)
        sys.exit(1)

    for seed in SEEDS:
        rd = Path(rundir_of(tag, seed))
        if (rd / "predictions.parquet").exists():
            print(f"[full] seed {seed}: {rd} already built — skipping")
            continue
        # one subprocess per seed: vLLM/TP2 memory is fully released between seeds
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--stage-infer-seed", seed,
               "--model-id", args.model_id, "--family-tag", tag]
        print(f"[full] seed {seed}: launching inference subprocess: {' '.join(cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0 or not (rd / "predictions.parquet").exists():
            print(f"FATAL: seed {seed} inference/build failed (rc={rc})")
            sys.exit(1)
    build_ensemble(tag, args.model_id)
    score_full(tag, args.model_id, pilot_info)


def stage_infer_seed(args):
    """Internal: one seed's inference + targeted ED postprocess (own process so the
    TP2 vLLM instance is torn down cleanly before the next seed starts)."""
    tag, seed = args.family_tag, args.stage_infer_seed
    check_pilot_gate(tag, args.model_id)  # defense in depth
    panel = load_panel()  # ED val+test, the committed 39,322-filing panel
    model_path, how = resolve_model(args.model_id)
    cls = FoldChatGenerator if needs_fold(args.model_id) else ri.VllmGenerator
    print(f"[seed {seed}] model {args.model_id} -> {model_path} ({how}); "
          f"generator={cls.__name__} TP={TP} max_model_len={MAX_MODEL_LEN}")
    gen = cls(model_path, MAX_MODEL_LEN, TP, MAX_TOKENS)
    do_infer(panel, rawdir_of(tag, seed), gen)
    build_run_dir(rawdir_of(tag, seed), rundir_of(tag, seed),
                  f"C6_llmtext_{tag}" + ("" if seed == "2026" else f"_s{seed}"),
                  {"seed_label": seed,
                   "prereg": "prereg-rfa v1.3 B2",
                   "system_fold": bool(getattr(gen, "system_fold", False)),
                   "chat_template_note": (
                       "Gemma has no system role: committed C6 system message folded "
                       "verbatim into the user turn (disclosed protocol delta)"
                       if getattr(gen, "system_fold", False) else
                       "native chat template, system message passed through"),
                   "sampling": {"temperature": 0.0, "retry_temperature": 0.2,
                                "max_tokens": MAX_TOKENS,
                                "max_model_len": MAX_MODEL_LEN, "tp": TP,
                                "guided_json": True}})


# ---------------------------------------------------------------------- dry run
def dry_run(args):
    ok = True

    def chk(cond, msg):
        nonlocal ok
        print(("  OK   " if cond else "  FAIL ") + msg)
        ok = ok and bool(cond)

    tag = args.family_tag
    print(f"DRY-RUN (no GPU, no writes) — family_tag={tag} model={args.model_id}")
    print("[env]")
    for k in ("SP500VOL_DATA_ROOT", "HF_HOME", "HF_HUB_OFFLINE",
              "TRANSFORMERS_OFFLINE", "VLLM_WORKER_MULTIPROC_METHOD"):
        print(f"       {k}={os.environ.get(k)}")
    print("[panel]")
    chk(Path(MANIFEST).exists(), f"manifest {MANIFEST}")
    if Path(MANIFEST).exists():
        panel = load_panel()
        chk(len(panel) > 0, f"ED panel: {len(panel)} filings "
            f"(committed panel = 39,322), splits "
            f"{panel.split.value_counts().to_dict()}")
        pil = load_panel("val").head(PILOT_N)
        chk(len(pil) == PILOT_N,
            f"pilot selection: first {PILOT_N} val filings by canonical sort "
            f"(first={pil.iloc[0].ticker}/{pil.iloc[0].accession}, "
            f"last={pil.iloc[-1].ticker}/{pil.iloc[-1].accession})")
        row = panel.iloc[0].to_dict()
        msgs = prompt_mod.build_messages(row, "«dry-run placeholder text»", VARIANT)
        chk(msgs[0]["role"] == "system" and msgs[1]["role"] == "user",
            "committed C6 prompt builds ([system, user])")
        if needs_fold(args.model_id):
            fm = fold_system(msgs)
            chk(len(fm) == 1 and fm[0]["role"] == "user"
                and fm[0]["content"].startswith(prompt_mod.SYSTEM_PROMPT + "\n\n"),
                "gemma system->user fold verified (verbatim prefix + blank line)")
    print("[data]")
    chk(Path(f"results/runs/A2_har_rv_full_{DISC}_seed2026/predictions.parquet")
        .exists(), "A2 event_driven run (labels + fh)")
    chk(ri.TEXT_CACHE.exists(), f"text cache {ri.TEXT_CACHE}")
    print("[committed anchors — needed by full mode]")
    for pth in ("results/tables/crossfamily_llama70_ens.csv",
                "results/tables/crossfamily_llm.csv",
                "results/tables/crossfamily_standalone.csv",
                "results/tables/crossfamily_llama70.csv",
                "results/tables/crossfamily_mistral24.csv"):
        chk(Path(pth).exists(), pth)
    for r in [QWEN_RUN] + list(LLA_RUNS.values()):
        chk(Path(f"results/runs/{r}/predictions.parquet").exists(),
            f"results/runs/{r}")
    print("[model resolution]")
    model_path, how = resolve_model(args.model_id)
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    resolved = how != "hub-id-unresolved"
    print(f"       {args.model_id} -> {model_path} ({how})")
    if offline:
        chk(resolved, "model resolvable under HF_HUB_OFFLINE=1")
    elif not resolved:
        print("  WARN model not in local HF cache (fine off-box; the box must have "
              "the mirror-prefetched snapshot)")
    print("[write-once state]")
    for pth in (pilot_json_of(tag), table_of(tag, "csv"), table_of(tag, "md")):
        print(f"       {'EXISTS (write-once guard would refuse)' if Path(pth).exists() else 'absent (ok)':<45} {pth}")
    pj = Path(pilot_json_of(tag))
    if pj.exists():
        d = json.loads(pj.read_text())
        print(f"       pilot gate: healthy={d.get('healthy')} "
              f"model_id={d.get('model_id')}")
    else:
        print("       pilot gate: not yet run — full mode would refuse")
    print("DRY-RUN " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------- selftest
def _selftest():
    import shutil
    import tempfile
    rng = np.random.default_rng(7)
    sandbox = Path(tempfile.mkdtemp(prefix="gemma27_selftest_"))
    print(f"[selftest] sandbox: {sandbox}")
    cwd0 = os.getcwd()
    failures = []

    def check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    # ---------- synthetic repo fixture ----------
    def synth_panel():
        tickers = [f"T{i:02d}" for i in range(40)]
        dv = pd.date_range("2023-01-02", periods=80, freq="B", tz="UTC")
        dt_ = pd.date_range("2023-06-01", periods=150, freq="B", tz="UTC")
        rows, k = [], 0
        for split, days in (("val", dv), ("test", dt_)):
            for d in days:
                for t in rng.choice(tickers, size=10, replace=False):
                    rows.append((t, d, split, f"A{k:07d}"))
                    k += 1
        base = pd.DataFrame(rows, columns=["ticker", "effective_trading_day",
                                           "split", "accession"])
        firm = {t: rng.normal(0, .25) for t in tickers}
        dayshock = {d: rng.normal(0, .15) for d in [*dv, *dt_]}
        base["text_path"] = base.ticker + "_" + base.accession
        base["filing_time_utc"] = base.effective_trading_day
        out = []
        for h in HORIZONS:
            b = base.copy()
            b["horizon_days"] = h
            idio = rng.normal(0, .30, len(b))
            ly = (np.log(0.25) + b.ticker.map(firm).values
                  + b.effective_trading_day.map(dayshock).values + idio)
            b["label_realised_vol"] = np.exp(ly)
            b["_ly"], b["_idio"] = ly, idio
            b["fh"] = np.exp(np.log(0.25) * 1.4
                             + 0.6 * (ly - np.log(0.25) - idio)
                             + rng.normal(0, .2, len(b)))
            out.append(b)
        return pd.concat(out, ignore_index=True)

    pan = synth_panel()

    def preds_strong(pan, jitter=0.0):
        f = np.exp(pan._ly - 0.5 * pan._idio + rng.normal(0, .10, len(pan))
                   + (rng.normal(0, jitter, len(pan)) if jitter else 0.0))
        return np.clip(f, pp.CLIP_LO, pp.CLIP_HI)

    def preds_null(pan):
        return np.clip(np.exp(np.log(0.25) + rng.normal(0, .3, len(pan))),
                       pp.CLIP_LO, pp.CLIP_HI)

    def preds_collapsed(pan):
        return np.full(len(pan), 0.25)

    def write_run(run_id, pan, f, stats=None):
        rd = Path(f"results/runs/{run_id}")
        rd.mkdir(parents=True, exist_ok=True)
        d = pan[["ticker", "accession", "horizon_days", "split",
                 "label_realised_vol", "effective_trading_day", "text_path",
                 "filing_time_utc"]].copy()
        d["prediction_realised_vol"] = np.asarray(f, float)
        d["feature_rv_22d"] = pan.label_realised_vol * np.exp(
            rng.normal(0, .1, len(pan)))
        d["run_id"], d["model_id"] = run_id, run_id
        d["dataset"], d["seed"], d["disclosure_subset"] = "full", 2026, DISC
        d["form"], d["item_subtype"], d["metadata_path"] = "8-K", "2.02", ""
        d["feature_rv_1d"] = d["feature_rv_5d"] = 0.25
        d.to_parquet(rd / "predictions.parquet", index=False)
        (rd / "metrics.json").write_text(json.dumps(pp.metrics_rows(d, DISC),
                                                    indent=2))
        (rd / "config.json").write_text(json.dumps(
            {"model_id": run_id, "stats": stats or
             {"parse_fail_rate": 0.0, "clipped_rate": 0.0}}, indent=2))
        return d

    def build_fixture(tag, gemma_kind="strong"):
        """Create the full synthetic repo tree in cwd (already chdir'ed)."""
        Path("results/tables").mkdir(parents=True, exist_ok=True)
        a2df = write_run(f"A2_har_rv_full_{DISC}_seed2026", pan, pan.fh.values)
        a2 = a2df.rename(columns={"prediction_realised_vol": "fh"})[
            KEY + ["split", "label_realised_vol", "fh", "effective_trading_day"]]
        # qwen anchor + committed csvs, generated THROUGH the same code path
        pq = write_run(QWEN_RUN, pan, preds_strong(pan))
        qrows = pd.DataFrame(m1_rows("qwen3_32b", pq, a2))
        qrows.insert(0, "disc", qrows.pop("disc"))
        qrows.to_csv("results/tables/crossfamily_llm.csv", index=False)
        qstd = []
        for h in HORIZONS:
            d = pq[(pq.split == "test") & (pq.horizon_days == h)]
            st = standalone_stats(d.label_realised_vol.to_numpy(),
                                  d.prediction_realised_vol.to_numpy())
            st.update(disc=DISC, family="qwen3_32b", h=h,
                      qlike_var_metricsjson=st["qlike_var"], metrics_sanity="PASS")
            qstd.append(st)
        pd.DataFrame(qstd).to_csv("results/tables/crossfamily_standalone.csv",
                                  index=False)
        # llama70 anchors: single + ens3 (ens = arithmetic mean of 3 jittered seeds)
        fseeds = [preds_strong(pan, jitter=0.02) for _ in range(3)]
        p70 = write_run(LLA_RUNS["llama70_awq"], pan, fseeds[0])
        fens = np.mean(np.column_stack(fseeds), axis=1)
        p70e = write_run(LLA_RUNS["llama70_awq_ens3"], pan, fens)
        r70 = pd.DataFrame(m1_rows("llama70_awq", p70, a2))
        r70e = pd.DataFrame(m1_rows("llama70_awq_ens3", p70e, a2))
        both = []
        for rr in (r70, r70e):
            rr = add_holm6(rr)
            rr["holm_family"] = "fixture Holm(6)"
            both.append(rr)
        pd.concat(both, ignore_index=True).to_csv(
            "results/tables/crossfamily_llama70_ens.csv", index=False)
        l70std = []
        for h in HORIZONS:
            d = p70[(p70.split == "test") & (p70.horizon_days == h)]
            st = standalone_stats(d.label_realised_vol.to_numpy(),
                                  d.prediction_realised_vol.to_numpy())
            st.update(disc=DISC, family="llama70_awq", h=h,
                      qlike_var_metricsjson=st["qlike_var"], metrics_sanity="PASS",
                      parse_fail_rate=0.0, clipped_rate=0.0)
            l70std.append(st)
        r70x = r70.merge(pd.DataFrame(l70std)[
            ["h"] + DIAG_COLS + ["qlike_var_metricsjson", "metrics_sanity",
                                 "parse_fail_rate", "clipped_rate"]], on="h")
        r70x.to_csv("results/tables/crossfamily_llama70.csv", index=False)
        # mistral24 committed anchor (carried unchanged; minimal but full columns)
        mis = pd.DataFrame(m1_rows("mistral24_ens3", write_run(
            "C6_llmtext_mistral24ens_fixture", pan, preds_null(pan)), a2))
        mis = add_holm6(mis)
        mis["holm_family"] = "fixture mistral Holm(6)"
        for c in DIAG_COLS + ["qlike_var_metricsjson"]:
            mis[c] = np.nan
        mis["metrics_sanity"] = "-"
        mis["parse_fail_rate"], mis["clipped_rate"] = 0.0, 0.0
        mis["flag"] = "bf16-ens3"
        mis.to_csv("results/tables/crossfamily_mistral24.csv", index=False)
        # gemma seed run dirs (pre-placed => full mode skips inference)
        kinds = {"strong": preds_strong, "null": preds_null,
                 "collapsed": preds_collapsed}
        for s in SEEDS:
            f = (kinds[gemma_kind](pan) if gemma_kind != "strong"
                 else preds_strong(pan, jitter=0.02))
            write_run(Path(rundir_of(tag, s)).name, pan, f)
        # healthy pilot json for the gate
        Path(pilot_json_of(tag)).write_text(json.dumps(
            {"healthy": True, "model_id": "fixture/model", "family_tag": tag,
             "n_docs": PILOT_N, "max_qlike_var": 0.5, "max_mode_share_pct": 5.0,
             "sampling": {"system_fold": True}}))
        return a2

    # ---------- unit tests: fold, branches, ladder, holm3, pilot selection ----------
    msgs = prompt_mod.build_messages(
        {"form": "8-K", "item_subtype": "2.02", "filing_date": "2024-01-05",
         "ticker": "ZZZ", "sections_json": None}, "BODY", VARIANT)
    fm = fold_system(msgs)
    check("fold_system folds [system,user] -> [user] with verbatim prefix",
          len(fm) == 1 and fm[0]["role"] == "user"
          and fm[0]["content"] == msgs[0]["content"] + "\n\n" + msgs[1]["content"]
          and msgs[0]["role"] == "system")  # original untouched
    check("needs_fold: gemma yes / glm no",
          needs_fold("unsloth/gemma-3-27b-it") and not needs_fold("zai-org/GLM-4-32B"))

    def mk(dm_firm, p_firm_holm, qv=1.0, ms=10.0):
        return pd.DataFrame({
            "h": [5, 10, 20], "dm_firm": dm_firm, "p_firm_holm": p_firm_holm,
            "dm_har": dm_firm, "p_har_holm": p_firm_holm, "p_har": p_firm_holm,
            "p_firm": p_firm_holm, "rel_firm": [-x for x in dm_firm],
            "rel_har": [-x for x in dm_firm],
            "qlike_var": [qv] * 3, "mode_share_pct": [ms] * 3})

    b, _, _ = decide_branch(mk([-3, -3, -1], [.01, .02, .30]), True)
    check("branch (a): 2/3 Holm(3)-sig & DM<0", b == "a")
    b, _, _ = decide_branch(mk([-3, -2, -1], [.06, .20, .30]), True)
    check("branch (b): 3/3 DM<0, <2 Holm-sig", b == "b")
    b, _, _ = decide_branch(mk([-3, 1, -1], [.06, .20, .30]), True)
    check("branch (c): healthy, neither (a) nor (b)", b == "c")
    b, _, _ = decide_branch(mk([-3, -3, -3], [.01, .01, .01]), False)
    check("branch (d): health fail dominates even a Holm-sig pattern", b == "d")

    t, _, _ = ladder(mk([-3, -3, -3], [.01, .01, .01]), "x", "q")
    check("ladder REPLICATES", t == "REPLICATES")
    la_dir = mk([-3, -3, -1], [.30, .30, .30])
    la_dir["p_firm"] = [.01, .01, .30]
    t, _, _ = ladder(la_dir, "x", "q")
    check("ladder DIRECTIONALLY REPLICATES", t == "DIRECTIONALLY REPLICATES")
    la_no = mk([1, 2, 3], [.9, .9, .9])
    t, _, _ = ladder(la_no, "x", "q")
    check("ladder DOES NOT REPLICATE", t == "DOES NOT REPLICATE")
    t, _, _ = ladder(mk([-1, 2, 3], [.9, .9, .9]), "x", "q")
    check("ladder PARTIAL/MIXED", t == "PARTIAL/MIXED")

    hh = add_holm3(pd.DataFrame({
        "h": [5, 10, 20], "p_har": [.01, .04, .5], "p_firm": [.02, .03, .9]}), "t")
    check("add_holm3: per-reference families of 3 (max adj = 3x raw)",
          np.isclose(hh.p_har_holm.iloc[0], .03)
          and np.isclose(hh.p_firm_holm.iloc[0], .06)
          and "p_har_holm6" in hh.columns)

    # pilot health formula
    y = np.exp(rng.normal(np.log(.25), .3, 3000))
    ok_h, _, _, _ = pilot_health(pd.DataFrame({
        "horizon_days": np.repeat([5, 10, 20], 1000),
        "label_realised_vol": y,
        "prediction_realised_vol": y * np.exp(rng.normal(0, .1, 3000))}))
    bad_h, _, mxq, mxm = pilot_health(pd.DataFrame({
        "horizon_days": np.repeat([5, 10, 20], 1000),
        "label_realised_vol": y,
        "prediction_realised_vol": np.full(3000, .25)}))
    check("pilot health: accurate forecaster HEALTH_PASS", ok_h)
    check("pilot health: constant forecaster HEALTH_FAIL via modal share",
          (not bad_h) and mxm == 100.0)

    # ---------- end-to-end in sandbox ----------
    try:
        os.chdir(sandbox)
        Path("results/e1_llm_forecast").mkdir(parents=True)
        tag = "gemmaX"
        build_fixture(tag, "strong")
        build_ensemble(tag, "fixture/model")
        pilot_info = check_pilot_gate(tag, "fixture/model")
        br, tier = score_full(tag, "fixture/model", pilot_info, anchor_pin=False,
                              model_desc="3-seed fixture ensemble")
        check("e2e strong: table + md written",
              Path(table_of(tag, "csv")).exists()
              and Path(table_of(tag, "md")).exists())
        check("e2e strong: branch (a) fired", br == "a")
        got = pd.read_csv(table_of(tag, "csv"))
        check("e2e strong: table carries qwen/llama70/mistral24 anchors + 2 new "
              "families",
              set(got.family) >= {"qwen3_32b", "llama70_awq", "llama70_awq_ens3",
                                  "mistral24_ens3", f"{tag}_bf16", f"{tag}_ens3"})
        md_txt = Path(table_of(tag, "md")).read_text()
        check("e2e strong: md discloses fold + Holm(3) + pilot gate",
              "folded" in md_txt and "Holm(3)" in md_txt and "pilot" in md_txt.lower())

        # write-once guard
        try:
            score_full(tag, "fixture/model", pilot_info, anchor_pin=False)
            check("write-once guard trips on second full write", False)
        except SystemExit as e:
            check("write-once guard trips on second full write", e.code == 3)
        try:
            run_pilot(argparse.Namespace(family_tag=tag, model_id="fixture/model"),
                      gen=None, n=10)
            check("write-once guard trips on existing pilot json", False)
        except SystemExit as e:
            check("write-once guard trips on existing pilot json", e.code == 3)

        # pilot gating refusals
        try:
            check_pilot_gate("nope", "fixture/model")
            check("full refuses without pilot json", False)
        except SystemExit as e:
            check("full refuses without pilot json", e.code == 2)
        Path(pilot_json_of("sick")).write_text(json.dumps(
            {"healthy": False, "model_id": "fixture/model"}))
        try:
            check_pilot_gate("sick", "fixture/model")
            check("full refuses on healthy=false (no override exists)", False)
        except SystemExit as e:
            check("full refuses on healthy=false (no override exists)", e.code == 2)
        try:
            check_pilot_gate(tag, "other/model")
            check("full refuses on pilot/full model mismatch", False)
        except SystemExit as e:
            check("full refuses on pilot/full model mismatch", e.code == 2)
        check("no --force-unhealthy style override is implemented",
              not any("force" in a or "skip-pilot" in a
                      for a in _build_parser()._option_string_actions))

        # G5 tamper: perturb one ensemble prediction and rescore (fresh outputs)
        ep = Path(ensdir_of(tag)) / "predictions.parquet"
        ed = pd.read_parquet(ep)
        ed.loc[ed.index[0], "prediction_realised_vol"] *= 1.01
        ed.to_parquet(ep, index=False)
        for f in (table_of(tag, "csv"), table_of(tag, "md")):
            Path(f).unlink()
        try:
            score_full(tag, "fixture/model", pilot_info, anchor_pin=False)
            check("G5 catches a tampered ensemble", False)
        except SystemExit as e:
            check("G5 catches a tampered ensemble", e.code == 1)
        # restore ens, then tamper the committed anchor csv -> G1'' must fail
        ep.unlink()
        build_ensemble(tag, "fixture/model")
        ref = pd.read_csv("results/tables/crossfamily_llama70_ens.csv")
        ref.loc[0, "rel_har"] += 1e-6
        ref.to_csv("results/tables/crossfamily_llama70_ens.csv", index=False)
        try:
            score_full(tag, "fixture/model", pilot_info, anchor_pin=False)
            check("G1'' catches a drifted committed anchor", False)
        except SystemExit as e:
            check("G1'' catches a drifted committed anchor", e.code == 1)
        ref.loc[0, "rel_har"] -= 1e-6
        ref.to_csv("results/tables/crossfamily_llama70_ens.csv", index=False)

        # unhealthy full run -> branch (d), still tabled, non-inferential
        tag2 = "gemmaY"
        for s in SEEDS:
            write_run(Path(rundir_of(tag2, s)).name, pan, preds_collapsed(pan))
        Path(pilot_json_of(tag2)).write_text(json.dumps(
            {"healthy": True, "model_id": "fixture/model", "family_tag": tag2,
             "n_docs": PILOT_N, "max_qlike_var": 0.5, "max_mode_share_pct": 5.0,
             "sampling": {"system_fold": True}}))
        build_ensemble(tag2, "fixture/model")
        pi2 = check_pilot_gate(tag2, "fixture/model")
        br2, _ = score_full(tag2, "fixture/model", pi2, anchor_pin=False,
                            model_desc="3-seed collapsed fixture ensemble")
        md2 = Path(table_of(tag2, "md")).read_text()
        check("e2e collapsed: branch (d) instrument-dead fired, still tabled",
              br2 == "d" and "instrument-dead" in md2
              and Path(table_of(tag2, "csv")).exists())
        check("e2e collapsed: verdict marked NON-INFERENTIAL", "NON-INFERENTIAL" in md2)

        # mock-generator pilot end-to-end (inference loop + selection determinism)
        man = pan[pan.split == "val"].drop_duplicates("text_path").copy()
        man["disclosure"], man["form"] = DISC, "8-K"
        man["item_subtype"], man["sections_json"] = "2.02", None
        man["filing_date"] = man.filing_time_utc.dt.date.astype(str)
        man["metadata_path"], man["token_count"] = "", 100
        man["feature_rv_1d"] = man["feature_rv_5d"] = 0.25
        man["feature_rv_22d"] = man.label_realised_vol
        man = man.sample(frac=1.0, random_state=0)  # shuffle: selection must re-sort
        man.to_parquet(MANIFEST, index=False)
        sel1 = pilot_panel(50)
        sel2 = pilot_panel(50)
        srt = man.sort_values(["filing_time_utc", "ticker", "accession"],
                              kind="mergesort")
        check("pilot selection: deterministic first-N by canonical sort "
              "(order-independent)",
              sel1.text_path.tolist() == sel2.text_path.tolist()
              == srt.head(50).text_path.tolist())
        tc = Path("results/_texts.parquet")
        pd.DataFrame({"text_path": man.text_path,
                      "text": "synthetic 8-K body"}).to_parquet(tc, index=False)
        old_tc = ri.TEXT_CACHE
        ri.TEXT_CACHE = tc
        try:
            ok_pilot = run_pilot(
                argparse.Namespace(family_tag="gemmaM", model_id="fixture/model"),
                gen=ri.MockGenerator(seed=2026), n=200)
            pj = json.loads(Path(pilot_json_of("gemmaM")).read_text())
            check("mock pilot: HEALTH_PASS + json schema",
                  ok_pilot and pj["healthy"] and pj["n_docs"] == 200
                  and set(pj["per_h"]) == {"5", "10", "20"}
                  and "selection" in pj and pj["split"].startswith("val"))

            class _ConstGen:
                name = "const"
                system_fold = False

                def generate(self, records, retry=False):
                    return [json.dumps({"vol_5d": .25, "vol_10d": .25,
                                        "vol_20d": .25})] * len(records)

            ok_bad = run_pilot(
                argparse.Namespace(family_tag="gemmaC", model_id="fixture/model"),
                gen=_ConstGen(), n=200)
            pj2 = json.loads(Path(pilot_json_of("gemmaC")).read_text())
            check("mock pilot: collapsed generator -> HEALTH_FAIL json",
                  (not ok_bad) and pj2["healthy"] is False
                  and pj2["max_mode_share_pct"] == 100.0)
        finally:
            ri.TEXT_CACHE = old_tc
    finally:
        os.chdir(cwd0)
        shutil.rmtree(sandbox, ignore_errors=True)

    print(f"\nSELFTEST {'PASS' if not failures else 'FAIL'} "
          f"({len(failures)} failure(s))")
    for f in failures:
        print("  FAILED:", f)
    sys.exit(0 if not failures else 1)


# -------------------------------------------------------------------------- cli
def _build_parser():
    ap = argparse.ArgumentParser(
        description="Prereg B2 (prereg-rfa v1.3): Gemma-3-27B fourth-family probe — "
                    "pilot health gate + full 3-seed ensemble. NO unhealthy override "
                    "exists by design.")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                    help="parameterised for the registered fallback (GLM-4-32B), "
                         "advanced only on hard failure — never on results")
    ap.add_argument("--family-tag", default=DEFAULT_TAG,
                    help="output naming; use the actual model's tag on fallback "
                         "(prereg v1.3: if downgraded, name after the actual model)")
    ap.add_argument("--pilot", action="store_true",
                    help="registered pilot: 2,000 val docs, health columns only")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate panel/prompt/model-path resolution; no GPU")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic-fixture selftest in a throwaway sandbox; no GPU")
    ap.add_argument("--stage-infer-seed", choices=SEEDS, default=None,
                    help=argparse.SUPPRESS)  # internal: one seed per subprocess
    return ap


def main():
    args = _build_parser().parse_args()
    if args.selftest:
        _selftest()
    elif args.dry_run:
        dry_run(args)
    elif args.pilot:
        run_pilot(args)
    elif args.stage_infer_seed:
        stage_infer_seed(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
