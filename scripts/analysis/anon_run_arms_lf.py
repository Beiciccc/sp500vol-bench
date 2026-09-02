"""C-anon LONG-FORM stretch (prereg-ea v1.4, configs/prereg_swap_lf_and_anon.md
§C-anon: "long-form as stretch, only when ED completes") — B2-ONLY. Run the single executed
long-form arm on the MASKED long-form (10-K/10-Q) inputs, only AFTER the ED
arms complete (the registered stretch condition).

EXECUTED ARM (run-dir conventions match the ED arms so the scorer
—anon_score.py --channel lf— treats it like any other run):

  --arm b2   B2-lf. TF-IDF + Ridge (configs/models/B2_tfidf_ridge.yaml), the
             EXACT scripts/train.py flow on the long_form panel — machinery
             reused VERBATIM from anon_run_arms.run_train_arm with the module
             re-scoped to long_form (masked RETRAIN, mirroring the ED B2 arm).
             CPU. Masked -> results/runs/B2_tfidf_ridge_anonmask_full_
             long_form_seed2026; --control retrains on UNMASKED text and
             compares against the committed B2_tfidf_ridge_full_long_form_
             seed2026 (the committed model.pkl is ABSENT, so the control IS
             the reproduction run) -> results/anon/g1_control_b2_lf.json.
             The json records exact-match rate + max|diff| as always, but the
             SCORER judges it by the v1.4 CPU gate: max|diff| <= 1e-8
             full-panel reproduction, NO deviation path (--record-g1-deviation
             is ED/GPU-arm-only; above tolerance the arm exits as G1-fail).

NOT RUN — all four decided BEFORE any LF statistic; disclosed at scoring
(anon_score.py --channel lf writes C5-lf into the table as not-executed rows):

  C5-lf — excluded pre-statistic (prereg-ea v1.4), DUAL-TRACK:
          HAR side = CONSTRUCTIVELY n/a — primary citation committed
          m1_multiseed.csv long_form C5_qwen3 seed-2026 rel_impr_pct =
          -1.0347/-3.1346/-6.6467 (h=5/10/20; the same single-seed basis the
          scorer uses), corroborated by deployable_combiner FIXED mean rel% =
          -0.85/-2.48/-5.97 (3-seed basis); all negative, so every HAR share
          cell is empty before execution under the registered n/a rule.
          firmID side = NO committed table carries that increment, so
          constructive n/a is NOT invoked there — excluded on GPU budget/
          scope (as C6-lf), and its execution could not change the branch
          adjudication (the share median takes HAR cells only). The ~2-3
          GPU-h masked pass buys no estimand. The --arm c5 implementation
          below is RETAINED (not deleted) for the record and for a possible
          prereg re-amendment; an invocation prints a not-executed notice
          first.
  C2-lf — two fixed-recipe FinBERT trainings (~20-30 GPU-h) for marginal
          defensive value; the ED C2 arm already covers the fine-tuned
          lineage (and the E-lf C2 arm is registered artefact-lost).
  C6-lf — the committed UNMASKED C6 long-form run EXISTS and is genuine
          (11,907/11,907 long-form val+test filings, real generations, not
          rv22 fill), so a masked C6-lf arm is well-defined; excluded on GPU
          budget/scope (another Qwen3-32B bf16 TP=2 block), NOT impossibility.
  (consequence) — with C5-lf out, NO neural text model is trained or run in
          this channel (B2 is a CPU classical model); the fine-tuned,
          frozen-embedding and prompted-LLM lineages are covered by the ED
          channel only.

RETAINED (not-executed) --arm c5 mechanics, for the record: committed head
ARTEFACT-LOST -> deterministic recipe rebuild + 1e-8 reproduction gate
(swap_longform_infer._prepare_c5_rebuild verbatim; E-lf staged heads adopted
when complete); --control = full-panel predict on ORIGINAL text + G1 vs
committed; default = masked INFERENCE (no masked retrain) on val+test with a
sibling masked embedding cache + patched text source (details below).

MASKED-INPUT MECHANICS (why this file exists instead of a flag on the ED
runner): C5 embeddings are disk-cached KEYED BY text_path and its text loader
streams the ORIGINAL text-cache parquet directly (qwen_llm._missing_texts),
bypassing the classical-text store injection. Serving masked text under the
original cache key would silently reuse ORIGINAL embeddings (a no-op "masked"
run). So for the masked pass this runner (a) repoints the embedding cache to a
SIBLING file <orig>__anonmask_lf.parquet — the original cache, warm from the
E-lf one-pass encode, stays byte-untouched and keeps serving the rebuild gate —
and (b) replaces qwen_llm._missing_texts with a coverage-asserted masked-store
reader. Both patches are applied ONLY AFTER the rebuild/reproduction gate has
run on the original machinery. Placeholder presence in the served text is
asserted (belt-and-braces against the silent-unmasked failure mode).

Box usage (export SP500VOL_DATA_ROOT=/data/sp500vol-data first; after
anon_mask_build.py --panel lf and after the ED arms):
  python scripts/analysis/anon_run_arms_lf.py --arm b2 --control        # CPU
  python scripts/analysis/anon_run_arms_lf.py --arm b2                  # CPU
  (--arm c5 is registered NOT-EXECUTED — do not run it on the box unless a
   prereg re-amendment reinstates the arm; an invocation prints the notice.)
Local (no GPU, nothing heavy):
  --selftest             synthetic masked-encode/cache-separation validation
  --arm b2 --selftest    store-injection + coverage check on the real panel
  --arm c5 --dry-run     preflight only (artefact/cache/store coverage)
"""
from __future__ import annotations

# thread caps BEFORE numpy/torch (box overrides by exporting its own values)
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from sp500vol.utils.paths import data_path  # noqa: E402

SEED = 2026
DISC = "long_form"
SPLITS = ("val", "test")
ANON_DIR = REPO / "results" / "anon"
COMMITTED = {
    "b2": REPO / "results" / "runs" / f"B2_tfidf_ridge_full_{DISC}_seed{SEED}",
    "c5": REPO / "results" / "runs" / f"C5_qwen3_full_{DISC}_seed{SEED}",
}
ELF_C5_DIR = REPO / "results" / "runs" / f"ELF_swap_C5_qwen3_full_{DISC}_seed{SEED}"
C5_ARM = "C5_qwen3"
HORIZONS = (5, 10, 20)


def _fatal(msg: str) -> None:
    raise SystemExit(f"[anon_lf] FATAL: {msg}")


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_masked_store() -> Path:
    return Path(data_path("processed", "_text_cache",
                          "filing_texts_anonmask_lf.parquet"))


# Reused machinery, imported by path (single source of truth):
#   anon_run_arms       — load_masked_store / rekey_store / inject_masked_store /
#                         g1_compare / run_train_arm (the ED b2/c2 train flow)
#   swap_longform_infer — prepare_arm (incl. _prepare_c5_rebuild, the v1.2
#                         rebuild-as-reproduction gate), predict_arm (resumable
#                         per-(split,horizon) parts), load_full_long_form_panel
_ED = None
_SLI = None


def ed_arms():
    global _ED
    if _ED is None:
        _ED = _load_by_path("anon_run_arms_for_lf",
                            REPO / "scripts" / "analysis" / "anon_run_arms.py")
    return _ED


def sli():
    global _SLI
    if _SLI is None:
        _SLI = _load_by_path("swap_longform_infer_for_lf",
                             REPO / "scripts" / "analysis" / "swap_longform_infer.py")
    return _SLI


def _placeholder_count(texts) -> int:
    return sum(t.count("[FIRM]") + t.count("[TICKER]") + t.count("[PERSON]")
               + t.count("[CIK]") + t.count("[PRODUCT]") for t in texts)


# ------------------------------------------------------------------ arm: b2
def run_b2(args) -> None:
    """B2-lf via the ED runner's run_train_arm, re-scoped to long_form.

    The module-global surgery below is the whole point of the reuse: DISC and
    the committed reference are switched to the long-form panel, and g1_compare
    is wrapped so the verdict lands in g1_control_b2_lf.json (never clobbering
    the ED g1_control_b2.json). Everything else — the train.py flow, the store
    injection, the coverage assert, the selftest — runs verbatim.
    """
    ed = ed_arms()
    ed.DISC = DISC
    ed.COMMITTED["b2"] = COMMITTED["b2"]
    orig_g1 = ed.g1_compare
    ed.g1_compare = lambda arm, new_run, committed: orig_g1(
        f"{arm}_lf", new_run, committed)

    ed.run_train_arm(args)

    if args.selftest or args.smoke:
        return
    # channel provenance stamp on the run dir the ED flow just wrote
    tag = "anonctrl" if args.control else "anonmask"
    run_dir = REPO / "results" / "runs" / \
        f"B2_tfidf_ridge_{tag}_full_{DISC}_seed{SEED}"
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        cfg.setdefault("anon", {})["channel"] = "long_form_stretch"
        cfg["anon"]["prereg"] = "prereg-ea v1.1 §C-anon (long_form stretch)"
        cfg_path.write_text(json.dumps(cfg, indent=2, default=str))


# ------------------------------------------------------------------ arm: c5
def _c5_staging_home(args) -> Path:
    """Rebuilt-head staging dir. Prefer the E-lf run's COMPLETE staged set so
    C-anon-lf provably uses the SAME reconstructed heads (the fingerprint check
    + re-run of the reproduction gate re-verify them either way); a partial
    E-lf set is never adopted (_prepare_c5_rebuild would wipe it)."""
    if args.c5_staging:
        return Path(args.c5_staging)
    elf_heads = ELF_C5_DIR / "rebuilt_heads"
    if len(list(elf_heads.glob("horizon_*.pt"))) == len(HORIZONS):
        print(f"[c5] adopting the E-lf complete staged head set: {elf_heads}")
        return ELF_C5_DIR
    return ANON_DIR / "c5_lf"


def _masked_cache_path(orig_cache: Path | None) -> Path:
    if orig_cache is None:
        _fatal("C5 was built with cache_embeddings=False — the committed "
               "convention requires the cache (its digest pins the encode "
               "dtype); refuse to guess.")
    return orig_cache.with_name(orig_cache.stem + "__anonmask_lf.parquet")


def _patch_masked_text_source(masked: dict[str, str]) -> None:
    """Replace qwen_llm._missing_texts (the ORIGINAL-cache parquet streamer)
    with a coverage-asserted masked-store reader. embed_dataframe resolves the
    name at qwen_llm module scope, so this patch covers every encode path."""
    import sp500vol.models.neural_text.qwen_llm as qwen_llm

    def masked_missing_texts(df, missing):
        out = {tp: masked[tp] for tp in missing if tp in masked}
        if len(out) != len(set(missing)):
            gaps = sorted(set(missing) - out.keys())
            _fatal(f"masked store missing {len(gaps)} requested docs "
                   f"(first: {gaps[0]}) — refuse to fall back to original text.")
        return out

    qwen_llm._missing_texts = masked_missing_texts
    print(f"[c5] text source patched: qwen_llm._missing_texts now serves the "
          f"masked store ({len(masked)} docs), coverage-asserted")


def _chunked_masked_encode(model, uniq_paths: list[str], chunk: int) -> None:
    """Encode masked docs `chunk` at a time; embed_dataframe persists the
    masked cache after every call, so a crash resumes at chunk granularity."""
    import sp500vol.models.neural_text.qwen_llm as qwen_llm

    cache = model._cache_path()
    store = qwen_llm._load_emb_store(cache) if cache is not None else {}
    missing = [tp for tp in uniq_paths if tp not in store]
    print(f"[c5] masked embedding cache {cache}: "
          f"{len(uniq_paths) - len(missing)}/{len(uniq_paths)} cached, "
          f"{len(missing)} to encode (chunk={chunk})")
    for i in range(0, len(missing), chunk):
        part = missing[i:i + chunk]
        model._encode(pd.DataFrame({"text_path": part}))
        print(f"[c5] masked encode {min(i + chunk, len(missing))}/"
              f"{len(missing)} docs (cache persisted)")


C5_NOT_EXECUTED_NOTE = (
    "[c5] NOTE (prereg-ea v1.4, decided PRE-STATISTIC): C5-lf is registered "
    "NOT-EXECUTED. HAR side constructively n/a — committed m1_multiseed.csv "
    "long_form C5_qwen3 seed-2026 rel_impr_pct = -1.0347/-3.1346/-6.6467 "
    "(primary; deployable_combiner FIXED -0.85/-2.48/-5.97 corroborates), "
    "all negative, so every HAR share cell is empty before execution; firmID "
    "side excluded on GPU budget/scope (no committed increment to invoke "
    "constructive n/a on) and cannot change the branch adjudication. This "
    "code path is retained for the record; running it for real requires a "
    "prereg re-amendment first. The scorer (anon_score.py --channel lf) "
    "ignores c5 run dirs and writes the arm as not-executed rows.")


def run_c5(args) -> None:
    print(C5_NOT_EXECUTED_NOTE)
    s = sli()
    train_mod = s._load_train_mod()
    panel = s.load_full_long_form_panel(train_mod)
    print(f"[c5] full long_form panel rows by split: "
          f"{panel.split.value_counts().to_dict()}")
    masked_mode = not args.control
    tag = "anonmask" if masked_mode else "anonctrl"
    run_dir = REPO / "results" / "runs" / f"{C5_ARM}_{tag}_full_{DISC}_seed{SEED}"
    staging_home = _c5_staging_home(args)

    # ---- frozen model: committed artefacts if present, else the v1.2
    #      recipe-rebuild + 1e-8 reproduction gate (verbatim E-lf machinery).
    #      swapped=None -> the one-pass cache warm covers the panel docs only.
    model, artefacts, problems = s.prepare_arm(
        C5_ARM, train_mod, panel, SEED, dry_run=args.dry_run,
        c5_tol=args.c5_tol, swapped=None, out_dir=staging_home)

    if args.dry_run:
        if masked_mode:
            _dry_run_masked_report(args, model, panel)
        status = "READY" if not problems else f"{len(problems)} problem(s)"
        print(f"[dry-run] c5 ({tag}): {status}")
        return

    run_dir.mkdir(parents=True, exist_ok=True)

    if not masked_mode:
        # ---- G1 control: full-panel predict on ORIGINAL text (warm cache) ----
        preds = model.predict(panel)
        out = panel[["ticker", "accession", "horizon_days", "split",
                     "label_realised_vol"]].copy()
        out["prediction_realised_vol"] = np.asarray(preds, dtype=float)
        if not np.isfinite(out["prediction_realised_vol"]).all():
            _fatal("non-finite control predictions")
        out.to_parquet(run_dir / "predictions.parquet", index=False)
        g1 = ed_arms().g1_compare("c5_lf", run_dir, COMMITTED["c5"])
        _write_c5_config(run_dir, tag, artefacts, args, extra={
            "g1": {"pass": g1["pass"], "exact_match_rate": g1["exact_match_rate"],
                   "max_abs_diff": g1["max_abs_diff"]}})
        print(f"[c5:{tag}] wrote {run_dir} ({len(out)} rows)")
        return

    # ------------------------------ masked inference ------------------------
    if model.device.type != "cuda":
        _fatal("masked C5 inference requires CUDA (the committed embedding "
               "convention is bf16/cuda; a CPU/MPS encode is not comparable). "
               "Set CUDA_VISIBLE_DEVICES to one GPU.")
    vt = panel[panel.split.isin(SPLITS)].copy().reset_index(drop=True)
    if vt.duplicated(["split", "horizon_days", "accession"]).any():
        _fatal("panel has duplicate (split, horizon, accession) rows")
    needed = set(vt["text_path"].astype(str))
    ed = ed_arms()
    masked, missing = ed.rekey_store(
        ed.load_masked_store(Path(args.masked_store)), needed)
    if missing:
        _fatal(f"masked store missing {len(missing)} of {len(needed)} "
               f"long-form val+test docs (first: {missing[0]}) — run "
               "anon_mask_build.py --panel lf to completion first.")

    # belt-and-braces: the served text must actually be masked
    sample = list(masked.values())[:50]
    n_ph = _placeholder_count(sample)
    if n_ph == 0:
        _fatal("no [FIRM]/[TICKER]/[PERSON]/[CIK]/[PRODUCT] placeholder in a "
               "50-doc sample of the masked store — wrong store?")
    print(f"[c5] masked store sanity: {n_ph} placeholders across 50 sample docs")

    # (a) sibling masked cache — the ORIGINAL cache stays untouched
    orig_cache = model._cache_path()
    masked_cache = _masked_cache_path(orig_cache)
    if masked_cache == orig_cache:
        _fatal("masked cache path equals the original cache path")
    orig_cache_sha = _sha256(orig_cache) if (orig_cache and orig_cache.exists()) \
        else None
    model._cache_path = (lambda mc=masked_cache: mc)  # instance attr wins
    print(f"[c5] embedding cache repointed: masked -> {masked_cache.name} "
          f"(original {orig_cache.name if orig_cache else None} frozen)")
    # (b) masked text source
    _patch_masked_text_source(masked)

    uniq = list(dict.fromkeys(vt["text_path"].astype(str)))
    _chunked_masked_encode(model, uniq, args.encode_chunk)

    out = s.predict_arm(f"{C5_ARM}_anonmask", model, vt, run_dir,
                        smoke=(args.smoke if args.smoke else None))

    # G3-analog: artefact-hash invariance through the masked pass
    g3 = {"gate": "G3-lf: rebuilt-head + original-cache hash invariance "
                  "through masked inference",
          "pass": True, "post": {},
          "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for name, rec in artefacts.items():
        p = Path(rec["path"]) if rec.get("path") else None
        pre = rec.get("sha256") or rec.get("sha256_pre")
        if p is None or pre is None or not p.exists():
            continue
        post = _sha256(p)
        g3["post"][name] = post
        if post != pre:
            g3["pass"] = False
            _fatal(f"G3 VIOLATION — artefact {name} changed during masked "
                   f"inference ({pre[:12]}… -> {post[:12]}…)")
    if orig_cache_sha is not None:
        post = _sha256(orig_cache)
        g3["post"]["original_emb_cache"] = post
        if post != orig_cache_sha:
            g3["pass"] = False
            _fatal("G3 VIOLATION — the ORIGINAL embedding cache changed during "
                   "the masked pass (cache separation failed)")
    ANON_DIR.mkdir(parents=True, exist_ok=True)
    (ANON_DIR / "g3_c5_lf_hash_invariance.json").write_text(
        json.dumps(g3, indent=2))
    print(f"[c5] G3-lf hash invariance PASS ({len(g3['post'])} artefacts, "
          f"original cache included) -> results/anon/g3_c5_lf_hash_invariance.json")

    if args.smoke:
        print(f"[c5:{tag}] SMOKE — parts under {run_dir}/parts, no final "
              "predictions.parquet written")
        return
    if len(out) != len(vt):
        _fatal(f"output rows {len(out)} != val+test panel rows {len(vt)}")
    out.to_parquet(run_dir / "predictions.parquet", index=False)
    _write_c5_config(run_dir, tag, artefacts, args, extra={
        "masked_cache": str(masked_cache),
        "n_masked_docs_served": len(masked),
        "rows_by_split": {k: int(v) for k, v in out.split.value_counts().items()},
        "frozen_head_note": "masked INFERENCE only — heads are the committed-"
                            "recipe rebuild gated at 1e-8 on ORIGINAL text; "
                            "no masked retrain (disclosed at scoring)",
    })
    print(f"[c5:{tag}] wrote {run_dir} ({len(out)} prediction rows, val+test)")


def _write_c5_config(run_dir: Path, tag: str, artefacts: dict, args,
                     extra: dict) -> None:
    (run_dir / "config.json").write_text(json.dumps({
        "model": C5_ARM, "dataset": "full", "disclosure": DISC, "seed": SEED,
        "anon": {"prereg": "prereg-ea v1.1 §C-anon (long_form stretch)",
                 "channel": "long_form_stretch", "mode": tag,
                 "masked_store": str(args.masked_store) if tag == "anonmask"
                 else None,
                 "artefacts": artefacts, **extra},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2, default=str))


def _dry_run_masked_report(args, model, panel) -> None:
    vt = panel[panel.split.isin(SPLITS)]
    needed = set(vt["text_path"].astype(str))
    store = Path(args.masked_store)
    if not store.exists():
        print(f"[dry-run] MISSING: masked store {store} — run "
              "anon_mask_build.py --panel lf first")
        return
    ed = ed_arms()
    masked, missing = ed.rekey_store(ed.load_masked_store(store), needed)
    print(f"[dry-run] masked store covers {len(masked)}/{len(needed)} "
          f"val+test docs" + (f" (MISSING {len(missing)})" if missing else ""))
    orig_cache = model._cache_path()
    mc = _masked_cache_path(orig_cache)
    n_cached = 0
    if mc.exists():
        keys = set(pd.read_parquet(mc, columns=["text_path"])
                   ["text_path"].astype(str))
        n_cached = sum(tp in keys for tp in needed)
    print(f"[dry-run] masked emb cache {mc.name}: {n_cached}/"
          f"{len(set(vt['text_path'].astype(str)))} unique docs cached; "
          f"the rest is ONE resumable GPU encode pass")


# ------------------------------------------------------------------ selftest
def _selftest() -> int:
    """Synthetic validation of the masked-inference mechanics (no GPU/HF/data):
      1. qwen_llm._missing_texts patch: embeddings derive from MASKED text;
      2. cache separation: original cache byte-frozen, masked cache created;
      3. chunked masked encode resumes (second call encodes nothing);
      4. g1 rename wrapper writes g1_control_b2_lf.json (scratch ANON dir);
      5. placeholder sanity guard fires on an unmasked store.
    """
    import sp500vol.models.neural_text.qwen_llm as qwen_llm
    from sp500vol.models.neural_text.qwen_llm import C5LLMProbe

    calls = {"n_encoded": 0}

    class FakeEnc:
        hidden_size = 8

        def encode(self, texts, *, batch_size=8):
            calls["n_encoded"] += len(texts)
            out = np.empty((len(texts), 8), dtype=np.float32)
            for i, t in enumerate(texts):
                hh = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
                out[i] = np.random.default_rng(hh).standard_normal(8)
            return out

    orig_missing_texts = qwen_llm._missing_texts
    with tempfile.TemporaryDirectory(prefix="anon_lf_selftest_") as td:
        td = Path(td)
        tps = [f"interim/full/10-K/000{i}/acc{i}.txt" for i in range(6)]
        orig_texts = {tp: f"Apple Inc reported quarter {i} results"
                      for i, tp in enumerate(tps)}
        masked_texts = {tp: f"[FIRM] reported quarter {i} results"
                        for i, tp in enumerate(tps)}

        def _mk(cache):
            m = C5LLMProbe(pretrained="stub/none", max_length=8, hidden_dim=4,
                           dropout=0.0, lr=1e-3, weight_decay=0.0, batch_size=8,
                           max_epochs=1, cache_embeddings=True, device="cpu",
                           checkpoint=False, seed=SEED, strategy="selftest")
            m._encoder = FakeEnc()
            m.embedding_dim_ = 8
            m._cache_path = (lambda c=cache: c)
            return m

        # original pass -> original cache
        oc, mc = td / "orig_cache.parquet", td / "orig_cache__anonmask_lf.parquet"
        qwen_llm._missing_texts = lambda df, missing: {
            tp: orig_texts[tp] for tp in missing}
        m1 = _mk(oc)
        e_orig = m1._encode(pd.DataFrame({"text_path": tps}))
        assert oc.exists()
        oc_sha = _sha256(oc)

        # masked pass -> sibling cache, patched source
        _patch_masked_text_source(masked_texts)
        m2 = _mk(mc)
        n_before = calls["n_encoded"]
        _chunked_masked_encode(m2, tps, chunk=2)
        e_mask = m2._encode(pd.DataFrame({"text_path": tps}))
        assert not np.allclose(e_orig, e_mask), \
            "masked embeddings equal original — text patch ineffective"
        exp = FakeEnc().encode([masked_texts[tp] for tp in tps])
        assert np.allclose(e_mask, exp), "embeddings not derived from masked text"
        assert mc.exists() and _sha256(oc) == oc_sha, \
            "original cache modified by the masked pass"
        print("[selftest] masked text patch + cache separation verified")

        # resume: everything cached -> zero new encodes
        n_mid = calls["n_encoded"]
        _chunked_masked_encode(m2, tps, chunk=2)
        assert calls["n_encoded"] == n_mid, "chunked encode did not resume"
        # in-process store poisoning guard: a FRESH store from disk must match
        qwen_llm._EMB_STORES.clear()
        m3 = _mk(mc)
        e_disk = m3._encode(pd.DataFrame({"text_path": tps}))
        assert np.allclose(e_disk, e_mask), "masked cache round-trip mismatch"
        print("[selftest] chunked masked encode resume + disk round-trip verified")

        # g1 rename wrapper (scratch ANON dir; never results/anon)
        ed = ed_arms()
        old_anon = ed.ANON_DIR
        try:
            ed.ANON_DIR = td / "anon"
            key = ["ticker", "accession", "horizon_days"]
            df = pd.DataFrame({"ticker": ["A", "B"], "accession": ["x", "y"],
                               "horizon_days": [5, 5], "split": ["val", "test"],
                               "prediction_realised_vol": [0.1, 0.2]})
            for d in ("new", "ref"):
                (td / d).mkdir()
                df.to_parquet(td / d / "predictions.parquet", index=False)
            orig_g1 = ed.g1_compare
            wrapped = lambda arm, new_run, committed: orig_g1(  # noqa: E731
                f"{arm}_lf", new_run, committed)
            v = wrapped("b2", td / "new", td / "ref")
            assert v["pass"] and (td / "anon" / "g1_control_b2_lf.json").exists()
            print("[selftest] g1 rename wrapper -> g1_control_b2_lf.json verified")
        finally:
            ed.ANON_DIR = old_anon

        # placeholder guard fires on an unmasked store
        assert _placeholder_count(list(orig_texts.values())) == 0
        assert _placeholder_count(list(masked_texts.values())) == len(tps)
        print("[selftest] placeholder sanity guard discriminates masked/unmasked")

    qwen_llm._missing_texts = orig_missing_texts
    print("[selftest] ALL PASS — nothing under results/ or the data root touched")
    return 0


# ------------------------------------------------------------------ cli
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["b2", "c5"],
                    help="required unless --selftest without --arm")
    ap.add_argument("--control", action="store_true",
                    help="G1 mode: run on UNMASKED text, compare vs the "
                         "committed long-form run (exact-match rate + "
                         "max|diff| per the v1.1 operationalisation)")
    ap.add_argument("--masked-store", default=str(default_masked_store()))
    ap.add_argument("--c5-tol", type=float, default=1e-8,
                    help="C5 recipe-rebuild reproduction-gate max relative "
                         "diff (prereg-ea v1.2 mechanism; loosening requires "
                         "disclosure at scoring)")
    ap.add_argument("--c5-staging", default=None,
                    help="rebuilt-head staging home (default: the E-lf run's "
                         "complete set when present, else results/anon/c5_lf)")
    ap.add_argument("--encode-chunk", type=int, default=512,
                    help="masked docs per encode+persist step (resume "
                         "granularity)")
    ap.add_argument("--dry-run", action="store_true",
                    help="c5: preflight artefacts/gate plan/store+cache "
                         "coverage; predict nothing")
    ap.add_argument("--smoke", type=int, default=0,
                    help="b2: keep N rows per (split,horizon), scratch run "
                         "dir; c5: N rows per part, no final parquet")
    ap.add_argument("--selftest", action="store_true",
                    help="without --arm: synthetic masked-encode validation; "
                         "with --arm b2: real-panel store-injection check "
                         "(no training)")
    args = ap.parse_args()

    if args.selftest and args.arm is None:
        return _selftest()
    if args.arm is None:
        ap.error("--arm is required (or --selftest)")
    if args.arm == "c5" and args.selftest:
        ap.error("c5 has no per-arm selftest — use the bare --selftest "
                 "(synthetic) and --dry-run (real preflight)")
    ANON_DIR.mkdir(parents=True, exist_ok=True)
    if args.arm == "b2":
        if args.dry_run:
            ap.error("--dry-run is c5-only; b2 preflight = --selftest")
        run_b2(args)
    else:
        run_c5(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
