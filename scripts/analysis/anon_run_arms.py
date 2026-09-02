"""C-anon step 2 (prereg-ea-v1.0, configs/prereg_swap_lf_and_anon.md §C-anon) —
run the three pre-registered arms on the MASKED event-driven (8-K) inputs.

Arms (all event_driven, seed 2026, run-dir conventions identical to the
committed originals so the scorer treats them like any other run):

  --arm c6   C6-masked. The e1 machinery VERBATIM (scripts/experiments/
             e1_llm_forecast/{prompt,run_inference,postprocess}.py imported by
             path, only the text source repointed to the masked store):
             identical JSON protocol, temperature 0, guided JSON when
             available, checkpoint every 500 filings, RESUMABLE (re-invoking
             skips finished (text_path, variant) pairs). Variant c6_text,
             subset event_driven. Postprocess -> results/runs/
             C6_llmtext_anonmask_full_event_driven_seed2026.
             MODEL NOTE: the prereg names Qwen3-32B-AWQ (single card); the
             COMMITTED C6 run's config.json records
             llm="/root/gpu-data/models/Qwen3-32B" (bf16, TP=2). The
             G1 bit-identity control is only meaningful with the committed
             weights/TP/vLLM env; --model/--tp are therefore explicit knobs
             and a control run with a different --model than the committed
             llm string requires --accept-model-mismatch.

  --arm c2   C2-masked. FinBERT-S1 event_driven RETRAIN under the FIXED
             recipe (configs/models/C2_finbert_s1.yaml verbatim: AdamW 8e-5,
             effective batch 128, 6% warmup, <=15 epochs early-stop patience
             3 — the base training config, NOT the ASHA-tuned arm), seed
             2026, then full-panel predict (train/val/test), exactly the
             scripts/train.py flow (its helpers are imported by path).
             Per-horizon checkpoints under <run_dir>/checkpoints make the
             retrain resumable. GPU.

  --arm b2   B2-masked. TF-IDF + Ridge (configs/models/B2_tfidf_ridge.yaml),
             same train.py flow. CPU.

  --control  G1 pipeline-invariance mode: identical run on the UNMASKED text
             (no store injection), run-dir *_anonctrl_*; predictions are then
             compared BIT-IDENTICALLY against the committed run
             ({C6_llmtext,C2_finbert_s1,B2_tfidf_ridge}_full_event_driven_
             seed2026). Verdict written to results/anon/g1_control_<arm>.json.
             The registered gate is exact equality; the comparison also
             reports exact-match rate and max |diff| so a documented deviation
             (GPU non-bit-determinism) can be adjudicated BEFORE scoring.

Masked-store injection (c2/b2): the shared text loader
(sp500vol.models.classical_text._text_dataset) resolves every text through a
process-level store keyed by the cache path; we PRELOAD that store with the
masked texts and no-op its persist hook, after asserting the store covers
EVERY text_path in the panel — so the untouched model code reads masked text
and the real on-disk cache can never be written. G3 (c6): masked-vs-committed
excerpt truncation stats -> results/anon/g3_truncation_stats.json.

Box usage (export SP500VOL_DATA_ROOT=/data/sp500vol-data first):
  python scripts/analysis/anon_run_arms.py --arm c6 --model <committed llm> --tp 2
  python scripts/analysis/anon_run_arms.py --arm c6 --control --model <same> --tp 2
  python scripts/analysis/anon_run_arms.py --arm c2
  python scripts/analysis/anon_run_arms.py --arm c2 --control
  python scripts/analysis/anon_run_arms.py --arm b2
  python scripts/analysis/anon_run_arms.py --arm b2 --control
Local (no GPU, nothing heavy):  --selftest [--arm c2] validates store
injection + coverage; --arm c6 --mock exercises the full c6 path with the
mock generator into a scratch out-dir (NOT results/runs).
"""
from __future__ import annotations

# thread caps BEFORE numpy/torch (box overrides by exporting its own values)
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from sp500vol.utils.paths import data_path  # noqa: E402

KEY = ["ticker", "accession", "horizon_days"]
SEED = 2026
DISC = "event_driven"
E1_DIR = REPO / "scripts" / "experiments" / "e1_llm_forecast"
ANON_DIR = REPO / "results" / "anon"
MANIFEST = REPO / "results" / "e1_llm_forecast" / "manifest_valtest.parquet"
COMMITTED_RAW = REPO / "results" / "e1_llm_forecast" / "raw"
COMMITTED = {
    "c6": REPO / "results" / "runs" / f"C6_llmtext_full_{DISC}_seed{SEED}",
    "c2": REPO / "results" / "runs" / f"C2_finbert_s1_full_{DISC}_seed{SEED}",
    "b2": REPO / "results" / "runs" / f"B2_tfidf_ridge_full_{DISC}_seed{SEED}",
}
MODEL_ID = {"c2": "C2_finbert_s1", "b2": "B2_tfidf_ridge"}


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def default_masked_store() -> Path:
    return Path(data_path("processed", "_text_cache",
                          "filing_texts_anonmask_ed.parquet"))


_MASK_MOD = None


def _path_key(p: str) -> str:
    """Root-invariant text_path key — imported from anon_mask_build.path_key
    (single source of truth). The masked store, the panel parquets and the e1
    manifest may carry different absolute roots for the same file (Mac
    /path/to/data-root/... vs box /data/...), so every lookup here matches on it."""
    global _MASK_MOD
    if _MASK_MOD is None:
        _MASK_MOD = _load_by_path("anon_mask_build_for_keys",
                                  REPO / "scripts" / "analysis" / "anon_mask_build.py")
    return _MASK_MOD.path_key(p)


def load_masked_store(path: Path) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(f"FATAL: masked store {path} not found — run "
                         "anon_mask_build.py first.")
    df = pd.read_parquet(path, columns=["text_path", "text"])
    return dict(zip(df["text_path"].astype(str), df["text"].astype(str),
                    strict=True))


def rekey_store(store: dict[str, str], targets: set[str]):
    """Re-key the masked store onto the TARGET text_path strings via the
    root-invariant path_key. Returns (dict target_path -> masked text,
    sorted missing target paths). Deterministic: on duplicate normalised
    cache keys the first occurrence wins."""
    norm: dict[str, str] = {}
    for k, v in store.items():
        nk = _path_key(k)
        if nk not in norm:
            norm[nk] = v
    out, missing = {}, []
    for t in targets:
        v = norm.get(_path_key(t))
        if v is None:
            missing.append(t)
        else:
            out[t] = v
    return out, sorted(missing)


def inject_masked_store(masked: dict[str, str], needed: set[str]) -> None:
    """Preload the shared text store with masked texts + disable persistence.

    `masked` must already be keyed by the panel's own text_path strings
    (rekey_store). Coverage is asserted FIRST: if any needed path were
    absent, load_texts would fall back to disk reads of ORIGINAL text and
    then persist the (masked) store over the real cache — both are made
    impossible here.
    """
    from sp500vol.models.classical_text import _text_dataset as tds

    missing = sorted(needed - masked.keys())
    if missing:
        raise SystemExit(f"FATAL: masked store missing {len(missing)} panel "
                         f"text_paths (first: {missing[0]}) — refuse to run.")
    tds._STORES[str(tds._default_cache_path())] = dict(masked)
    tds._persist = lambda *a, **k: None  # belt-and-braces: never write the cache
    print(f"[inject] masked store injected: {len(masked)} docs "
          f"(covers all {len(needed)} panel docs); cache persistence disabled")


# ------------------------------------------------------------------ G1 compare
def g1_compare(arm: str, new_run: Path, committed: Path) -> dict:
    a = pd.read_parquet(new_run / "predictions.parquet")
    b = pd.read_parquet(committed / "predictions.parquet")
    cols = KEY + ["split", "prediction_realised_vol"]
    m = a[cols].merge(b[cols], on=KEY + ["split"], suffixes=("_new", "_ref"))
    x = m["prediction_realised_vol_new"].to_numpy(float)
    y = m["prediction_realised_vol_ref"].to_numpy(float)
    same = (x == y) | (np.isnan(x) & np.isnan(y))
    diffs = np.abs(x - y)[~same]
    out = {
        "arm": arm, "gate": "G1 unmasked control vs committed, bit-identical",
        "new_run": str(new_run), "committed": str(committed),
        "n_new": int(len(a)), "n_committed": int(len(b)), "n_joined": int(len(m)),
        "n_exact": int(same.sum()),
        "exact_match_rate": float(same.mean()) if len(m) else float("nan"),
        "max_abs_diff": float(diffs.max()) if len(diffs) else 0.0,
        "rows_align": bool(len(a) == len(b) == len(m)),
        "pass": bool(len(a) == len(b) == len(m) and same.all()),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    ANON_DIR.mkdir(parents=True, exist_ok=True)
    (ANON_DIR / f"g1_control_{arm}.json").write_text(json.dumps(out, indent=2))
    print(f"[G1:{arm}] {'PASS' if out['pass'] else 'FAIL'} — exact "
          f"{out['n_exact']}/{out['n_joined']} rows, max|diff| "
          f"{out['max_abs_diff']:.3e} -> results/anon/g1_control_{arm}.json")
    return out


# ------------------------------------------------------------------ arm: c6
def ed_manifest_paths() -> set[str]:
    m = pd.read_parquet(MANIFEST, columns=["text_path", "disclosure"])
    return set(m[m["disclosure"] == DISC]["text_path"])


def run_c6(args) -> None:
    ri = _load_by_path("anon_e1_run_inference", E1_DIR / "run_inference.py")
    pp = _load_by_path("anon_e1_postprocess", E1_DIR / "postprocess.py")
    masked_mode = not args.control
    tag = "anonmask" if masked_mode else "anonctrl"
    need = ed_manifest_paths()

    committed_llm = None
    cfg_path = COMMITTED["c6"] / "config.json"
    if cfg_path.exists():
        committed_llm = json.loads(cfg_path.read_text()).get("llm")
    if committed_llm and args.model != committed_llm:
        print(f"[c6] WARNING: --model {args.model!r} != committed llm "
              f"{committed_llm!r} (bf16, TP=2 on the committed run).")
        if args.control and not args.accept_model_mismatch:
            raise SystemExit(
                "FATAL: G1 control with a different model than the committed "
                "run cannot be bit-identical by construction. Re-run with "
                f"--model {committed_llm!r} (and --tp 2), or pass "
                "--accept-model-mismatch to record a deviation explicitly.")

    manifest_path = MANIFEST
    if masked_mode:
        store = Path(args.masked_store)
        rekeyed, missing = rekey_store(load_masked_store(store), need)
        if missing and not args.mock:
            raise SystemExit(f"FATAL: masked store missing {len(missing)} of "
                             f"{len(need)} ED manifest docs (root-invariant "
                             f"match; first: {missing[0]}).")
        if missing:  # mock validation against a partial (smoke) store
            covered = set(rekeyed)
            print(f"[c6] MOCK: smoke store covers {len(covered)}/{len(need)} "
                  "ED docs — writing a temp manifest restricted to covered docs")
            m = pd.read_parquet(MANIFEST)
            m = m[m["text_path"].isin(covered)]
            manifest_path = Path(os.environ.get("TMPDIR", "/tmp")) / \
                "anon_c6_mock_manifest.parquet"
            m.to_parquet(manifest_path, index=False)
            need = covered
        # Serve masked texts straight from the re-keyed dict: the manifest's
        # text_path strings may carry a different root than the store's, so
        # the exact-key parquet streamer is replaced, not just repointed.
        ri.stream_texts = lambda tps: {p: rekeyed[p] for p in tps if p in rekeyed}
        print(f"[c6] masked store -> {store} "
              f"(serves all {len(need)} target docs via root-invariant keys)")

    out_dir = Path(args.out_dir or (ANON_DIR / f"raw_c6_{tag}"))
    if args.mock:
        out_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"anon_c6_mock_{tag}"
        print(f"[c6] MOCK mode -> scratch out dir {out_dir}")
    ns = SimpleNamespace(
        manifest=str(manifest_path), model=args.model, variant="c6_text",
        subset=DISC, out_dir=str(out_dir), checkpoint_every=500,
        pilot=0, limit=args.limit, mock=args.mock, seed=SEED,
        max_model_len=8192, max_tokens=120, tp=args.tp, thinking=False)
    ri.run(ns)

    done = ri.load_done(out_dir)
    n_done = sum((tp, "c6_text") in done for tp in need)
    complete = n_done == len(need)
    print(f"[c6] progress: {n_done}/{len(need)} ED filings done "
          f"({'COMPLETE' if complete else 'resumable — re-invoke to continue'})")
    if args.limit or args.mock:
        print("[c6] limit/mock run — skipping postprocess/G1/G3 (validation only)")
        return
    if not complete:
        return

    # ---- postprocess into a standard run dir (event_driven only) ----
    pp.DISCLOSURES = (DISC,)  # ED-only per prereg scope; no relabelled duplicates
    pp.build_runs(SimpleNamespace(raw_dir=str(out_dir),
                                  out_root=str(REPO / "results" / "runs"),
                                  model_suffix=f"_{tag}", on_missing="rv22"))
    new_run = REPO / "results" / "runs" / f"C6_llmtext_{tag}_full_{DISC}_seed{SEED}"

    if masked_mode:
        write_g3(out_dir, need)
    else:
        g1_compare("c6", new_run, COMMITTED["c6"])


def write_g3(masked_raw: Path, need: set[str]) -> None:
    """G3: excerpt/truncation stats, masked vs committed raw (c6_text, ED)."""
    def collect(raw_dir: Path) -> pd.DataFrame:
        parts = sorted(Path(raw_dir).glob("part-*.parquet"))
        df = pd.concat([pd.read_parquet(
            p, columns=["text_path", "variant", "prompt_chars",
                        "excerpt_source", "parse_ok"]) for p in parts])
        df = df[(df["variant"] == "c6_text") & df["text_path"].isin(need)]
        return df.drop_duplicates("text_path", keep="last")

    m, c = collect(masked_raw), collect(COMMITTED_RAW)

    def stats(d: pd.DataFrame) -> dict:
        return {
            "n": int(len(d)),
            "prompt_chars_mean": float(d.prompt_chars.mean()),
            "prompt_chars_median": float(d.prompt_chars.median()),
            "prompt_chars_p95": float(d.prompt_chars.quantile(0.95)),
            "excerpt_source_pct": {k: float(100 * v) for k, v in
                                   d.excerpt_source.value_counts(normalize=True)
                                   .items()},
            "parse_ok_pct": float(100 * d.parse_ok.mean()),
        }

    out = {"gate": "G3 excerpt truncation comparable, masked vs committed",
           "masked": stats(m), "committed": stats(c),
           "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    ANON_DIR.mkdir(parents=True, exist_ok=True)
    (ANON_DIR / "g3_truncation_stats.json").write_text(json.dumps(out, indent=2))
    print(f"[G3] masked prompt_chars median "
          f"{out['masked']['prompt_chars_median']:.0f} vs committed "
          f"{out['committed']['prompt_chars_median']:.0f} "
          f"-> results/anon/g3_truncation_stats.json")


# ------------------------------------------------------------------ arm: c2/b2
def run_train_arm(args) -> None:
    """Replicates scripts/train.py's flow verbatim through its own helpers
    (identical recipe/data chain), with the masked store injected first."""
    arm = args.arm
    model_id = MODEL_ID[arm]
    masked_mode = not args.control
    tag = "anonmask" if masked_mode else "anonctrl"
    run_id = f"{model_id}_{tag}_full_{DISC}_seed{SEED}"
    run_dir = REPO / "results" / "runs" / run_id
    train_mod = _load_by_path("anon_train_mod", REPO / "scripts" / "train.py")
    from sp500vol.utils import seed_everything

    data = train_mod._load_dataset("full")
    data = train_mod._filter_disclosure(data, DISC)
    data = train_mod._assign_splits(data, "full")
    data = train_mod._drop_invalid_rows(data)
    if args.smoke:
        data = (data.groupby(["split", "horizon_days"], group_keys=False)
                .head(args.smoke).reset_index(drop=True))
        run_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"{run_id}_smoke"
        print(f"[{arm}] SMOKE {args.smoke}/group -> scratch run dir {run_dir}")
    needed = set(data["text_path"].astype(str))

    if masked_mode:
        rekeyed, missing = rekey_store(load_masked_store(Path(args.masked_store)),
                                       needed)
        if args.selftest and missing:
            # local validation against a partial (smoke) store: shrink the
            # panel to covered docs — NEVER allowed outside --selftest.
            cov = data["text_path"].astype(str).isin(rekeyed.keys())
            print(f"[selftest] SMOKE store covers {int(cov.sum())}/{len(data)} "
                  "panel rows — restricting the selftest to covered docs")
            data = data[cov].reset_index(drop=True)
            needed = set(data["text_path"].astype(str))
            if not len(data):
                raise SystemExit("SELFTEST FAIL: store covers no panel doc")
        inject_masked_store(rekeyed, needed)

    if args.selftest:
        _selftest(data, masked_mode)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    cfg = train_mod._load_yaml(REPO / "configs" / "models" / f"{model_id}.yaml")
    model = train_mod._build_model(model_id, cfg, dataset="full",
                                   run_dir=run_dir, seed=SEED)
    train_rows = data[data["split"] == "train"].copy()
    val_rows = data[data["split"] == "val"].copy()
    print(f"[{arm}:{tag}] fit: {len(train_rows)} train / {len(val_rows)} val rows "
          f"(fixed recipe, seed {SEED})")
    model.fit(train_rows, train_rows["label_realised_vol"].to_numpy(),
              X_val=val_rows,
              y_val=(val_rows["label_realised_vol"].to_numpy()
                     if not val_rows.empty else None))

    val_curves = getattr(model, "val_curves_", None)
    if val_curves:
        (run_dir / "val_curves.json").write_text(json.dumps(
            {str(k): v for k, v in val_curves.items()}, indent=2, default=str))

    predictions = data.copy()
    predictions["prediction_realised_vol"] = model.predict(predictions)
    predictions["run_id"] = run_id
    predictions["model_id"] = model_id
    predictions["dataset"] = "full"
    predictions["seed"] = SEED
    predictions["disclosure_subset"] = DISC
    predictions["feature_rv_1d"] = train_mod._feature_rv_1d(predictions)
    cols = train_mod._prediction_columns(predictions)
    predictions[cols].to_parquet(run_dir / "predictions.parquet", index=False)
    model.save(run_dir / "model.pkl")
    metrics = train_mod._metrics_by_group(predictions)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "model": model_id, "dataset": "full", "disclosure": DISC, "seed": SEED,
        "model_config": cfg,
        "anon": {"prereg": "prereg-ea-v1.0 §C-anon", "mode": tag,
                 "masked_store": str(args.masked_store) if masked_mode else None},
    }, indent=2, default=str))
    print(f"[{arm}:{tag}] wrote {run_dir} ({len(predictions)} prediction rows)")

    if not masked_mode and not args.smoke:
        g1_compare(arm, run_dir, COMMITTED[arm])


def _selftest(data: pd.DataFrame, masked_mode: bool) -> None:
    """Cheap injection check: load_texts must return the injected masked text
    and must not rewrite the on-disk cache."""
    from sp500vol.models.classical_text import _text_dataset as tds

    cache = Path(tds._default_cache_path())
    mtime = cache.stat().st_mtime if cache.exists() else None
    sample = data.drop_duplicates("text_path").head(5).reset_index(drop=True)
    got = tds.load_texts(sample)
    store = tds._STORES.get(str(tds._default_cache_path()), {})
    ok = all(got[i] == store.get(str(sample["text_path"].iloc[i]))
             for i in range(len(sample))) if masked_mode else len(got) == len(sample)
    if cache.exists() and mtime is not None and cache.stat().st_mtime != mtime:
        raise SystemExit("SELFTEST FAIL: on-disk text cache was modified!")
    if masked_mode and not ok:
        raise SystemExit("SELFTEST FAIL: load_texts did not return masked text")
    n_ph = sum(t.count("[FIRM]") + t.count("[TICKER]") + t.count("[PERSON]")
               for t in got)
    print(f"SELFTEST OK: load_texts served {'masked' if masked_mode else 'real'} "
          f"text for {len(got)} sample docs (placeholders seen: {n_ph}); "
          "cache untouched.")


# ------------------------------------------------------------------ cli
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["c6", "c2", "b2"])
    ap.add_argument("--control", action="store_true",
                    help="G1 mode: run on UNMASKED text, compare bit-identical "
                         "to the committed run")
    ap.add_argument("--masked-store", default=str(default_masked_store()))
    ap.add_argument("--model", default="Qwen/Qwen3-32B-AWQ",
                    help="c6 only. Prereg names Qwen3-32B-AWQ; the committed "
                         "run used its config.json llm path (bf16, TP=2) — "
                         "see module docstring before choosing.")
    ap.add_argument("--tp", type=int, default=1, help="c6 tensor parallel size")
    ap.add_argument("--out-dir", default=None, help="c6 raw checkpoint dir "
                    "(default results/anon/raw_c6_<tag>)")
    ap.add_argument("--accept-model-mismatch", action="store_true",
                    help="c6 control: proceed although --model != committed llm "
                         "(records an explicit deviation; G1 will fail)")
    ap.add_argument("--mock", action="store_true",
                    help="c6: mock generator, scratch out dir (local validation)")
    ap.add_argument("--limit", type=int, default=0,
                    help="c6: stratified N-filing sample (with --mock only)")
    ap.add_argument("--smoke", type=int, default=0,
                    help="c2/b2: keep N rows per (split,horizon), scratch run dir")
    ap.add_argument("--selftest", action="store_true",
                    help="c2/b2: validate store injection + coverage, no training")
    args = ap.parse_args()

    if args.limit and not args.mock:
        raise SystemExit("--limit is a --mock-only validation knob (the real "
                         "run must cover the full ED manifest).")
    ANON_DIR.mkdir(parents=True, exist_ok=True)
    if args.arm == "c6":
        run_c6(args)
    else:
        run_train_arm(args)


if __name__ == "__main__":
    main()
