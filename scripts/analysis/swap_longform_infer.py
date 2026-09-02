"""E-lf STEP 2/3 — re-inference of FROZEN long-form models on SWAPPED documents.

Pre-registration: configs/prereg_swap_lf_and_anon.md §E-lf (tag prereg-ea-v1.0,
amended prereg-ea-v1.2 BEFORE any E-lf statistic: C2 artefact-lost; C5 head
rebuild-as-reproduction). ZERO TRAINING on swapped data. Box-executable: device
from environment (CUDA_VISIBLE_DEVICES), resumable per (arm, split, horizon)
part, checkpoint-hash invariance recorded.

Arms (frozen artefacts -> swapped-document predictions):
  B2_tfidf_ridge stored model.pkl preferred; else deterministic refit per the
                 committed recipe on the ORIGINAL train split, gated by exact
                 reproduction of the committed val/test predictions. CPU.
  C5_qwen3       frozen Qwen/Qwen3-Embedding-8B embedder + per-horizon heads.
                 Stored checkpoints/model.pkl load strictly when present; when
                 the head files are LOST (the box state), the heads are
                 deterministically REBUILT with the committed recipe on the
                 ORIGINAL train split and gated by reproduction of the
                 committed predictions.parquet (max rel diff --c5-tol, default
                 1e-8) — the exact B2 rebuild-as-reproduction mechanism
                 (prereg-ea-v1.2). The cold box embedding cache is warmed in
                 ONE GPU pass over original + swapped docs together. 1 GPU.
  C2_finbert_s1  ARTEFACT-LOST (prereg-ea-v1.2): the three horizon checkpoints
                 existed only on the box and were deleted; no local copy was
                 ever held. Retraining a stand-in would violate the §E-lf
                 zero-training principle, so the arm is registered as
                 not-executed and is EXCLUDED from --arm all. Selecting it
                 explicitly still fails fatally on the missing checkpoints.

# =============================================================================
# DISCIPLINE — READ BEFORE TOUCHING THIS FILE (predict_winner_test convention)
#   * never calls .fit() on swapped data; the ONLY fits permitted anywhere are
#     the B2 / C5 recipe rebuilds on the ORIGINAL train split, and only when
#     the stored artefacts are absent, and only if they reproduce the committed
#     predictions to --b2-tol / --c5-tol (default 1e-8) — otherwise FATAL
#     (rebuild-as-reproduction, not a training degree of freedom);
#   * computes, prints, stores NO accuracy/error/loss statistic on ANY split
#     (prediction-vs-prediction reproduction diffs are permitted; label-vs-
#     prediction comparisons are NOT — the C5 rebuild's early-stopping val loss
#     is the committed recipe's own internal mechanism, never a readout);
#   * output parquet carries exactly the pre-registered columns
#     [ticker, accession, horizon_days, split, label_realised_vol,
#      prediction_realised_vol]; labels pass through UNTOUCHED;
#   * G3: sha256 of every model artefact is recorded BEFORE prediction and
#     re-hashed AFTER — any change is FATAL (no-retraining proof).
# The single pre-registered scoring happens in swap_longform_score.py.
# =============================================================================

Run on the box (from repo root, after swap_longform_build.py):
  .venv/bin/python scripts/analysis/swap_longform_infer.py --arm all --dry-run
  .venv/bin/python scripts/analysis/swap_longform_infer.py --arm B2_tfidf_ridge  # CPU
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/analysis/swap_longform_infer.py --arm C5_qwen3  # 1 GPU
Local validation:  --selftest   (synthetic corpus; no GPU, no HF, no results/ writes)
"""
from __future__ import annotations

# --- thread caps BEFORE numpy/torch (box overrides by exporting its own) -----
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DISC = "long_form"
SPLITS = ("val", "test")
OUT_COLS = ["ticker", "accession", "horizon_days", "split",
            "label_realised_vol", "prediction_realised_vol"]
ARM_KIND = {
    "B2_tfidf_ridge": "sklearn",
    "C5_qwen3": "c5",
    "C2_finbert_s1": "neural",
}
ARM_ORDER = ("B2_tfidf_ridge", "C5_qwen3", "C2_finbert_s1")  # cheap first
# prereg-ea-v1.2: C2 checkpoints physically lost -> arm registered not-executed;
# '--arm all' therefore covers the two arms whose frozen artefacts are recoverable.
ARMS_ALL_V12 = ("B2_tfidf_ridge", "C5_qwen3")
C2_LOST_NOTE = (
    "[infer] NOTE (prereg-ea-v1.2): C2_finbert_s1 is registered ARTEFACT-LOST — its "
    "three horizon checkpoints existed only on the box and were deleted (no local copy "
    "was ever held); retraining a stand-in would violate the §E-lf zero-training "
    "principle. '--arm all' runs B2_tfidf_ridge + C5_qwen3 only; the coverage "
    "degradation is disclosed at scoring (swap_longform_score.py).")


def _fatal(msg: str) -> None:
    raise SystemExit(f"[swap_longform_infer] FATAL: {msg}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_train_mod():
    """Import scripts/train.py by file path — the EXACT loader/factory chain the
    committed runs used (same convention as predict_winner_test via asha_hpo)."""
    spec = importlib.util.spec_from_file_location("elf_train", REPO / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_full_long_form_panel(train_mod) -> pd.DataFrame:
    """Full long-form panel with TRUE split labels — the committed loader chain."""
    data = train_mod._load_dataset("full")
    data = train_mod._filter_disclosure(data, DISC)
    data = train_mod._assign_splits(data, "full")
    data = train_mod._drop_invalid_rows(data)
    return data.reset_index(drop=True)


def apply_swap(panel: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Return the val+test rows with text_path EXCHANGED per the manifest.

    The manifest supplies the (split, horizon, accession) -> partner_accession
    map; the swapped text_path is taken from the PANEL's own row of the partner
    (path strings stay native to this machine's panel — cache keys stay valid).
    Train rows are untouched and excluded (M1 needs val+test only).
    """
    vt = panel[panel.split.isin(SPLITS)].copy().reset_index(drop=True)
    if vt.duplicated(["split", "horizon_days", "accession"]).any():
        _fatal("panel has duplicate (split, horizon, accession) rows")
    man = manifest[["split", "horizon_days", "accession", "partner_accession",
                    "partner_ticker", "swapped"]].copy()
    if man.duplicated(["split", "horizon_days", "accession"]).any():
        _fatal("manifest has duplicate (split, horizon, accession) rows")

    # key-set equality: the manifest must describe EXACTLY this panel
    pk = set(map(tuple, vt[["split", "horizon_days", "accession"]].itertuples(index=False)))
    mk = set(map(tuple, man[["split", "horizon_days", "accession"]].itertuples(index=False)))
    if pk != mk:
        _fatal(f"manifest/panel key mismatch: panel-only={len(pk - mk)}, "
               f"manifest-only={len(mk - pk)} — rebuild the manifest on this data root")

    m = vt.merge(man, on=["split", "horizon_days", "accession"],
                 how="left", validate="one_to_one")
    if m.partner_accession.isna().any():
        _fatal("unmatched rows after manifest merge")
    lookup = m.set_index(["split", "horizon_days", "accession"])["text_path"]
    part_keys = list(zip(m["split"], m["horizon_days"], m["partner_accession"]))
    m["text_path_orig"] = m["text_path"]
    m["text_path"] = lookup.loc[part_keys].to_numpy()
    # partner ticker sanity (manifest built from the same universe)
    tick = m.set_index(["split", "horizon_days", "accession"])["ticker"]
    if not (tick.loc[part_keys].to_numpy() == m["partner_ticker"].to_numpy()).all():
        _fatal("partner_ticker mismatch between manifest and panel")
    n_sw = int(m.swapped.sum())
    print(f"[infer] swap applied: {n_sw}/{len(m)} val+test rows exchanged "
          f"({n_sw / len(m):.4f})")
    return m


# --------------------------------------------------------------------------- #
# strict checkpoint loading (predict_winner_test pattern)                      #
# --------------------------------------------------------------------------- #
def _meta_diff(expected: dict, actual: dict) -> str:
    keys = sorted(set(expected) | set(actual))
    lines = [f"    {k}: expected={expected.get(k, '<absent>')!r}  "
             f"checkpoint={actual.get(k, '<absent>')!r}"
             for k in keys if expected.get(k, "<absent>") != actual.get(k, "<absent>")]
    return "\n".join(lines) or "    (dicts differ but no key-level diff found)"


def strict_load_checkpoints(model, horizons, n_train_by_h, *, dry_run=False):
    """Load every per-horizon checkpoint with a STRICT fingerprint match.
    Returns (paths, sha256s, problems). Any mismatch is fatal unless dry_run."""
    from sp500vol.models.neural_text import _train_utils as train_utils

    problems, paths, hashes = [], {}, {}

    def _problem(msg: str) -> None:
        if dry_run:
            problems.append(msg)
            print(f"[dry-run] MISSING/MISMATCH: {msg}")
        else:
            _fatal(msg)

    for h in horizons:
        if h not in n_train_by_h:
            _fatal(f"horizon {h} has no train rows — no checkpoint can exist for it")
        path = train_utils.horizon_checkpoint_path(model, h)
        if path is None:
            _fatal(f"could not derive checkpoint path for horizon {h}")
        paths[h] = path
        if not path.exists():
            _problem(f"checkpoint missing: {path}")
            continue
        payload = train_utils._torch_load(path)
        if not isinstance(payload, dict) or "meta" not in payload or "state" not in payload:
            _problem(f"checkpoint payload invalid (no meta/state): {path}")
            continue
        expected = train_utils.checkpoint_meta(model, horizon=h, n_train=int(n_train_by_h[h]))
        if payload["meta"] != expected:
            _problem(f"checkpoint fingerprint mismatch for horizon {h} ({path}):\n"
                     + _meta_diff(expected, payload["meta"]))
            continue
        if not isinstance(payload["state"], dict):
            _problem(f"checkpoint state invalid for horizon {h}: {path}")
            continue
        model.models_[h] = payload["state"]
        hashes[str(h)] = _sha256(path)
        print(f"[infer] loaded horizon {h} checkpoint: {path.name} "
              f"(sha256 {hashes[str(h)][:12]}…)")
    return paths, hashes, problems


def _load_from_model_pkl(arm, kind, cfg, pkl, horizons, seed, *, dry_run):
    """Fallback frozen artefact when checkpoints/ is absent: the run's own
    model.pkl (written by the training process; carries the identical per-horizon
    states). Strictness: constructor parameters must match the committed config
    and the horizons must cover the panel — any mismatch is FATAL."""
    if kind == "c5":
        from sp500vol.models.neural_text.qwen_llm import C5LLMProbe as CLS
    else:
        from sp500vol.models.neural_text.finbert_s1 import FinBertS1 as CLS
    print(f"[infer] {arm}: checkpoints/ absent — loading frozen states from {pkl}")
    model = CLS.load(pkl)
    enc = cfg.get("encoder", {})
    problems = []

    def _check(name, got, want):
        if got != want:
            msg = f"{arm} model.pkl {name} mismatch: pkl={got!r} vs config={want!r}"
            if dry_run:
                problems.append(msg)
                print(f"[dry-run] MISSING/MISMATCH: {msg}")
            else:
                _fatal(msg)

    _check("pretrained", model.encoder_cfg.pretrained, str(enc.get("pretrained")))
    _check("max_length", int(model.encoder_cfg.max_length), int(enc.get("max_length", 512)))
    _check("seed", model.seed, seed)
    have = {int(h) for h in model.models_}
    if not set(int(h) for h in horizons) <= have:
        _check("horizons", sorted(have), sorted(int(h) for h in horizons))
    artefacts = {"model_pkl": {"path": str(pkl), "sha256": _sha256(pkl),
                               "mode": "model_pkl_fallback"}}
    return model, artefacts, problems


# --------------------------------------------------------------------------- #
# per-arm model preparation                                                    #
# --------------------------------------------------------------------------- #
def prepare_arm(arm, train_mod, panel, seed, *, dry_run=False, b2_tol=1e-8,
                c5_tol=1e-8, swapped=None, out_dir=None, run_dir_override=None):
    """Build the FROZEN model for `arm`. Returns (model, artefact_hashes, problems).
    Never trains on swapped data; see module discipline header."""
    kind = ARM_KIND[arm]
    run_dir = Path(run_dir_override) if run_dir_override else (
        REPO / "results" / "runs" / f"{arm}_full_{DISC}_seed{seed}")
    if not run_dir.exists():
        _fatal(f"committed run dir missing: {run_dir}")
    cfg = train_mod._load_yaml(REPO / "configs" / "models" / f"{arm}.yaml")

    if kind == "sklearn":
        return _prepare_b2(arm, train_mod, cfg, panel, run_dir, seed,
                           dry_run=dry_run, tol=b2_tol)

    # neural / c5: same factory the training run used
    from sp500vol.utils import seed_everything
    seed_everything(seed)
    model = train_mod._build_model(arm, cfg, dataset="full", run_dir=run_dir, seed=seed)
    if not getattr(model, "checkpoint", False) or getattr(model, "checkpoint_dir", None) is None:
        _fatal(f"{arm}: model built without checkpointing — nothing to load")
    train_rows = panel[panel.split == "train"]
    n_train_by_h = train_rows.groupby(train_rows.horizon_days.astype(int)).size().to_dict()
    horizons = sorted(panel.horizon_days.astype(int).unique().tolist())
    print(f"[infer] {arm}: horizons={horizons}  n_train per horizon={n_train_by_h}")

    from sp500vol.models.neural_text import _train_utils as train_utils
    n_present = sum(
        1 for h in horizons
        if (p := train_utils.horizon_checkpoint_path(model, h)) is not None and p.exists())
    have_ckpts = n_present == len(horizons)
    pkl = run_dir / "model.pkl"
    if not have_ckpts and pkl.exists():
        # fallback frozen artefact: the training process's own model.pkl (holds the
        # same per-horizon states); constructor-parameter consistency is enforced
        model, artefacts, problems = _load_from_model_pkl(
            arm, kind, cfg, pkl, horizons, seed, dry_run=dry_run)
    elif kind == "c5" and n_present == 0 and not pkl.exists():
        # prereg-ea-v1.2: C5 head artefacts LOST -> deterministic recipe rebuild on
        # the ORIGINAL train split, gated by reproduction of the committed
        # predictions (the exact B2 mechanism). Self-contained (incl. the one-pass
        # cache warm-up over original + swapped docs), so return directly.
        return _prepare_c5_rebuild(arm, model, panel, run_dir, seed,
                                   out_dir=out_dir, tol=c5_tol, dry_run=dry_run,
                                   swapped=swapped)
    else:
        # complete sets load strictly; a PARTIAL committed set stays FATAL for C5
        # too (mixed committed/rebuilt head provenance is not registered)
        paths, hashes, problems = strict_load_checkpoints(
            model, horizons, n_train_by_h, dry_run=dry_run)
        artefacts = {f"checkpoint_h{h}": {"path": str(p), "sha256": hashes.get(str(h))}
                     for h, p in paths.items()}

    if kind == "c5":
        # embedding-cache coverage report (informational; misses => GPU encode)
        cache_path = model._cache_path()
        uniq = panel.loc[panel.split.isin(SPLITS), "text_path"].astype(str).unique()
        n_hit = 0
        if cache_path is not None and cache_path.exists():
            keys = set(pd.read_parquet(cache_path, columns=["text_path"])
                       ["text_path"].astype(str))
            n_hit = sum(tp in keys for tp in uniq)
            artefacts["emb_cache"] = {"path": str(cache_path),
                                      "sha256_pre": _sha256(cache_path),
                                      "n_unique_docs": int(len(uniq)),
                                      "n_cached": int(n_hit)}
            print(f"[infer] C5 embedding cache: {n_hit}/{len(uniq)} unique val+test "
                  f"docs cached at {cache_path.name} (device={model.device.type})")
        else:
            artefacts["emb_cache"] = {"path": str(cache_path), "sha256_pre": None,
                                      "n_unique_docs": int(len(uniq)), "n_cached": 0}
            # WARNING, not a problem: the frozen encoder re-encode is deterministic
            # and pre-registered as acceptable (full GPU pass on a cold cache).
            tag = "dry-run" if dry_run else "infer"
            print(f"[{tag}] WARNING: C5 embedding cache MISSING at {cache_path} — all "
                  f"{len(uniq)} docs will be re-encoded on GPU (frozen encoder; "
                  f"deterministic; acceptable, not a blocker)")
    return model, artefacts, problems


def _prepare_c5_rebuild(arm, model, panel, run_dir, seed, *, out_dir, tol,
                        dry_run, swapped):
    """C5 head artefacts lost (no checkpoints/, no model.pkl) — prereg-ea-v1.2.

    Deterministically REBUILD the per-horizon heads with the COMMITTED recipe
    (scripts/train.py call order: seed_everything -> encode -> per-horizon heads
    with the config's early stopping) on the ORIGINAL (unswapped) train split,
    then REQUIRE reproduction of the committed predictions.parquet to within
    `tol` max relative diff BEFORE the model touches any swapped document — the
    exact B2 refit-gate mechanism. Rebuild-as-reproduction: the gate makes this
    a reconstruction of the frozen artefact, not a training degree of freedom.

    The box embedding cache is cold, so ORIGINAL + SWAPPED documents are encoded
    in ONE GPU pass up front and both are cached (the rebuild fit, the gate
    predict and the swapped predict then run entirely on cache hits).
    """
    from sp500vol.models.neural_text import _train_utils as train_utils

    problems: list[str] = []
    committed = run_dir / "predictions.parquet"
    cache_path = model._cache_path()
    # one-pass encode set: every original doc (train+val+test) + every swapped doc
    union = list(dict.fromkeys(
        panel["text_path"].astype(str).tolist()
        + ([] if swapped is None else swapped["text_path"].astype(str).tolist())))
    n_cached = 0
    if cache_path is not None and cache_path.exists():
        keys = set(pd.read_parquet(cache_path, columns=["text_path"])
                   ["text_path"].astype(str))
        n_cached = sum(tp in keys for tp in union)
    print(f"[infer] {arm}: head artefacts ABSENT (no checkpoints/, no model.pkl) — "
          f"prereg-ea-v1.2 recipe-rebuild path (rebuild-as-reproduction)")
    print(f"[infer] {arm}: embedding cache coverage {n_cached}/{len(union)} unique "
          f"original+swapped docs (cache={cache_path})")
    if not committed.exists():
        msg = f"{arm}: committed predictions missing: {committed} — reproduction gate impossible"
        if dry_run:
            problems.append(msg)
            print(f"[dry-run] MISSING/MISMATCH: {msg}")
        else:
            _fatal(msg)

    if dry_run:
        if n_cached < len(union):
            print(f"[dry-run] WARNING: {len(union) - n_cached} docs uncached — ONE full "
                  f"GPU encode pass (original+swapped together) runs before the rebuild "
                  f"(frozen encoder; deterministic; acceptable, not a blocker)")
        print(f"[dry-run] {arm}: plan = deterministic head rebuild (committed recipe, "
              f"seed {seed}, ORIGINAL train split) -> reproduction gate vs committed "
              f"predictions (max rel diff {tol:.1e}, all splits) -> swapped-document "
              f"inference")
        artefacts = {"rebuild": {"mode": "recipe_rebuild_planned", "tol": tol,
                                 "n_union_docs": int(len(union)),
                                 "n_cached": int(n_cached)}}
        return model, artefacts, problems

    if model.cache_embeddings and model.device.type != "cuda":
        _fatal(f"{arm}: rebuild requires CUDA — the committed embedding convention is "
               f"bf16/cuda (the cache filename digest encodes it); a CPU/MPS encode "
               f"cannot reproduce the committed predictions and would waste days "
               f"before failing the gate. Set CUDA_VISIBLE_DEVICES to one GPU.")

    # All-or-nothing staging: heads are rebuilt under the ELF run dir (never inside
    # the committed run dir). A PARTIAL staged set is wiped: resuming horizon k+1
    # after a checkpoint-skip of horizon k would desync the seeded RNG stream from
    # the committed straight-through run. A COMPLETE staged set may resume (the
    # reproduction gate below re-verifies it in full either way).
    if out_dir is None:
        _fatal(f"{arm}: rebuild path needs out_dir for head staging")
    staging = Path(out_dir) / "rebuilt_heads"
    model.checkpoint_dir = staging
    horizons = sorted(panel.horizon_days.astype(int).unique().tolist())
    staged_paths = {h: train_utils.horizon_checkpoint_path(model, h) for h in horizons}
    n_staged = sum(1 for p in staged_paths.values() if p is not None and p.exists())
    if 0 < n_staged < len(horizons):
        print(f"[infer] {arm}: partial staged rebuild ({n_staged}/{len(horizons)}) — "
              f"wiping for a deterministic straight-through retrain")
        for p in staged_paths.values():
            if p is not None and p.exists():
                p.unlink()
    elif n_staged == len(horizons):
        print(f"[infer] {arm}: complete staged rebuild found — resuming (fingerprint "
              f"check at load; the reproduction gate re-verifies everything)")

    # Committed call order (scripts/train.py): seed_everything -> encode -> heads.
    from sp500vol.utils import seed_everything
    seed_everything(seed)
    if union:
        print(f"[infer] {arm}: warming embedding cache — ONE pass over {len(union)} "
              f"unique original+swapped docs ({len(union) - n_cached} to encode)")
        model._encode(pd.DataFrame({"text_path": union}))

    # Recipe fit on the ORIGINAL train split — the exact scripts/train.py call
    # (X_val drives the config's early stopping, part of the committed recipe).
    # ZERO swapped data enters this fit.
    train_rows = panel[panel.split == "train"].copy()
    val_rows = panel[panel.split == "val"].copy()
    model.fit(train_rows, train_rows["label_realised_vol"].to_numpy(),
              X_val=val_rows,
              y_val=val_rows["label_realised_vol"].to_numpy() if not val_rows.empty else None)

    # Epoch-trajectory diagnostic vs the committed val_curves.json (epoch INDICES
    # only — no loss value is printed or stored); the gate below is the arbiter.
    curves_path = run_dir / "val_curves.json"
    if curves_path.exists() and getattr(model, "val_curves_", None):
        committed_curves = json.loads(curves_path.read_text())
        for h in horizons:
            ours = model.val_curves_.get(h) or model.val_curves_.get(str(h))
            ref_c = committed_curves.get(str(h))
            if ours and ref_c:
                ob = max((e["epoch"] for e in ours if e.get("is_best")), default=None)
                rb = max((e["epoch"] for e in ref_c if e.get("is_best")), default=None)
                tag = "match" if (len(ours), ob) == (len(ref_c), rb) else "DIFFER"
                print(f"[infer] {arm}: h={h} rebuild epochs {len(ours)} (best {ob}) vs "
                      f"committed {len(ref_c)} (best {rb}) — {tag}")

    # REPRODUCTION GATE (prediction-vs-prediction; labels enter NO statistic):
    # predict over the FULL panel in the committed row order (scripts/train.py
    # predicted the whole data frame), so batch composition matches the committed
    # pass exactly; compare ALL splits against the committed parquet.
    key4 = ["ticker", "accession", "horizon_days", "split"]
    ref = pd.read_parquet(committed)[key4 + ["prediction_realised_vol"]]
    if ref.duplicated(key4).any() or panel.duplicated(key4).any():
        _fatal(f"{arm}: duplicate {key4} keys — reproduction join unsafe")
    preds = model.predict(panel)
    got = panel[key4].copy()
    got["pred_new"] = np.asarray(preds, dtype=float)
    j = got.merge(ref, on=key4, how="inner", validate="one_to_one")
    if len(j) != len(got):
        _fatal(f"{arm}: reproduction join lost rows ({len(j)}/{len(got)})")
    diff = np.abs(j.pred_new.to_numpy() - j.prediction_realised_vol.to_numpy())
    rel = diff / np.maximum(np.abs(j.prediction_realised_vol.to_numpy()), 1e-12)
    max_rel = float(rel.max())
    print(f"[infer] {arm}: reproduction gate (recipe_rebuild) max|Δpred|/|pred| = "
          f"{max_rel:.3e} (tol {tol:.1e}) over {len(j)} rows, ALL splits")
    if max_rel > tol:
        _fatal(f"{arm}: reproduction gate FAILED (max rel diff {max_rel:.3e} > {tol:.1e}) "
               f"— the committed C5 heads were NOT reconstructed; do not proceed. Check "
               f"the box GPU model / torch+transformers versions against the committed "
               f"run's env.json (NVIDIA A100-SXM4-40GB, python 3.11.15) and that "
               f"CUDA_VISIBLE_DEVICES pins ONE gpu; staged heads left in {staging} "
               f"for forensics")

    artefacts = {}
    for h in horizons:
        p = train_utils.horizon_checkpoint_path(model, h)
        artefacts[f"rebuilt_checkpoint_h{h}"] = {
            "path": str(p), "sha256": _sha256(p), "mode": "recipe_rebuild"}
    artefacts["rebuild_gate"] = {"mode": "recipe_rebuild",
                                 "reproduction_max_rel_diff": max_rel, "tol": tol,
                                 "n_rows_compared": int(len(j))}
    if cache_path is not None and cache_path.exists():
        keys = set(pd.read_parquet(cache_path, columns=["text_path"])
                   ["text_path"].astype(str))
        n_after = sum(tp in keys for tp in union)
        if n_after != len(union):
            _fatal(f"{arm}: embedding cache incomplete after warm-up "
                   f"({n_after}/{len(union)}) — swapped inference would re-encode")
        artefacts["emb_cache"] = {"path": str(cache_path),
                                  "sha256_pre": _sha256(cache_path),
                                  "n_unique_docs": int(len(union)),
                                  "n_cached": int(n_after)}
    print(f"[infer] {arm}: recipe rebuild PASSED the reproduction gate — heads staged "
          f"at {staging}; proceeding to swapped-document inference")
    return model, artefacts, problems


def _prepare_b2(arm, train_mod, cfg, panel, run_dir, seed, *, dry_run, tol):
    """B2: stored model.pkl preferred; else recipe refit on ORIGINAL train rows,
    gated by exact reproduction of the committed val/test predictions."""
    from sp500vol.models.classical_text import TfidfRidge

    problems: list[str] = []
    pkl = run_dir / "model.pkl"
    committed = run_dir / "predictions.parquet"
    if not committed.exists():
        _fatal(f"{arm}: committed predictions missing: {committed}")

    if dry_run:
        print(f"[dry-run] {arm}: model.pkl {'EXISTS' if pkl.exists() else 'MISSING -> recipe refit + reproduction gate'}")
        return None, {"model_pkl": {"path": str(pkl),
                                    "sha256": _sha256(pkl) if pkl.exists() else None,
                                    "mode": "stored" if pkl.exists() else "refit"}}, problems

    if pkl.exists():
        model = TfidfRidge.load(pkl)
        mode, sha = "stored", _sha256(pkl)
        print(f"[infer] {arm}: loaded stored model.pkl (sha256 {sha[:12]}…)")
    else:
        # deterministic recipe refit on the ORIGINAL train split (frozen-model
        # reconstruction, NOT training on swapped data; train rows untouched)
        print(f"[infer] {arm}: model.pkl absent — recipe refit on ORIGINAL train split")
        from sp500vol.utils import seed_everything
        seed_everything(seed)
        model = train_mod._build_model(arm, cfg, dataset="full", run_dir=run_dir, seed=seed)
        train_rows = panel[panel.split == "train"].copy()
        model.fit(train_rows, train_rows["label_realised_vol"].to_numpy())
        mode, sha = "refit", None

    # REPRODUCTION GATE (prediction-vs-prediction; no label enters any statistic):
    # the reconstructed/loaded model must reproduce the committed val+test
    # predictions on the ORIGINAL documents to within `tol`.
    ref = pd.read_parquet(committed)
    ref = ref[ref.split.isin(SPLITS)][["ticker", "accession", "horizon_days", "split",
                                       "prediction_realised_vol"]]
    vt = panel[panel.split.isin(SPLITS)].copy().reset_index(drop=True)
    preds = np.empty(len(vt), dtype=float)
    for (split, h), grp in vt.groupby(["split", "horizon_days"]):
        preds[grp.index.to_numpy()] = model.predict(grp)
    got = vt[["ticker", "accession", "horizon_days", "split"]].copy()
    got["pred_new"] = preds
    j = got.merge(ref, on=["ticker", "accession", "horizon_days", "split"],
                  how="inner", validate="one_to_one")
    if len(j) != len(got):
        _fatal(f"{arm}: reproduction join lost rows ({len(j)}/{len(got)})")
    diff = np.abs(j.pred_new.to_numpy() - j.prediction_realised_vol.to_numpy())
    rel = diff / np.maximum(np.abs(j.prediction_realised_vol.to_numpy()), 1e-12)
    max_rel = float(rel.max())
    print(f"[infer] {arm}: reproduction gate ({mode}) max|Δpred|/|pred| = {max_rel:.3e} "
          f"(tol {tol:.1e}) over {len(j)} val+test rows")
    if max_rel > tol:
        _fatal(f"{arm}: reproduction gate FAILED (max rel diff {max_rel:.3e} > {tol:.1e}) "
               f"— frozen model NOT reconstructed; do not proceed (check sklearn version "
               f"vs run_dir/env.json, or ship model.pkl)")
    artefacts = {"model_pkl": {"path": str(pkl), "sha256": sha, "mode": mode,
                               "reproduction_max_rel_diff": max_rel, "tol": tol}}
    return model, artefacts, problems


# --------------------------------------------------------------------------- #
# prediction loop (resumable per (split, horizon) part)                        #
# --------------------------------------------------------------------------- #
def predict_arm(arm, model, swapped, out_dir, *, smoke=None):
    """Predict swapped val+test rows per (split, horizon) part; resume on rerun.

    Per-(split,horizon) parts are REQUIRED for correctness of B1/B2-style models:
    their predict() dedups texts by accession (one text per filing), while the
    swap is PER-HORIZON — the same accession may carry different partners'
    documents at different horizons. Within one horizon accessions are unique,
    so the dedup is harmless.
    """
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    pred_cols = ["ticker", "accession", "horizon_days", "split",
                 "label_realised_vol", "text_path"]
    frames = []
    for split in SPLITS:
        for h in sorted(swapped.horizon_days.astype(int).unique()):
            tag = f"{split}_h{h}" + ("" if smoke is None else f"_smoke{smoke}")
            part_path = parts_dir / f"{tag}.parquet"
            if part_path.exists():
                frames.append(pd.read_parquet(part_path))
                print(f"[infer] {arm}: part {tag} exists — resumed")
                continue
            rows = swapped[(swapped.split == split)
                           & (swapped.horizon_days == h)][pred_cols].reset_index(drop=True)
            if smoke is not None:
                rows = rows.head(smoke).reset_index(drop=True)
            if rows.duplicated("accession").any():
                _fatal(f"{arm}: duplicate accession within part {tag} — dedup unsafe")
            print(f"[infer] {arm}: predicting part {tag} ({len(rows)} rows)")
            preds = model.predict(rows)
            if not np.isfinite(np.asarray(preds, dtype=float)).all():
                _fatal(f"{arm}: non-finite predictions in part {tag}")
            out = rows[OUT_COLS[:-1]].copy()
            out["prediction_realised_vol"] = np.asarray(preds, dtype=float)
            tmp = part_path.with_suffix(".parquet.tmp")
            out.to_parquet(tmp, index=False)
            os.replace(tmp, part_path)
            frames.append(out)
    return pd.concat(frames, ignore_index=True)


def run_arm(arm, train_mod, panel, swapped, manifest_sha, args):
    seed = args.seed
    out_dir = REPO / "results" / "runs" / f"ELF_swap_{arm}_full_{DISC}_seed{seed}"
    sentinel = out_dir / "_DONE"
    fname = ("predictions_swapped.parquet" if args.smoke is None
             else f"predictions_swapped_smoke{args.smoke}.parquet")
    if sentinel.exists() and args.smoke is None:
        print(f"[infer] {arm}: sentinel {sentinel} exists — already done, skipping")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    model, artefacts, problems = prepare_arm(
        arm, train_mod, panel, seed, dry_run=args.dry_run, b2_tol=args.b2_tol,
        c5_tol=args.c5_tol, swapped=swapped, out_dir=out_dir)
    if args.dry_run:
        status = "READY" if not problems else f"{len(problems)} problem(s)"
        print(f"[dry-run] {arm}: {status}")
        return problems

    # pin one token cache across parts (C2 path; harmless elsewhere)
    if hasattr(model, "_new_token_cache"):
        model._tok_cache = model._new_token_cache()
        model._tok_cache_pinned = True

    out = predict_arm(arm, model, swapped, out_dir, smoke=args.smoke)

    # output guards: counts + finiteness only — never label-vs-prediction
    if list(out.columns) != OUT_COLS:
        _fatal(f"{arm}: output columns drifted: {list(out.columns)}")
    n_expected = len(swapped) if args.smoke is None else None
    if n_expected is not None and len(out) != n_expected:
        _fatal(f"{arm}: output rows {len(out)} != swapped panel rows {n_expected}")

    # G3 — artefact-hash invariance: re-hash AFTER prediction; any change is fatal
    g3 = {"pass": True, "post": {}}
    for name, rec in artefacts.items():
        p = Path(rec["path"]) if rec.get("path") else None
        pre = rec.get("sha256") or rec.get("sha256_pre")
        if p is None or pre is None or not p.exists():
            continue
        post = _sha256(p)
        g3["post"][name] = post
        if post != pre:
            g3["pass"] = False
            _fatal(f"{arm}: G3 VIOLATION — artefact {name} changed during inference "
                   f"({pre[:12]}… -> {post[:12]}…)")
    print(f"[infer] {arm}: G3 artefact-hash invariance — PASS "
          f"({len(g3['post'])} artefacts re-hashed unchanged)")

    out_path = out_dir / fname
    out.to_parquet(out_path, index=False)
    meta = {
        "prereg": "configs/prereg_swap_lf_and_anon.md §E-lf (prereg-ea-v1.0 + v1.2 amendment)",
        "arm": arm, "seed": seed, "smoke": args.smoke,
        "manifest_sha256": manifest_sha,
        "rows_out_by_split": {k: int(v) for k, v in out.split.value_counts().items()},
        "n_swapped_rows": int(swapped.swapped.sum()),
        "artefacts": artefacts,
        "g3_hash_invariance": g3,
        "output": str(out_path),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / fname.replace(".parquet", "_meta.json")).write_text(
        json.dumps(meta, indent=2, default=str))
    if args.smoke is None:
        sentinel.write_text(meta["generated_utc"])
    print(f"[infer] {arm}: wrote {len(out)} rows -> {out_path}")
    print(f"[infer] {arm}: predictions only — no statistic was computed on any split")


# --------------------------------------------------------------------------- #
# selftest — synthetic corpus; no GPU, no HF, nothing under results/           #
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    print("[selftest] synthetic end-to-end validation (B2 flow + strict-fingerprint logic)")
    from swap_longform_build import build_manifest
    from sp500vol.models.classical_text import TfidfRidge

    rng = np.random.default_rng(2026)
    with tempfile.TemporaryDirectory(prefix="elf_selftest_") as td:
        td = Path(td)
        # --- tiny corpus: 4 firms x 6 days x 2 horizons, distinct doc contents
        firms = ["AAA", "BBB", "CCC", "DDD"]
        vocab = {"AAA": "alpha growth revenue", "BBB": "beta risk litigation",
                 "CCC": "gamma supply demand", "DDD": "delta merger guidance"}
        rows = []
        acc_i = 0
        for d in range(9):
            day = pd.Timestamp("2020-01-06") + pd.Timedelta(days=d)
            split = "train" if d < 3 else ("val" if d < 6 else "test")
            for f in firms:
                acc = f"acc{acc_i:04d}"
                acc_i += 1
                doc = td / f"{acc}.txt"
                doc.write_text(f"{vocab[f]} filing {acc} " + " ".join(
                    rng.choice(list("abcdefgh"), size=20)))
                for h in (5, 10):
                    rows.append({"ticker": f, "accession": acc, "horizon_days": h,
                                 "split": split,
                                 "label_realised_vol": float(0.1 + 0.05 * rng.random()),
                                 "filing_time_utc": pd.Timestamp(day, tz="UTC"),
                                 "effective_trading_day": day,
                                 "text_path": str(doc)})
        panel = pd.DataFrame(rows)

        manifest, stats = build_manifest(panel, horizons=(5, 10), min_rows=(4, 4))
        assert manifest.swapped.any(), "selftest manifest swapped nothing"
        print(f"[selftest] manifest: {len(manifest)} rows, "
              f"{int(manifest.swapped.sum())} swapped")

        swapped = apply_swap(panel, manifest)
        # per-horizon partner correctness: row's text_path == partner row's original
        chk = swapped.merge(
            panel[panel.split.isin(SPLITS)][["split", "horizon_days", "accession", "text_path"]]
            .rename(columns={"accession": "partner_accession", "text_path": "tp_partner"}),
            on=["split", "horizon_days", "partner_accession"], how="left")
        assert (chk.text_path == chk.tp_partner).all(), "swap application wrong"
        print("[selftest] apply_swap: partner text_path exchange verified")

        # --- B2 flow: fit on ORIGINAL train, predict original vs swapped;
        #     document-swap must equal forecast-swap for a text-pure model.
        model = TfidfRidge(max_features=50, ridge_alpha=1.0, log_target=True)
        tr = panel[panel.split == "train"]
        model.fit(tr, tr.label_realised_vol.to_numpy())
        out_dir = td / "out"
        got = predict_arm("B2_selftest", model, swapped, out_dir)
        vt = panel[panel.split.isin(SPLITS)].reset_index(drop=True).copy()
        orig = np.empty(len(vt))
        for (s, h), grp in vt.groupby(["split", "horizon_days"]):
            orig[grp.index.to_numpy()] = model.predict(grp)
        vt["pred_orig"] = orig
        j = got.merge(manifest[["split", "horizon_days", "accession", "partner_accession"]],
                      on=["split", "horizon_days", "accession"])
        j = j.merge(vt[["split", "horizon_days", "accession", "pred_orig"]]
                    .rename(columns={"accession": "partner_accession"}),
                    on=["split", "horizon_days", "partner_accession"])
        max_d = float(np.abs(j.prediction_realised_vol - j.pred_orig).max())
        assert max_d < 1e-12, f"document-swap != forecast-swap for text-pure model ({max_d})"
        print(f"[selftest] B2 equivalence: re-inferred swapped preds == permuted "
              f"original preds (max|Δ|={max_d:.2e})")
        # resume path: second call must reuse parts untouched
        got2 = predict_arm("B2_selftest", model, swapped, out_dir)
        assert np.allclose(got.prediction_realised_vol, got2.prediction_realised_vol)
        print("[selftest] resume-from-parts verified")

        # --- strict fingerprint logic on a dummy checkpoint (no HF/GPU)
        import torch
        from sp500vol.models.neural_text import _train_utils as train_utils
        from sp500vol.models.neural_text.encoders import EncoderConfig

        class Dummy:
            name = "DUMMY_arm"
            encoder_cfg = EncoderConfig(pretrained="none/none", max_length=8)
            seed = 2026
            strategy = "S1"
            checkpoint = True
            checkpoint_dir = td / "ckpt"
            hidden_dim, dropout, lr = 4, 0.0, 1e-3
            models_: dict = {}

        dummy = Dummy()
        state = {"head_state": {"w": torch.zeros(2)}}
        train_utils.save_horizon_checkpoint(dummy, horizon=5, n_train=12, state=state)
        _, hashes, probs = strict_load_checkpoints(dummy, [5], {5: 12}, dry_run=True)
        assert not probs and 5 in dummy.models_, "matching fingerprint failed to load"
        dummy2 = Dummy()
        dummy2.models_ = {}
        dummy2.seed = 2027  # tampered -> meta mismatch
        _, _, probs2 = strict_load_checkpoints(dummy2, [5], {5: 12}, dry_run=True)
        assert probs2, "fingerprint mismatch NOT detected"
        print("[selftest] strict checkpoint fingerprint: match loads, mismatch aborts")

        # --- C5 recipe-rebuild + reproduction gate (prereg-ea-v1.2): CPU only,
        #     stubbed encoder (no GPU/HF). Verifies: deterministic rebuild passes
        #     the gate; a complete staged set resumes via fingerprint load; a
        #     PARTIAL staged set is wiped (all-or-nothing); dry-run plans without
        #     problems; tampered committed predictions fail the gate fatally.
        import shutil

        from sp500vol.models.neural_text.qwen_llm import C5LLMProbe
        from sp500vol.utils import seed_everything

        def _fake_encode(df):
            # deterministic per text_path; consumes NO global RNG stream
            out = np.empty((len(df), 16), dtype=np.float32)
            for i, tp in enumerate(df["text_path"].astype(str)):
                hh = int(hashlib.sha256(tp.encode()).hexdigest()[:8], 16)
                out[i] = np.random.default_rng(hh).standard_normal(16).astype(np.float32)
            return out

        def _mk_c5(ckpt_dir):
            m = C5LLMProbe(pretrained="stub/none", max_length=8, hidden_dim=8,
                           dropout=0.1, lr=1e-3, weight_decay=0.0, batch_size=8,
                           max_epochs=3, early_stopping=True, es_patience=2,
                           es_min_delta=0.0, mixed_precision="no", warmup_ratio=0.0,
                           log_target=True, cache_embeddings=False, device="cpu",
                           checkpoint=True, checkpoint_dir=ckpt_dir, seed=2026,
                           strategy="C5_selftest")
            m.embedding_dim_ = 16
            m._encode = _fake_encode
            return m

        c5_run_dir = td / "c5_run"
        c5_run_dir.mkdir()
        elf_dir = td / "c5_elf_out"
        # simulate the COMMITTED run (scripts/train.py order): seed -> fit(train,
        # X_val=val) -> predict(FULL panel) -> predictions.parquet; the heads are
        # then "lost" (their dir is simply never consulted again)
        seed_everything(2026)
        ref_model = _mk_c5(td / "c5_committed_ckpts")
        tr_c5 = panel[panel.split == "train"].copy()
        va_c5 = panel[panel.split == "val"].copy()
        ref_model.fit(tr_c5, tr_c5.label_realised_vol.to_numpy(),
                      X_val=va_c5, y_val=va_c5.label_realised_vol.to_numpy())
        committed_c5 = panel[["ticker", "accession", "horizon_days", "split"]].copy()
        committed_c5["prediction_realised_vol"] = ref_model.predict(panel)
        committed_c5.to_parquet(c5_run_dir / "predictions.parquet", index=False)

        # 1) rebuild from nothing must reproduce the committed predictions
        m2 = _mk_c5(td / "c5_placeholder")
        _, arte, probs3 = _prepare_c5_rebuild(
            "C5_selftest", m2, panel, c5_run_dir, 2026, out_dir=elf_dir, tol=1e-8,
            dry_run=False, swapped=swapped)
        assert not probs3
        assert arte["rebuild_gate"]["reproduction_max_rel_diff"] <= 1e-8
        staged = sorted((elf_dir / "rebuilt_heads").glob("horizon_*.pt"))
        assert len(staged) == 2, f"expected 2 staged heads, got {staged}"
        print(f"[selftest] C5 recipe-rebuild: gate PASSED (max rel diff "
              f"{arte['rebuild_gate']['reproduction_max_rel_diff']:.2e})")

        # 2) all-or-nothing: a PARTIAL staged set is wiped, retrained, re-gated
        staged[0].unlink()
        m3 = _mk_c5(td / "c5_placeholder2")
        _, arte3, _ = _prepare_c5_rebuild(
            "C5_selftest", m3, panel, c5_run_dir, 2026, out_dir=elf_dir, tol=1e-8,
            dry_run=False, swapped=swapped)
        assert arte3["rebuild_gate"]["reproduction_max_rel_diff"] <= 1e-8
        print("[selftest] C5 recipe-rebuild: partial staging wiped; straight-through "
              "retrain reproduced again")

        # 3) a COMPLETE staged set resumes via fingerprint load; gate re-verifies
        m6 = _mk_c5(td / "c5_placeholder3")
        _, arte6, _ = _prepare_c5_rebuild(
            "C5_selftest", m6, panel, c5_run_dir, 2026, out_dir=elf_dir, tol=1e-8,
            dry_run=False, swapped=swapped)
        assert arte6["rebuild_gate"]["reproduction_max_rel_diff"] <= 1e-8
        print("[selftest] C5 recipe-rebuild: complete staged set resumed and re-gated")

        # 4) dry-run reports the plan and no problems
        m4 = _mk_c5(td / "c5_placeholder4")
        _, _, probs4 = _prepare_c5_rebuild(
            "C5_selftest", m4, panel, c5_run_dir, 2026, out_dir=td / "c5_elf_dry",
            tol=1e-8, dry_run=True, swapped=swapped)
        assert probs4 == [], f"dry-run rebuild plan should not be a problem: {probs4}"
        print("[selftest] C5 recipe-rebuild: dry-run plans with zero problems")

        # 5) tampered committed predictions must FAIL the gate fatally
        bad = committed_c5.copy()
        bad.loc[bad.index[0], "prediction_realised_vol"] *= 1.001
        bad.to_parquet(c5_run_dir / "predictions.parquet", index=False)
        shutil.rmtree(elf_dir / "rebuilt_heads")
        m5 = _mk_c5(td / "c5_placeholder5")
        try:
            _prepare_c5_rebuild("C5_selftest", m5, panel, c5_run_dir, 2026,
                                out_dir=elf_dir, tol=1e-8, dry_run=False,
                                swapped=swapped)
            raise AssertionError("tampered committed predictions passed the gate")
        except SystemExit as e:
            assert "reproduction gate FAILED" in str(e), f"wrong failure: {e}"
        print("[selftest] C5 recipe-rebuild: tampered committed predictions -> "
              "gate FATAL")

    print("[selftest] ALL PASS — nothing under results/ was touched")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", default="all",
                    choices=["all", *ARM_ORDER], help="which frozen arm to re-infer")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--manifest", type=str, default=None,
                    help="manifest parquet (default: $SP500VOL_DATA_ROOT/processed/"
                         "swap_lf/swap_manifest_long_form.parquet)")
    ap.add_argument("--smoke", type=int, default=None, metavar="N",
                    help="keep N rows per (split,horizon); *_smoke outputs; no sentinel")
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight: verify panel/manifest alignment, checkpoint "
                         "fingerprints, C5 cache coverage, B2 artefacts; predict nothing")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic validation; no GPU/HF; touches nothing under results/")
    ap.add_argument("--b2-tol", type=float, default=1e-8,
                    help="B2 reproduction-gate max relative diff (loosening requires "
                         "disclosure in the scoring md)")
    ap.add_argument("--c5-tol", type=float, default=1e-8,
                    help="C5 recipe-rebuild reproduction-gate max relative diff "
                         "(prereg-ea-v1.2; loosening requires disclosure in the "
                         "scoring md)")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    from sp500vol.utils.paths import data_path
    manifest_path = (Path(args.manifest) if args.manifest else
                     data_path("processed", "swap_lf", "swap_manifest_long_form.parquet"))
    if not manifest_path.exists():
        _fatal(f"manifest not found: {manifest_path} — run swap_longform_build.py first")
    manifest = pd.read_parquet(manifest_path)
    manifest_sha = _sha256(manifest_path)
    print(f"[infer] manifest: {manifest_path} ({len(manifest)} rows, "
          f"sha256 {manifest_sha[:12]}…)")

    train_mod = _load_train_mod()
    panel = load_full_long_form_panel(train_mod)
    print(f"[infer] full long_form panel rows by split: "
          f"{panel.split.value_counts().to_dict()}")
    swapped = apply_swap(panel, manifest)

    if args.arm == "all":
        arms = list(ARMS_ALL_V12)
        print(C2_LOST_NOTE)
    else:
        arms = [args.arm]
    any_problem = False
    for arm in arms:
        print(f"\n[infer] ===== arm {arm} =====")
        probs = run_arm(arm, train_mod, panel, swapped, manifest_sha, args)
        if args.dry_run and probs:
            any_problem = True
    if args.dry_run:
        print(f"\n[dry-run] {'PROBLEMS FOUND — see above' if any_problem else 'ALL READY'}")
        return 3 if any_problem else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
