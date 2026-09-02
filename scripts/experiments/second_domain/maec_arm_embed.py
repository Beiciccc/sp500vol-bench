#!/usr/bin/env python
"""MAEC audit — frozen-encoder embedding step, GPU box side (prereg
configs/prereg_maec_audit.md §5-4, OPEN-5: Qwen3-Emb-8B mean-pool + ridge).

Mirror of yelp_arm_embed.py `encode` (same embedder family, same vLLM pooling
runner, same head-only truncation discipline) collapsed to the MAEC scale:
3,443 calls fit one single-GPU pass, so there is ONE output parquet with ONE
row per call_id — exactly what the CPU fit stage consumes:

    maec_baseline_text.py --arms qwen_emb --embeddings <out>

ENCODER CONTRACT (fit-stage discipline: no label is read, no metric computed):
  * text     : <call dir>/text.txt, whitespace-normalised sentence-per-line ->
               single spaces — byte-identical to maec_baseline_text.load_texts,
               so the frozen-embedding arm reads the exact string the TF-IDF
               arm was fit on.
  * truncate : HEAD-only (OPEN-12 discipline). First the maec_prompt char
               budget (TRANSCRIPT_CHAR_BUDGET = 48,000 chars ~ 12k tokens under
               chars/4), then a real-tokenizer re-check truncates to
               max_model_len - 8 tokens KEEPING THE HEAD (vLLM's native
               truncate_prompt_tokens keeps the TAIL and is deliberately NOT
               used). Both truncation trigger counts are disclosed in
               <out>.meta.json (median transcript ~2.7k tokens: rare).
  * pooling  : MEAN over the truncated transcript (§5-4 pins "mean-pool"),
               requested explicitly via vLLM override_pooler_config; if the
               installed vLLM predates PoolerConfig the model-default pooler is
               used and DISCLOSED in the meta sidecar (the Yelp arm relied on
               the same runner="pooling"/convert="embed" path).
  * output   : parquet [call_id, emb_0 .. emb_{D-1}] float32, one row per
               manifest call (all 3,443 — the fit stage inner-merges onto the
               panel, so §3-excluded calls are simply never consumed), plus
               <out>.meta.json provenance (model, dim, pooling, truncations).

BOX LAUNCH (single GPU, offline HF cache; row16 launch.sh env conventions):
    cd /root/gpu-data/repo
    export HF_HOME=/root/gpu-data/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
    CUDA_VISIBLE_DEVICES=0 /root/gpu-data/venvs/main/bin/python \
        scripts/experiments/second_domain/maec_arm_embed.py \
        --manifest /root/gpu-data/second-domain/earnings_calls/maec_manifest.parquet \
        --texts-root /root/gpu-data/second-domain/earnings_calls/MAEC/MAEC_Dataset \
        --out /root/gpu-data/second-domain/earnings_calls/maec_qwen3emb.parquet \
        --model Qwen/Qwen3-Embedding-8B
(Local-path equivalents on the Mac: /path/to/data-root/second-domain/... .)
"""
from __future__ import annotations

import os

# thread caps before numpy (<= 4; box may export higher values first)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maec_prompt import TRANSCRIPT_CHAR_BUDGET  # noqa: E402  (frozen budget)

EC_LOCAL = "/path/to/data-root/second-domain/earnings_calls"


def load_text(path: Path) -> str:
    """maec_baseline_text.load_texts normalisation: sentence-per-line -> spaces."""
    return " ".join(path.read_text(encoding="utf-8", errors="replace").split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=f"{EC_LOCAL}/maec_manifest.parquet")
    ap.add_argument("--texts-root", default=None,
                    help="MAEC_Dataset root; text = <root>/<call_id>/text.txt "
                         "(default: use the manifest's absolute `path` column)")
    ap.add_argument("--out", default=f"{EC_LOCAL}/maec_qwen3emb.parquet")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--max-model-len", type=int, default=16_384)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--limit", type=int, default=0,
                    help="first N calls only (smoke run)")
    args = ap.parse_args()

    t0 = time.time()
    mf = pd.read_parquet(args.manifest)[["call_id", "path", "n_chars"]]
    assert not mf["call_id"].duplicated().any(), "duplicate call_id in manifest"
    mf = mf.sort_values("call_id", kind="mergesort").reset_index(drop=True)
    if args.limit:
        mf = mf.head(args.limit)
    print(f"[encode] {len(mf):,} calls, model={args.model}")

    texts, n_char_trunc = [], 0
    for r in mf.itertuples(index=False):
        p = (Path(args.texts_root) / r.call_id / "text.txt" if args.texts_root
             else Path(r.path))
        t = load_text(p)
        assert len(t) > 0, f"empty transcript for {r.call_id} at {p}"
        if len(t) > TRANSCRIPT_CHAR_BUDGET:          # head-only char budget
            t = t[:TRANSCRIPT_CHAR_BUDGET]
            n_char_trunc += 1
        texts.append(t)
    print(f"[encode] char-budget ({TRANSCRIPT_CHAR_BUDGET:,}) head-truncations: "
          f"{n_char_trunc}")

    # real-tokenizer re-check: HEAD-keeping truncation to max_model_len - 8
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    budget = args.max_model_len - 8
    n_tok_trunc = 0
    for i, t in enumerate(texts):
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) > budget:
            texts[i] = tok.decode(ids[:budget], skip_special_tokens=True)
            n_tok_trunc += 1
    print(f"[encode] tokenizer head-truncations to {budget:,} tokens: {n_tok_trunc}")

    from vllm import LLM
    try:
        from vllm.config import PoolerConfig
        _pc = PoolerConfig(pooling_type="MEAN")
    except Exception:
        _pc = None
    base = dict(model=args.model, max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_mem, enforce_eager=False)
    # Progressive kwarg fallback across vLLM versions: newer renamed
    # override_pooler_config -> pooler_config; older used task="embed".
    attempts = []
    if _pc is not None:
        attempts += [({"runner": "pooling", "convert": "embed", "pooler_config": _pc},
                      "MEAN (pooler_config, §5-4)"),
                     ({"runner": "pooling", "convert": "embed", "override_pooler_config": _pc},
                      "MEAN (override_pooler_config, §5-4)")]
    attempts += [({"runner": "pooling", "convert": "embed"},
                  "model-default (runner=pooling/convert=embed; MEAN unavailable, DISCLOSED)"),
                 ({"task": "embed"},
                  "model-default (task=embed legacy; MEAN unavailable, DISCLOSED)")]
    llm, pooling = None, None
    for kw, label in attempts:
        try:
            llm = LLM(**base, **kw)
            pooling = label
            break
        except TypeError as e:
            print(f"[warn] LLM({sorted(kw)}) rejected: {e} — trying next fallback")
    if llm is None:
        raise RuntimeError("no LLM constructor variant accepted — vLLM API mismatch")
    print(f"[encode] pooling mode: {pooling}")
    embed_fn = getattr(llm, "embed", None) or llm.encode
    outs = embed_fn(texts)
    def _vec(o):
        e = getattr(o.outputs, "embedding", None)
        if e is None:
            e = o.outputs.data
            e = e.tolist() if hasattr(e, "tolist") else e
        return e
    emb = np.asarray([_vec(o) for o in outs], dtype=np.float32)
    assert emb.shape[0] == len(mf) and np.isfinite(emb).all(), \
        "embedding count/finiteness check failed"

    df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    df.insert(0, "call_id", mf["call_id"].to_numpy())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prereg": "configs/prereg_maec_audit.md §5-4 (OPEN-5)",
        "model": args.model, "pooling": pooling,
        "n_calls": int(len(mf)), "dim": int(emb.shape[1]),
        "max_model_len": args.max_model_len,
        "char_budget": TRANSCRIPT_CHAR_BUDGET,
        "n_char_truncated": int(n_char_trunc),
        "token_budget": budget, "n_token_truncated": int(n_tok_trunc),
        "text_normalisation": "whitespace-join (maec_baseline_text.load_texts)",
        "truncation": "HEAD-only at both stages (OPEN-12 discipline)",
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[encode] wrote {out}  shape={emb.shape}  "
          f"({time.time() - t0:.0f}s); meta -> {out}.meta.json")


if __name__ == "__main__":
    main()
