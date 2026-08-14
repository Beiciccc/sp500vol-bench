#!/usr/bin/env python
"""MAEC audit — prompted-arm vLLM batch runner + v-space collect step (prereg
configs/prereg_maec_audit.md §5-2 prompted text arm / §5-3 identity probe,
tag prereg-maec-v1.0).

The prompt / parse / clip machinery is FROZEN in maec_prompt.py (build_messages,
parse_output, to_v, JSON_SCHEMA, RETRY_SUFFIX); this runner only moves data.
The e1_llm_forecast/run_inference.py protocol is kept where §5-2 pins it
(guided JSON when available else regex fallback + ONE retry pass at temp 0.2,
temperature 0 first pass, thinking OFF, real-tokenizer budget re-fit,
checkpoint-every 500, resumable) — but the data plumbing is MAEC's own.

Two subcommands:

  infer   (GPU box; --mock runs without vLLM for local validation)
      Call list = unique call_ids of maec_panel.parquet ACROSS BOTH alignments
      (verified identical; 3,410 of the 3,443 manifest calls — the difference
      is exactly the prereg §3 exclusions: 32 stubs excluded from ALL arms
      (§3.1) + ambiguity/price-gate calls that can never join the panel; the
      shortfall is printed when --manifest is given). Each call is prompted
      ONCE per variant — one generation covers all four horizons (§2.2) and
      predictions are split-invariant, so train/val/test all get rows.
      Variants (maec_prompt): maec_text (§5-2, head-12k-token transcript) and
      maec_identity (§5-3 zero-content probe: ticker + CRSP comnam + date).
      Raw parquet parts (one row per call x variant):
        [call_id, ticker, call_date, variant, model_name, raw_output,
         vol_ann_pct_3, vol_ann_pct_7, vol_ann_pct_15, vol_ann_pct_30,
         parse_ok, retry_used, prompt_chars, transcript_truncated]
      vol_ann_pct_* are the parsed annualised-% forecasts (pre-clip; the frozen
      [3,300]% clip + v-conversion happen in `collect`).

  collect (local Mac, CPU; fit-stage discipline — no metric, no label read
      beyond copying the panel's label column into the loader schema)
      Consolidates part files and emits, per variant, the §5-2 post-step:
        v_hat = to_v(ann_pct) = ln(clip(ann,3,300)%/100/sqrt(252))    (frozen)
      joined onto the panel rows for BOTH alignments in ONE loader file
      (KEY = (permno, call_date, horizon) is identical across alignments —
      asserted; rows carry the PRIMARY label so the protocol's primary-merge
      label assert holds, plus any shifted-only KEY rows, disclosed):
        results/second_domain/maec/preds/preds_prompted_qwen.parquet
        results/second_domain/maec/preds/preds_identity_probe.parquet
      schema [permno, call_date, horizon, call_id, alignment, split, label,
      prediction, arm] — the maec_protocol.py loader contract. Parse failures
      are filled with the arm's val-split mean PREDICTION per horizon (Yelp
      collect precedent; counts + fill values disclosed in
      llm_collect_stats.json). --emit-published additionally writes
      preds_<arm>_published.parquet (Table-5 year-panel row assignment via
      maec_baseline_text.assign_published — split-invariant predictions need
      no refit) for the maec_published_scorer.py G1 reading; default OFF.

BOX LAUNCH (single GPU, offline HF cache; row16 launch.sh env conventions;
sequence per the audit plan: embeddings first, then text arm, then probe):
    cd /root/rivermind-data/repo
    export HF_HOME=/root/rivermind-data/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
    # (1) §5-2 full-transcript text arm
    CUDA_VISIBLE_DEVICES=0 /root/rivermind-data/venvs/main/bin/python \
        scripts/experiments/second_domain/maec_llm_infer.py infer \
        --panel /root/rivermind-data/second-domain/earnings_calls/maec_panel.parquet \
        --texts-root /root/rivermind-data/second-domain/earnings_calls/MAEC/MAEC_Dataset \
        --variant maec_text --model Qwen/Qwen3-32B-AWQ --tp 1 \
        --out-dir /root/rivermind-data/second-domain/earnings_calls/maec_llm_raw
    # (2) §5-3 zero-content identity probe (same out-dir; resume-safe, keyed
    #     on (call_id, variant))
    CUDA_VISIBLE_DEVICES=0 /root/rivermind-data/venvs/main/bin/python \
        scripts/experiments/second_domain/maec_llm_infer.py infer \
        --panel /root/rivermind-data/second-domain/earnings_calls/maec_panel.parquet \
        --texts-root /root/rivermind-data/second-domain/earnings_calls/MAEC/MAEC_Dataset \
        --variant maec_identity --model Qwen/Qwen3-32B-AWQ --tp 1 \
        --out-dir /root/rivermind-data/second-domain/earnings_calls/maec_llm_raw

COLLECT (local Mac, after rsync-back of maec_llm_raw):
    .venv/bin/python scripts/experiments/second_domain/maec_llm_infer.py collect \
        --raw-dir /Volumes/Z/second-domain/earnings_calls/maec_llm_raw \
        --panel /Volumes/Z/second-domain/earnings_calls/maec_panel.parquet \
        --out-dir results/second_domain/maec/preds

Local mock validation (no GPU; synthetic fixture, never real data):
    ... infer --mock --panel <synthetic> --out-dir <scratch>  then  collect ...
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
from maec_prompt import (
    CLIP_PCT_HI,
    CLIP_PCT_LO,
    HORIZONS,
    JSON_SCHEMA,
    MAX_MODEL_LEN,
    RETRY_SUFFIX,
    VOL_KEYS,
    build_excerpt,
    build_messages,
    horizon_of,
    parse_output,
    to_v,
)

REPO = Path(__file__).resolve().parents[3]
KEY = ["permno", "call_date", "horizon"]
CLIP_V_LO, CLIP_V_HI = float(np.log(1e-4)), 0.0     # §5 combiner clip range
VARIANTS = ("maec_text", "maec_identity")
ARM_OF = {"maec_text": "prompted_qwen", "maec_identity": "identity_probe"}
EC_LOCAL = "/Volumes/Z/second-domain/earnings_calls"
PCT_COLS = [f"vol_ann_pct_{h}" for h in HORIZONS]


# ==================================================================== generators
class MockGenerator:
    """No-GPU stand-in: plausible annualised-% JSON; a fraction of first-pass
    outputs is deliberately unparseable so the retry path is exercised."""

    name = "mock"

    def __init__(self, seed: int = 2026, garbage_rate: float = 0.2):
        self.rng = np.random.default_rng(seed)
        self.garbage_rate = garbage_rate

    def generate(self, records: list[dict], retry: bool = False) -> list[str]:
        outs = []
        for _rec in records:
            if not retry and self.rng.random() < self.garbage_rate:
                outs.append("Volatility depends on many macro factors.")
                continue
            base = float(np.exp(self.rng.normal(np.log(30.0), 0.4)))
            vols = {k: round(float(np.clip(base * np.exp(self.rng.normal(0, 0.1)),
                                           5.0, 250.0)), 2) for k in VOL_KEYS}
            outs.append(json.dumps(vols))
        return outs


class VllmGenerator:
    """vLLM offline batch (e1 VllmGenerator port, §5-2 contract): temperature 0
    (0.2 on the single retry pass), guided JSON when GuidedDecodingParams
    exists else plain decoding + the frozen regex parse, Qwen3 thinking OFF,
    real-tokenizer budget re-fit shrinking ONLY the <<<transcript>>> block."""

    def __init__(self, model: str, max_model_len: int, tp: int, max_tokens: int,
                 gpu_mem: float = 0.92):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        self.name = model
        self._SamplingParams = SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.llm = LLM(model=model, max_model_len=max_model_len,
                       tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem,
                       enforce_eager=False, trust_remote_code=True)
        self.max_model_len = max_model_len
        self.max_tokens = max_tokens
        self.guided = None
        try:  # structured output API drifts across vLLM versions — degrade gracefully
            from vllm.sampling_params import GuidedDecodingParams
            self.guided = GuidedDecodingParams(json=JSON_SCHEMA)
        except Exception:
            print("[warn] GuidedDecodingParams unavailable -> plain decoding "
                  "+ frozen regex parse")

    def _params(self, retry: bool):
        kw = dict(temperature=0.0 if not retry else 0.2, max_tokens=self.max_tokens)
        if self.guided is not None:
            kw["guided_decoding"] = self.guided
        return self._SamplingParams(**kw)

    def _template(self, messages: list[dict]) -> str:
        try:  # Qwen3: thinking MUST stay off or max_tokens goes to <think>
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    def _fit_budget(self, messages: list[dict]) -> str:
        """maec_prompt promise: 'box runner re-checks with the real tokenizer
        and re-fits to max_model_len' — e1 _fit_budget port (chars/4
        under-truncates numeric-dense text)."""
        budget = self.max_model_len - self.max_tokens - 32
        prompt = self._template(messages)
        n = len(self.tokenizer.encode(prompt))
        for _ in range(6):
            if n <= budget:
                return prompt
            u = messages[-1]["content"]
            i0, i1 = u.find("<<<"), u.rfind(">>>")
            if i0 >= 0 and i1 > i0:
                exc = u[i0 + 3:i1]
                keep = max(500, int(len(exc) * (budget / n) * 0.9))
                messages[-1]["content"] = u[:i0 + 3] + exc[:keep] + u[i1:]
            else:
                keep = max(500, int(len(u) * (budget / n) * 0.9))
                messages[-1]["content"] = u[:keep]
            prompt = self._template(messages)
            n = len(self.tokenizer.encode(prompt))
        return prompt

    def generate(self, records: list[dict], retry: bool = False) -> list[str]:
        prompts = [self._fit_budget(r["messages"]) for r in records]
        outs = self.llm.generate(prompts, self._params(retry))
        return [o.outputs[0].text for o in outs]


# ======================================================================== infer
def load_calls(panel_path: str, manifest: str | None) -> pd.DataFrame:
    cols = ["call_id", "ticker", "call_date", "company_name", "text_path"]
    panel = pd.read_parquet(panel_path, columns=cols + ["alignment"])
    calls = (panel.drop_duplicates("call_id")[cols]
             .sort_values(["call_date", "call_id"], kind="mergesort")
             .reset_index(drop=True))
    # the same call must resolve identically under both alignments
    chk = panel.drop_duplicates(["call_id", "ticker", "company_name"])
    assert not chk["call_id"].duplicated().any(), \
        "call_id maps to conflicting ticker/company_name across panel rows"
    assert calls["company_name"].notna().all() and (calls["company_name"] != "").all()
    if manifest:
        n_mf = len(pd.read_parquet(manifest, columns=["call_id"]))
        print(f"[infer] {len(calls):,} unique panel calls of {n_mf:,} manifest "
              f"calls — shortfall {n_mf - len(calls)} = prereg §3 exclusions "
              f"(32 stubs excluded from ALL arms §3.1 + ambiguity/price-gate "
              f"drops; those calls can never be scored)")
    else:
        print(f"[infer] {len(calls):,} unique panel calls (both alignments)")
    return calls


def load_done(out_dir: Path) -> set[tuple[str, str]]:
    done = set()
    for f in sorted(out_dir.glob("part-*.parquet")):
        d = pd.read_parquet(f, columns=["call_id", "variant"])
        done.update(zip(d["call_id"], d["variant"], strict=False))
    return done


def _flush(gen, chunk: list[dict], out_dir: Path, part_idx: int) -> int:
    """Generate a chunk, ONE retry pass on parse failures (frozen RETRY_SUFFIX,
    temp 0.2), checkpoint a part file."""
    raw = gen.generate(chunk, retry=False)
    parsed = [parse_output(t) for t in raw]
    fail_ix = [i for i, p in enumerate(parsed) if p is None]
    if fail_ix:
        retry_recs = []
        for i in fail_ix:
            rec = dict(chunk[i])
            rec["messages"] = [chunk[i]["messages"][0],
                               {"role": "user",
                                "content": chunk[i]["messages"][1]["content"]
                                + RETRY_SUFFIX}]
            retry_recs.append(rec)
        raw2 = gen.generate(retry_recs, retry=True)
        for j, i in enumerate(fail_ix):
            p2 = parse_output(raw2[j])
            if p2 is not None:
                parsed[i], raw[i] = p2, raw2[j]
    rows = []
    for rec, rtext, p in zip(chunk, raw, parsed, strict=False):
        r = {"call_id": rec["row"]["call_id"], "ticker": rec["row"]["ticker"],
             "call_date": rec["row"]["call_date"], "variant": rec["variant"],
             "model_name": gen.name, "raw_output": rtext[:2000]}
        for k in VOL_KEYS:
            r[f"vol_ann_pct_{horizon_of(k)}"] = p[k] if p else np.nan
        r.update(parse_ok=p is not None, retry_used=False,
                 prompt_chars=len(rec["messages"][1]["content"]),
                 transcript_truncated=rec["truncated"])
        rows.append(r)
    for i in fail_ix:
        rows[i]["retry_used"] = True
    pd.DataFrame(rows).to_parquet(out_dir / f"part-{part_idx:05d}.parquet",
                                  index=False)
    return part_idx + 1


def cmd_infer(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calls = load_calls(args.panel, args.manifest)
    if args.limit:
        calls = calls.head(args.limit)
        print(f"[infer] --limit {args.limit}: first {len(calls)} calls")
    variants = list(VARIANTS) if args.variant == "both" else [args.variant]

    done = load_done(out_dir)
    if done:
        print(f"[infer] resume: {len(done)} (call_id, variant) pairs already done")
    pending = [(row, v) for row in calls.to_dict("records") for v in variants
               if (row["call_id"], v) not in done]
    if not pending:
        print("[infer] nothing to do")
        return
    print(f"[infer] pending: {len(pending)} generations "
          f"({len({r['call_id'] for r, _ in pending})} calls x {variants})")

    need_text = {r["call_id"]: r for r, v in pending if v == "maec_text"}
    texts = {}
    for cid, row in need_text.items():
        p = (Path(args.texts_root) / cid / "text.txt" if args.texts_root
             else Path(row["text_path"]))
        texts[cid] = " ".join(p.read_text(encoding="utf-8",
                                          errors="replace").split())
        assert len(texts[cid]) > 0, f"empty transcript for {cid} at {p}"

    gen = (MockGenerator(seed=args.seed) if args.mock else
           VllmGenerator(args.model, args.max_model_len, args.tp,
                         args.max_tokens))

    part_idx = len(list(out_dir.glob("part-*.parquet")))
    chunk, t0, n_done = [], time.time(), 0
    for row, variant in pending:
        full_text = texts.get(row["call_id"], "")
        truncated = False
        if variant == "maec_text":
            _exc, _src, truncated = build_excerpt(full_text)   # OPEN-12 disclosure
        chunk.append({"row": row, "variant": variant, "truncated": truncated,
                      "messages": build_messages(row, full_text, variant)})
        if len(chunk) >= args.checkpoint_every:
            part_idx = _flush(gen, chunk, out_dir, part_idx)
            n_done += len(chunk)
            rate = n_done / (time.time() - t0)
            eta_h = (len(pending) - n_done) / max(rate, 1e-9) / 3600
            print(f"  {n_done}/{len(pending)} done, {rate:.2f} gen/s, "
                  f"ETA {eta_h:.1f}h")
            chunk = []
    if chunk:
        _flush(gen, chunk, out_dir, part_idx)
    print("[infer] run complete")


# ====================================================================== collect
def cmd_collect(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = sorted(Path(args.raw_dir).glob("part-*.parquet"))
    assert parts, f"no part-*.parquet under {args.raw_dir}"
    raw = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
    n0 = len(raw)
    raw = raw.drop_duplicates(["call_id", "variant"], keep="first")
    if len(raw) != n0:
        print(f"[collect] deduped {n0 - len(raw)} duplicate (call_id, variant) rows")

    panel = pd.read_parquet(args.panel,
                            columns=KEY + ["call_id", "alignment", "split", "label"])
    prim = panel[panel["alignment"] == "primary"][KEY + ["call_id", "split", "label"]]
    shif = panel[panel["alignment"] == "shifted"][KEY + ["call_id", "split", "label"]]
    extra = shif.merge(prim[KEY], on=KEY, how="left", indicator=True)
    extra = extra[extra["_merge"] == "left_only"].drop(columns="_merge")
    base = pd.concat([prim.assign(alignment="primary"),
                      extra.assign(alignment="shifted")], ignore_index=True)
    assert not base.duplicated(KEY).any()
    print(f"[collect] loader row base: {len(prim):,} primary rows "
          f"+ {len(extra):,} shifted-only KEY rows (expected 0; disclosed)")

    stats = {}
    for variant, g in raw.groupby("variant"):
        arm = ARM_OF[str(variant)]
        # §5-2 frozen conversion: clip [3,300]% -> sigma_daily -> v = ln sigma
        long = []
        for h in HORIZONS:
            v = g[f"vol_ann_pct_{h}"].map(
                lambda x: to_v(x) if np.isfinite(x) else np.nan)
            long.append(pd.DataFrame({"call_id": g["call_id"], "horizon": h,
                                      "prediction": v.astype(float)}))
        long = pd.concat(long, ignore_index=True)

        missing_calls = set(base["call_id"]) - set(g["call_id"])
        assert not missing_calls, (
            f"{arm}: {len(missing_calls)} panel calls have NO raw inference row "
            f"(box run incomplete?) e.g. {sorted(missing_calls)[:5]}")
        d = base.merge(long, on=["call_id", "horizon"], how="left",
                       validate="m:1")
        # parse failures -> val-split mean PREDICTION per horizon (Yelp
        # precedent; predictions only — no label is used)
        fills = {}
        for h in HORIZONS:
            mh = d["horizon"] == h
            nan_h = mh & d["prediction"].isna()
            fv = d.loc[mh & (d["split"] == "val") & d["prediction"].notna(),
                       "prediction"].mean()
            if not np.isfinite(fv):        # degenerate: all val rows failed
                fv = d.loc[mh & d["prediction"].notna(), "prediction"].mean()
            d.loc[nan_h, "prediction"] = fv
            fills[str(h)] = {"n_filled": int(nan_h.sum()),
                             "fill_value_v": float(fv)}
        d["prediction"] = np.clip(d["prediction"].astype(float),
                                  CLIP_V_LO, CLIP_V_HI)
        assert np.isfinite(d["prediction"]).all()
        d["arm"] = arm
        d = d[KEY + ["call_id", "alignment", "split", "label", "prediction", "arm"]]
        fp = out_dir / f"preds_{arm}.parquet"
        d.to_parquet(fp, index=False)
        stats[arm] = {
            "rows": len(d), "n_calls": int(g["call_id"].nunique()),
            "model_name": str(g["model_name"].iloc[0]),
            "parse_ok_rate": float(g["parse_ok"].mean()),
            "retry_used": int(g["retry_used"].sum()),
            "transcript_truncated_calls": int(g["transcript_truncated"].sum()),
            "clip_pct_range": [CLIP_PCT_LO, CLIP_PCT_HI],
            "nan_fills_per_horizon": fills,
            "shifted_only_key_rows": len(extra),
        }
        print(f"[collect] {arm}: {len(d):,} rows -> {fp.name}  "
              f"(parse_ok {100 * stats[arm]['parse_ok_rate']:.2f}%, "
              f"truncated {stats[arm]['transcript_truncated_calls']}, "
              f"fills {[f['n_filled'] for f in fills.values()]})")

        if args.emit_published:
            # split-invariant predictions under the §4 v1.1 Table-5 assignment
            # (no refit); schema = preds_tfidf_published.parquet
            from maec_baseline_text import assign_published
            full = pd.read_parquet(args.panel)
            pub, dropped = assign_published(
                full[full["alignment"] == "primary"].reset_index(drop=True))
            pubm = pub[KEY + ["call_id", "alignment", "year_panel", "split",
                              "label", "v_past_match"]].merge(
                long, on=["call_id", "horizon"], how="left", validate="m:1")
            for h in HORIZONS:
                mh = pubm["horizon"] == h
                pubm.loc[mh & pubm["prediction"].isna(), "prediction"] = \
                    fills[str(h)]["fill_value_v"]
            pubm["prediction"] = np.clip(pubm["prediction"].astype(float),
                                         CLIP_V_LO, CLIP_V_HI)
            pubm["arm"] = f"{arm}_published"
            fpub = out_dir / f"preds_{arm}_published.parquet"
            pubm.to_parquet(fpub, index=False)
            stats[arm]["published_rows"] = len(pubm)
            stats[arm]["published_dropped_after_test_end"] = dropped
            print(f"[collect] {arm}: published assignment -> {fpub.name} "
                  f"({len(pubm):,} rows)")

    (out_dir / "llm_collect_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[collect] stats -> {out_dir / 'llm_collect_stats.json'}")


# ========================================================================== cli
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("infer", help="vLLM batch inference (box; --mock local)")
    i.add_argument("--panel", default=f"{EC_LOCAL}/maec_panel.parquet")
    i.add_argument("--manifest", default=None,
                   help="optional maec_manifest.parquet for the 3,443-call "
                        "shortfall disclosure")
    i.add_argument("--texts-root", default=None,
                   help="MAEC_Dataset root; text = <root>/<call_id>/text.txt "
                        "(default: the panel's absolute text_path)")
    i.add_argument("--out-dir", required=True)
    i.add_argument("--variant", choices=[*VARIANTS, "both"], default="both")
    i.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    i.add_argument("--tp", type=int, default=1)
    i.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    i.add_argument("--max-tokens", type=int, default=160)
    i.add_argument("--checkpoint-every", type=int, default=500)
    i.add_argument("--limit", type=int, default=0, help="first N calls (smoke)")
    i.add_argument("--mock", action="store_true", help="no GPU/vLLM")
    i.add_argument("--seed", type=int, default=2026)

    c = sub.add_parser("collect", help="v-space conversion + loader preds (local)")
    c.add_argument("--raw-dir", required=True)
    c.add_argument("--panel", default=f"{EC_LOCAL}/maec_panel.parquet")
    c.add_argument("--out-dir", default=str(REPO / "results/second_domain/maec/preds"))
    c.add_argument("--emit-published", action="store_true",
                   help="also write preds_<arm>_published.parquet (Table-5 "
                        "assignment; split-invariant, no refit) for the G1 "
                        "published reading")

    args = ap.parse_args()
    {"infer": cmd_infer, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
