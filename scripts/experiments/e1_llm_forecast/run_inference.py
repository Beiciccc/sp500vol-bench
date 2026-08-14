"""E1 / C6+D4 — box-side vLLM batch runner for the generative-LLM vol forecaster.

Two subcommands:

  build-manifest   (run LOCALLY, on the Mac, before shipping to the box)
      Joins A2_har_rv combined predictions (val+test only — no training is ever done,
      so train filings are never sent to the LLM) with aligned_filings.parquet to get
      sections_json / token_count / filing_date. One row per FILING (splits are
      constant across horizons — verified). Ship the resulting parquet to the box
      together with the text cache.

        .venv/bin/python scripts/experiments/e1_llm_forecast/run_inference.py \\
            build-manifest --out results/e1_llm_forecast/manifest_valtest.parquet

  run              (on the GPU box; also runs locally with --mock)
      vLLM offline batch inference, temperature=0, max_tokens 120, guided JSON if the
      installed vLLM supports it, else robust regex parsing with ONE retry pass on
      parse failures. Streams filing text from the cache parquet with pyarrow
      iter_batches (never loads the 3.3GB file whole). Checkpoints a parquet part
      file every --checkpoint-every filings; re-running skips (text_path, variant)
      pairs already present in --out-dir (resumable).

        SP500VOL_DATA_ROOT=/data/sp500vol-data python run_inference.py run \\
            --manifest manifest_valtest.parquet --model Qwen/Qwen3-32B-AWQ \\
            --variant both --out-dir raw_outputs/

      --pilot N   : first N TEST filings, stratified by form (go/no-go pilot).
      --limit N   : stratified sample over val+test (local mock validation).
      --mock      : replace the vLLM generator with a plausible-JSON mock (no GPU).

Raw output schema (one row per filing x variant):
  text_path, variant, model_name, raw_output, vol_5d, vol_10d, vol_20d,
  parse_ok, retry_used, prompt_chars, excerpt_source
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt import (
    JSON_SCHEMA,
    RETRY_SUFFIX,
    build_excerpt,
    build_messages,
    parse_output,
)

DATA_ROOT = Path(os.environ.get("SP500VOL_DATA_ROOT", "/Volumes/Z/sp500vol-data"))
TEXT_CACHE = DATA_ROOT / "processed" / "_text_cache" / "filing_texts.parquet"
ALIGNED = DATA_ROOT / "processed" / "full" / "aligned_filings.parquet"
A2_COMBINED = "results/runs/A2_har_rv_full_combined_seed2026/predictions.parquet"
VARIANTS = ("c6_text", "d4_fused")
CONTROL_VARIANTS = ("c6_dateonly", "c6_datefirm")  # P0-2 contamination controls
MANIFEST_COLS = [
    "ticker", "form", "item_subtype", "accession", "filing_time_utc", "filing_date",
    "effective_trading_day", "split", "disclosure", "text_path", "metadata_path",
    "feature_rv_1d", "feature_rv_5d", "feature_rv_22d", "sections_json", "token_count",
]


# ===================================================================== manifest
def build_manifest(a2_path: str, out: str) -> pd.DataFrame:
    a2 = pd.read_parquet(a2_path)
    a2 = a2[a2["split"].isin(["val", "test"])]
    # one row per filing: splits & rv lags are constant across the 3 horizons
    filings = a2.sort_values("horizon_days").drop_duplicates(subset=["text_path"]).copy()
    aligned = pd.read_parquet(
        ALIGNED, columns=["text_path", "filing_date", "sections_json", "token_count"]
    ).drop_duplicates(subset=["text_path"])
    m = filings.merge(aligned, on="text_path", how="left", validate="1:1")
    m["disclosure"] = np.where(m["form"].eq("8-K"), "event_driven", "long_form")
    m = m[MANIFEST_COLS].sort_values(["filing_time_utc", "ticker", "accession"])
    n_missing = int(m["sections_json"].isna().sum() + m["filing_date"].isna().sum())
    est_prompt_tokens = np.minimum(m["token_count"].fillna(0), 5300).sum() + 700 * len(m)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out, index=False)
    print(f"manifest: {len(m)} filings -> {out}")
    print(f"  by split: {m['split'].value_counts().to_dict()}")
    print(f"  by form:  {m['form'].value_counts().to_dict()}")
    print(f"  rows with missing aligned join: {n_missing}")
    print(f"  est prompt tokens per variant:  {est_prompt_tokens/1e6:.0f}M")
    return m


# ===================================================================== generators
class MockGenerator:
    """No-GPU stand-in: vol = clip(rv_22d * lognormal random walk). ~4% of first-pass
    outputs are deliberately unparseable to exercise the retry path (retries succeed)."""

    name = "mock"

    def __init__(self, seed: int = 2026, garbage_rate: float = 0.04):
        self.rng = np.random.default_rng(seed)
        self.garbage_rate = garbage_rate

    def generate(self, records: list[dict], retry: bool = False) -> list[str]:
        outs = []
        for rec in records:
            row = rec["row"]
            if not retry and self.rng.random() < self.garbage_rate:
                outs.append("As an AI language model, volatility depends on many factors.")
                continue
            base = float(row.get("feature_rv_22d") or 0.25)
            vols = {}
            for k in ("vol_5d", "vol_10d", "vol_20d"):
                v = base * float(np.exp(self.rng.normal(0.0, 0.30)))
                vols[k] = round(float(np.clip(v, 0.03, 3.0)), 4)
            outs.append(json.dumps(vols))
        return outs


class VllmGenerator:
    """vLLM offline batch generator. temperature=0, max_tokens ~120, guided JSON when
    the installed vLLM exposes GuidedDecodingParams; else plain decoding + regex parse."""

    def __init__(self, model: str, max_model_len: int, tp: int, max_tokens: int,
                 gpu_mem: float = 0.92, thinking: bool = False):
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
        self.thinking = thinking
        self.guided = None
        try:  # vLLM >= 0.6.6 structured output; API drifts, so degrade gracefully
            from vllm.sampling_params import GuidedDecodingParams
            self.guided = GuidedDecodingParams(json=JSON_SCHEMA)
        except Exception:
            print("[warn] GuidedDecodingParams unavailable -> plain decoding + regex parse")

    def _params(self, retry: bool):
        kw = dict(temperature=0.0 if not retry else 0.2, max_tokens=self.max_tokens)
        if self.guided is not None:
            kw["guided_decoding"] = self.guided
        return self._SamplingParams(**kw)

    def _template(self, messages: list[dict]) -> str:
        try:  # Qwen3: MUST disable thinking mode or 120 tokens go to <think>
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.thinking)
        except TypeError:  # tokenizers without the kwarg
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    def _fit_budget(self, messages: list[dict]) -> str:
        """Render the prompt; if it exceeds the context budget, shrink ONLY the
        filing excerpt (between <<< and >>> in the user message) and re-render.
        The CHARS_PER_TOKEN=4.0 char cap under-truncates numeric-dense filings
        (~2.5 chars/token), which crashed the first pilot at 8193 tokens."""
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
            else:  # no excerpt markers: truncate the tail of the user content
                keep = max(500, int(len(u) * (budget / n) * 0.9))
                messages[-1]["content"] = u[:keep]
            prompt = self._template(messages)
            n = len(self.tokenizer.encode(prompt))
        return prompt

    def generate(self, records: list[dict], retry: bool = False) -> list[str]:
        prompts = [self._fit_budget(r["messages"]) for r in records]
        outs = self.llm.generate(prompts, self._params(retry))
        return [o.outputs[0].text for o in outs]


# ===================================================================== text streaming
def stream_texts(text_paths: set[str]) -> dict[str, str]:
    """Stream ONLY the needed rows out of the 3.3GB cache; never load it whole."""
    got: dict[str, str] = {}
    pf = pq.ParquetFile(TEXT_CACHE)
    for batch in pf.iter_batches(batch_size=2048, columns=["text_path", "text"]):
        tp = batch.column("text_path").to_pylist()
        keep = [i for i, p in enumerate(tp) if p in text_paths]
        if keep:
            tx = batch.column("text").take(keep).to_pylist()
            for i, j in enumerate(keep):
                got[tp[j]] = tx[i]
        if len(got) == len(text_paths):
            break
    missing = text_paths - got.keys()
    if missing:
        print(f"[warn] {len(missing)} text_paths not found in cache; skipped")
    return got


# ===================================================================== run
def select_rows(m: pd.DataFrame, args) -> pd.DataFrame:
    if args.subset != "all":
        m = m[m["disclosure"] == args.subset]
    if args.pilot:
        test = m[m["split"] == "test"].sort_values(["filing_time_utc", "ticker"])
        parts, n = [], args.pilot
        fracs = test["form"].value_counts(normalize=True)
        for form, frac in fracs.items():
            k = max(1, int(round(n * frac)))
            parts.append(test[test["form"] == form].head(k))
        m = pd.concat(parts).drop_duplicates(subset=["text_path"]).head(n)
        print(f"pilot: {len(m)} test filings, forms {m['form'].value_counts().to_dict()}")
    elif args.limit:
        rng = np.random.default_rng(2026)
        parts = []
        for (_f, _s), g in m.groupby(["form", "split"]):
            k = max(1, int(round(args.limit * len(g) / len(m))))
            parts.append(g.iloc[rng.choice(len(g), size=min(k, len(g)), replace=False)])
        m = pd.concat(parts).drop_duplicates(subset=["text_path"]).head(args.limit)
        print(f"limit: {len(m)} filings, {m.groupby(['form','split']).size().to_dict()}")
    return m


def load_done(out_dir: Path) -> set[tuple[str, str]]:
    done = set()
    for f in sorted(out_dir.glob("part-*.parquet")):
        d = pd.read_parquet(f, columns=["text_path", "variant"])
        done.update(zip(d["text_path"], d["variant"], strict=False))
    return done


def run(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = select_rows(pd.read_parquet(args.manifest), args)
    variants = (VARIANTS if args.variant == "both"
                else CONTROL_VARIANTS if args.variant == "controls"
                else (args.variant,))

    done = load_done(out_dir)
    if done:
        print(f"resume: {len(done)} (text_path, variant) pairs already done")

    # pending work items, one per filing x variant
    pending_rows = []
    for row in m.to_dict("records"):
        for v in variants:
            if (row["text_path"], v) not in done:
                pending_rows.append((row, v))
    if not pending_rows:
        print("nothing to do")
        return
    print(f"pending: {len(pending_rows)} generations "
          f"({len({r[0]['text_path'] for r in pending_rows})} filings x {len(variants)} variants)")

    texts = stream_texts({r[0]["text_path"] for r in pending_rows})

    if args.mock:
        gen = MockGenerator(seed=args.seed)
    else:
        gen = VllmGenerator(args.model, args.max_model_len, args.tp, args.max_tokens,
                            thinking=getattr(args, "thinking", False))

    part_idx = len(list(out_dir.glob("part-*.parquet")))
    chunk, t0, n_done = [], time.time(), 0
    for row, variant in pending_rows:
        text = texts.get(row["text_path"])
        if text is None:
            continue
        excerpt, src = build_excerpt(row["form"], row.get("sections_json"), text)
        chunk.append({
            "row": row, "variant": variant, "excerpt_source": src,
            "messages": build_messages(row, text, variant),
        })
        if len(chunk) >= args.checkpoint_every:
            part_idx = _flush(gen, chunk, out_dir, part_idx)
            n_done += len(chunk)
            rate = n_done / (time.time() - t0)
            eta_h = (len(pending_rows) - n_done) / max(rate, 1e-9) / 3600
            print(f"  {n_done}/{len(pending_rows)} done, {rate:.2f} gen/s, ETA {eta_h:.1f}h")
            chunk = []
    if chunk:
        _flush(gen, chunk, out_dir, part_idx)
    print("run complete")


def _flush(gen, chunk: list[dict], out_dir: Path, part_idx: int) -> int:
    """Generate a chunk (with one retry pass on parse failures) and checkpoint it."""
    raw = gen.generate(chunk, retry=False)
    parsed = [parse_output(t) for t in raw]
    # retry pass: rebuild prompts with the strict reminder, slightly warmer sampling
    fail_ix = [i for i, p in enumerate(parsed) if p is None]
    if fail_ix:
        retry_recs = []
        for i in fail_ix:
            rec = dict(chunk[i])  # keep the original excerpt; append a strict reminder
            rec["messages"] = [chunk[i]["messages"][0],
                               {"role": "user",
                                "content": chunk[i]["messages"][1]["content"] + RETRY_SUFFIX}]
            retry_recs.append(rec)
        raw2 = gen.generate(retry_recs, retry=True)
        for j, i in enumerate(fail_ix):
            p2 = parse_output(raw2[j])
            if p2 is not None:
                parsed[i], raw[i] = p2, raw2[j]
    rows = []
    for rec, rtext, p in zip(chunk, raw, parsed, strict=False):
        rows.append({
            "text_path": rec["row"]["text_path"],
            "variant": rec["variant"],
            "model_name": gen.name,
            "raw_output": rtext[:2000],
            "vol_5d": p["vol_5d"] if p else np.nan,
            "vol_10d": p["vol_10d"] if p else np.nan,
            "vol_20d": p["vol_20d"] if p else np.nan,
            "parse_ok": p is not None,
            "retry_used": False,
            "prompt_chars": len(rec["messages"][1]["content"]),
            "excerpt_source": rec["excerpt_source"],
        })
    for i in fail_ix:
        rows[i]["retry_used"] = True
    pd.DataFrame(rows).to_parquet(out_dir / f"part-{part_idx:05d}.parquet", index=False)
    return part_idx + 1


# ===================================================================== cli
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-manifest", help="build val+test filings manifest (local)")
    b.add_argument("--a2", default=A2_COMBINED)
    b.add_argument("--out", default="results/e1_llm_forecast/manifest_valtest.parquet")

    r = sub.add_parser("run", help="batch inference (box; --mock for local validation)")
    r.add_argument("--manifest", required=True)
    r.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    r.add_argument("--variant", choices=["c6_text", "d4_fused", "both", "c6_dateonly", "c6_datefirm", "controls", "c6_para1", "c6_para2"], default="both")
    r.add_argument("--subset", choices=["long_form", "event_driven", "all"], default="all")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--checkpoint-every", type=int, default=500)
    r.add_argument("--pilot", type=int, default=0,
                   help="first N TEST filings, stratified by form")
    r.add_argument("--limit", type=int, default=0,
                   help="stratified val+test sample of N filings (mock validation)")
    r.add_argument("--mock", action="store_true", help="mock generator, no GPU/vLLM")
    r.add_argument("--seed", type=int, default=2026)
    r.add_argument("--max-model-len", type=int, default=8192)
    r.add_argument("--max-tokens", type=int, default=120)
    r.add_argument("--tp", type=int, default=1)
    r.add_argument("--thinking", action="store_true",
                   help="enable Qwen3 thinking mode (elicitation arm; pair with --max-tokens 2048)")

    args = ap.parse_args()
    if args.cmd == "build-manifest":
        build_manifest(args.a2, args.out)
    else:
        run(args)


if __name__ == "__main__":
    main()
