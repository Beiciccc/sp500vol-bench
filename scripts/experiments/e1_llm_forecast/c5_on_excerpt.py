"""P1 control (mock-review): C5-on-C6-excerpt — encode the EXACT curated excerpts
that C6 (prompted Qwen3-32B) reads, with the same-lineage Qwen3-Embedding-8B, train
a per-horizon ridge head, and write a standard run dir C5x_qwen3exc.

Purpose: the "prompting > pooled embeddings" dissociation (C6 genuine increment vs
C5_qwen3 none) confounds three things: input curation (C5 saw head-truncated full
text, C6 sees curated 1A/7/7A sections), mechanism (prompting vs embedding), and
head training. This control fixes the INPUT to C6's curated excerpt: if embeddings
of the same curated text still add ~nothing in M1, input curation is not the
explanation and the mechanism (prompting) claim stands. Head here is ridge (not
C5's MLP) — documented; ridge-on-embeddings is the standard linear-probe reading.

Box-side, single GPU, ~1-2h for ~31.6k long_form filings (train+val+test).
Run: /root/rivermind-data/repo/.venv/bin/python scripts/experiments/e1_llm_forecast/c5_on_excerpt.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt import build_excerpt  # noqa: E402

DATA_ROOT = Path(os.environ.get("SP500VOL_DATA_ROOT", "/Volumes/Z/sp500vol-data"))
TEXT_CACHE = DATA_ROOT / "processed" / "_text_cache" / "filing_texts.parquet"
ALIGNED = DATA_ROOT / "processed" / "full" / "aligned_filings.parquet"
A2_LF = "results/runs/A2_har_rv_full_long_form_seed2026/predictions.parquet"
MODEL = "Qwen/Qwen3-Embedding-8B"
OUT_RUN = "results/runs/C5x_qwen3exc_full_long_form_seed2026"
EMB_CKPT = "results/e1_llm_forecast/_c5x_embeddings.npz"
MAX_LEN = 4096
BATCH = 16
EPS = 1e-8


def stream_texts(paths: set[str]) -> dict[str, str]:
    out = {}
    pf = pq.ParquetFile(TEXT_CACHE)
    for batch in pf.iter_batches(batch_size=2048, columns=["text_path", "text"]):
        mask = pc.is_in(batch.column("text_path"), value_set=__import__("pyarrow").array(list(paths)))
        sub = batch.filter(mask)
        for p, t in zip(sub.column("text_path").to_pylist(), sub.column("text").to_pylist()):
            out[p] = t or ""
    return out


def build_filings() -> tuple[pd.DataFrame, pd.DataFrame]:
    a2 = pd.read_parquet(A2_LF)
    filings = a2.sort_values("horizon_days").drop_duplicates("text_path").copy()
    aligned = pd.read_parquet(ALIGNED, columns=["text_path", "sections_json"]).drop_duplicates("text_path")
    filings = filings.merge(aligned, on="text_path", how="left")
    print(f"long_form filings: {len(filings)} (splits {filings.split.value_counts().to_dict()})")

    # ---- excerpts (identical builder to C6's prompts)
    paths = set(filings.text_path)
    print("streaming texts...")
    texts = stream_texts(paths)
    excerpts = []
    for _, r in filings.iterrows():
        exc, _src = build_excerpt(r["form"], r.get("sections_json"), texts.get(r["text_path"], ""))
        excerpts.append(exc)
    filings["excerpt"] = excerpts
    del texts
    print(f"excerpts built; median chars {int(np.median([len(e) for e in excerpts]))}")
    return a2, filings


def encode_shard(filings: pd.DataFrame, shard_idx: int, num_shards: int) -> None:
    """Encode a contiguous shard of the (deterministically ordered) filings on the
    current CUDA device; save to a shard-specific npz. Resumable per shard."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    ckpt = EMB_CKPT.replace(".npz", f".shard{shard_idx}of{num_shards}.npz")
    filings = filings.sort_values("text_path").reset_index(drop=True)
    lo = len(filings) * shard_idx // num_shards
    hi = len(filings) * (shard_idx + 1) // num_shards
    part = filings.iloc[lo:hi]
    print(f"shard {shard_idx}/{num_shards}: rows {lo}:{hi} ({len(part)})")

    if Path(ckpt).exists():
        z = np.load(ckpt, allow_pickle=True)
        emb, done_paths = z["emb"], list(z["paths"])
        print(f"resume: {len(done_paths)} embeddings loaded")
    else:
        emb, done_paths = np.zeros((0, 0), dtype=np.float32), []
    todo = part[~part.text_path.isin(set(done_paths))]
    if len(todo):
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, padding_side="left")
        model = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                          torch_dtype=torch.bfloat16, device_map="cuda:0")
        model.eval()
        chunks = []
        with torch.inference_mode():
            for i in range(0, len(todo), BATCH):
                batch_texts = todo.excerpt.iloc[i:i + BATCH].tolist()
                enc = tok(batch_texts, truncation=True, max_length=MAX_LEN,
                          padding=True, return_tensors="pt").to("cuda:0")
                h = model(**enc).last_hidden_state  # (B, T, H); left-pad -> last token real
                v = h[:, -1, :].float()
                v = torch.nn.functional.normalize(v, dim=-1)
                chunks.append(v.cpu().numpy().astype(np.float32))
                if (i // BATCH) % 50 == 0:
                    print(f"  encoded {i + len(batch_texts)}/{len(todo)}", flush=True)
                if (i // BATCH) % 100 == 99:  # periodic checkpoint
                    new = np.concatenate(chunks)
                    allemb = new if emb.size == 0 else np.concatenate([emb, new])
                    allp = done_paths + todo.text_path.iloc[:i + BATCH].tolist()
                    np.savez(ckpt, emb=allemb, paths=np.array(allp, dtype=object))
        new = np.concatenate(chunks)
        emb = new if emb.size == 0 else np.concatenate([emb, new])
        done_paths = done_paths + todo.text_path.tolist()
    np.savez(ckpt, emb=emb, paths=np.array(done_paths, dtype=object))
    print(f"SHARD_DONE {shard_idx}/{num_shards} n={len(done_paths)}")


def assemble_and_train(a2: pd.DataFrame, filings: pd.DataFrame, num_shards: int) -> None:
    emb_list, path_list = [], []
    for k in range(num_shards):
        ckpt = EMB_CKPT.replace(".npz", f".shard{k}of{num_shards}.npz")
        z = np.load(ckpt, allow_pickle=True)
        emb_list.append(z["emb"]); path_list.extend(list(z["paths"]))
    emb = np.concatenate(emb_list)
    missing = set(filings.text_path) - set(path_list)
    if missing:
        raise SystemExit(f"ASSEMBLE_FAIL: {len(missing)} filings lack embeddings")
    order = {p: i for i, p in enumerate(path_list)}
    X_all = emb[[order[p] for p in filings.text_path]]
    print(f"embeddings assembled: {X_all.shape}")

    # ---- ridge head per horizon (log target + Duan smearing, B2/A2 conventions)
    rows = []
    for h in (5, 10, 20):
        ah = a2[a2.horizon_days == h].merge(filings[["text_path"]].assign(_idx=np.arange(len(filings))),
                                            on="text_path", how="inner")
        X = X_all[ah._idx.values]
        y = np.log(np.clip(ah.label_realised_vol.values, EPS, None))
        tr, va = ah.split == "train", ah.split == "val"
        from sklearn.linear_model import Ridge
        best, best_q = None, np.inf
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            r = Ridge(alpha=alpha).fit(X[tr.values], y[tr.values])
            pv = r.predict(X[va.values])
            q = float(np.mean(np.exp(y[va.values] - pv) - (y[va.values] - pv) - 1))  # qlike-ish on logs
            if q < best_q:
                best_q, best, best_alpha = q, r, alpha
        resid = y[tr.values] - best.predict(X[tr.values])
        smear = float(np.mean(np.exp(resid)))
        pred = np.exp(best.predict(X)) * smear
        out = ah[["ticker", "accession", "horizon_days", "split", "form", "item_subtype",
                  "filing_time_utc", "effective_trading_day", "label_realised_vol",
                  "feature_rv_1d", "feature_rv_5d", "feature_rv_22d", "text_path",
                  "metadata_path"]].copy()
        out["prediction_realised_vol"] = np.clip(pred, 0.01, 5.0)
        rows.append(out)
        print(f"h={h}: alpha={best_alpha} smear={smear:.3f}")
    res = pd.concat(rows, ignore_index=True)
    res["run_id"] = "C5x_qwen3exc_full_long_form_seed2026"
    res["model_id"] = "C5x_qwen3exc"
    res["dataset"] = "full"
    res["seed"] = 2026
    res["disclosure_subset"] = "long_form"
    cols = ["run_id", "model_id", "dataset", "seed", "disclosure_subset", "split", "ticker",
            "form", "item_subtype", "accession", "filing_time_utc", "effective_trading_day",
            "horizon_days", "label_realised_vol", "prediction_realised_vol",
            "feature_rv_1d", "feature_rv_5d", "feature_rv_22d", "text_path", "metadata_path"]
    res = res[cols]
    Path(OUT_RUN).mkdir(parents=True, exist_ok=True)
    res.to_parquet(f"{OUT_RUN}/predictions.parquet", index=False)

    def q_var(y, f):
        a = np.clip(np.asarray(y) ** 2, 1e-8, None); b = np.clip(np.asarray(f) ** 2, 1e-8, None)
        return float(np.mean(a / b - np.log(a / b) - 1))
    metrics = []
    for split in ("train", "val", "test"):
        for h in (5, 10, 20):
            g = res[(res.split == split) & (res.horizon_days == h)]
            e = g.prediction_realised_vol - g.label_realised_vol
            metrics.append({"split": split, "disclosure_subset": "long_form", "horizon_days": h,
                            "n": int(len(g)), "mae": float(e.abs().mean()),
                            "rmse": float(np.sqrt((e ** 2).mean())),
                            "r2": float(1 - (e ** 2).sum() / ((g.label_realised_vol - g.label_realised_vol.mean()) ** 2).sum()),
                            "qlike": q_var(g.label_realised_vol, g.prediction_realised_vol)})
    json.dump(metrics, open(f"{OUT_RUN}/metrics.json", "w"), indent=1)
    json.dump({"model_id": "C5x_qwen3exc", "note": "P1 control: Qwen3-Embedding-8B on the exact "
               "C6 curated excerpts (1A/7/7A else head), last-token pool + L2, ridge head "
               "(log target + Duan smearing). Isolates input-curation from the prompting mechanism."},
              open(f"{OUT_RUN}/config.json", "w"), indent=1)
    print("C5X_DONE", OUT_RUN, "test qlike:",
          [round(m["qlike"], 3) for m in metrics if m["split"] == "test"])


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-idx", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=2)
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()
    a2, filings = build_filings()
    if args.assemble:
        assemble_and_train(a2, filings, args.num_shards)
    elif args.shard_idx is not None:
        encode_shard(filings, args.shard_idx, args.num_shards)
    else:
        raise SystemExit("pass --shard-idx K or --assemble")


if __name__ == "__main__":
    main()
