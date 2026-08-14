"""Yelp second-domain: frozen-embedding text arm (Qwen3-Embedding-8B + ridge head).

Mirrors the SEC benchmark's C5x arm (same embedder family applied to the exact text
the protocol scores) so the two domains carry a structurally symmetric challenger.
Protocol position is identical to the TF-IDF arm: fit on train, alpha chosen on val,
raw test predictions; the combiner/identity/placebo cascade runs via yelp_protocol.py
--text-arm qwen3emb_chrono.

Two subcommands (run encode once per GPU shard, then fit on CPU):

  encode --panel P --out-dir D --model M --shard I --num-shards N
      Deduplicates (entity_id, event_time) texts, takes shard I of N (sorted, strided),
      embeds with vLLM task="embed", writes emb_I.npy + keys_I.parquet into D.

  fit --panel P --emb-dir D --out-dir PREDS
      Stacks shards, maps embeddings onto panel rows, fits one ridge per horizon
      (train fit, val-selected alpha), writes preds_qwen3emb_chrono.parquet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLIP_LO, CLIP_HI = 1.0, 5.0
ALPHAS = tuple(float(a) for a in np.logspace(-2, 5, 15))
MAX_CHARS = 12_000  # ~3k tokens; text length p90 is ~1.2k words so truncation is rare


def clip(x):
    return np.clip(np.asarray(x, dtype=np.float64), CLIP_LO, CLIP_HI)


def mse(y, f):
    return float(np.mean((np.asarray(y, float) - np.asarray(f, float)) ** 2))


def unique_events(panel: pd.DataFrame) -> pd.DataFrame:
    ev = (panel.drop_duplicates(["entity_id", "event_time"])
          [["entity_id", "event_time", "text"]]
          .sort_values(["entity_id", "event_time"], kind="mergesort")
          .reset_index(drop=True))
    ev["text"] = ev.text.str.slice(0, MAX_CHARS)
    return ev


def cmd_encode(args) -> None:
    panel = pd.read_parquet(args.panel, columns=["entity_id", "event_time", "text"])
    ev = unique_events(panel)
    shard = ev.iloc[args.shard::args.num_shards].reset_index(drop=True)
    print(f"[encode] shard {args.shard}/{args.num_shards}: "
          f"{len(shard):,} of {len(ev):,} unique events")

    from vllm import LLM
    llm = LLM(model=args.model, runner="pooling", convert="embed",
              max_model_len=4096, gpu_memory_utilization=0.92, enforce_eager=False)
    outs = llm.embed(shard.text.tolist())
    emb = np.asarray([o.outputs.embedding for o in outs], dtype=np.float32)
    assert emb.shape[0] == len(shard) and np.isfinite(emb).all()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"emb_{args.shard}.npy", emb)
    shard[["entity_id", "event_time"]].to_parquet(out / f"keys_{args.shard}.parquet",
                                                  index=False)
    print(f"[encode] wrote emb_{args.shard}.npy {emb.shape}")


class _RidgeGram:
    """Ridge fit via the Gram matrix: X^T X computed ONCE, then each alpha is a
    cheap (d x d) solve. Mathematically identical to sklearn's cholesky Ridge;
    ~15x cheaper than refitting per alpha on a 2-core box. Intercept handled by
    centering (standardised in the normal equations)."""

    def __init__(self, coef, intercept):
        self.coef_, self.intercept_ = coef, intercept

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


def _gram_xty(X, rows, y, chunk=20000):
    """Accumulate centred X^T X and X^T y over row CHUNKS in float64, never
    materialising a float64 copy of the whole subset. Peak extra memory is one
    chunk (chunk x d x 8 bytes). Memory-frugal for the 16 GB no-GPU box."""
    d = X.shape[1]
    xm = np.zeros(d); n = len(rows)
    for i in range(0, n, chunk):                       # mean over the subset
        xm += X[rows[i:i + chunk]].sum(0, dtype=np.float64)
    xm /= n
    ym = float(np.asarray(y, np.float64).mean())
    gram = np.zeros((d, d)); xty = np.zeros(d)
    for i in range(0, n, chunk):
        xc = X[rows[i:i + chunk]].astype(np.float64) - xm
        yc = np.asarray(y[i:i + chunk], np.float64) - ym
        gram += xc.T @ xc
        xty += xc.T @ yc
    return gram, xty, xm, ym


def fit_ridge_select(X, tr_rows, ytr, X_val_rows, yv):
    gram, xty, xm, ym = _gram_xty(X, tr_rows, ytr)
    eye = np.eye(gram.shape[0])
    Xv = X[X_val_rows].astype(np.float64)              # val subset is small
    best = None
    for a in ALPHAS:
        coef = np.linalg.solve(gram + a * eye, xty)
        intercept = ym - xm @ coef
        v = mse(yv, clip(Xv @ coef + intercept))
        if best is None or v < best[0]:
            best = (v, a, _RidgeGram(coef, intercept))
    _, alpha, model = best
    return model, alpha


def cmd_fit(args) -> None:
    emb_dir = Path(args.emb_dir)
    kfs = sorted(emb_dir.glob("keys_*.parquet"))
    keys = [pd.read_parquet(kf) for kf in kfs]
    shapes = [np.load(emb_dir / f"emb_{kf.stem.split('_')[1]}.npy", mmap_mode="r").shape
              for kf in kfs]
    ntot, d = sum(s[0] for s in shapes), shapes[0][1]
    X = np.empty((ntot, d), dtype=np.float32)          # preallocate once (no vstack doubling)
    off = 0
    for kf, s in zip(kfs, shapes, strict=False):
        src = np.load(emb_dir / f"emb_{kf.stem.split('_')[1]}.npy")
        X[off:off + s[0]] = src
        off += s[0]
        del src
    key = pd.concat(keys, ignore_index=True)
    assert len(key) == len(X) and not key.duplicated().any()
    key["row"] = np.arange(len(key))
    print(f"[fit] {len(key):,} unique events embedded, dim {d}", flush=True)

    panel = pd.read_parquet(args.panel, columns=["entity_id", "event_time", "split",
                                                 "horizon_months", "label"])
    panel = panel.merge(key, on=["entity_id", "event_time"], validate="m:1")
    assert panel.row.notna().all(), "panel rows missing embeddings"

    frames, metrics = [], []
    for h in sorted(panel.horizon_months.unique()):
        d = panel[panel.horizon_months == h]
        tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
        tr_rows = tr.row.to_numpy(); va_rows = va.row.to_numpy(); te_rows = te.row.to_numpy()
        model, alpha = fit_ridge_select(X, tr_rows, tr.label.to_numpy(float),
                                        va_rows, va.label.to_numpy(float))
        f_va = clip(model.predict(X[va_rows].astype(np.float64)))
        f_te = clip(model.predict(X[te_rows].astype(np.float64)))
        m = mse(te.label.to_numpy(float), f_te)
        metrics.append({"horizon_months": int(h), "alpha": alpha,
                        "mse_test_qwen3emb_chrono": m})
        print(f"[fit] h={h}m alpha={alpha:g} test MSE={m:.4f}")
        for sub, pred, split in ((va, f_va, "val"), (te, f_te, "test")):
            out = sub[["entity_id", "event_time", "horizon_months", "label"]].copy()
            out["split"], out["prediction"], out["arm"] = split, pred, "qwen3emb_chrono"
            frames.append(out[["entity_id", "event_time", "split", "horizon_months",
                               "label", "prediction", "arm"]])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(
        out_dir / "preds_qwen3emb_chrono.parquet", index=False)
    (out_dir / "qwen3emb_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[fit] wrote {out_dir}/preds_qwen3emb_chrono.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--panel", required=True)
    e.add_argument("--out-dir", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--shard", type=int, required=True)
    e.add_argument("--num-shards", type=int, default=4)
    f = sub.add_parser("fit")
    f.add_argument("--panel", required=True)
    f.add_argument("--emb-dir", required=True)
    f.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    cmd_encode(args) if args.cmd == "encode" else cmd_fit(args)


if __name__ == "__main__":
    main()
