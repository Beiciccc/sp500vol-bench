"""Pre-registered ASHA HPO harness (configs/hpo_arm.yaml; spec results/HPO_ARM_SPEC.md).

Deterministic Sobol sampling + successive-halving over the archived training
pipeline (scripts/train.py helpers, exactly the pilot arm's import pattern).

Physical test firewall: split=="test" rows are DROPPED before anything trains;
the val years are re-split at val_fit_end — val-fit rows keep split "val"
(early stopping + rung promotion), val-select rows are relabelled into the now
vacant "test" slot, so the untouched pipeline scores them as out-of-sample.
A manifest hash of the surviving row keys is stored per trial.

Stages (driven per-rung by a shell driver; every stage is idempotent):
  plan    --task T1a                          -> results/hpo/<task>/trials.json
  search  --task T1a --rung 0 --shard 0 --num-shards 4   (one GPU per shard)
  promote --task T1a --rung 0                 -> survivors_rung1.json
  select  --task T1a                          -> selection.json (Track-A top-2)

Each trial retrains from scratch to its rung's epoch cap (no checkpoint
resume: reproducible, sharding-invariant; budgeted in the spec's upper bound).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

# FD-exhaustion guard: sequential trials in one process accumulate DataLoader
# shared-memory file descriptors under the default 'file_descriptor' strategy
# ("Too many open files" -> pin-memory thread death). 'file_system' shares by
# name, not FD; pair with a raised ulimit -n in the driver.
import torch

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass
try:
    import resource
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(1_048_576, _hard), _hard))
except Exception:
    pass

from sp500vol.evaluation.metrics import qlike
from sp500vol.utils import seed_everything

CFG = yaml.safe_load((REPO / "configs" / "hpo_arm.yaml").read_text())
HPO_ROOT = REPO / "results" / "hpo"
_TOK_CACHES: dict = {}  # process-level, keyed (model, disclosure)

TASKS = {
    "T1a": dict(model="C2_finbert_s1", disclosure="long_form", n=32),
    "T1c": dict(model="C2_finbert_s1", disclosure="event_driven", n=24, rung0_frac=0.30),
    "T4": dict(model="C4_longformer", disclosure="long_form", n=16,
               rungs=[2, 5, 10]),
    "T3d2lf": dict(model="D2_gated_fusion", disclosure="long_form", n=48, fusion=True),
    "T3d2ed": dict(model="D2_gated_fusion", disclosure="event_driven", n=48, fusion=True),
    "T3d1lf": dict(model="D1_concat_mlp", disclosure="long_form", n=32, fusion=True),
    "T3d1ed": dict(model="D1_concat_mlp", disclosure="event_driven", n=32, fusion=True),
}
# T1b/T1d (demeaned strata) reuse this harness with the row2_demeaned target
# transform once T1a/T1c winners exist (warm start); T2/T5 run their own
# closed-form grids (separate light scripts).


def _train_mod():
    spec = importlib.util.spec_from_file_location("hpo_train", REPO / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sobol_trials(task: str) -> list[dict]:
    """Deterministic Sobol draw over the pre-registered 8-dim space."""
    from scipy.stats import qmc
    t = TASKS[task]
    n = t["n"]
    space = CFG["space"]
    dims = ["lr", "head_lr_mult", "weight_decay", "head_hidden_dim",
            "head_dropout", "freeze_mode", "effective_batch", "objective"]
    if t.get("fusion"):
        dims = [d for d in dims if d != "freeze_mode"]
    eng = qmc.Sobol(d=len(dims), scramble=False, seed=CFG["sampler"]["seed"])
    u = eng.random(n)
    trials = []
    for k in range(n):
        cfg = {}
        for j, d in enumerate(dims):
            s = space[d]
            if s["dist"] == "log_uniform":
                cfg[d] = float(np.exp(np.log(s["lo"]) + u[k, j] * (np.log(s["hi"]) - np.log(s["lo"]))))
            else:
                vals = s["values"]
                cfg[d] = vals[min(int(u[k, j] * len(vals)), len(vals) - 1)]
        trials.append({"trial": k, **cfg})
    return trials


def rungs_for(task: str) -> list[int]:
    return TASKS[task].get("rungs", CFG["asha"]["rungs_epochs"])


def task_dir(task: str) -> Path:
    d = HPO_ROOT / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def prepare_data(train_mod, model_id: str, disclosure: str, rung0_frac=None, rung=None):
    """Load panel exactly as train.py, then apply the pre-registered firewall."""
    data = train_mod._load_dataset("full")
    data = train_mod._filter_disclosure(data, disclosure)
    data = train_mod._assign_splits(data, "full")
    data = train_mod._drop_invalid_rows(data)
    n_test = int((data["split"] == "test").sum())
    data = data[data["split"] != "test"].copy()          # physical firewall
    day = pd.to_datetime(data["effective_trading_day"])
    sel = (data["split"] == "val") & (day > pd.Timestamp(CFG["val_split"]["val_fit_end"]))
    # Real test rows were dropped above; the vacated label is reused for val-select so
    # the untouched pipeline scores it out-of-sample. Renamed on read-back below: no
    # selection ever sees a test row (manifest hash per trial proves it).
    data.loc[sel, "split"] = "val_select"
    if rung == 0 and rung0_frac:                          # stratified early-rung subsample
        tr = data[data.split == "train"]
        keep = (tr.sort_values("effective_trading_day", kind="mergesort")
                  .groupby(pd.to_datetime(tr["effective_trading_day"]).dt.year, group_keys=False)
                  .apply(lambda g: g.iloc[:: max(1, round(1 / rung0_frac))]))
        data = pd.concat([keep, data[data.split != "train"]], ignore_index=True)
    key = data[["accession", "horizon_days"]].astype(str).agg("|".join, axis=1)
    manifest = hashlib.sha256("\n".join(sorted(key)).encode()).hexdigest()
    return data, manifest, n_test


def trial_cfg(train_mod, model_id: str, trial: dict, max_epochs: int) -> dict:
    cfg = train_mod._load_yaml(REPO / "configs" / "models" / f"{model_id}.yaml")
    tr = cfg.setdefault("training", {})
    tr["lr"] = trial["lr"]
    tr["weight_decay"] = trial["weight_decay"]
    tr["head_lr_mult"] = trial.get("head_lr_mult", 1.0)
    tr["freeze_mode"] = trial.get("freeze_mode", "none")
    tr["objective"] = trial["objective"]
    tr["max_epochs"] = max_epochs
    tr["early_stopping"] = True
    tr["es_patience"] = 3
    base_bs = int(tr.get("batch_size", 128))
    eff = int(trial["effective_batch"])
    tr["batch_size"] = min(base_bs, eff)
    tr["grad_accumulation_steps"] = max(1, eff // tr["batch_size"])
    hd = cfg.setdefault("head", {})
    hd["hidden_dim"] = trial["head_hidden_dim"]
    hd["dropout"] = trial["head_dropout"]
    return cfg


def vol_unit_qlike(pred: pd.DataFrame, split: str) -> float:
    g = pred[pred["split"] == split]
    y = g["label_realised_vol"].to_numpy(float)
    f = g["prediction_realised_vol"].to_numpy(float)
    return float(qlike(y ** 2, f ** 2))


def run_trial(train_mod, task: str, trial: dict, rung: int, seed: int = 2026) -> dict:
    t = TASKS[task]
    max_epochs = rungs_for(task)[rung]
    rid = f"{task}_trial{trial['trial']:03d}_rung{rung}"
    if seed != 2026:
        rid += f"_s{seed}"
    out = task_dir(task) / rid
    done = out / "summary.json"
    if done.exists():
        return json.loads(done.read_text())
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    cfg = trial_cfg(train_mod, t["model"], trial, max_epochs)
    data, manifest, n_test_dropped = prepare_data(
        train_mod, t["model"], t["disclosure"],
        rung0_frac=t.get("rung0_frac"), rung=rung)
    model = train_mod._build_model(t["model"], cfg, dataset="full", run_dir=out, seed=seed)
    if t.get("fusion"):  # fusion ctors predate the objective kwarg; inject post-hoc
        model.objective = trial["objective"]
    # Process-level token-cache: tokenisation is identical across trials of a task
    # (encoder + max_length are benchmark dims, not hyperparameters), so pin one
    # cache per (model, disclosure) and every trial after the first skips it.
    if hasattr(model, "_new_token_cache"):
        key = (t["model"], t["disclosure"])
        cache = _TOK_CACHES.get(key)
        if cache is None:
            cache = model._new_token_cache()
            _TOK_CACHES[key] = cache
        model._tok_cache = cache
        model._tok_cache_pinned = True
    tr_rows = data[data.split == "train"]
    va_rows = data[data.split == "val"]
    model.fit(tr_rows, tr_rows["label_realised_vol"].to_numpy(),
              X_val=va_rows, y_val=va_rows["label_realised_vol"].to_numpy())
    pred = data.copy()
    pred["prediction_realised_vol"] = model.predict(pred)
    pred[["accession", "horizon_days", "split", "label_realised_vol",
          "prediction_realised_vol"]].to_parquet(out / "predictions.parquet", index=False)
    summary = {
        "rid": rid, "task": task, "rung": rung, "epochs_cap": max_epochs,
        "trial": trial, "manifest_sha256": manifest, "n_test_dropped": n_test_dropped,
        "val_fit_qlike": vol_unit_qlike(pred, "val"),
        "val_select_qlike": vol_unit_qlike(pred, "val_select"),
    }
    done.write_text(json.dumps(summary, indent=2))
    return summary


def survivors(task: str, rung: int) -> list[int]:
    if rung == 0:
        return [t["trial"] for t in json.loads((task_dir(task) / "trials.json").read_text())]
    return json.loads((task_dir(task) / f"survivors_rung{rung}.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plan", "search", "promote", "select"])
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--rung", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--only-trial", type=int, default=None,
                    help="run exactly this trial id (seed-validation retrains)")
    args = ap.parse_args()
    td = task_dir(args.task)

    if args.stage == "plan":
        trials = sobol_trials(args.task)
        (td / "trials.json").write_text(json.dumps(trials, indent=1))
        print(f"[plan] {args.task}: {len(trials)} trials -> {td}/trials.json")
        return

    trials = {t["trial"]: t for t in json.loads((td / "trials.json").read_text())}

    if args.stage == "search":
        train_mod = _train_mod()
        alive = ([args.only_trial] if args.only_trial is not None
                 else survivors(args.task, args.rung))
        mine = alive[args.shard::args.num_shards]
        print(f"[search] {args.task} rung {args.rung}: shard {args.shard}/{args.num_shards} "
              f"-> {len(mine)} of {len(alive)} trials")
        for k in mine:
            s = run_trial(train_mod, args.task, trials[k], args.rung, seed=args.seed)
            print(f"  trial {k:03d}: val_fit {s['val_fit_qlike']:.4f}")
        return

    if args.stage == "promote":
        alive = survivors(args.task, args.rung)
        scored = []
        for k in alive:
            f = td / f"{args.task}_trial{k:03d}_rung{args.rung}" / "summary.json"
            scored.append((json.loads(f.read_text())["val_fit_qlike"], k))
        scored.sort()
        keep = [k for _, k in scored[: math.ceil(len(scored) / CFG["asha"]["eta"])]]
        nxt = td / f"survivors_rung{args.rung + 1}.json"
        nxt.write_text(json.dumps(sorted(keep)))
        print(f"[promote] {args.task} rung {args.rung}: {len(alive)} -> {len(keep)} -> {nxt}")
        return

    if args.stage == "select":
        last = len(rungs_for(args.task)) - 1
        alive = survivors(args.task, last)
        rows = []
        for k in alive:
            f = td / f"{args.task}_trial{k:03d}_rung{last}" / "summary.json"
            s = json.loads(f.read_text())
            rows.append({"trial": k, "val_fit": s["val_fit_qlike"],
                         "val_select": s["val_select_qlike"], **trials[k]})
        df = pd.DataFrame(rows).sort_values(
            ["val_select", "lr", "weight_decay"],
            ascending=[True, True, False])  # pre-registered tiebreak
        df.to_csv(td / "final_rung_table.csv", index=False)
        top2 = df.head(2).to_dict("records")
        (td / "selection.json").write_text(json.dumps(
            {"track_a_top2": top2, "rule": CFG["selection"]["track_a"]}, indent=1))
        print(f"[select] {args.task}: top-2 by val_select ->\n{df.head(2).to_string(index=False)}")
        return


if __name__ == "__main__":
    main()
