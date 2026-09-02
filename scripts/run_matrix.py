"""Orchestrate the 4xA100 C+D rerun in three gated stages — one run PER GPU, parallel.

A 4xA100 box runs four independent runs concurrently (one per GPU via
CUDA_VISIBLE_DEVICES); the 144-run matrix is data-parallel-BY-TASK, not DDP.

Stages:
  smoke  — every model x long_form, `train.py --smoke` (tiny data, real batch/
           seq-len). Crash check before burning full runs; parallel across GPUs.
  pilot  — 3 representative FULL runs + convergence gate (validates lr/epochs on
           real val-curves before the full matrix).
  full   — the 16 x 3 x 3 = 144 matrix, dispatched seed-batch by seed-batch so a
           config that diverges on one seed skips its remaining seeds. Per-run
           gate; resumable; gate over explicit targets (no-evidence => FAIL).

Usage:
    python scripts/run_matrix.py --stage smoke
    python scripts/run_matrix.py --stage pilot
    python scripts/run_matrix.py --stage full --gpus 4
    python scripts/run_matrix.py --stage full --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from queue import Empty, Queue

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "results" / "runs"

# 16 C+D models (A+B reused, not rerun)
MODELS = [
    "C1_bert_s1", "C1_bert_s2",
    "C2_finbert_s1", "C2_finbert_s2", "C2_finbert_s3", "C2_finbert_s4",
    "C3_roberta_s1", "C4_longformer",
    "C5_qwen3", "C5_gteqwen2", "C5_e5mistral",
    "D1_concat_mlp", "D2_gated_fusion",
    "D3_qwen3", "D3_gteqwen2", "D3_e5mistral",
]
DISCLOSURES = ["long_form", "event_driven", "combined"]
SEEDS = [2026, 2027, 2028]

# 3 representative pilots: distinct convergence characters
PILOT = [
    ("C2_finbert_s1", "long_form", 2026),  # fine-tune, lr-sensitive
    ("C4_longformer", "long_form", 2026),  # long sequence -> highest lr risk
    ("C5_qwen3", "long_form", 2026),       # frozen-embedding probe (head-only)
]

# training-config keys whose change must invalidate a prior run — NOT just max_epochs,
# so a post-pilot lr (or batch/warmup) change forces a re-run instead of keeping stale.
_FINGERPRINT_KEYS = (
    "lr", "max_epochs", "warmup_ratio", "es_patience", "es_min_delta",
    "batch_size", "grad_accumulation_steps", "mixed_precision", "weight_decay",
)


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "check_convergence", REPO_ROOT / "scripts" / "analysis" / "check_convergence.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GATE = _load_gate()


def run_id(model: str, disclosure: str, seed: int, dataset: str, smoke: bool) -> str:
    rid = f"{model}_{dataset}_{disclosure}_seed{seed}"
    return rid + "_smoke" if smoke else rid


def _config_fingerprint(model: str) -> dict | None:
    """Current training-config fingerprint for a model; None for non-neural (A/B)."""
    cfg = OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / "configs" / "models" / f"{model}.yaml"), resolve=True
    )
    tr = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    if not isinstance(tr, dict) or "max_epochs" not in tr:
        return None
    return {k: tr.get(k) for k in _FINGERPRINT_KEYS}


def _run_fingerprint(run_dir: Path) -> dict | None:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return None
    try:
        tr = json.loads(cfg_path.read_text(encoding="utf-8"))["model_config"]["training"]
    except (KeyError, ValueError, TypeError):
        return None
    return {k: tr.get(k) for k in _FINGERPRINT_KEYS}


def is_done(rid: str, model: str) -> bool:
    """Done only if outputs exist AND (for neural models) the run's FULL training
    fingerprint matches the current config — so a post-pilot lr/epoch/batch change
    re-runs the cell instead of silently keeping stale predictions."""
    d = RUNS_DIR / rid
    if not ((d / "predictions.parquet").exists() and (d / "metrics.json").exists()):
        return False
    expected = _config_fingerprint(model)
    if expected is None:  # A/B — output presence is enough
        return True
    return _run_fingerprint(d) == expected


def train_one(
    model: str, disclosure: str, seed: int, dataset: str, smoke: bool, gpu_id: int | None
) -> int:
    cmd = [
        sys.executable, "scripts/train.py",
        "--model", model, "--dataset", dataset, "--disclosure", disclosure, "--seed", str(seed),
    ]
    if smoke:
        cmd.append("--smoke")
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False).returncode


def gate_one(rid: str) -> str:
    res = GATE.classify_run(RUNS_DIR / rid)
    return res["verdict"] if res else "CONVERGED"


def _dispatch(jobs, dataset, *, smoke, n_gpus, diverged, dlock, gpu_ids=None):
    """Run (model, disclosure, seed) jobs across worker GPUs, ONE run per GPU.
    `gpu_ids` pins workers to explicit PHYSICAL GPU ids (e.g. [1, 3]); when None,
    defaults to range(n_gpus). Mutates `diverged` (set of (model, disclosure)) under
    dlock; returns a list of (rid, status, verdict). Diverged-aware skip applies only
    to full runs."""
    q: Queue = Queue()
    for j in jobs:
        q.put(j)
    results: list[tuple[str, str, str | None]] = []
    rlock = threading.Lock()

    def record(rid, status, verdict):
        with rlock:
            results.append((rid, status, verdict))

    def worker(gpu_id):
        while True:
            try:
                model, disclosure, seed = q.get_nowait()
            except Empty:
                return
            try:
                rid = run_id(model, disclosure, seed, dataset, smoke)
                if not smoke and is_done(rid, model):
                    v = gate_one(rid)
                    if v == "DIVERGED":  # a stale-but-done diverged run still gates its siblings
                        with dlock:
                            diverged.add((model, disclosure))
                    record(rid, "done", v)
                    print(f"  [gpu{gpu_id}] done: {rid} [{v}]", flush=True)
                    continue
                if not smoke:
                    with dlock:
                        bad = (model, disclosure) in diverged
                    if bad:
                        record(rid, "skipped", None)
                        print(f"  [gpu{gpu_id}] SKIP (config diverged): {rid}", flush=True)
                        continue
                rc = train_one(model, disclosure, seed, dataset, smoke, gpu_id)
                if rc != 0:
                    record(rid, "crashed", None)
                    print(f"  [gpu{gpu_id}] CRASH(rc={rc}): {rid}", flush=True)
                    continue
                v = gate_one(rid)
                record(rid, "trained", v)
                print(f"  [gpu{gpu_id}] {rid}: {v}", flush=True)
                if not smoke and v == "DIVERGED":
                    with dlock:
                        diverged.add((model, disclosure))
                    print(f"     !! {model}/{disclosure} diverged — pending seeds skip", flush=True)
            finally:
                q.task_done()

    ids = list(gpu_ids) if gpu_ids is not None else list(range(n_gpus))
    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def _gate(targets) -> int:
    """Gate over EXPLICIT run-id targets. No produced run => no evidence => FAIL (2),
    so a stage can never green-light the next on an empty result set."""
    existing = [t for t in targets if (RUNS_DIR / t).exists()]
    if not existing:
        print("gate: no runs produced — no evidence, treating as FAIL")
        return 2
    return subprocess.run(
        [sys.executable, "scripts/analysis/check_convergence.py", *existing],
        cwd=REPO_ROOT, check=False,
    ).returncode


def stage_smoke(dataset, models, n_gpus, dry) -> int:
    print(f"=== STAGE smoke: {len(models)} models x long_form, {n_gpus} GPUs parallel ===")
    jobs = [(m, "long_form", 2026) for m in models]
    if dry:
        for m, d, s in jobs:
            print(f"  [plan] {run_id(m, d, s, dataset, True)}")
        return 0
    results = _dispatch(
        jobs, dataset, smoke=True, n_gpus=n_gpus, diverged=set(), dlock=threading.Lock()
    )
    crashed = [rid for rid, st, _ in results if st == "crashed"]
    if crashed:
        print(f"\nSMOKE FAILED — crashed: {crashed}")
        return 1
    print("\nSMOKE PASS — all model paths ran end-to-end")
    return 0


def stage_pilot(dataset, n_gpus, dry) -> int:
    print(f"=== STAGE pilot: {len(PILOT)} full runs + gate, {n_gpus} GPUs parallel ===")
    if dry:
        for m, d, s in PILOT:
            print(f"  [plan] {run_id(m, d, s, dataset, False)}")
        return 0
    _dispatch(
        list(PILOT), dataset, smoke=False, n_gpus=n_gpus, diverged=set(), dlock=threading.Lock()
    )
    print("\n--- convergence gate (pilot) ---")
    rc = _gate([run_id(m, d, s, dataset, False) for m, d, s in PILOT])
    verdict = {0: "all CONVERGED — clear to launch full", 1: "UNDERFIT/SUSPECT — review lr/epochs",
               2: "DIVERGED/no-evidence — fix before full"}.get(rc, "?")
    print(f"\nPILOT gate: {verdict}")
    return rc


def stage_full(dataset, models, disclosures, seeds, n_gpus, dry, *, flat=False, gpu_ids=None) -> int:
    total = len(models) * len(disclosures) * len(seeds)
    where = f"GPUs {gpu_ids}" if gpu_ids is not None else f"{n_gpus} GPUs"
    mode = "FLAT single-queue" if flat else "seed-batched"
    print(f"=== STAGE full: {len(models)}x{len(disclosures)}x{len(seeds)} = {total}, {where} parallel, {mode} ===")
    if dry:
        for s in seeds:
            for m in models:
                for d in disclosures:
                    print(f"  [plan] {run_id(m, d, s, dataset, False)}")
        return 0
    diverged: set = set()
    dlock = threading.Lock()
    tally = {"trained": 0, "done": 0, "skipped": 0, "crashed": 0}
    diverged_verdicts = 0
    # FLAT: one queue over ALL seeds (no per-seed barrier — keeps every GPU busy to the
    # end, fixing the tail-idle where a slow model leaves spare GPUs waiting). Seeds are
    # ordered first so the diverged-skip (a config diverging on an earlier seed skips its
    # later seeds) still mostly fires. seed-batched: the original gated mode.
    batches = ([[(m, d, s) for s in seeds for m in models for d in disclosures]] if flat
               else [[(m, d, seed) for m in models for d in disclosures] for seed in seeds])
    for jobs in batches:
        label = "FLAT" if flat else f"seed {jobs[0][2]}"
        print(f"\n--- {label} batch ({len(jobs)} runs) ---")
        results = _dispatch(
            jobs, dataset, smoke=False, n_gpus=n_gpus, diverged=diverged, dlock=dlock, gpu_ids=gpu_ids
        )
        for _, st, v in results:
            if st in tally:
                tally[st] += 1
            if v == "DIVERGED":
                diverged_verdicts += 1
    print(
        f"\nFULL summary: {tally} diverged_verdicts={diverged_verdicts} "
        f"diverged_configs={len(diverged)}"
    )
    print("\n--- final convergence gate ---")
    targets = [
        run_id(m, d, s, dataset, False) for m in models for d in disclosures for s in seeds
    ]
    return _gate(targets)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["smoke", "pilot", "full"])
    parser.add_argument("--dataset", default="full")
    parser.add_argument("--gpus", type=int, default=4, help="parallel workers, one run per GPU")
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=None,
                        help="explicit PHYSICAL GPU ids to pin workers to (e.g. --gpu-ids 1 3). "
                             "Overrides --gpus. Lets a second orchestrator use only the idle GPUs.")
    parser.add_argument("--flat", action="store_true",
                        help="full stage: one queue over ALL seeds (no per-seed barrier) so no "
                             "GPU idles waiting for a slow model's tail.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--disclosures", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    models = args.models or MODELS
    disclosures = args.disclosures or DISCLOSURES
    seeds = args.seeds or SEEDS
    gpu_ids = args.gpu_ids if args.gpu_ids else None
    n_gpus = len(gpu_ids) if gpu_ids else max(1, args.gpus)

    if args.stage == "smoke":
        return stage_smoke(args.dataset, models, n_gpus, args.dry_run)
    if args.stage == "pilot":
        return stage_pilot(args.dataset, n_gpus, args.dry_run)
    return stage_full(args.dataset, models, disclosures, seeds, n_gpus, args.dry_run,
                      flat=args.flat, gpu_ids=gpu_ids)


if __name__ == "__main__":
    raise SystemExit(main())
