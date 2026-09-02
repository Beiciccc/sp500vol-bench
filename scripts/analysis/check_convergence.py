"""Convergence gate — read per-run val_curves.json + config.json, classify each
run as CONVERGED / UNDERFIT_HIT_CAP / DIVERGED so a 30-step smoke (which only
proves "did not crash") is never mistaken for "did converge".

Why this exists: the rerun uses an aggressive linearly-scaled lr (8e-5 @ eff
batch 128) with a max_epochs cap + early stopping. The only reliable read on
whether lr/epochs are right is the SHAPE of the validation curve:

  * CONVERGED       — early stop fired (best epoch before the last run epoch),
                      or training stopped under the epoch cap. lr/epochs OK.
  * UNDERFIT_HIT_CAP— hit max_epochs with the LAST epoch still the best, i.e.
                      val loss was still improving when the cap cut it off
                      -> epochs too few, raise the cap.
  * DIVERGED        — NaN loss, a NaN val after finite vals, or val loss
                      rebounding to SEVERAL TIMES its min (>=3x) -> lr too high.
                      A modest early-stopping patience tail (~<=2x min) is healthy.

Per (model, horizon) is classified; the run's verdict is the worst horizon.

Usage:
    python scripts/analysis/check_convergence.py                       # all runs
    python scripts/analysis/check_convergence.py --glob 'C4_longformer_full_*'
    python scripts/analysis/check_convergence.py results/runs/C2_finbert_s1_full_long_form_seed2026

Exit code: 0 all CONVERGED, 1 any UNDERFIT_HIT_CAP/SUSPECT, 2 any DIVERGED
(so it can gate an orchestration step).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "results" / "runs"

# severity ordering — a run's verdict is the worst (highest) of its horizons
SEVERITY = {"CONVERGED": 0, "NO_DATA": 0, "SUSPECT": 1, "UNDERFIT_HIT_CAP": 2, "DIVERGED": 3}
EXIT_FOR_SEVERITY = {0: 0, 1: 1, 2: 1, 3: 2}


def _divergence_reason(train, val, finite_val, *, rebound: float, min_epochs: int) -> str | None:
    """A divergence reason for this curve, or None if it is healthy.

    NaN train loss anywhere; a NaN val *after* finite vals (blew up mid-run); or a
    val loss ending *persistently* (last two epochs) at a LARGE multiple of its own
    minimum — lr too high / training broke. The persistence check avoids flagging a
    single noisy point.

    The rebound multiple is intentionally generous (default 3x the min, i.e.
    rebound=2.0). Early stopping with patience>=2 GUARANTEES the last epochs sit above
    the min (that is why it stopped) and restores the best-epoch weights, so a modest
    patience tail — empirically up to ~1.9x the min on the aggressive lr used here — is
    a perfectly healthy CONVERGED run, not divergence. Only a val rebounding to several
    times its min (or NaN) signals a genuinely broken run worth re-tuning.
    """
    if any(t is None for t in train):
        return "nan_train_loss"
    if finite_val and any(v is None for v in val):
        return "nan_val_loss"
    if len(finite_val) >= min_epochs and min(finite_val) > 0:
        threshold = min(finite_val) * (1 + rebound)
        if all(v > threshold for v in finite_val[-2:]):
            return "val_rebound"
    return None


def classify_horizon(
    records: list[dict],
    max_epochs: int,
    *,
    rebound: float = 2.0,
    min_epochs_for_rebound: int = 3,
) -> tuple[str, dict]:
    """Classify one horizon's per-epoch curve. Pure function (unit-tested).

    Convergence is judged from the epoch of the MINIMUM val loss (the weights the
    model actually restores), not from is_best — which is unreliable when there is
    no validation signal (every epoch flags is_best=True under fixed-epoch mode).
    """
    if not records:
        return "NO_DATA", {}
    recs = sorted(records, key=lambda r: int(r["epoch"]))
    max_seen = max(int(r["epoch"]) for r in recs)
    train = [r.get("train_loss") for r in recs]
    val = [r.get("val_loss") for r in recs]
    finite_pairs = [(int(r["epoch"]), r["val_loss"]) for r in recs if r.get("val_loss") is not None]
    finite_val = [v for _, v in finite_pairs]

    detail = {"max_seen": max_seen, "max_epochs": max_epochs}

    reason = _divergence_reason(
        train, val, finite_val, rebound=rebound, min_epochs=min_epochs_for_rebound
    )
    if reason:
        return "DIVERGED", {**detail, "reason": reason}

    # No validation signal at all -> cannot judge convergence (don't default CONVERGED)
    if not finite_pairs:
        return "SUSPECT", {**detail, "reason": "no_val_signal"}

    min_val_epoch, min_val = min(finite_pairs, key=lambda p: p[1])
    detail.update(min_val=min_val, final_val=finite_val[-1], min_val_epoch=min_val_epoch)

    # Hit the cap AND the best (min) val is the last epoch => still improving when cut off
    if max_seen >= max_epochs and min_val_epoch >= max_seen:
        return "UNDERFIT_HIT_CAP", detail
    return "CONVERGED", detail


EXPECTED_HORIZONS = ("5", "10", "20")


def _is_neural_run(run_dir: Path) -> bool:
    """True if config.json marks a neural/fusion run (training.max_epochs present)."""
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return False
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return "max_epochs" in cfg["model_config"]["training"]
    except (KeyError, ValueError, TypeError):
        return False


def classify_run(run_dir: Path) -> dict | None:
    """Convergence verdict for a run, or None for a non-neural (A/B) run.

    A neural run missing val_curves.json, or missing some expected horizons (e.g.
    a checkpoint-resumed run that skipped already-trained horizons and so never
    re-recorded their curves), is SUSPECT — never silently CONVERGED.
    """
    vc_path = run_dir / "val_curves.json"
    neural = _is_neural_run(run_dir)
    if not vc_path.exists():
        if neural:
            return {
                "run": run_dir.name,
                "verdict": "SUSPECT",
                "max_epochs": _read_max_epochs(run_dir),
                "horizons": {"-": ("SUSPECT", {"reason": "neural run missing val_curves.json"})},
            }
        return None  # A/B model — nothing to gate
    curves = json.loads(vc_path.read_text(encoding="utf-8"))
    max_epochs = _read_max_epochs(run_dir)

    per_h: dict[str, tuple[str, dict]] = {}
    # expected horizons absent from the curve (resume gap) -> SUSPECT
    for hz in EXPECTED_HORIZONS:
        if hz not in curves:
            per_h[hz] = ("SUSPECT", {"reason": "horizon missing from val_curves"})
    for hz, records in curves.items():
        per_h[str(hz)] = classify_horizon(records, max_epochs)

    worst = "CONVERGED"
    for verdict, _ in per_h.values():
        if SEVERITY[verdict] > SEVERITY[worst]:
            worst = verdict
    return {"run": run_dir.name, "verdict": worst, "max_epochs": max_epochs, "horizons": per_h}


def _read_max_epochs(run_dir: Path, default: int = 15) -> int:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return default
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return int(cfg["model_config"]["training"]["max_epochs"])
    except (KeyError, ValueError, TypeError):
        return default


def d_verdict_short(v: str) -> str:
    return {
        "CONVERGED": "ok",
        "UNDERFIT_HIT_CAP": "CAP",
        "DIVERGED": "DIV",
        "SUSPECT": "?",
        "NO_DATA": "-",
    }[v]


def _write_md(rows: list[dict], path: Path) -> None:
    lines = [
        "# Convergence gate",
        "",
        "Per run, worst-horizon verdict from the validation curve. `(best/seen)` = best epoch "
        "/ last epoch run; cap = max_epochs.",
        "",
        "- **CONVERGED** — early-stopped or stopped under cap; lr/epochs OK.",
        "- **UNDERFIT_HIT_CAP** — hit cap with last epoch still best -> raise max_epochs.",
        "- **DIVERGED** — NaN / val rebound -> lower lr.",
        "",
        "| run | verdict | cap | per-horizon (verdict, best/seen) |",
        "|---|---|--:|---|",
    ]
    for r in sorted(rows, key=lambda x: (-SEVERITY[x["verdict"]], x["run"])):
        per_h = " ".join(
            f"h{hz}:{d_verdict_short(v)}({d.get('min_val_epoch', '?')}/{d.get('max_seen', '?')})"
            for hz, (v, d) in r["horizons"].items()
        )
        lines.append(f"| {r['run']} | **{r['verdict']}** | {r['max_epochs']} | {per_h} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="*", help="run dirs or run_ids (default: --glob)")
    parser.add_argument("--glob", default="*", help="glob under results/runs/ (default: all)")
    parser.add_argument(
        "--out-md", type=Path, default=REPO_ROOT / "results/tables/convergence_gate.md"
    )
    args = parser.parse_args()

    if args.targets:
        dirs = [Path(t) if Path(t).is_dir() else RUNS_DIR / t for t in args.targets]
    else:
        # glob mode excludes *_smoke runs (their capped max_epochs always "hits the
        # cap" and would pollute the gate); pass them explicitly as targets to inspect.
        dirs = sorted(
            d for d in RUNS_DIR.glob(args.glob) if d.is_dir() and not d.name.endswith("_smoke")
        )

    rows = [r for d in dirs if (r := classify_run(d)) is not None]
    if not rows:
        print("no runs with val_curves.json to gate")
        return 0

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    _write_md(rows, args.out_md)

    worst_sev = 0
    counts: dict[str, int] = {}
    for r in sorted(rows, key=lambda x: (-SEVERITY[x["verdict"]], x["run"])):
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        worst_sev = max(worst_sev, SEVERITY[r["verdict"]])
        flag = "" if r["verdict"] == "CONVERGED" else "  <<<"
        print(f"{r['verdict']:18s} {r['run']}{flag}")
    print(f"\n{dict(counts)} -> {args.out_md}")
    return EXIT_FOR_SEVERITY[worst_sev]


if __name__ == "__main__":
    raise SystemExit(main())
