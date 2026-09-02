"""TASK A4 — Cost vs accuracy efficiency frontier.

Aggregates GPU-hours per model across the 3 seeds x 3 disclosures (SUM for total,
MEAN for per-run) from cost.json, reads best test QLIKE per model on long_form from
metrics.json (min over horizons, plus the 3-horizon vector), and identifies the
Pareto frontier on (accuracy=best_qlike lower-better, cost=gpu_hours_total).

A/B baselines are CPU / seed-invariant: reported as ~0 GPU-hours (their recorded
total_seconds are CPU wall-clock, not GPU) so they anchor the cheap end.

Writes results/tables/cost_accuracy.{csv,md}.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "runs"
TABLES = ROOT / "results" / "tables"

HORIZONS = (5, 10, 20)
SEEDS_CD = (2026, 2027, 2028)
DISCLOSURES = ("long_form", "event_driven", "combined")

# block membership
A_MODELS = ["A1_hv", "A2_har_rv", "A3_garch", "A4_egarch", "A5_arima"]
B_MODELS = ["B1_bow_ridge", "B2_tfidf_ridge", "B3_lm_linear", "B4_lm_features"]
C_MODELS = ["C1_bert_s1", "C1_bert_s2", "C2_finbert_s1", "C2_finbert_s2",
            "C2_finbert_s3", "C2_finbert_s4", "C3_roberta_s1", "C4_longformer",
            "C5_qwen3", "C5_gteqwen2", "C5_e5mistral"]
D_MODELS = ["D1_concat_mlp", "D2_gated_fusion", "D3_qwen3", "D3_gteqwen2", "D3_e5mistral"]


def block_of(m):
    return m[0]


def read_cost(run_id):
    f = RUNS / run_id / "cost.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    return d.get("total_gpu_hours", 0.0), d.get("total_seconds", 0.0)


def read_qlike_long_form(run_id):
    """Return {h: qlike} for split==test, disclosure_subset==long_form."""
    f = RUNS / run_id / "metrics.json"
    if not f.exists():
        return {}
    rows = json.load(open(f))
    out = {}
    for r in rows:
        if r.get("split") == "test" and r.get("disclosure_subset") == "long_form" \
                and r.get("horizon_days") in HORIZONS:
            out[r["horizon_days"]] = r["qlike"]
    return out


def gather():
    recs = []
    # A / B: seed-invariant (seed2026 only), CPU -> report ~0 GPU-hours
    for m in A_MODELS + B_MODELS:
        secs_total = 0.0
        for d in DISCLOSURES:
            c = read_cost(f"{m}_full_{d}_seed2026")
            if c:
                secs_total += c[1]
        q = read_qlike_long_form(f"{m}_full_long_form_seed2026")
        recs.append(dict(model=m, block=block_of(m),
                         gpu_hours_total=0.0, gpu_hours_per_run=0.0,
                         cpu_seconds_total=secs_total, n_runs=len(DISCLOSURES),
                         qlike=q))
    # C / D: 3 seeds x 3 disclosures = 9 GPU runs each
    for m in C_MODELS + D_MODELS:
        gh_total = 0.0
        n = 0
        for d in DISCLOSURES:
            for s in SEEDS_CD:
                c = read_cost(f"{m}_full_{d}_seed{s}")
                if c:
                    gh_total += c[0]
                    n += 1
        # best long_form QLIKE: average over the 3 seeds per horizon, then min over horizons
        per_h = {h: [] for h in HORIZONS}
        for s in SEEDS_CD:
            q = read_qlike_long_form(f"{m}_full_long_form_seed{s}")
            for h in HORIZONS:
                if h in q:
                    per_h[h].append(q[h])
        q_mean = {h: (sum(v) / len(v) if v else float("nan")) for h, v in per_h.items()}
        recs.append(dict(model=m, block=block_of(m),
                         gpu_hours_total=gh_total,
                         gpu_hours_per_run=(gh_total / n if n else 0.0),
                         cpu_seconds_total=0.0, n_runs=n, qlike=q_mean))
    return recs


def pareto(recs):
    """Lower gpu_hours_total AND lower best_qlike is better. Mark non-dominated."""
    pts = []
    for r in recs:
        bq = r["best_qlike"]
        cost = r["gpu_hours_total"]
        pts.append((r, cost, bq))
    for r, cost, bq in pts:
        dominated = False
        for r2, cost2, bq2 in pts:
            if r2 is r:
                continue
            # r2 dominates r if cheaper-or-equal and more-accurate-or-equal, strict in one
            if cost2 <= cost and bq2 <= bq and (cost2 < cost or bq2 < bq):
                dominated = True
                break
        r["on_pareto_frontier"] = not dominated
    return recs


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    recs = gather()
    for r in recs:
        q = r["qlike"]
        vals = [q[h] for h in HORIZONS if h in q and q[h] == q[h]]
        r["best_qlike"] = min(vals) if vals else float("nan")
    recs = pareto(recs)
    # sort by best_qlike (accuracy) ascending, tie-break cheaper
    recs.sort(key=lambda r: (r["best_qlike"], r["gpu_hours_total"]))

    # CSV
    import csv
    cols = ["model", "block", "gpu_hours_total", "gpu_hours_per_run",
            "qlike_long_form_h5", "qlike_long_form_h10", "qlike_long_form_h20",
            "best_qlike", "on_pareto_frontier"]
    csv_path = TABLES / "cost_accuracy.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in recs:
            q = r["qlike"]
            w.writerow([
                r["model"], r["block"],
                f"{r['gpu_hours_total']:.3f}", f"{r['gpu_hours_per_run']:.3f}",
                f"{q.get(5, float('nan')):.4f}", f"{q.get(10, float('nan')):.4f}",
                f"{q.get(20, float('nan')):.4f}", f"{r['best_qlike']:.4f}",
                "yes" if r["on_pareto_frontier"] else "no",
            ])

    # totals
    tot_c = sum(r["gpu_hours_total"] for r in recs if r["block"] == "C")
    tot_d = sum(r["gpu_hours_total"] for r in recs if r["block"] == "D")
    tot_cd = tot_c + tot_d

    # MD
    md = []
    md.append("# Cost vs Accuracy — Efficiency Frontier (Task A4)\n")
    md.append("Cost = total GPU-hours summed across all runs of that model "
              "(C/D: 3 seeds x 3 disclosures = 9 runs; per-run = mean). "
              "Accuracy = best test QLIKE on **long_form** (min over horizons 5/10/20; "
              "C/D QLIKE averaged across the 3 seeds per horizon first). "
              "A/B are CPU / seed-invariant baselines reported at ~0 GPU-hours to anchor "
              "the cheap end (their CPU wall-clock seconds noted separately).\n")
    md.append(f"**Total GPU-hours across all C runs = {tot_c:.1f}; "
              f"all D runs = {tot_d:.1f}; C+D = {tot_cd:.1f}.**\n")
    md.append("| Rank | Model | Block | GPU-h total | GPU-h/run | "
              "QLIKE h5 | QLIKE h10 | QLIKE h20 | Best QLIKE | Pareto |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(recs, 1):
        q = r["qlike"]
        pf = "**YES**" if r["on_pareto_frontier"] else "no"
        md.append(f"| {i} | {r['model']} | {r['block']} | "
                  f"{r['gpu_hours_total']:.2f} | {r['gpu_hours_per_run']:.3f} | "
                  f"{q.get(5, float('nan')):.4f} | {q.get(10, float('nan')):.4f} | "
                  f"{q.get(20, float('nan')):.4f} | {r['best_qlike']:.4f} | {pf} |")
    md.append("\n### CPU baselines — recorded wall-clock seconds (reported at ~0 GPU-h)")
    md.append("| Model | Block | CPU seconds (sum over 3 disclosures) |")
    md.append("|---|---|---|")
    for r in recs:
        if r["block"] in ("A", "B"):
            md.append(f"| {r['model']} | {r['block']} | {r['cpu_seconds_total']:.1f} |")
    md_path = TABLES / "cost_accuracy.md"
    open(md_path, "w").write("\n".join(md) + "\n")

    return recs, tot_c, tot_d, tot_cd, csv_path, md_path


if __name__ == "__main__":
    recs, tot_c, tot_d, tot_cd, csv_path, md_path = main()
    print(f"total C GPU-h={tot_c:.2f}  D GPU-h={tot_d:.2f}  C+D={tot_cd:.2f}")
    print(f"wrote {csv_path}\nwrote {md_path}")
    pf = [r["model"] for r in recs if r["on_pareto_frontier"]]
    print("Pareto frontier:", pf)
    for r in recs:
        print(f"{r['model']:16s} blk={r['block']} gpu_h={r['gpu_hours_total']:7.2f} "
              f"per_run={r['gpu_hours_per_run']:6.3f} best_qlike={r['best_qlike']:.4f} "
              f"pareto={'Y' if r['on_pareto_frontier'] else '.'}")
