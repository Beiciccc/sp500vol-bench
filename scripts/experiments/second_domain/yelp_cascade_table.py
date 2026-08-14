#!/usr/bin/env python
"""THE second-domain cascade table (SECOND_DOMAIN_PLAN.md §2) — assembly + gates.

Five rows, columns = per-horizon test MSE + month-clustered DM p:
  1  naive pooled-split text gain (the field's standard design; identity share =
     how much of that apparent gain a ZERO-TEXT entity-mean-only predictor recovers
     under the SAME pooled split)
  2  chronological text-alone (log-recalibrated) vs recalibrated AR
  3  AR + text combiner delta vs recalibrated AR
  4  AR + entity-mean delta vs recalibrated AR (identity control, zero text)
  5  AR + entity-mean + text RESIDUAL delta, month-clustered DM p + two-way p +
     label-shuffle placebo — the honest headline cell; MDE in the note.

Inputs (all produced upstream; synthetic vs real is upstream's --data-root only):
  --protocol   protocol_results.json      (yelp_protocol.py)
  --preds-dir  preds_tfidf_naive_pooled.parquet + baseline_metrics.json
               (yelp_baseline_text.py)
  --panel      canonical panel parquet    (yelp_build_panel.py; gate G1 counts and
               the naive entity-mean-only arm's fit set)

The naive entity-mean-only arm is computed HERE: the pooled 80/20 eval rows come
from preds_tfidf_naive_pooled.parquet; the fit set is the panel complement; the
predictor is the fit-set mean label per business (fallback: fit-set global mean).
The recomputed pooled-mean and text MSEs are asserted equal to baseline_metrics.json.

Gates auto-filled from the numbers (SANITY section):
  G1 panel shape; G2 recalibration + AR beats global-mean/last-value;
  G3 naive arm credits text; G4 placebos clean; G5 MDE <= naive apparent gain.
On a synthetic run (protocol_results.json carries oracle_injected_dmse) a RECOVERY
section additionally verifies the machinery against the KNOWN injected structure.

Outputs: results/tables/yelp_cascade.{csv,md} (suffix "_<tag>" when --tag is not
REAL, so a synthetic run can never overwrite the real table). Exit code 1 if a
hard machinery gate fails.

Run from repo root:
    .venv/bin/python scripts/experiments/second_domain/yelp_cascade_table.py \
        [--tag SYNTHETIC]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
G4_DM_MAX = 2.0          # placebo |mean DM| threshold (forecast_combination.py gate)
REC_BAND = (0.25, 1.10)  # d5 / oracle-given-text band at the primary horizon
REC_NULL_REL = 0.30      # % floor: sub-MDE horizons must not invent a residual


def fmt_p(p):
    return "n/a" if p is None or (isinstance(p, float) and np.isnan(p)) else f"{p:.4f}"


def naive_entity_arm(panel_path, preds_dir, metrics):
    """Zero-text entity-mean-only arm under the SAME pooled random split as the
    naive text arm; returns per-horizon dict with the identity share."""
    panel = pd.read_parquet(panel_path,
                            columns=["entity_id", "event_time", "horizon_months", "label"])
    ev = pd.read_parquet(preds_dir / "preds_tfidf_naive_pooled.parquet")
    out = {}
    for h, met in metrics.items():
        ph = panel[panel.horizon_months == h]
        eh = ev[ev.horizon_months == h]
        assert len(eh) == met["naive_n_test"], "naive eval rows do not match metrics"
        key = ["entity_id", "event_time"]
        eval_keys = pd.MultiIndex.from_frame(eh[key])
        fit = ph[~pd.MultiIndex.from_frame(ph[key]).isin(eval_keys)]
        assert len(fit) + len(eh) == len(ph), "naive fit/eval split does not partition"

        base = float(fit.label.mean())
        y = eh.label.to_numpy(float)
        mse_base = float(np.mean((y - base) ** 2))
        mse_text = float(np.mean((y - eh.prediction.to_numpy(float)) ** 2))
        assert abs(mse_base - met["mse_naive_pooled_mean"]) < 1e-8, \
            "recomputed pooled-mean MSE disagrees with baseline_metrics.json"
        assert abs(mse_text - met["mse_naive_text"]) < 1e-8, \
            "recomputed naive text MSE disagrees with baseline_metrics.json"

        emap = fit.groupby("entity_id")["label"].mean()
        pred_ent = eh.entity_id.map(emap).fillna(base).to_numpy(float)
        mse_ent = float(np.mean((y - pred_ent) ** 2))
        share = (100.0 * (mse_base - mse_ent) / (mse_base - mse_text)
                 if mse_base > mse_text else float("nan"))
        out[h] = {"mse_base": mse_base, "mse_text": mse_text,
                  "mse_entity_mean": mse_ent,
                  "gain_text_pct": 100.0 * (mse_base - mse_text) / mse_base,
                  "gain_entity_pct": 100.0 * (mse_base - mse_ent) / mse_base,
                  "identity_share_naive_pct": share}
    return out


def build_rows(hs, res, metrics, naive):
    """Tidy per-(row, horizon) records for the CSV and the markdown table."""
    recs = []
    for h in hs:
        r, m, nv = res[str(h)], metrics[h], naive[h]
        recs += [
            dict(row=1, arm="naive pooled-split text (random 80/20; field design)",
                 reference="pooled mean", h=h, mse=nv["mse_text"],
                 delta_rel_pct=nv["gain_text_pct"], dm_p=None,
                 note=f"identity share (entity-mean-only, zero text) = "
                      f"{nv['identity_share_naive_pct']:.0f}% of the apparent gain"),
            dict(row=2, arm="chronological text-alone (log-recalibrated)",
                 reference="recalibrated AR f_R", h=h, mse=r["mse"]["T"],
                 delta_rel_pct=r["text_alone"]["delta_rel_pct"],
                 dm_p=r["text_alone"]["p"], note=f"DM {r['text_alone']['dm']:+.2f}"),
            dict(row=3, arm="AR + text combiner f_U",
                 reference="recalibrated AR f_R", h=h, mse=r["mse"]["U"],
                 delta_rel_pct=r["ar_text"]["delta_rel_pct"], dm_p=r["ar_text"]["p"],
                 note=f"DM {r['ar_text']['dm']:+.2f}; two-way p="
                      f"{fmt_p(r['ar_text']['p_2way'])}"),
            dict(row=4, arm="AR + entity-mean (identity control, zero text)",
                 reference="recalibrated AR f_R", h=h, mse=r["mse"]["Re"],
                 delta_rel_pct=r["entity"]["delta_rel_pct"], dm_p=r["entity"]["p"],
                 note=f"DM {r['entity']['dm']:+.2f}; chrono identity share "
                      f"{r['entity']['identity_share_chrono_pct']:.0f}% of row-3 gain"),
            dict(row=5, arm="AR + entity-mean + text (RESIDUAL text increment)",
                 reference="AR + entity-mean f_Re", h=h, mse=r["mse"]["Ue"],
                 delta_rel_pct=r["residual"]["delta_rel_pct"], dm_p=r["residual"]["p"],
                 note=f"DM {r['residual']['dm']:+.2f}; two-way p="
                      f"{fmt_p(r['residual']['p_2way'])}; placebo mean DM "
                      f"{r['placebo']['row5_shuffle']['mean_dm']:+.2f} "
                      f"(mean p={r['placebo']['row5_shuffle']['mean_p']:.3f})"),
        ]
    return pd.DataFrame(recs)


def gates(hs, res, metrics, naive, panel_path):
    """G1-G5 auto-filled from the numbers; returns list of (gate, pass, detail)."""
    panel = pd.read_parquet(panel_path, columns=["entity_id", "event_time",
                                                 "horizon_months", "split"])
    base = panel.drop_duplicates(["entity_id", "event_time"])
    n_ent, n_ev = base.entity_id.nunique(), len(base)
    cells = all(res[str(h)]["n_val"] >= 100 and res[str(h)]["n_test"] >= 30 for h in hs)
    g = [("G1 panel shape", n_ent >= 100 and n_ev >= 10_000 and cells,
          f"{n_ent:,} entities, {n_ev:,} events; val>=100 & test>=30 per horizon: "
          f"{cells}")]

    ok2, det2 = True, []
    for h in hs:
        m, b = metrics[h], res[str(h)]["recal_b"]
        ok = (0.5 <= b <= 1.5 and m["mse_test_ar_recal"] < m["mse_test_global_mean"]
              and m["mse_test_ar_recal"] < m["mse_test_last_value"])
        ok2 &= ok
        det2.append(f"h={h}: b={b:.3f}, AR recal {m['mse_test_ar_recal']:.4f} vs "
                    f"global {m['mse_test_global_mean']:.4f} / last "
                    f"{m['mse_test_last_value']:.4f}")
    g.append(("G2 AR baseline sane", ok2, "; ".join(det2)))

    ok3 = all(naive[h]["gain_text_pct"] > 0 for h in hs)
    g.append(("G3 naive arm credits text", ok3,
              "; ".join(f"h={h}: {naive[h]['gain_text_pct']:+.1f}%" for h in hs)))

    # G4 primary gate = the PRE-REGISTERED label-shuffle placebo (SECOND_DOMAIN_PLAN.md
    # names it the main placebo; within-month text-swap is the OPTIONAL diagnostic and
    # is reported in full as G4b, not folded into the pass/fail gate).
    ok4, worst = True, 0.0
    for h in hs:
        for k in ("row3_shuffle", "row5_shuffle"):
            pl = res[str(h)]["placebo"][k]
            worst = max(worst, abs(pl["mean_dm"]))
            ok4 &= abs(pl["mean_dm"]) < G4_DM_MAX and pl["mean_p"] > 0.05
    g.append(("G4 label-shuffle placebos clean (pre-registered primary)", ok4,
              f"max |mean DM| = {worst:.2f} (threshold {G4_DM_MAX}); all mean p > .05: {ok4}"))
    swap_det = []
    for h in hs:
        for k in ("row3_swap", "row5_swap"):
            pl = res[str(h)]["placebo"][k]
            flag = " (BORDERLINE)" if pl["mean_p"] <= 0.05 or abs(pl["mean_dm"]) >= G4_DM_MAX else ""
            swap_det.append(f"h={h} {k}: mean DM {pl['mean_dm']:+.2f}, mean p {pl['mean_p']:.3f}{flag}")
    g.append(("G4b within-month text-swap (diagnostic, fully disclosed)", True,
              "; ".join(swap_det)))

    ok5, det5 = True, []
    for h in hs:
        mde = max(res[str(h)]["mde"]["ar_stage_rel_pct"],
                  res[str(h)]["mde"]["entity_stage_rel_pct"])
        ok = mde <= naive[h]["gain_text_pct"]
        ok5 &= ok
        det5.append(f"h={h}: MDE {mde:.2f}% vs naive gain "
                    f"{naive[h]['gain_text_pct']:.1f}%")
    g.append(("G5 MDE <= naive apparent gain", ok5, "; ".join(det5)))
    return g


def recovery(hs, res, metrics, naive):
    """SYNTHETIC-only machinery-validation checks against the KNOWN injected
    structure (oracle benchmark computed by yelp_protocol.py from truth_months)."""
    h1 = min(hs)
    r1 = res[str(h1)]
    checks = []

    b_ok = all(0.5 <= res[str(h)]["recal_b"] <= 1.5
               and res[str(h)]["mse"]["R"] <= 1.02 * metrics[h]["mse_test_ar_raw"]
               for h in hs)
    checks.append(("R1 combiner recalibration sane (b in [0.5,1.5]; near-neutral on "
                   "an already-calibrated baseline)", b_ok,
                   "; ".join(f"h={h}: b={res[str(h)]['recal_b']:.3f}, "
                             f"MSE {metrics[h]['mse_test_ar_raw']:.4f}->"
                             f"{res[str(h)]['mse']['R']:.4f}" for h in hs)))

    t = r1["text_alone"]
    checks.append((f"R2 text-alone loses chronologically (h={h1})",
                   t["delta_rel_pct"] < 0 and t["dm"] > 0 and t["p"] < 0.05,
                   f"{t['delta_rel_pct']:+.2f}%, DM {t['dm']:+.2f}, p={t['p']:.4f}"))

    share = min(naive[h]["identity_share_naive_pct"] for h in hs)
    absorb = [h for h in hs
              if res[str(h)]["entity"]["delta_rel_pct"] > 0
              and res[str(h)]["entity"]["dm"] < 0 and res[str(h)]["entity"]["p"] < 0.05
              and res[str(h)]["entity"]["identity_share_chrono_pct"] >= 50.0]
    checks.append(("R3 entity-mean absorbs the entity effect",
                   share >= 50.0 and len(absorb) > 0,
                   f"naive identity share >= {share:.0f}% at every horizon (>=50 "
                   f"required); chronological absorption (row-4 sig. + chrono share "
                   f">=50%) at h in {absorb or 'NONE'}; "
                   + "; ".join(f"h={h}: row4 {res[str(h)]['entity']['delta_rel_pct']:+.2f}% "
                               f"(p={res[str(h)]['entity']['p']:.4f}), chrono share "
                               f"{res[str(h)]['entity']['identity_share_chrono_pct']:.0f}%"
                               for h in hs)))

    # R4 yardstick = the ORACLE-GIVEN-TEXT benchmark (leaky test-fit projection of
    # the same reference on f_text): the deployed val-fit combiner must recover a
    # substantial fraction of what is actually IN the text forecast. The DGP truth
    # (oracle_injected_dmse) instead measures how much of the injected signal the
    # TF-IDF fixture arm extracted into f_text — reported as a diagnostic.
    ok4, det4 = True, []
    for h in hs:
        r = res[str(h)]
        rr = r["residual"]
        orc_rel = rr["oracle_given_text_rel_pct"]
        frac = rr["recovered_frac_of_text_oracle"]
        extract = (100.0 * rr["oracle_given_text_dmse"] / r["oracle_injected_dmse"]
                   if r["oracle_injected_dmse"] else float("nan"))
        if h == h1:
            ok = (rr["dm"] < 0 and rr["p"] < 0.05
                  and REC_BAND[0] <= frac <= REC_BAND[1])
        else:  # sub-MDE horizon: must not invent a residual that is not there
            ok = abs(rr["delta_rel_pct"]) <= max(REC_NULL_REL, 1.2 * abs(orc_rel))
        ok4 &= ok
        det4.append(f"h={h}: residual {rr['delta_rel_pct']:+.2f}% vs text-oracle "
                    f"{orc_rel:+.2f}% (recovered {100 * frac:.0f}%, p={rr['p']:.4f}; "
                    f"fixture arm extracted {extract:.0f}% of the DGP-injected signal)")
    checks.append((f"R4 residual text effect recovered (h={h1} band "
                   f"{REC_BAND[0]:.2f}-{REC_BAND[1]:.2f}x text-oracle; other horizons "
                   f"no invented residual)", ok4, "; ".join(det4)))

    ok5 = all(abs(res[str(h)]["placebo"][k]["mean_dm"]) < G4_DM_MAX
              and res[str(h)]["placebo"][k]["mean_p"] > 0.05
              for h in hs for k in ("row3_shuffle", "row5_shuffle",
                                    "row3_swap", "row5_swap"))
    checks.append(("R5 placebos kill the increment", ok5,
                   "label-shuffle + within-month swap, both stages, all horizons"))

    ok6, det6 = True, []
    for h in hs:
        inj = res[str(h)]["injection"]
        ad = [x for x in inj["targets"] if x["adaptive"]][0]
        conv = all(x["converged"] for x in inj["targets"])
        ok = (conv and ad["kappa"] > 0 and ad["ar"]["detect"]
              and inj["s_within_entity_max_absmean"] < 1e-9)
        ok6 &= ok
        det6.append(f"h={h}: converged={conv}, adaptive {ad['target_pct']:.2f}% "
                    f"(genuine injection kappa={ad['kappa']:+.4f}) detect "
                    f"AR={ad['ar']['detect']}; entity-stage transmission "
                    f"detect={ad['entity']['detect']} (deployed loading "
                    f"kappa_ent={ad['entity']['kappa']:+.4f}, reported not gated)")
    checks.append(("R6 injection machinery (oracle, disclosed): calibration "
                   "converges, injected signal detected at the AR stage", ok6,
                   "; ".join(det6)))
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol",
                    default=str(REPO / "results/second_domain/protocol_results.json"))
    ap.add_argument("--preds-dir", default=str(REPO / "results/second_domain/preds"))
    ap.add_argument("--panel",
                    default=str(REPO / "results/second_domain/yelp_panel.parquet"))
    ap.add_argument("--out-stem", default=str(REPO / "results/tables/yelp_cascade"))
    ap.add_argument("--tag", default=None,
                    help="banner + filename suffix; defaults to the protocol tag")
    args = ap.parse_args()

    prot = json.loads(Path(args.protocol).read_text())
    res = prot["horizons"]
    hs = sorted(int(k) for k in res)
    preds_dir = Path(args.preds_dir)
    metrics = {m["horizon_months"]: m for m in
               json.loads((preds_dir / "baseline_metrics.json").read_text())["horizons"]}
    naive = naive_entity_arm(args.panel, preds_dir, metrics)

    tag = (args.tag or prot.get("tag") or "REAL").upper()
    synthetic = bool(prot.get("synthetic")) or tag == "SYNTHETIC"
    if synthetic:
        tag = "SYNTHETIC"
    banner = ("SYNTHETIC FIXTURE — MACHINERY VALIDATION ONLY (known injected DGP; "
              "never citable as a Yelp result)" if synthetic
              else "Yelp Open Dataset — business-month rating forecasting")

    rows = build_rows(hs, res, metrics, naive)
    g = gates(hs, res, metrics, naive, args.panel)
    rec = recovery(hs, res, metrics, naive) if synthetic else []

    # ------------------------------------------------------------- markdown table
    md = [f"# Yelp second-domain cascade table [{tag}]\n", f"> {banner}\n"]
    hdr = "| # | arm | reference |"
    sep = "|---|---|---|"
    for h in hs:
        hdr += f" h={h}m MSE | Δ rel% | DM p |"
        sep += "---|---|---|"
    md += [hdr, sep]
    for i in range(1, 6):
        sub = {int(r.h): r for _, r in rows[rows.row == i].iterrows()}
        any_r = next(iter(sub.values()))
        line = f"| {i} | {any_r.arm} | {any_r.reference} |"
        for h in hs:
            r = sub[h]
            line += f" {r.mse:.4f} | {r.delta_rel_pct:+.2f} | {fmt_p(r.dm_p)} |"
        md.append(line)
    notes = [f"row {i} ({h}m): {r.note}" for i in range(1, 6)
             for h, r in {int(x.h): x for _, x in rows[rows.row == i].iterrows()}.items()]
    md.append("\n**Row notes.** " + " · ".join(notes))
    md.append(
        "\n**Table note (pre-registered).** Loss = squared error on stars (MSE); "
        "inference = month-clustered DM (HAC lag = h-1 months, HLN, t(n_months-1)); "
        "robustness = business x month two-way CGM. Combiner weights are validation-"
        "fit and test-frozen; entity means use train+val observed monthly stars only. "
        "MDE (80% power, 5% size, signal-injection methodology with a DISCLOSED "
        "oracle entity-orthogonal injection): "
        + "; ".join(f"h={h}m: AR stage {res[str(h)]['mde']['ar_stage_rel_pct']:.2f}%, "
                    f"entity stage {res[str(h)]['mde']['entity_stage_rel_pct']:.2f}%"
                    for h in hs)
        + ". Boundary: "
        + "; ".join(f"h={h}m {res[str(h)]['boundary_overlap_val_rows']} val rows with "
                    f"outcome windows crossing the test start "
                    f"(embargo={'on' if res[str(h)]['embargo_val'] else 'off'})"
                    for h in hs) + ".")

    md.append("\n## SANITY (gates auto-filled from the numbers)\n")
    md.append("| gate | verdict | detail |\n|---|---|---|")
    for name, ok, det in g:
        md.append(f"| {name} | {'PASS' if ok else '**FAIL**'} | {det} |")

    if synthetic:
        md.append("\n## RECOVERY — machinery validation against the KNOWN injected "
                  "structure\n")
        md.append("| check | verdict | detail |\n|---|---|---|")
        for name, ok, det in rec:
            md.append(f"| {name} | {'PASS' if ok else '**FAIL**'} | {det} |")

    h1 = min(hs)
    r1, n1 = res[str(h1)], naive[h1]
    md.append("\n## HONEST HEADLINE (auto-filled)\n")
    md.append(
        f"Under the field-standard pooled random split the text model appears to cut "
        f"MSE by {n1['gain_text_pct']:.1f}% (row 1, h={h1}m); a zero-text business-"
        f"mean predictor recovers {n1['identity_share_naive_pct']:.0f}% of that "
        f"apparent gain under the same split. Under the chronological protocol, "
        f"text alone loses to the recalibrated AR baseline "
        f"({r1['text_alone']['delta_rel_pct']:+.1f}%, row 2); with the entity-mean "
        f"identity control in the reference, the residual text increment is "
        f"{r1['residual']['delta_rel_pct']:+.2f}% (month-clustered DM "
        f"{r1['residual']['dm']:+.2f}, p={r1['residual']['p']:.4f}; two-way p="
        f"{fmt_p(r1['residual']['p_2way'])}; label-shuffle placebo mean DM "
        f"{r1['placebo']['row5_shuffle']['mean_dm']:+.2f}, mean p="
        f"{r1['placebo']['row5_shuffle']['mean_p']:.3f}; MDE "
        f"{max(r1['mde']['ar_stage_rel_pct'], r1['mde']['entity_stage_rel_pct']):.2f}% "
        f"at 80% power).")

    # ------------------------------------------------------------------- outputs
    suffix = "" if tag == "REAL" else f"_{tag.lower()}"
    stem = Path(f"{args.out_stem}{suffix}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    rows.insert(0, "tag", tag)
    rows.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text("\n".join(md))

    # ------------------------------------------------------------ console output
    print("\n" + "=" * 78)
    print(f"YELP SECOND-DOMAIN CASCADE TABLE  [{tag}]")
    print(banner)
    print("=" * 78)
    show = rows[["row", "arm", "h", "mse", "delta_rel_pct", "dm_p"]].copy()
    show["dm_p"] = show.dm_p.map(fmt_p)
    print(show.to_string(index=False,
                         formatters={"mse": "{:.4f}".format,
                                     "delta_rel_pct": "{:+.2f}".format}))
    print("\nSANITY gates:")
    for name, ok, det in g:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {det}")
    hard_fail = [name for name, ok, _ in g if not ok]
    if synthetic:
        print("\nRECOVERY (machinery validation, SYNTHETIC):")
        for name, ok, det in rec:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {det}")
        hard_fail += [name for name, ok, _ in rec if not ok]
        verdict = "PASS" if not hard_fail else "FAIL"
        print(f"\nMACHINERY VALIDATION: {verdict}")
    print(f"\nwrote {stem.with_suffix('.csv')} and {stem.with_suffix('.md')}")
    if hard_fail:
        print(f"FAILED: {', '.join(hard_fail)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
