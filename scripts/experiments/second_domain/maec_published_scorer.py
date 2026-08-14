#!/usr/bin/env python
"""MAEC audit — published-convention arm reading + sanity gates G1/G2
(prereg configs/prereg_maec_audit.md §4 v1.1 + §7, tag prereg-maec-v1.0).

SCORING STAGE, SINGLE-SHOT (§6.5 discipline): this script computes the
published-convention readouts on REAL test rows and therefore runs ONCE, before any
maec_protocol.py arm run (G1 is the audit's precondition gate). It refuses to
overwrite an existing output; reruns are bug-fixes only and require
--force-rerun --reason '...' (the reason is logged into the output JSON and
must be echoed in the prereg revision record).

WHAT IT READS (per §4 v1.1): the published convention is the MAEC (CIKM 2020)
Table 5 year-panel split (2015 / 2016 / 2017-18, chronological 7:1:2 pinned
inside each panel), alignment day-1-start unadjusted (= the primary labels).
Readings are the PUBLISHED style — MSE(v) on each year panel's TEST rows,
NO recalibration, NO combination, NO clustered inference (these readings are
descriptive; they never enter Holm or "win" prose, §6.2):
  (a) raw V_past^(n) as the forecast (the baseline Yu et al. discarded at
      MSE 1.12) — column v_past_match of the published preds file, clipped to
      the §5 v-range only (a no-op for build-clipped features);
  (b) each text arm STANDALONE — column `prediction` of
      preds_<arm>_published.parquet (now: tfidf; later: prompted/qwen_emb from
      analogous files, same schema, same Table-5 row assignment).

GATES (§7):
  G1 (sign replication, audit precondition): at least one text arm's
     standalone MSE(v) beats raw V_past^(n) SOMEWHERE — operationalised as
     >= 1 (year-panel x horizon) test cell with mse_text < mse_raw (the pooled
     comparison is also reported). If ALL §5 text arms fail G1 the audit
     downgrades to the §8 branch-D baseline-recalibration-only scope — no
     mid-course arm swaps or prompt retries.
     NOTE: a FAIL verdict is FINAL only when every §5 headline text arm has a
     published reading in the same run; a partial-arm FAIL is provisional.
  G2 (magnitude gate): our raw V_past MSE(v), POOLED over all published test
     rows (all 3 year panels x 4 horizons, row-equal-weight — aggregation
     choice disclosed; per-cell and per-horizon ratios reported alongside),
     must be within ratio [1/3, 3] of Yu et al.'s reported 1.12 (panel differs
     — theirs MAEC-15/16 subset, ours the full release — hence a magnitude
     gate, not an equality gate). FAIL -> audit the §2 label construction
     first; no tables may be emitted (SS7: no table may be emitted while a gate is failing).

Usage (from repo root; the ONLY mode executed at build time is --selftest):
    .venv/bin/python scripts/experiments/second_domain/maec_published_scorer.py \
        [--preds-dir results/second_domain/maec/preds] [--arms tfidf] \
        [--out results/second_domain/maec/published_readings.json] \
        [--force-rerun --reason '...']  |  --selftest
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
CLIP_LO, CLIP_HI = float(np.log(1e-4)), 0.0          # v-units (§5)
KEY = ["permno", "call_date", "horizon"]
HORIZONS = (3, 7, 15, 30)
YEAR_PANELS = ("2015", "2016", "2017-18")             # §4 v1.1 Table-5 panels
YU_RAW_MSE = 1.12                                     # Yu et al. raw V_past scale
G2_BAND = (1.0 / 3.0, 3.0)                            # §7 magnitude band
NEEDED = ["year_panel", "split", "label", "v_past_match", "prediction", *KEY]
DEFAULT_OUT = REPO / "results/second_domain/maec/published_readings.json"


def mse(y, f) -> float:
    return float(np.mean((np.asarray(y, float) - np.asarray(f, float)) ** 2))


def clip_v(f):
    return np.clip(np.asarray(f, float), CLIP_LO, CLIP_HI)


def single_shot_guard(out_path: Path, force: bool, reason: str | None) -> None:
    """§6.5 single-shot discipline (disclosure override = --force-rerun --reason)."""
    if out_path.exists():
        if not (force and reason):
            sys.exit(f"REFUSED (§6.5 single-shot): {out_path} exists. Reruns are "
                     f"bug-fixes only — pass --force-rerun --reason '...' and log "
                     f"the diff in the prereg revision record.")
        print(f"[§6.5] force-rerun of {out_path}; reason: {reason}")


def load_arm(preds_dir: Path, arm: str) -> pd.DataFrame:
    fp = preds_dir / f"preds_{arm}_published.parquet"
    assert fp.exists(), f"missing published preds file for arm {arm!r}: {fp}"
    df = pd.read_parquet(fp)
    missing = set(NEEDED) - set(df.columns)
    assert not missing, f"{fp.name} missing columns {sorted(missing)}"
    assert not df.duplicated(KEY).any(), f"{fp.name}: duplicate keys"
    return df


def published_readings(frames: dict[str, pd.DataFrame]) -> dict:
    """§4 published-convention readings on TEST rows + §7 G1/G2 verdicts.
    `frames`: arm name -> published preds frame (train/val rows are ignored)."""
    arms = list(frames)
    assert arms, "no arms given"
    tests = {}
    for a, f in frames.items():
        t = f[f["split"] == "test"].copy()
        assert len(t) > 0, f"arm {a}: no test rows"
        assert t["year_panel"].isin(YEAR_PANELS).all(), \
            f"arm {a}: year_panel outside the Table-5 panels"
        assert np.isfinite(t[["label", "v_past_match", "prediction"]]
                           .to_numpy(float)).all(), f"arm {a}: non-finite values"
        tests[a] = t

    # G5 spirit (§3.4): identical test row-set AND identical raw input across arms
    base = arms[0]
    key0 = tests[base][KEY].sort_values(KEY).reset_index(drop=True)
    for a in arms[1:]:
        ka = tests[a][KEY].sort_values(KEY).reset_index(drop=True)
        assert ka.equals(key0), \
            f"G5 FAIL: published test row-set of arm {a!r} != {base!r}"
        m = tests[base][KEY + ["v_past_match"]].merge(
            tests[a][KEY + ["v_past_match"]], on=KEY, validate="1:1")
        assert np.allclose(m["v_past_match_x"], m["v_past_match_y"]), \
            f"G5 FAIL: v_past_match differs between arms {base!r} and {a!r}"

    # ---- raw V_past^(n) readings (no recalibration), from the base frame ----
    d0 = tests[base]
    raw_cells = {}
    for (yp, h), g in d0.groupby(["year_panel", "horizon"]):
        raw_cells[f"{yp}_h{int(h)}"] = {
            "n_test": len(g),
            "mse_raw_vpast": mse(g["label"], clip_v(g["v_past_match"]))}
    raw_pooled = mse(d0["label"], clip_v(d0["v_past_match"]))
    raw_per_h = {str(int(h)): mse(g["label"], clip_v(g["v_past_match"]))
                 for h, g in d0.groupby("horizon")}

    # ---- text arms standalone, per (year panel x horizon) cell ----
    arm_out = {}
    for a in arms:
        cells = {}
        for (yp, h), g in tests[a].groupby(["year_panel", "horizon"]):
            ck = f"{yp}_h{int(h)}"
            m = mse(g["label"], g["prediction"])
            cells[ck] = {"mse_text_standalone": m,
                         "mse_raw_vpast": raw_cells[ck]["mse_raw_vpast"],
                         "n_test": raw_cells[ck]["n_test"],
                         "text_beats_raw": bool(m < raw_cells[ck]["mse_raw_vpast"])}
        pooled = mse(tests[a]["label"], tests[a]["prediction"])
        n_beat = sum(c["text_beats_raw"] for c in cells.values())
        arm_out[a] = {"cells": cells, "pooled_mse_text": pooled,
                      "pooled_beats_raw": bool(pooled < raw_pooled),
                      "n_cells_beating_raw": int(n_beat),
                      "n_cells": len(cells)}

    # ---- G1 (§7): >= 1 text arm beats raw V_past somewhere (per-cell) ----
    g1_pass = any(v["n_cells_beating_raw"] >= 1 for v in arm_out.values())
    g1 = {"pass": bool(g1_pass),
          "definition": ("PASS iff >=1 text arm has mse_text < mse_raw_vpast in "
                         ">=1 (year_panel x horizon) TEST cell; pooled comparison "
                         "reported alongside (descriptive)"),
          "arms_scored": arms,
          "per_arm_cells_beating_raw": {a: v["n_cells_beating_raw"]
                                        for a, v in arm_out.items()},
          "per_arm_pooled_beats_raw": {a: v["pooled_beats_raw"]
                                       for a, v in arm_out.items()},
          "finality_note": ("a FAIL is FINAL only if every §5 headline text arm "
                            "(tfidf, prompted_qwen, qwen_emb) is in arms_scored; "
                            "otherwise provisional")}

    # ---- G2 (§7): pooled raw V_past MSE(v) within [1/3, 3] of Yu's 1.12 ----
    ratio = raw_pooled / YU_RAW_MSE
    cell_ratios = {k: v["mse_raw_vpast"] / YU_RAW_MSE for k, v in raw_cells.items()}
    g2 = {"pass": bool(G2_BAND[0] <= ratio <= G2_BAND[1]),
          "our_raw_vpast_mse_pooled": raw_pooled,
          "yu_reference_mse": YU_RAW_MSE,
          "ratio_pooled": ratio, "band": list(G2_BAND),
          "aggregation": ("pooled over ALL published test rows (3 year panels x "
                          "4 horizons, row-equal-weight) — disclosed choice; "
                          "per-horizon and per-cell ratios alongside"),
          "ratio_per_horizon": {h: m / YU_RAW_MSE for h, m in raw_per_h.items()},
          "ratio_per_cell_min": float(min(cell_ratios.values())),
          "ratio_per_cell_max": float(max(cell_ratios.values()))}

    return {"raw_vpast": {"cells": raw_cells, "pooled_mse": raw_pooled,
                          "per_horizon_pooled_mse": raw_per_h},
            "arms": arm_out, "G1": g1, "G2": g2}


def print_verdicts(res: dict) -> None:
    print("\n--- published-convention readings (MSE(v), test rows, descriptive) ---")
    for ck in sorted(res["raw_vpast"]["cells"]):
        c = res["raw_vpast"]["cells"][ck]
        line = f"{ck:>12}: raw_vpast {c['mse_raw_vpast']:.4f} (n={c['n_test']})"
        for a, v in res["arms"].items():
            t = v["cells"][ck]
            line += (f"  |  {a} {t['mse_text_standalone']:.4f}"
                     f"{' <raw' if t['text_beats_raw'] else ''}")
        print(line)
    print(f"{'pooled':>12}: raw_vpast {res['raw_vpast']['pooled_mse']:.4f}"
          + "".join(f"  |  {a} {v['pooled_mse_text']:.4f}"
                    for a, v in res["arms"].items()))

    g1, g2 = res["G1"], res["G2"]
    if g1["pass"]:
        print(f"\nG1 (§7) PASS — text beats raw V_past somewhere: "
              f"cells beating raw per arm = {g1['per_arm_cells_beating_raw']}")
    else:
        print(f"\nG1 (§7) FAIL — NO text arm beats raw V_past in any "
              f"(year-panel x horizon) cell among arms scored "
              f"{g1['arms_scored']}. If this is the FULL §5 text-arm set, the "
              f"audit downgrades to §8 branch D (baseline-recalibration-only); "
              f"do NOT swap arms or retry prompts (§7). Otherwise: provisional "
              f"until all arms have published readings.")
    verdict = "PASS" if g2["pass"] else "FAIL"
    print(f"G2 (§7) {verdict} — our pooled raw V_past MSE(v) = "
          f"{g2['our_raw_vpast_mse_pooled']:.4f} vs Yu et al. {YU_RAW_MSE}; "
          f"ratio {g2['ratio_pooled']:.3f} "
          f"{'in' if g2['pass'] else 'OUTSIDE'} [{G2_BAND[0]:.3f}, {G2_BAND[1]}]"
          + ("" if g2["pass"] else
             " -> audit the §2 label construction FIRST; no tables (§7)."))


# ------------------------------------------------------------------- selftest
def _synthetic_arm(rng, raw_mse_target: float, text_noise: float,
                   n_cell: int = 60) -> pd.DataFrame:
    """One synthetic published-preds frame. raw forecast = label + bias + eps
    with bias^2 + 0.01 ~= raw_mse_target; text pred = label + text_noise*z.
    Train/val rows carry ABSURD predictions to prove the test-only filter."""
    bias = float(np.sqrt(max(raw_mse_target - 0.01, 1e-6)))
    rows = []
    pid = 10_000
    for yp in YEAR_PANELS:
        y0 = int(yp[:4])
        for h in HORIZONS:
            for split, n in (("train", 10), ("val", 5), ("test", n_cell)):
                lab = -4.0 + 0.5 * rng.standard_normal(n)
                vp = lab + bias + 0.1 * rng.standard_normal(n)
                pred = (lab + text_noise * rng.standard_normal(n)
                        if split == "test" else np.full(n, 99.0))
                for i in range(n):
                    rows.append({
                        "permno": pid + i, "call_date": pd.Timestamp(f"{y0}-06-01")
                        + pd.Timedelta(days=i), "horizon": h, "year_panel": yp,
                        "split": split, "label": lab[i], "v_past_match": vp[i],
                        "prediction": pred[i]})
        pid += 1000
    return pd.DataFrame(rows)


def selftest() -> None:
    print("=== maec_published_scorer --selftest (synthetic, in-memory) ===")
    t0 = time.time()
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

    # case A: raw at the Yu scale, text clearly better -> G1 PASS, G2 PASS
    rng = np.random.default_rng(2026)
    fa = _synthetic_arm(rng, raw_mse_target=YU_RAW_MSE, text_noise=0.2)
    ra = published_readings({"syn_text": fa})
    check("case A: G1 PASS (text beats raw somewhere)", ra["G1"]["pass"],
          f"cells={ra['G1']['per_arm_cells_beating_raw']}")
    check("case A: G2 PASS (raw at Yu scale)", ra["G2"]["pass"],
          f"ratio={ra['G2']['ratio_pooled']:.3f}")
    check("case A: all 12 cells beat raw",
          ra["arms"]["syn_text"]["n_cells_beating_raw"] == 12)
    check("case A: train/val rows excluded (absurd preds not scored)",
          ra["arms"]["syn_text"]["pooled_mse_text"] < 1.0,
          f"pooled={ra['arms']['syn_text']['pooled_mse_text']:.3f}")

    # case B: raw far below the Yu scale, text pure noise -> G1 FAIL, G2 FAIL
    rng = np.random.default_rng(7)
    fb = _synthetic_arm(rng, raw_mse_target=0.02, text_noise=3.0)
    rb = published_readings({"syn_text": fb})
    check("case B: G1 FAIL (no cell beats raw)", not rb["G1"]["pass"],
          f"cells={rb['G1']['per_arm_cells_beating_raw']}")
    check("case B: G2 FAIL (ratio below 1/3)", not rb["G2"]["pass"],
          f"ratio={rb['G2']['ratio_pooled']:.3f}")

    # arm row-set mismatch must hard-fail (G5 spirit, §3.4)
    fb2 = fb[fb["permno"] != int(fb.loc[fb.split == "test", "permno"].iloc[0])]
    try:
        published_readings({"a1": fb, "a2": fb2})
        check("row-set mismatch across arms raises", False)
    except AssertionError as e:
        check("row-set mismatch across arms raises", "G5 FAIL" in str(e))

    # §6.5 single-shot guard: refuse on existing output unless force+reason
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"_pubscorer_guard_{os.getpid()}.json"
    tmp.write_text("{}")
    try:
        try:
            single_shot_guard(tmp, force=False, reason=None)
            check("single-shot guard refuses existing output", False)
        except SystemExit:
            check("single-shot guard refuses existing output", True)
        single_shot_guard(tmp, force=True, reason="selftest")   # must not exit
        check("guard override with --force-rerun --reason passes", True)
    finally:
        tmp.unlink(missing_ok=True)

    n_pass = sum(ok for _, ok in checks)
    print(f"\nselftest: {n_pass}/{len(checks)} checks passed in {time.time()-t0:.1f}s")
    if n_pass != len(checks):
        sys.exit(1)
    print("SELFTEST PASS")


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir",
                    default=str(REPO / "results/second_domain/maec/preds"))
    ap.add_argument("--arms", default="tfidf",
                    help="comma list; each needs preds_<arm>_published.parquet")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--tag", default="REAL")
    ap.add_argument("--force-rerun", action="store_true",
                    help="§6.5 single-shot override; requires --reason")
    ap.add_argument("--reason", default=None,
                    help="revision-log reason for a --force-rerun")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic self-test exercising both gate outcomes")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    out_path = Path(args.out)
    single_shot_guard(out_path, args.force_rerun, args.reason)

    t0 = time.time()
    preds_dir = Path(args.preds_dir)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    frames = {a: load_arm(preds_dir, a) for a in arms}
    res = published_readings(frames)

    out = {"tag": args.tag, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "prereg": "configs/prereg_maec_audit.md §4 v1.1 + §7 (G1/G2)",
           "preds_dir": str(preds_dir), "arms": arms,
           "force_rerun_reason": args.reason,
           "inference_note": ("published-convention readings are DESCRIPTIVE "
                              "(§6.2): no clustering, no Holm, never 'win' prose"),
           **res}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    print(f"=== [{args.tag}] maec_published_scorer — arms {arms}, "
          f"done in {time.time()-t0:.1f}s ===")
    print_verdicts(res)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
