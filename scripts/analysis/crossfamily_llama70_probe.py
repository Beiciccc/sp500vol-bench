"""Prereg B2 rider (configs/prereg_residual_family_audit.md v1.3, tag prereg-rfa-v1.3,
"Rider: 70B zero-content probe") — the zero-content date+ticker probe through Llama-3.1-70B.

Quant reviewer's verbatim ask: "date+ticker term inside its reference, reconciling the
Table 6 149/103% probe cell with the replication claim". This script runs the C6
contamination arm's date+ticker ZERO-CONTENT prompt — template VERBATIM from
scripts/experiments/e1_llm_forecast/prompt.py, variant `c6_datefirm` (form + items,
filing date, ticker, "(No filing text is provided.)"; same fields, no document text;
that template is the one llm_contamination.py's committed C6_datefirm runs used) —
through hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 (int4: the SAME precision
as the committed llama70 fulltext runs, disclosed — internal consistency of the
comparison), single seed 2026, TP=2, on the SAME event_driven panel as the committed
crossfamily_llama70 runs (the 39,322-filing ED manifest, val+test).

READOUTS — DESCRIPTIVE ONLY (prereg v1.3: "descriptive, no branches"; no prereg branch fires, no
Holm family beyond what the template does — Holm only WITHIN each block, exactly as
llm_contamination.py does):
  1. probe M1 rel% vs the single recalibrated HAR and vs the HAR+firm-identity
     reference per horizon (the verbatim M1 block of crossfamily_llama70.py:
     log-space combiner val-fit test-frozen, day-clustered DM, HAC lag h-1, HLN);
  2. side-by-side with the COMMITTED llama70-ens fulltext rel% (anchor cells read from
     results/tables/crossfamily_llama70_ens.csv; the single-seed llama70 rows are
     carried as robustness denominators);
  3. probe-share = probe_rel% / fulltext_rel% per horizon and reference, with the
     >100% convention documented: a share above 100% means the zero-content probe
     ALONE reproduces more than the fulltext increment in that cell (the Table 6
     149/103% convention); shares are only well-identified where the fulltext
     denominator is >= 1% and raw-significant (llm_contamination.md's
     stable-denominator rule) — flagged per cell;
  4. text-beyond-identity: joint reference [1, log fHAR, log f_probe] (val-fit,
     test-frozen); the unrestricted model adds log f_fulltext(llama70-ens) — whether
     the committed fulltext increment survives with the SAME-model date+ticker
     forecast inside the reference (the llm_contamination.py joint block, with
     f_datefirm := this probe and f_fulltext := the committed llama70 ensemble).

SANITY GATES (HARD RULE — abort before writing tables):
  GP1 (the G1'' convention of crossfamily_mistral24.py): the committed llama70
      single-seed AND ens3 M1 rows, recomputed on this exact code path, must
      reproduce results/tables/crossfamily_llama70_ens.csv to machine precision
      (rtol 1e-12) — the anchor cells are verified, then carried unchanged
      (prereg v1.3: committed readings are not re-derived, only anchored);
  GP2 the probe M1 cells must sit on the identical test panel as the committed
      llama70 cells (n_test equal per horizon) — a mismatch means panel drift.

OUTPUT (WRITE-ONCE, single shot): results/tables/crossfamily_llama70_probe.{csv,md}.

BOX ENV (2xA100-40G; auto-defaulted when /root/gpu-data exists):
  export SP500VOL_DATA_ROOT=/root/gpu-data/sp500vol-data
  export HF_HOME=/root/gpu-data/hf
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # box has no HF egress: cache only
  VLLM_WORKER_MULTIPROC_METHOD=spawn is set in-script (fork deadlocks with TP2).

RUN (on the box, from the repo root /root/gpu-data/repo):
  /root/gpu-data/venvs/main/bin/python scripts/analysis/crossfamily_llama70_probe.py
LOCAL (no GPU):
  .venv/bin/python scripts/analysis/crossfamily_llama70_probe.py --dry-run
  .venv/bin/python scripts/analysis/crossfamily_llama70_probe.py --selftest
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # fork deadlocks w/ TP2
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
if os.path.isdir("/root/gpu-data"):  # box defaults (launch.sh convention)
    os.environ.setdefault("SP500VOL_DATA_ROOT", "/root/gpu-data/sp500vol-data")
    os.environ.setdefault("HF_HOME", "/root/gpu-data/hf")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
sys.path.insert(0, str(_REPO / "scripts" / "experiments" / "e1_llm_forecast"))
import forecast_combination as fc  # noqa: E402
import clustered_dm as cdm  # noqa: E402
import prompt as prompt_mod  # noqa: E402
import run_inference as ri  # noqa: E402
import postprocess as pp  # noqa: E402

KEY = ["ticker", "accession", "horizon_days"]
EPS = 1e-8
HORIZONS = (5, 10, 20)
RTOL = 1e-12    # machine-precision anchor gate (GP1, the G1'' convention)
DISC = "event_driven"
VARIANT = "c6_datefirm"      # the C6 contamination arm's zero-content prompt, VERBATIM
SEED = 2026                  # single seed (temperature-0 protocol; not a sampler seed)
# committed B1/C6 sampling stack — FIXED, not CLI-exposed (protocol must not drift):
TP = 2
MAX_MODEL_LEN = 8192
MAX_TOKENS = 120
CHECKPOINT_EVERY = 500

DEFAULT_MODEL_ID = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
MANIFEST = "results/e1_llm_forecast/manifest_valtest.parquet"
RAW_DIR = Path("results/e1_llm_forecast/raw_datefirm_llama70")
PROBE_RUN = "C6_datefirm_llama70_full_event_driven_seed2026"
PROBE_FAM = "llama70_datefirm_probe"
LLA_RUNS = {  # committed llama70 anchors, recomputed for GP1
    "llama70_awq": "C6_llmtext_llama70_full_event_driven_seed2026",
    "llama70_awq_ens3": "C6_llmtext_llama70ens_full_event_driven_seed2026",
}
ENS_FAM, SINGLE_FAM = "llama70_awq_ens3", "llama70_awq"
CSV_OUT = "results/tables/crossfamily_llama70_probe.csv"
MD_OUT = "results/tables/crossfamily_llama70_probe.md"

M1_COLS = ["n_test", "n_days", "rel_har", "dm_har", "p_har",
           "rel_firm", "dm_firm", "p_firm", "g_text"]


def assert_writeonce(*paths):
    hit = [p for p in paths if Path(p).exists()]
    if hit:
        print(f"WRITE-ONCE guard (prereg v1.3 'single shot'): output(s) already exist — "
              f"refusing to overwrite: {hit}\n"
              "If a rerun is genuinely intended, inspect and move the existing file(s) "
              "manually first; this script never overwrites its registered outputs.")
        sys.exit(3)


# --------------------------------------------------- verbatim B1 statistics block
def ols(y, X):  # verbatim from crossfamily_llama70.py
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def holm(ps):  # verbatim from crossfamily_llama70.py
    ps = np.asarray(ps, float)
    n = len(ps)
    order = np.argsort(ps)
    out = np.empty(n)
    for rank, idx in enumerate(order):
        out[idx] = ps[idx] * (n - rank)
    run = 0.0
    for idx in order:
        run = max(run, out[idx])
        out[idx] = min(run, 1.0)
    return out


def close(a, b):  # verbatim from crossfamily_llama70.py
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) <= RTOL * max(abs(a), abs(b), 1.0)


def L(x):
    return np.log(np.clip(np.asarray(x, float), EPS, None))


def m1_rows(fam, preds, a2):
    """M1 block — verbatim from crossfamily_llama70.py / crossfamily_mistral24.py."""
    t = preds[KEY + ["prediction_realised_vol"]].rename(
        columns={"prediction_realised_vol": "ft"})
    rows = []
    for h in HORIZONS:
        m = a2[a2.horizon_days == h].merge(t[t.horizon_days == h], on=KEY).dropna()
        v, te = m[m.split == "val"], m[m.split == "test"]
        y = te.label_realised_vol.values
        fR, fU, g = fc.log_combo(v.label_realised_vol.values, v.fh.values,
                                 v.ft.values, te.fh.values, te.ft.values)
        qR, qU = fc.qlike(y, fR), fc.qlike(y, fU)
        rel = 100 * np.mean(qR - qU) / np.mean(qR)
        dm, pv, nd = cdm.dm_test_clustered(qU, qR, te.effective_trading_day.values, h)
        # firm-identity-augmented reference (val-window firm mean spec)
        fm = v.groupby("ticker").label_realised_vol.mean()
        gmean = v.label_realised_vol.mean()
        fid_v = v.ticker.map(fm).fillna(gmean).values
        fid_t = te.ticker.map(fm).fillna(gmean).values
        ly = L(v.label_realised_vol.values)
        bR = ols(ly, np.column_stack([np.ones(len(v)), L(v.fh.values), L(fid_v)]))
        bU = ols(ly, np.column_stack([np.ones(len(v)), L(v.fh.values), L(fid_v),
                                      L(v.ft.values)]))
        fRf = np.exp(bR[0] + bR[1] * L(te.fh.values) + bR[2] * L(fid_t))
        fUf = np.exp(bU[0] + bU[1] * L(te.fh.values) + bU[2] * L(fid_t)
                     + bU[3] * L(te.ft.values))
        qRf, qUf = fc.qlike(y, fRf), fc.qlike(y, fUf)
        relf = 100 * np.mean(qRf - qUf) / np.mean(qRf)
        dmf, pf, _ = cdm.dm_test_clustered(qUf, qRf, te.effective_trading_day.values, h)
        rows.append(dict(disc=DISC, family=fam, h=h, n_test=len(te), n_days=nd,
                         rel_har=rel, dm_har=dm, p_har=pv,
                         rel_firm=relf, dm_firm=dmf, p_firm=pf, g_text=g))
    return rows


def probe_share(probe_rel, fulltext_rel):
    """probe-share = probe_rel% / fulltext_rel%, in percent. NaN when the fulltext
    denominator is ~0 (unidentified). Values may exceed 100% (Table 6 convention:
    the zero-content probe alone reproduces MORE than the fulltext increment)."""
    if not np.isfinite(fulltext_rel) or abs(fulltext_rel) < 1e-12:
        return float("nan")
    return float(100.0 * probe_rel / fulltext_rel)


# ------------------------------------------------------------ inference (no text)
def resolve_model(model_id):
    """Same resolution ladder as crossfamily_gemma27.py: local dir > HF cache
    snapshot under $HF_HOME/hub > raw hub id (only usable online)."""
    p = Path(model_id)
    if p.is_dir():
        return str(p), "local-dir"
    hub = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"
    cache = hub / ("models--" + model_id.replace("/", "--")) / "snapshots"
    if cache.is_dir():
        snaps = sorted([d for d in cache.iterdir()
                        if d.is_dir() and (d / "config.json").exists()])
        if snaps:
            return str(snaps[-1]), "hf-cache-snapshot"
    return model_id, "hub-id-unresolved"


def load_panel():
    """The committed llama70 ED panel: event_driven manifest rows, val+test."""
    m = pd.read_parquet(MANIFEST)
    m = m[m.disclosure == DISC]
    return m.sort_values(["filing_time_utc", "ticker", "accession"],
                         kind="mergesort").reset_index(drop=True)


def do_infer(panel, out_dir, gen, checkpoint_every=CHECKPOINT_EVERY):
    """Zero-content probe inference: build_messages(variant=c6_datefirm) VERBATIM —
    no document text is read at all. Resumable via part-*.parquet; retry pass and
    output schema come from the committed run_inference._flush."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = ri.load_done(out_dir)
    pending = [r for r in panel.to_dict("records")
               if (r["text_path"], VARIANT) not in done]
    print(f"[infer] {out_dir}: {len(panel)} filings, {len(pending)} pending "
          f"({len(done)} resumed)")
    if not pending:
        return
    part_idx = len(list(out_dir.glob("part-*.parquet")))
    chunk, n_done = [], 0
    for row in pending:
        chunk.append({"row": row, "variant": VARIANT, "excerpt_source": "none",
                      "messages": prompt_mod.build_messages(row, "", VARIANT)})
        if len(chunk) >= checkpoint_every:
            part_idx = ri._flush(gen, chunk, out_dir, part_idx)
            n_done += len(chunk)
            print(f"  [infer] {n_done}/{len(pending)} done")
            chunk = []
    if chunk:
        ri._flush(gen, chunk, out_dir, part_idx)
    print(f"[infer] {out_dir}: complete")


def build_run_dir(raw_dir, run_id, llm_note_extra=None):
    """Targeted event_driven postprocess for the c6_datefirm raw outputs — the exact
    build-runs logic of postprocess.py (fill rv22, clip [0.03,3.0], stats, metrics),
    restricted to the ED disclosure."""
    raw = pp.load_raw(str(raw_dir))
    rv = raw[raw.variant == VARIANT]
    a2 = pd.read_parquet(f"results/runs/A2_har_rv_full_{DISC}_seed2026/predictions.parquet")
    base = a2[a2["split"].isin(["val", "test"])].copy()
    base = base.merge(rv[["text_path", "vol_5d", "vol_10d", "vol_20d", "parse_ok"]],
                      on="text_path", how="inner")
    volmap = {5: "vol_5d", 10: "vol_10d", 20: "vol_20d"}
    pred = np.full(len(base), np.nan)
    for h, col in volmap.items():
        ix = base["horizon_days"] == h
        pred[ix.to_numpy()] = base.loc[ix, col].to_numpy()
    n_all, n_miss = len(base), int(np.isnan(pred).sum())
    pred = np.where(np.isnan(pred), base["feature_rv_22d"].to_numpy(), pred)  # rv22
    n_clip = int(((pred < pp.CLIP_LO) | (pred > pp.CLIP_HI)).sum())
    base["prediction_realised_vol"] = np.clip(pred, pp.CLIP_LO, pp.CLIP_HI)
    base["run_id"], base["model_id"] = run_id, "C6_datefirm_llama70"
    base["dataset"], base["seed"] = "full", 2026
    base["disclosure_subset"] = DISC
    out = base[pp.PRED_COLS].reset_index(drop=True)
    rd = Path(f"results/runs/{run_id}")
    rd.mkdir(parents=True, exist_ok=True)
    out.to_parquet(rd / "predictions.parquet", index=False)
    (rd / "metrics.json").write_text(json.dumps(pp.metrics_rows(out, DISC), indent=2))
    (rd / "config.json").write_text(json.dumps({
        "model_id": "C6_datefirm_llama70",
        "note": ("Prereg B2 rider (prereg-rfa v1.3): zero-content date+ticker probe "
                 "through Llama-3.1-70B AWQ-INT4 — the C6 contamination arm's "
                 "c6_datefirm prompt VERBATIM (no filing text), same precision as "
                 "the committed llama70 fulltext runs, single seed 2026."
                 + (" " + llm_note_extra if llm_note_extra else "")),
        "variant": VARIANT,
        "llm": str(rv["model_name"].iloc[0]),
        "clip_range": [pp.CLIP_LO, pp.CLIP_HI],
        "on_missing": "rv22",
        "stats": {
            "n_rows": int(len(out)),
            "n_filings": int(out["text_path"].nunique()),
            "parse_fail_rows": n_miss,
            "parse_fail_rate": round(n_miss / n_all, 4) if n_all else float("nan"),
            "clipped_rows": n_clip,
            "clipped_rate": round(n_clip / max(n_all - n_miss, 1), 4),
        },
        "sampling": {"temperature": 0.0, "retry_temperature": 0.2,
                     "max_tokens": MAX_TOKENS, "max_model_len": MAX_MODEL_LEN,
                     "tp": TP, "guided_json": True, "seed": SEED},
    }, indent=2))
    print(f"[build-run] wrote {rd}  rows={len(out)} "
          f"parse_fail={n_miss} clip_rate={n_clip / max(n_all - n_miss, 1):.4f}")
    return out


# ---------------------------------------------------------------------- scoring
def score(model_id):
    assert_writeonce(CSV_OUT, MD_OUT)
    a2 = fc.load("A2_har_rv", DISC)[KEY + ["split", "label_realised_vol",
                                           "prediction_realised_vol",
                                           "effective_trading_day"]] \
        .rename(columns={"prediction_realised_vol": "fh"})

    # ---- GP1 (G1'' convention): committed llama70 anchors reproduce exactly ----
    ref_l = pd.read_csv("results/tables/crossfamily_llama70_ens.csv")
    preds_l = {}
    gp1_bad = []
    for fam, run in LLA_RUNS.items():
        preds_l[fam] = pd.read_parquet(f"results/runs/{run}/predictions.parquet")
        mine_all = pd.DataFrame(m1_rows(fam, preds_l[fam], a2))
        for _, r in ref_l[ref_l.family == fam].iterrows():
            mine = mine_all[(mine_all.disc == r.disc) & (mine_all.h == r.h)]
            if len(mine) != 1:
                gp1_bad.append((r.disc, fam, int(r.h), "row missing"))
                continue
            for c in M1_COLS:
                if not close(mine[c].iloc[0], r[c]):
                    gp1_bad.append((r.disc, fam, int(r.h), c,
                                    float(mine[c].iloc[0]), float(r[c])))
    if gp1_bad:
        print(f"SANITY GP1 FAIL ({len(gp1_bad)} mismatches vs committed "
              f"crossfamily_llama70_ens.csv):")
        for b in gp1_bad[:20]:
            print("  ", b)
        sys.exit(1)
    print(f"SANITY GP1 PASS: all {len(ref_l)} committed llama70 rows (single-seed AND "
          f"ens3) reproduced to machine precision (rtol {RTOL:g}) on columns "
          f"{M1_COLS} — anchors verified, carried unchanged")

    # ---- probe M1 rows + within-block Holm(6) (template convention) ----
    probe = pd.read_parquet(f"results/runs/{PROBE_RUN}/predictions.parquet")
    pr = pd.DataFrame(m1_rows(PROBE_FAM, probe, a2)).sort_values("h") \
        .reset_index(drop=True)
    ps6 = holm(np.concatenate([pr.p_har.values, pr.p_firm.values]))
    pr["p_har_holm"], pr["p_firm_holm"] = ps6[:len(pr)], ps6[len(pr):]
    pr["holm_family"] = ("probe block Holm(6): 3 horizons x {HAR, HAR+firmID} — "
                         "within-block only (llm_contamination.py convention); "
                         "DESCRIPTIVE, no prereg branch")

    ens_ref = ref_l[ref_l.family == ENS_FAM].sort_values("h").reset_index(drop=True)
    sin_ref = ref_l[ref_l.family == SINGLE_FAM].sort_values("h").reset_index(drop=True)

    # ---- GP2: identical test panel (n_test per horizon must match committed) ----
    gp2_bad = [(int(h), int(a), int(b)) for h, a, b in
               zip(pr.h, pr.n_test, ens_ref.n_test) if int(a) != int(b)]
    if gp2_bad:
        print("SANITY GP2 FAIL: probe test panel differs from the committed llama70 "
              f"panel (h, n_probe, n_committed): {gp2_bad}")
        sys.exit(1)
    print(f"SANITY GP2 PASS: probe cells sit on the identical test panel as the "
          f"committed llama70 cells (n_test = "
          f"{'/'.join(str(int(x)) for x in pr.n_test)})")

    # ---- probe shares vs the committed fulltext rel% ----
    share_rows = []
    for i, h in enumerate(HORIZONS):
        p_, e_, s_ = pr.iloc[i], ens_ref.iloc[i], sin_ref.iloc[i]
        well = bool(e_.rel_har >= 1.0 and e_.dm_har < 0 and e_.p_har < .05)
        share_rows.append(dict(
            disc=DISC, block="probe_share", h=h,
            probe_rel_har=p_.rel_har, fulltext_ens_rel_har=e_.rel_har,
            share_har_pct=probe_share(p_.rel_har, e_.rel_har),
            probe_rel_firm=p_.rel_firm, fulltext_ens_rel_firm=e_.rel_firm,
            share_firm_pct=probe_share(p_.rel_firm, e_.rel_firm),
            share_har_pct_vs_single=probe_share(p_.rel_har, s_.rel_har),
            share_firm_pct_vs_single=probe_share(p_.rel_firm, s_.rel_firm),
            denominator_well_identified=well,
        ))
    shares = pd.DataFrame(share_rows)

    # ---- text-beyond-identity: joint reference [1, log fHAR, log f_probe] ----
    pf = probe[KEY + ["prediction_realised_vol"]].rename(
        columns={"prediction_realised_vol": "f_probe"})
    ff = preds_l[ENS_FAM][KEY + ["prediction_realised_vol"]].rename(
        columns={"prediction_realised_vol": "f_fulltext"})
    joint_rows = []
    for h in HORIZONS:
        m = a2[a2.horizon_days == h] \
            .merge(pf[pf.horizon_days == h], on=KEY) \
            .merge(ff[ff.horizon_days == h], on=KEY).dropna()
        v, te = m[m.split == "val"], m[m.split == "test"]
        yv, yt = v.label_realised_vol.values, te.label_realised_vol.values
        XR_v = np.column_stack([np.ones(len(v)), L(v.fh), L(v.f_probe)])
        XU_v = np.column_stack([XR_v, L(v.f_fulltext)])
        bR, bU = ols(L(yv), XR_v), ols(L(yv), XU_v)
        fRj = np.exp(bR[0] + bR[1] * L(te.fh) + bR[2] * L(te.f_probe))
        fUj = np.exp(bU[0] + bU[1] * L(te.fh) + bU[2] * L(te.f_probe)
                     + bU[3] * L(te.f_fulltext))
        lRj, lUj = fc.qlike(yt, fRj), fc.qlike(yt, fUj)
        dmj, pj, ndj = cdm.dm_test_clustered(lUj, lRj,
                                             te.effective_trading_day.values, h)
        qRj, qUj = float(lRj.mean()), float(lUj.mean())
        joint_rows.append(dict(disc=DISC, block="beyond_identity",
                               arm="fulltext_ens_beyond_probe", h=h,
                               n_test=len(te), n_days=ndj,
                               qlike_R=qRj, qlike_U=qUj,
                               rel_pct=100.0 * (qRj - qUj) / qRj,
                               dm_clu=dmj, p_raw=pj, g_text=float(bU[3])))
    joint = pd.DataFrame(joint_rows)
    joint["p_holm"] = holm(joint.p_raw.fillna(1.0).values)  # within-block Holm(3)

    # ---- assemble csv (long format, block column) ----
    pr_out = pr.copy()
    pr_out.insert(1, "block", "probe_m1")
    anchors = pd.concat([sin_ref, ens_ref], ignore_index=True)
    anchors.insert(1, "block", "fulltext_anchor_committed")
    out = pd.concat([pr_out, anchors, shares, joint], ignore_index=True)

    n_beyond_raw = int(((joint.dm_clu < 0) & (joint.p_raw < .05)).sum())
    n_beyond_holm = int(((joint.dm_clu < 0) & (joint.p_holm < .05)).sum())
    n_probe_pos_har = int((pr.rel_har > 0).sum())
    retain = [100.0 * j / e if abs(e) > 1e-12 else float("nan")
              for j, e in zip(joint.rel_pct, ens_ref.rel_har)]

    def sig(dm, p):
        return "**" if (dm < 0 and p < 0.05) else ""

    md = [
        "# 70B zero-content date+ticker probe — prereg B2 rider (prereg-rfa v1.3)",
        "",
        "Reconciles the Table 6 probe cell (date+ticker reproducing >100% of a "
        "fulltext increment on the Qwen arm) with the llama70 replication claim, by "
        "running the SAME zero-content probe through the replication family itself. "
        "**Descriptive readout — no prereg branch fires; Holm only within blocks** "
        "(the llm_contamination.py template convention).",
        "",
        "## Disclosures",
        "",
        f"- **Model / precision**: {model_id} (weight-only **AWQ-INT4** — the SAME "
        "precision as the committed llama70 fulltext runs, so the probe-vs-fulltext "
        "comparison is internally consistent; disclosed exactly as the committed "
        "runs disclose it). vLLM TP=2, temperature-0 protocol, guided JSON, "
        "clip [0.03,3.0], on_missing=rv22 — byte-identical stack to the committed "
        "runs.",
        "- **Prompt**: the C6 contamination arm's `c6_datefirm` template VERBATIM "
        "(scripts/experiments/e1_llm_forecast/prompt.py): form type + items, filing "
        "date, ticker, the line \"(No filing text is provided.)\" and the identical "
        "task text — the SAME fields, NO document text. This is the prompt behind "
        "the committed C6_datefirm (Qwen) runs in llm_contamination.md.",
        "- **Seed**: single seed 2026 (the probe is a control readout; the committed "
        "fulltext anchor is the 3-seed ensemble — llama70 seed jitter moved rel% by "
        "<0.04pp, see crossfamily_llama70_ens.csv — and single-seed fulltext "
        "denominators are reported as robustness).",
        "- **Panel**: the identical event_driven panel as the committed "
        "crossfamily_llama70 runs (verified by GP2: equal n_test per horizon).",
        "- **>100% convention**: probe-share = probe rel% / committed fulltext rel% "
        "per cell; a value ABOVE 100% means the zero-content probe alone reproduces "
        "MORE than the fulltext increment in that cell. Shares are quotable only "
        "where the denominator is well-identified (fulltext rel% >= 1% AND raw-"
        "significant, the llm_contamination stable-denominator rule) — flagged per "
        "cell; small denominators inflate shares mechanically.",
        "- **No new Holm family**: Holm within the probe M1 block (6 cells) and "
        "within the beyond-identity block (3 cells) only, as the template does; "
        "committed anchor cells carry their committed values unchanged (GP1-anchored,"
        " not re-derived).",
        "",
        "## 1. Probe M1 vs the committed fulltext increment (side by side)",
        "",
        "rel% on volatility-unit QLIKE; `**` = clustered DM<0, raw p<.05. fulltext = "
        "committed llama70_awq_ens3 (crossfamily_llama70_ens.csv).",
        "",
        "| h | probe rel% vs HAR | DM | probe rel% vs HAR+firmID | DM | "
        "fulltext rel% vs HAR | fulltext rel% vs HAR+firmID | probe-share vs HAR | "
        "probe-share vs firmID | well-identified denom |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|",
    ]
    for i, h in enumerate(HORIZONS):
        p_, e_, s_ = pr.iloc[i], ens_ref.iloc[i], shares.iloc[i]
        md.append(
            f"| {h} | {p_.rel_har:+.2f}%{sig(p_.dm_har, p_.p_har)} | "
            f"{p_.dm_har:+.2f} | {p_.rel_firm:+.2f}%{sig(p_.dm_firm, p_.p_firm)} | "
            f"{p_.dm_firm:+.2f} | {e_.rel_har:+.2f}% | {e_.rel_firm:+.2f}% | "
            f"{s_.share_har_pct:.0f}% | {s_.share_firm_pct:.0f}% | "
            f"{'yes' if s_.denominator_well_identified else 'NO (do not quote)'} |")
    md += [
        "",
        f"(probe Holm(6) within block: min Holm p vs HAR = {pr.p_har_holm.min():.4g},"
        f" vs firmID = {pr.p_firm_holm.min():.4g}; single-seed fulltext robustness "
        "denominators in the csv: share_har_pct_vs_single / "
        "share_firm_pct_vs_single.)",
        "",
        "## 2. Text beyond identity — joint reference [1, log fHAR, log f_probe] "
        "(+ log f_fulltext)",
        "",
        "The same-model identity control for the REPLICATION family: the reference "
        "already contains everything llama70 produces from date+ticker alone, so any "
        "residual fulltext increment must come from the filing text.",
        "",
        "| h | n_test | n_days | QLIKE(R') | QLIKE(U') | rel% | DM(clu) | p raw | "
        "p Holm(3) | g_text | retained share of fulltext-vs-HAR rel% |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for i, r in joint.iterrows():
        md.append(f"| {int(r.h)} | {int(r.n_test)} | {int(r.n_days)} | "
                  f"{r.qlike_R:.4f} | {r.qlike_U:.4f} | "
                  f"{r.rel_pct:+.2f}%{sig(r.dm_clu, r.p_holm)} | {r.dm_clu:+.2f} | "
                  f"{r.p_raw:.2e} | {r.p_holm:.4f} | {r.g_text:+.3f} | "
                  f"{retain[i]:.0f}% |")
    md += [
        "",
        "## Bottom line (descriptive — feeds the Table 6 reconciliation sentence)",
        "",
        f"- The zero-content probe carries a positive vs-HAR increment in "
        f"{n_probe_pos_har}/3 horizons (identity/era memory, zero filing content); "
        "its share of the committed fulltext increment is the probe-share column "
        "above (>100% possible by convention).",
        f"- With the same-model date+ticker forecast INSIDE the reference, the "
        f"committed llama70-ens fulltext still adds in {n_beyond_raw}/3 horizons at "
        f"raw p<.05 ({n_beyond_holm}/3 after within-block Holm(3)), retaining "
        f"{min(retain):.0f}%-{max(retain):.0f}% of the uncontrolled fulltext-vs-HAR "
        "rel% per cell — this text-beyond-identity readout, not the raw probe-share,"
        " is the number the replication claim rests on.",
        "- No prereg branch fires on this table (registered as descriptive; "
        "prereg-rfa v1.3 §B2 rider).",
        "",
        "## SANITY",
        "",
        f"- GP1 PASS (G1'' convention): all {len(ref_l)} committed "
        "crossfamily_llama70_ens.csv rows (single-seed AND ens3) reproduced on this "
        f"exact code path to machine precision (rtol {RTOL:g}) on columns {M1_COLS};"
        " anchor cells then carried unchanged.",
        f"- GP2 PASS: probe cells sit on the identical test panel as the committed "
        f"llama70 cells (n_test = {'/'.join(str(int(x)) for x in pr.n_test)}).",
        "",
    ]

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(CSV_OUT, index=False)
    Path(MD_OUT).write_text("\n".join(md))
    print(f"wrote {CSV_OUT} / {MD_OUT}")
    print(pr[["family", "h", "n_test", "rel_har", "dm_har", "p_har",
              "rel_firm", "dm_firm", "p_firm"]].to_string(index=False))
    print("\nprobe shares (%):")
    print(shares[["h", "share_har_pct", "share_firm_pct",
                  "denominator_well_identified"]].to_string(index=False))
    print("\nbeyond identity:")
    print(joint[["h", "rel_pct", "dm_clu", "p_raw", "p_holm",
                 "g_text"]].to_string(index=False))
    return out


# ----------------------------------------------------------------- orchestration
def run_probe(args, gen=None):
    assert_writeonce(CSV_OUT, MD_OUT)
    needed = ["results/tables/crossfamily_llama70_ens.csv", MANIFEST,
              f"results/runs/A2_har_rv_full_{DISC}_seed2026/predictions.parquet"] + \
             [f"results/runs/{r}/predictions.parquet" for r in LLA_RUNS.values()]
    missing = [p for p in needed if not Path(p).exists()]
    if missing:
        print("FATAL: required committed inputs missing:", missing)
        sys.exit(1)
    rd = Path(f"results/runs/{PROBE_RUN}")
    if not (rd / "predictions.parquet").exists():
        panel = load_panel()
        if gen is None:
            model_path, how = resolve_model(args.model_id)
            print(f"[probe] model {args.model_id} -> {model_path} ({how}); "
                  f"TP={TP} max_model_len={MAX_MODEL_LEN} seed={SEED}")
            gen = ri.VllmGenerator(model_path, MAX_MODEL_LEN, TP, MAX_TOKENS)
        do_infer(panel, RAW_DIR, gen)
        build_run_dir(RAW_DIR, PROBE_RUN)
    else:
        print(f"[probe] {rd} already built — skipping inference")
    score(args.model_id)


# ---------------------------------------------------------------------- dry run
def dry_run(args):
    ok = True

    def chk(cond, msg):
        nonlocal ok
        print(("  OK   " if cond else "  FAIL ") + msg)
        ok = ok and bool(cond)

    print(f"DRY-RUN (no GPU, no writes) — model={args.model_id}")
    print("[env]")
    for k in ("SP500VOL_DATA_ROOT", "HF_HOME", "HF_HUB_OFFLINE",
              "TRANSFORMERS_OFFLINE", "VLLM_WORKER_MULTIPROC_METHOD"):
        print(f"       {k}={os.environ.get(k)}")
    print("[panel]")
    chk(Path(MANIFEST).exists(), f"manifest {MANIFEST}")
    if Path(MANIFEST).exists():
        panel = load_panel()
        chk(len(panel) > 0, f"ED panel: {len(panel)} filings (committed llama70 "
            f"panel = 39,322), splits {panel.split.value_counts().to_dict()}")
        row = panel.iloc[0].to_dict()
        msgs = prompt_mod.build_messages(row, "", VARIANT)
        u = msgs[1]["content"]
        chk(msgs[0]["role"] == "system"
            and f"- Ticker: {row['ticker']}" in u
            and "(No filing text is provided.)" in u
            and "Filing excerpt" not in u,
            "c6_datefirm prompt builds VERBATIM (ticker + date, zero content, "
            "no excerpt block)")
        print("       prompt head: " + u[:120].replace("\n", " | "))
    print("[committed inputs]")
    chk(Path("results/tables/crossfamily_llama70_ens.csv").exists(),
        "results/tables/crossfamily_llama70_ens.csv (fulltext anchor)")
    chk(Path(f"results/runs/A2_har_rv_full_{DISC}_seed2026/predictions.parquet")
        .exists(), "A2 event_driven run (labels + fh)")
    for r in LLA_RUNS.values():
        chk(Path(f"results/runs/{r}/predictions.parquet").exists(),
            f"results/runs/{r}")
    print("[model resolution]")
    model_path, how = resolve_model(args.model_id)
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    resolved = how != "hub-id-unresolved"
    print(f"       {args.model_id} -> {model_path} ({how})")
    if offline:
        chk(resolved, "model resolvable under HF_HUB_OFFLINE=1")
    elif not resolved:
        print("  WARN model not in local HF cache (fine off-box; the box already "
              "holds the committed llama70 AWQ snapshot)")
    print("[write-once state]")
    for pth in (CSV_OUT, MD_OUT):
        print(f"       {'EXISTS (write-once guard would refuse)' if Path(pth).exists() else 'absent (ok)':<45} {pth}")
    print(f"       probe run dir "
          f"{'EXISTS (inference would be skipped)' if Path(f'results/runs/{PROBE_RUN}/predictions.parquet').exists() else 'absent (inference would run)'}")
    print("DRY-RUN " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------- selftest
def _selftest():
    import shutil
    import tempfile
    rng = np.random.default_rng(11)
    sandbox = Path(tempfile.mkdtemp(prefix="l70probe_selftest_"))
    print(f"[selftest] sandbox: {sandbox}")
    cwd0 = os.getcwd()
    failures = []

    def check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    # ---- unit: verbatim prompt shape (imported template, asserted fields) ----
    row = {"form": "8-K", "item_subtype": "2.02", "filing_date": "2024-01-05",
           "ticker": "XYZ", "sections_json": None}
    msgs = prompt_mod.build_messages(row, "SHOULD NOT APPEAR", VARIANT)
    u = msgs[1]["content"]
    check("c6_datefirm prompt: system+user, ticker+date, no filing text",
          msgs[0]["role"] == "system" and "- Ticker: XYZ" in u
          and "- Filing date: 2024-01-05" in u
          and "(No filing text is provided.)" in u
          and "SHOULD NOT APPEAR" not in u and "Filing excerpt" not in u)
    do_msgs = prompt_mod.build_messages(row, "", "c6_dateonly")
    check("c6_datefirm differs from c6_dateonly ONLY by the ticker line",
          u.replace("- Ticker: XYZ\n", "") == do_msgs[1]["content"])

    # ---- unit: probe-share convention ----
    check("probe_share: 1.5/1.0 -> 150% (>100% convention)",
          np.isclose(probe_share(1.5, 1.0), 150.0))
    check("probe_share: ~0 denominator -> NaN (unidentified)",
          np.isnan(probe_share(0.5, 0.0)))
    check("probe_share: sign preserved for negative numerator",
          np.isclose(probe_share(-0.5, 1.0), -50.0))

    # ---- synthetic fixtures ----
    def synth_panel():
        tickers = [f"T{i:02d}" for i in range(40)]
        dv = pd.date_range("2023-01-02", periods=80, freq="B", tz="UTC")
        dt_ = pd.date_range("2023-06-01", periods=150, freq="B", tz="UTC")
        rows, k = [], 0
        for split, days in (("val", dv), ("test", dt_)):
            for d in days:
                for t in rng.choice(tickers, size=10, replace=False):
                    rows.append((t, d, split, f"A{k:07d}"))
                    k += 1
        base = pd.DataFrame(rows, columns=["ticker", "effective_trading_day",
                                           "split", "accession"])
        firm = {t: rng.normal(0, .25) for t in tickers}
        dayshock = {d: rng.normal(0, .15) for d in [*dv, *dt_]}
        base["text_path"] = base.ticker + "_" + base.accession
        base["filing_time_utc"] = base.effective_trading_day
        out = []
        for h in HORIZONS:
            b = base.copy()
            b["horizon_days"] = h
            idio = rng.normal(0, .30, len(b))
            ly = (np.log(0.25) + b.ticker.map(firm).values
                  + b.effective_trading_day.map(dayshock).values + idio)
            b["label_realised_vol"] = np.exp(ly)
            b["_ly"], b["_idio"] = ly, idio
            b["fh"] = np.exp(np.log(0.25) * 1.4
                             + 0.6 * (ly - np.log(0.25) - idio)
                             + rng.normal(0, .2, len(b)))
            out.append(b)
        return pd.concat(out, ignore_index=True)

    pan = synth_panel()

    def write_run(run_id, f, stats=None):
        rd = Path(f"results/runs/{run_id}")
        rd.mkdir(parents=True, exist_ok=True)
        d = pan[["ticker", "accession", "horizon_days", "split",
                 "label_realised_vol", "effective_trading_day", "text_path",
                 "filing_time_utc"]].copy()
        d["prediction_realised_vol"] = np.asarray(f, float)
        d["feature_rv_22d"] = pan.label_realised_vol * np.exp(
            rng.normal(0, .1, len(pan)))
        d["run_id"], d["model_id"] = run_id, run_id
        d["dataset"], d["seed"], d["disclosure_subset"] = "full", 2026, DISC
        d["form"], d["item_subtype"], d["metadata_path"] = "8-K", "2.02", ""
        d["feature_rv_1d"] = d["feature_rv_5d"] = 0.25
        d.to_parquet(rd / "predictions.parquet", index=False)
        (rd / "metrics.json").write_text(json.dumps(pp.metrics_rows(d, DISC),
                                                    indent=2))
        (rd / "config.json").write_text(json.dumps(
            {"model_id": run_id, "stats": stats or
             {"parse_fail_rate": 0.0, "clipped_rate": 0.0}}, indent=2))
        return d

    # fulltext: firm/day + half the idio; probe: firm/day ONLY (identity-like)
    def f_fulltext(jit=0.0):
        return np.clip(np.exp(pan._ly - 0.5 * pan._idio
                              + rng.normal(0, .10, len(pan))
                              + (rng.normal(0, jit, len(pan)) if jit else 0.0)),
                       pp.CLIP_LO, pp.CLIP_HI)

    def f_probe():
        return np.clip(np.exp(pan._ly - 1.0 * pan._idio
                              + rng.normal(0, .12, len(pan))),
                       pp.CLIP_LO, pp.CLIP_HI)

    try:
        os.chdir(sandbox)
        Path("results/tables").mkdir(parents=True)
        Path("results/e1_llm_forecast").mkdir(parents=True)
        a2df = write_run(f"A2_har_rv_full_{DISC}_seed2026", pan.fh.values)
        a2 = a2df.rename(columns={"prediction_realised_vol": "fh"})[
            KEY + ["split", "label_realised_vol", "fh", "effective_trading_day"]]
        seeds = [f_fulltext(jit=0.02) for _ in range(3)]
        write_run(LLA_RUNS["llama70_awq"], seeds[0])
        write_run(LLA_RUNS["llama70_awq_ens3"],
                  np.mean(np.column_stack(seeds), axis=1))
        both = []
        for fam, run in LLA_RUNS.items():
            rr = pd.DataFrame(m1_rows(
                fam, pd.read_parquet(f"results/runs/{run}/predictions.parquet"), a2))
            ps6 = holm(np.concatenate([rr.p_har.values, rr.p_firm.values]))
            rr["p_har_holm"], rr["p_firm_holm"] = ps6[:3], ps6[3:]
            rr["holm_family"] = "fixture Holm(6)"
            both.append(rr)
        pd.concat(both, ignore_index=True).to_csv(
            "results/tables/crossfamily_llama70_ens.csv", index=False)

        # mock inference end-to-end for the control variant (no text cache needed)
        man = pan.drop_duplicates("text_path").copy()
        man["disclosure"], man["form"] = DISC, "8-K"
        man["item_subtype"], man["sections_json"] = "2.02", None
        man["filing_date"] = man.filing_time_utc.dt.date.astype(str)
        man["metadata_path"], man["token_count"] = "", 100
        man["feature_rv_1d"] = man["feature_rv_5d"] = 0.25
        man["feature_rv_22d"] = man.label_realised_vol
        man.to_parquet(MANIFEST, index=False)

        class _ProbeGen:  # deterministic mock: emits the fixture probe forecast
            name = "mock-probe"

            def __init__(self):
                self.lut = {}
                for h in HORIZONS:
                    sub = pan[pan.horizon_days == h].set_index("text_path")
                    fp = f_probe_by_h[h]
                    for tp_, v_ in zip(sub.index, fp):
                        self.lut.setdefault(tp_, {})[f"vol_{h}d"] = float(v_)

            def generate(self, records, retry=False):
                outs = []
                for rec in records:
                    outs.append(json.dumps(self.lut[rec["row"]["text_path"]]))
                return outs

        fp_all = f_probe()
        f_probe_by_h = {h: fp_all[(pan.horizon_days == h).to_numpy()]
                        for h in HORIZONS}
        run_probe(argparse.Namespace(model_id="fixture/llama70-awq"),
                  gen=_ProbeGen())
        check("e2e: csv + md written",
              Path(CSV_OUT).exists() and Path(MD_OUT).exists())
        got = pd.read_csv(CSV_OUT)
        check("e2e: all four blocks present",
              set(got.block.dropna()) == {"probe_m1", "fulltext_anchor_committed",
                                          "probe_share", "beyond_identity"})
        bi = got[got.block == "beyond_identity"]
        check("e2e: beyond-identity block = 3 cells, fulltext adds beyond the "
              "probe (DM<0, raw p<.05) in the constructed fixture",
              len(bi) == 3 and bool((bi.dm_clu < 0).all())
              and bool((bi.p_raw < .05).all()))
        sh = got[got.block == "probe_share"]
        check("e2e: probe shares finite with well-identified denominators",
              len(sh) == 3 and sh.share_har_pct.notna().all()
              and sh.denominator_well_identified.all())
        md_txt = Path(MD_OUT).read_text()
        check("e2e: md documents the >100% convention + descriptive/no-branch",
              ">100%" in md_txt and "no prereg branch" in md_txt.lower())
        check("e2e: probe run dir built via the postprocess path (rv22/clip/stats)",
              Path(f"results/runs/{PROBE_RUN}/config.json").exists()
              and json.load(open(f"results/runs/{PROBE_RUN}/config.json"))
              ["stats"]["parse_fail_rate"] == 0.0)

        # write-once guard
        try:
            score("fixture/llama70-awq")
            check("write-once guard trips on second write", False)
        except SystemExit as e:
            check("write-once guard trips on second write", e.code == 3)

        # GP1 tamper -> abort
        for f in (CSV_OUT, MD_OUT):
            Path(f).unlink()
        ref = pd.read_csv("results/tables/crossfamily_llama70_ens.csv")
        ref.loc[0, "rel_har"] += 1e-6
        ref.to_csv("results/tables/crossfamily_llama70_ens.csv", index=False)
        try:
            score("fixture/llama70-awq")
            check("GP1 catches a drifted committed anchor", False)
        except SystemExit as e:
            check("GP1 catches a drifted committed anchor", e.code == 1)
        ref.loc[0, "rel_har"] -= 1e-6
        ref.to_csv("results/tables/crossfamily_llama70_ens.csv", index=False)

        # GP2 tamper: drop one test day from the probe run -> panel drift -> abort
        pq_path = Path(f"results/runs/{PROBE_RUN}/predictions.parquet")
        full = pd.read_parquet(pq_path)
        drop_day = full[full.split == "test"].effective_trading_day.iloc[0]
        full[~((full.split == "test")
               & (full.effective_trading_day == drop_day))].to_parquet(
            pq_path, index=False)
        try:
            score("fixture/llama70-awq")
            check("GP2 catches probe/committed panel drift", False)
        except SystemExit as e:
            check("GP2 catches probe/committed panel drift", e.code == 1)
        full.to_parquet(pq_path, index=False)  # restore

        # restored state must score cleanly again (fresh single shot)
        score("fixture/llama70-awq")
        check("restored state scores cleanly (single shot preserved)",
              Path(CSV_OUT).exists())
    finally:
        os.chdir(cwd0)
        shutil.rmtree(sandbox, ignore_errors=True)

    print(f"\nSELFTEST {'PASS' if not failures else 'FAIL'} "
          f"({len(failures)} failure(s))")
    for f in failures:
        print("  FAILED:", f)
    sys.exit(0 if not failures else 1)


# -------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(
        description="Prereg B2 rider (prereg-rfa v1.3): zero-content date+ticker "
                    "probe through Llama-3.1-70B AWQ-INT4 — descriptive readout, "
                    "no branches.")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                    help="int4 AWQ, the committed llama70 runs' precision (disclosed)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate panel/prompt/model-path resolution; no GPU")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic-fixture selftest in a throwaway sandbox; no GPU")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.dry_run:
        dry_run(args)
    else:
        run_probe(args)


if __name__ == "__main__":
    main()
