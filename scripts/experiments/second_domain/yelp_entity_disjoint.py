"""ACTION A, MONEY EXPERIMENT — split-based entity controls are blind to pretrained priors.

THE CLAIM. The field's standard fix for entity leakage is to make the split
entity-disjoint (group-wise CV; Saeb et al. 2017; Chaibub Neto et al. 2017): no entity
appears in both train and test, so a fitted model cannot memorise an entity's mean and
call it text understanding. That fix works on FITTED arms by construction.

It cannot work on a ZERO-SHOT PROMPTED arm, and the reason is mechanical, not empirical:
a prompted foundation model never sees the training split at all. Its knowledge of
"Joe's Pizza in Phoenix" was acquired from a pretraining corpus that no split of OUR
panel can touch. Splitting is a statement about our data; the model's entity prior is a
statement about its data. So an entity-disjoint split leaves the prompted arm's identity
channel completely intact while removing the fitted arms' — and a benchmark builder who
runs the standard fix will believe they have controlled a shortcut that is still there.

THE TEST. Run the identical audit twice on the same universe and the same rows, changing
ONLY the split rule:
  A) shared-entity split (chronological; entities recur across train and test)
  B) entity-disjoint split (businesses partitioned; the field's standard fix)
and read the ZERO-CONTENT IDENTITY PROBE (business name + city + categories + month, no
review text) under both. The probe is pure identity by construction, so it is the
instrument that detects whether the split removed identity.

PREDICTION (pre-registered here, before running):
  - fitted arms (TF-IDF, frozen-embedding): entity-disjoint REMOVES their entity edge;
    the entity-mean control loses its power because there is no test-entity mean to learn.
  - prompted arm and its probe: entity-disjoint changes NOTHING; the probe keeps whatever
    gain it had, because its identity knowledge is pretrained, not fitted.
  => the two arm classes move in OPPOSITE directions under the standard fix.
FALSIFIED IF: the probe's gain collapses under the entity-disjoint split too (then the
split does control pretrained priors and the whole claim dies), or if the fitted arms
keep their edge (then the split does not work at all and the contrast is uninformative).

Universe: the 2,000-business subsample for which zero-shot 70B predictions exist, over the
months those predictions cover. Prompted predictions are split-invariant by construction
(zero-shot), so the SAME prediction vector is scored under both splits — which is exactly
the point being demonstrated. Fitted arms are refit under each split.

Run:  .venv/bin/python scripts/experiments/second_domain/yelp_entity_disjoint.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "yproto", str(Path(__file__).with_name("yelp_protocol.py")))
_yproto = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_yproto)
dm_test_month, dm_test_2way = _yproto.dm_test_month, _yproto.dm_test_2way

PANEL = "/path/to/data-root/second-domain/yelp/run1/yelp_panel.parquet"
PREDS = Path("results/second_domain/preds")
FEATS = ["ar_last_mean", "ar_roll3_mean", "ar_roll12_mean", "ar_log_n_reviews"]
CLIP_LO, CLIP_HI = 1.0, 5.0
ALPHAS_AR = tuple(float(a) for a in np.logspace(-3, 3, 13))
ALPHAS_TXT = tuple(float(a) for a in np.logspace(-1, 4, 11))
EPS = 1e-8
OUT = Path("results/tables")


def clip(x):
    return np.clip(np.asarray(x, float), CLIP_LO, CLIP_HI)


def mse(y, f):
    return float(np.mean((np.asarray(y, float) - np.asarray(f, float)) ** 2))


def rel(a, b):
    return 100.0 * (a - b) / a


def se(y, f):
    """Per-row squared error — the loss whose mean is the reported MSE."""
    return (np.asarray(y, float) - np.asarray(f, float)) ** 2


def entity_bucket(eid: str) -> int:
    """Deterministic entity-disjoint assignment: 60/20/20 by hash of the id."""
    h = int(hashlib.sha256(eid.encode()).hexdigest()[:8], 16) % 100
    return 0 if h < 60 else (1 if h < 80 else 2)


def log_recal(yv, fv):
    ly = np.log(np.clip(yv, EPS, None)); lf = np.log(np.clip(fv, EPS, None))
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(lf)), lf]), ly, rcond=None)
    return float(beta[0]), float(beta[1])


def apply_recal(a, b, f):
    return clip(np.exp(a + b * np.log(np.clip(np.asarray(f, float), EPS, None))))


def log_combo(yv, fpv, ftv, fpt, ftt):
    """Val-fit log-space nested combiner, frozen to test (the paper's instrument)."""
    ly = np.log(np.clip(yv, EPS, None))
    A = lambda p, t: np.column_stack([np.ones(len(p)), np.log(np.clip(p, EPS, None)),
                                      np.log(np.clip(t, EPS, None))])
    bU, *_ = np.linalg.lstsq(A(fpv, ftv), ly, rcond=None)
    bR, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(fpv)), np.log(np.clip(fpv, EPS, None))]),
                             ly, rcond=None)
    fR = clip(np.exp(bR[0] + bR[1] * np.log(np.clip(fpt, EPS, None))))
    fU = clip(np.exp(A(fpt, ftt) @ bU))
    return fR, fU


def fit_ridge(Xtr, ytr, Xv, yv, alphas):
    best = None
    for a in alphas:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        v = mse(yv, clip(m.predict(Xv)))
        if best is None or v < best[0]:
            best = (v, m)
    return best[1]


def run_split(panel, prompted, split_name, h):
    """Fit AR + TF-IDF under this split; score every arm on this split's test rows."""
    d = panel[panel.horizon_months == h]
    tr, va, te = (d[d.sp == s] for s in ("train", "val", "test"))
    if min(len(tr), len(va), len(te)) < 200:
        return []
    ytr, yv, yt = (s.label.to_numpy(float) for s in (tr, va, te))

    # --- AR baseline (fitted) + val-fit recalibration
    sc = StandardScaler().fit(tr[FEATS])
    ar = fit_ridge(sc.transform(tr[FEATS]), ytr, sc.transform(va[FEATS]), yv, ALPHAS_AR)
    f_ar_v, f_ar_t = clip(ar.predict(sc.transform(va[FEATS]))), clip(ar.predict(sc.transform(te[FEATS])))
    a, b = log_recal(yv, f_ar_v)
    fRv, fRt = apply_recal(a, b, f_ar_v), apply_recal(a, b, f_ar_t)

    # --- entity-mean control (the zero-text identity forecast) -----------------
    # Under the entity-disjoint split a test entity has NO train/val rows, so the
    # entity mean is undefined and falls back to the global mean: the control is
    # mechanically dead. That is the fix working on the fitted side.
    seen = pd.concat([tr, va])
    emap = seen.groupby("entity_id").label.mean()
    g = float(seen.label.mean())
    em_v = va.entity_id.map(emap).fillna(g).to_numpy(float)
    em_t = te.entity_id.map(emap).fillna(g).to_numpy(float)
    cov = float(te.entity_id.isin(emap.index).mean())
    lyv = np.log(np.clip(yv, EPS, None))
    Av = np.column_stack([np.ones(len(fRv)), np.log(np.clip(fRv, EPS, None)),
                          np.log(np.clip(em_v, EPS, None))])
    be, *_ = np.linalg.lstsq(Av, lyv, rcond=None)
    fRe_t = clip(np.exp(np.column_stack([np.ones(len(fRt)), np.log(np.clip(fRt, EPS, None)),
                                         np.log(np.clip(em_t, EPS, None))]) @ be))
    months_t = te.event_time.to_numpy()
    ents_t = te.entity_id.to_numpy()

    def scored(arm, kind, fR_, fU_):
        """rel% plus month-clustered and two-way DM of loss(f_U) vs loss(f_R):
        DM < 0 means the augmented forecast is better (text/identity helps)."""
        lR, lU = se(yt, fR_), se(yt, fU_)
        dm, p, nm = dm_test_month(lU, lR, months_t, h)
        tw = dm_test_2way(lU - lR, ents_t, months_t, h)
        dm2, p2 = tw["dm_2way"], tw["p_2way"]
        return {"split": split_name, "h": h, "arm": arm, "kind": kind,
                "rel_vs_recalAR": rel(float(lR.mean()), float(lU.mean())),
                "dm_month": dm, "p_month": p, "n_months": nm,
                "dm_2way": dm2, "p_2way": p2,
                "entity_coverage_test": cov, "n_test": len(te)}

    rows = [scored("entity-mean control (zero text)", "identity", fRt, fRe_t)]

    # --- fitted text arm: TF-IDF (refit under this split) ---------------------
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)
    Ttr = vec.fit_transform(tr.text)
    txt = fit_ridge(Ttr, ytr, vec.transform(va.text), yv, ALPHAS_TXT)
    ft_v, ft_t = clip(txt.predict(vec.transform(va.text))), clip(txt.predict(vec.transform(te.text)))
    fR, fU = log_combo(yv, f_ar_v, ft_v, f_ar_t, ft_t)
    rows.append(scored("TF-IDF (fitted)", "fitted", fR, fU))

    # --- zero-shot prompted arms: predictions are SPLIT-INVARIANT -------------
    for arm, col in (("70B prompted (zero-shot)", "llm70_chrono"),
                     ("70B identity probe (zero-content)", "llm70_probe")):
        pv = va[col].to_numpy(float); pt = te[col].to_numpy(float)
        if np.isnan(pv).any() or np.isnan(pt).any():
            continue
        fR, fU = log_combo(yv, f_ar_v, pv, f_ar_t, pt)
        rows.append(scored(arm, "zero-shot", fR, fU))
    return rows


def main():
    panel = pd.read_parquet(PANEL)
    # universe = the entities/rows for which zero-shot 70B predictions exist
    pro = {}
    for arm, f in (("llm70_chrono", "preds_llm70_chrono.parquet"),
                   ("llm70_probe", "preds_llm70_probe.parquet")):
        p = pd.read_parquet(PREDS / f)[["entity_id", "event_time", "horizon_months", "prediction"]]
        pro[arm] = p.rename(columns={"prediction": arm})
    m = pro["llm70_chrono"].merge(pro["llm70_probe"],
                                  on=["entity_id", "event_time", "horizon_months"], validate="1:1")
    panel = panel.merge(m, on=["entity_id", "event_time", "horizon_months"], validate="1:1")
    print(f"universe: {len(panel):,} rows, {panel.entity_id.nunique():,} entities "
          f"(rows with zero-shot 70B predictions)")

    out = []
    # --- SPLIT A: shared-entity (chronological), the paper's protocol ---------
    a = panel.copy()
    a["sp"] = a["split"]                      # val -> val, test -> test (train has no preds)
    # the prompted preds cover val+test only; use val as the fitting window and
    # test as evaluation, and carve a train window out of val's earlier months.
    cut = a[a.sp == "val"].event_time.quantile(0.5)
    a.loc[(a.sp == "val") & (a.event_time <= cut), "sp"] = "train"
    shared_ents = set(a[a.sp == "train"].entity_id) & set(a[a.sp == "test"].entity_id)
    print(f"[A shared-entity] entities in BOTH train and test: {len(shared_ents):,}")

    # --- SPLIT B: entity-disjoint (the field's standard fix) -----------------
    b = panel.copy()
    bucket = b.entity_id.map(entity_bucket)
    b["sp"] = np.select([bucket == 0, bucket == 1], ["train", "val"], "test")
    overlap = set(b[b.sp == "train"].entity_id) & set(b[b.sp == "test"].entity_id)
    print(f"[B entity-disjoint] entities in BOTH train and test: {len(overlap):,} (must be 0)")
    assert not overlap, "entity-disjoint split leaked an entity"

    for h in sorted(panel.horizon_months.unique()):
        out += run_split(a, pro, "A: shared-entity (chronological)", int(h))
        out += run_split(b, pro, "B: entity-disjoint (standard fix)", int(h))

    df = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "yelp_entity_disjoint.csv", index=False)
    piv = df.pivot_table(index=["arm", "kind"], columns=["split", "h"],
                         values="rel_vs_recalAR")
    print("\n=== combiner gain vs recalibrated AR (%), by split ===")
    print(piv.round(3).to_string())
    print("\n=== month-clustered DM (negative = the arm helps) and p ===")
    for _, r in df.iterrows():
        print(f"{r.split:36s} h={r.h} {r.arm:34s} rel={r.rel_vs_recalAR:+7.3f}%  "
              f"DM_month={r.dm_month:+6.2f} (p={r.p_month:.4f}, n={r.n_months} months)  "
              f"DM_2way={r.dm_2way:+6.2f} (p={r.p_2way:.4f})")
    print("\n=== test-entity coverage by the entity-mean control ===")
    print(df.groupby("split").entity_coverage_test.first().to_string())
    print(f"\nwrote {OUT}/yelp_entity_disjoint.csv")


if __name__ == "__main__":
    main()
