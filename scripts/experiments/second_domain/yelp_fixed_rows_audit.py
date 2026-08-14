"""BLEEDING-STOP — the one-change audit, done so that it is actually one change.

WHY THIS EXISTS. The first version (yelp_entity_disjoint.py) compared a chronological
split against an entity-disjoint split. Those two rules assign DIFFERENT rows to test, so
the comparison confounded the split rule with the scoring sample, its size, and its power.
Four independent reviewers caught it, and they were right; one wrote: "a one-change audit
whose one change also changes the scoring sample is not a one-change audit."

THE FIX. Hold the scored rows FIXED and vary only what the fitted arms are allowed to
learn from:
  * TEST ROWS: the entity-disjoint test bucket's rows, in the test window. Identical in
    both conditions, row for row.
  * CONDITION A (entity SEEN): the fitted arms may train on earlier rows of THOSE SAME
    businesses — the leak the field worries about.
  * CONDITION B (entity UNSEEN): the fitted arms train only on other businesses — the
    field's standard fix (group-wise CV).
Everything else is held: same rows scored, same labels, same prompts, same reference,
same combiner, same clustering, same horizons. The zero-shot arms' prediction vectors are
byte-identical across conditions by construction — that is the point being demonstrated,
not an assumption.

So the only thing that moves is whether a test business's own history was available to
the fitted arms. Now the p-values ARE comparable, because n is the same.

PREDICTION (recorded before running, as the falsifier for the claim):
  going from SEEN to UNSEEN should cost the fitted arms and the entity-mean control their
  entity edge, and should cost the zero-shot probe NOTHING, because the probe's identity
  knowledge is pretrained and was never in our training rows to withhold.
FALSIFIED IF: the probe's gain also collapses (then splits DO reach pretrained priors), or
the fitted arms keep their edge (then the fix does not work and the contrast is vacuous).

Run:  .venv/bin/python scripts/experiments/second_domain/yelp_fixed_rows_audit.py
"""
from __future__ import annotations

import hashlib
import importlib.util as _ilu
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

_spec = _ilu.spec_from_file_location("yproto", str(Path(__file__).with_name("yelp_protocol.py")))
_yproto = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_yproto)
dm_test_month, dm_test_2way, monthly_mean = _yproto.dm_test_month, _yproto.dm_test_2way, _yproto.monthly_mean

PANEL = "/Volumes/Z/second-domain/yelp/run1/yelp_panel.parquet"
PREDS = Path("results/second_domain/preds")
FEATS = ["ar_last_mean", "ar_roll3_mean", "ar_roll12_mean", "ar_log_n_reviews"]
CLIP_LO, CLIP_HI, EPS = 1.0, 5.0, 1e-8
ALPHAS_AR = tuple(float(a) for a in np.logspace(-3, 3, 13))
ALPHAS_TXT = tuple(float(a) for a in np.logspace(-1, 4, 11))
N_BOOT = 2000
N_DRAWS = 12   # size-matched redraws of condition B
RNG = np.random.default_rng(2026)


def clip(x):
    return np.clip(np.asarray(x, float), CLIP_LO, CLIP_HI)


def mse(y, f):
    return float(np.mean((np.asarray(y, float) - np.asarray(f, float)) ** 2))


def se(y, f):
    return (np.asarray(y, float) - np.asarray(f, float)) ** 2


def bucket(eid: str) -> int:
    return int(hashlib.sha256(eid.encode()).hexdigest()[:8], 16) % 100


def fit_ridge(Xtr, ytr, Xv, yv, alphas):
    best = None
    for a in alphas:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        v = mse(yv, clip(m.predict(Xv)))
        if best is None or v < best[0]:
            best = (v, m)
    return best[1]


def log_combo(yv, fpv, ftv, fpt, ftt):
    ly = np.log(np.clip(yv, EPS, None))
    def A(p, t):
        return np.column_stack([np.ones(len(p)), np.log(np.clip(p, EPS, None)),
                                np.log(np.clip(t, EPS, None))])
    bU, *_ = np.linalg.lstsq(A(fpv, ftv), ly, rcond=None)
    bR, *_ = np.linalg.lstsq(
        np.column_stack([np.ones(len(fpv)), np.log(np.clip(fpv, EPS, None))]), ly, rcond=None)
    fR = clip(np.exp(bR[0] + bR[1] * np.log(np.clip(fpt, EPS, None))))
    fU = clip(np.exp(A(fpt, ftt) @ bU))
    return fR, fU


def rel_ci(lR, lU, months, n_boot=N_BOOT):
    """Month-block bootstrap CI for the relative gain 100*(mean lR - mean lU)/mean lR.
    Resamples whole months, which is the clustering unit the DM uses."""
    dfm = pd.DataFrame({"lR": lR, "lU": lU, "m": pd.PeriodIndex(pd.to_datetime(months), freq="M")})
    g = dfm.groupby("m")[["lR", "lU"]].mean()
    R, U = g.lR.to_numpy(), g.lU.to_numpy()
    n = len(R)
    point = 100.0 * (R.mean() - U.mean()) / R.mean()
    idx = RNG.integers(0, n, size=(n_boot, n))
    Rb, Ub = R[idx].mean(1), U[idx].mean(1)
    boots = 100.0 * (Rb - Ub) / Rb
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def run_condition(train_pool, va, te, prompted_cols, cond, h, rows, draw=0):
    """Fit AR + TF-IDF on train_pool, score on the FIXED te rows."""
    ytr, yv, yt = (s.label.to_numpy(float) for s in (train_pool, va, te))
    sc = StandardScaler().fit(train_pool[FEATS])
    ar = fit_ridge(sc.transform(train_pool[FEATS]), ytr, sc.transform(va[FEATS]), yv, ALPHAS_AR)
    f_ar_v, f_ar_t = clip(ar.predict(sc.transform(va[FEATS]))), clip(ar.predict(sc.transform(te[FEATS])))

    seen = pd.concat([train_pool, va])
    emap = seen.groupby("entity_id").label.mean()
    gmean = float(seen.label.mean())
    em_v = va.entity_id.map(emap).fillna(gmean).to_numpy(float)
    em_t = te.entity_id.map(emap).fillna(gmean).to_numpy(float)
    cov = float(te.entity_id.isin(emap.index).mean())

    months_t, ents_t = te.event_time.to_numpy(), te.entity_id.to_numpy()

    def score(arm, kind, ftv, ftt):
        fR, fU = log_combo(yv, f_ar_v, ftv, f_ar_t, ftt)
        lR, lU = se(yt, fR), se(yt, fU)
        pt, lo, hi = rel_ci(lR, lU, months_t)
        dm, p, nm = dm_test_month(lU, lR, months_t, h)
        tw = dm_test_2way(lU - lR, ents_t, months_t, h)
        rows.append({"condition": cond, "draw": draw, "h": h, "arm": arm, "kind": kind,
                     "rel_pct": pt, "ci_lo": lo, "ci_hi": hi,
                     "dm_month": dm, "p_month": p, "n_months": nm,
                     "dm_2way": tw["dm_2way"], "p_2way": tw["p_2way"],
                     "entity_coverage_test": cov, "n_test": len(te),
                     "n_train": len(train_pool)})

    score("entity-mean control (zero text)", "identity", em_v, em_t)
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)
    Ttr = vec.fit_transform(train_pool.text)
    txt = fit_ridge(Ttr, ytr, vec.transform(va.text), yv, ALPHAS_TXT)
    score("TF-IDF (fitted)", "fitted",
          clip(txt.predict(vec.transform(va.text))), clip(txt.predict(vec.transform(te.text))))
    for arm, col in prompted_cols:
        score(arm, "zero-shot", va[col].to_numpy(float), te[col].to_numpy(float))


def main():
    panel = pd.read_parquet(PANEL)
    pro = {}
    for arm, f in (("llm70_chrono", "preds_llm70_chrono.parquet"),
                   ("llm70_probe", "preds_llm70_probe.parquet")):
        p = pd.read_parquet(PREDS / f)[["entity_id", "event_time", "horizon_months", "prediction"]]
        pro[arm] = p.rename(columns={"prediction": arm})
    m = pro["llm70_chrono"].merge(pro["llm70_probe"],
                                  on=["entity_id", "event_time", "horizon_months"], validate="1:1")
    panel = panel.merge(m, on=["entity_id", "event_time", "horizon_months"], validate="1:1")
    panel["b"] = panel.entity_id.map(bucket)

    # Three time windows. W3 (latest) is the FIXED evaluation set: rows of the held-out
    # entity bucket. Both conditions score exactly these rows.
    q1, q2 = panel.event_time.quantile([0.40, 0.65])
    W1 = panel.event_time <= q1
    W2 = (panel.event_time > q1) & (panel.event_time <= q2)
    W3 = panel.event_time > q2
    TEST_E = panel.b >= 80          # held-out entities
    OTHER_E = panel.b < 60          # never scored

    # Restrict the fixed evaluation set to entities that HAVE history: for an entity with
    # no earlier rows there is nothing to leak, so both conditions would be identical and
    # the row only dilutes the contrast. This is the population the claim is about.
    _hist = set(panel[TEST_E & (W1 | W2)].entity_id)
    fixed_test = panel[TEST_E & W3 & panel.entity_id.isin(_hist)]
    # Condition A: the whole fitted pipeline (arm, alpha, combiner, entity mean) is
    # entity-matched to the test set — the leak the field worries about.
    A_train, A_val = panel[TEST_E & W1], panel[TEST_E & W2]
    # Condition B: the whole fitted pipeline never sees a test entity — group-wise CV.
    B_train, B_val = panel[OTHER_E & W1], panel[OTHER_E & W2]

    print(f"FIXED test rows: {len(fixed_test):,} over {fixed_test.entity_id.nunique():,} businesses"
          f" — identical in both conditions")
    print(f"A (entity SEEN)  : train {len(A_train):,} / val {len(A_val):,}, "
          f"{A_train.entity_id.nunique():,} businesses = the test businesses")
    print(f"B (entity UNSEEN): train {len(B_train):,} / val {len(B_val):,}, "
          f"{B_train.entity_id.nunique():,} businesses, disjoint from test")
    assert not (set(B_train.entity_id) & set(fixed_test.entity_id)), "B leaked a test entity"
    assert not (set(B_val.entity_id) & set(fixed_test.entity_id)), "B val leaked a test entity"
    assert set(fixed_test.entity_id) <= set(A_train.entity_id) | set(A_val.entity_id), \
        "A must have history for every scored entity"

    prompted = [("70B prompted (zero-shot)", "llm70_chrono"),
                ("70B identity probe (zero-content)", "llm70_probe")]
    rows = []
    for h in sorted(panel.horizon_months.unique()):
        f_te = fixed_test[fixed_test.horizon_months == h]
        if len(f_te) < 200:
            continue
        a_tr = A_train[A_train.horizon_months == h]
        a_va = A_val[A_val.horizon_months == h]
        if min(len(a_tr), len(a_va)) < 200:
            continue
        run_condition(a_tr, a_va, f_te, prompted, "A: entity SEEN in training", int(h), rows)

        # SIZE-MATCHED condition B. The first version let B train on 3x more rows and
        # entities than A, so "the fix" was confounded with more data — which is the
        # opposite-signed confound to the one the reviewers caught. Here B draws a random
        # subset of OTHER entities sized to A's entity count, repeated N_DRAWS times, so
        # the only systematic difference left is whether the scored entities are in train.
        n_ent_a = a_tr.entity_id.nunique()
        pool_ents = np.array(sorted(set(B_train.entity_id) & set(B_val.entity_id)))
        for draw in range(N_DRAWS):
            rng = np.random.default_rng(3000 + draw)
            pick = set(rng.choice(pool_ents, size=min(n_ent_a, len(pool_ents)), replace=False))
            b_tr = B_train[(B_train.horizon_months == h) & B_train.entity_id.isin(pick)]
            b_va = B_val[(B_val.horizon_months == h) & B_val.entity_id.isin(pick)]
            if min(len(b_tr), len(b_va)) < 200:
                continue
            run_condition(b_tr, b_va, f_te, prompted,
                          "B: entity UNSEEN (size-matched)", int(h), rows, draw=draw)

    df = pd.DataFrame(rows)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/tables/yelp_fixed_rows_audit.csv", index=False)
    print("\n=== SAME test rows; B size-matched to A and redrawn "
          f"{N_DRAWS}x (median [min,max] over draws) ===")
    for h in sorted(df.h.unique()):
        print(f"\n-- h={h} months, n_test={int(df[df.h==h].n_test.iloc[0])} --")
        for arm in ["70B identity probe (zero-content)", "70B prompted (zero-shot)",
                    "TF-IDF (fitted)", "entity-mean control (zero text)"]:
            A = df[(df.h == h) & (df.arm == arm) & df.condition.str.startswith("A")]
            B = df[(df.h == h) & (df.arm == arm) & df.condition.str.startswith("B")]
            if not len(A) or not len(B):
                continue
            a0 = A.iloc[0]
            sig = int((B.p_month < 0.05).sum())
            print(f"  {arm:34s} A {a0.rel_pct:+7.3f}% (p={a0.p_month:.3f})"
                  f"  ->  B {B.rel_pct.median():+7.3f}% "
                  f"[{B.rel_pct.min():+.3f},{B.rel_pct.max():+.3f}]  "
                  f"sig in {sig}/{len(B)} draws  median n_train={int(B.n_train.median())}"
                  f" vs A {int(a0.n_train)}")
    print("\n=== entity-mean control coverage of the (fixed) test entities ===")
    print(df.groupby("condition").entity_coverage_test.first().to_string())
    print("\nwrote results/tables/yelp_fixed_rows_audit.csv")


if __name__ == "__main__":
    main()
