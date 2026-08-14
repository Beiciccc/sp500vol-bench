from math import erf, sqrt

import numpy as np
import pandas as pd

KEY=["ticker","accession","horizon_days"]
def preds(run,disc):
    p=pd.read_parquet(f"results/runs/{run}_full_{disc}_seed2026/predictions.parquet")
    return p[p.split=="test"]
def norm_p(t):  # two-sided normal p
    return 2*(1-0.5*(1+erf(abs(t)/sqrt(2))))
def ols_hac(y,X,L):
    XtXi=np.linalg.inv(X.T@X); beta=XtXi@X.T@y; r=y-X@beta
    S=X*r[:,None]; Om=S.T@S
    for l in range(1,L+1):
        w=1-l/(L+1); G=S[l:].T@S[:-l]; Om+=w*(G+G.T)
    cov=XtXi@Om@XtXi; return beta, np.sqrt(np.diag(cov))
TEXT={"long_form":["B1_bow_ridge","B2_tfidf_ridge","B3_lm_linear","B4_lm_features","C1_bert_s1","C2_finbert_s1","C2_finbert_s2","C2_finbert_s3","C2_finbert_s4","C3_roberta_s1","C4_longformer"],
      "event_driven":["B1_bow_ridge","B2_tfidf_ridge","B3_lm_linear","B4_lm_features","C2_finbert_s1"]}
md=["# E1 Forecast-encompassing regression: RV = a + b*f_HAR + g*f_text + e (test, HAC lag h-1)",
    "g = coefficient on the text-only forecast conditional on the HAR-RV forecast. g not significant (p>=0.05) => text carries NO incremental information beyond HAR-RV.\n"]
total=0; sig_pos=0; sig_neg=0
for disc,models in TEXT.items():
    har=preds("A2_har_rv",disc)[KEY+["prediction_realised_vol","label_realised_vol","filing_time_utc"]].rename(columns={"prediction_realised_vol":"fhar"})
    print(f"\n=== {disc} ===")
    md.append(f"\n## {disc}\n| model | h | g (text coef) | HAC t | p | verdict |\n|---|---|---|---|---|---|")
    for m in models:
        c=preds(m,disc)[KEY+["prediction_realised_vol"]].rename(columns={"prediction_realised_vol":"ftext"})
        d=har.merge(c,on=KEY).sort_values("filing_time_utc")
        line=f"  {m:16s}"
        for h in (5,10,20):
            s=d[d.horizon_days==h]; y=s.label_realised_vol.to_numpy()
            X=np.column_stack([np.ones(len(s)),s.fhar.to_numpy(),s.ftext.to_numpy()])
            beta,se=ols_hac(y,X,max(1,h-1)); g=beta[2]; t=g/se[2]; p=norm_p(t)
            total+=1
            if p<0.05 and g>0: sig_pos+=1; v="text adds (g>0,sig)"
            elif p<0.05 and g<0: sig_neg+=1; v="text harms (g<0,sig)"
            else: v="no incremental (n.s.)"
            line+=f"  h{h}:g={g:+.3f} t={t:+.2f} p={p:.3f}"
            md.append(f"| {m} | {h} | {g:+.3f} | {t:+.2f} | {p:.4f} | {v} |")
        print(line)
print(f"\nTOTAL {total} regressions: g sig POSITIVE (text adds)={sig_pos}; g sig NEGATIVE (text harms)={sig_neg}; no incremental (n.s.)={total-sig_pos-sig_neg}")
md.append(f"\n**Summary:** of {total} encompassing tests, {sig_pos} show a significantly positive text coefficient (incremental value), {sig_neg} significantly negative, {total-sig_pos-sig_neg} no significant incremental information beyond HAR-RV.")
open("results/tables/encompassing_regression.md","w").write("\n".join(md))
print("wrote results/tables/encompassing_regression.md")
