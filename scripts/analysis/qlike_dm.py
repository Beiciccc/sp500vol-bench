import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from sp500vol.evaluation.dm_test import dm_test
KEY = ["ticker","accession","horizon_days"]; EPS=1e-8
def preds(run, disc):
    p = pd.read_parquet(f"results/runs/{run}_full_{disc}_seed2026/predictions.parquet")
    return p[p.split=="test"]
def qlike(y,f):
    y=np.clip(np.asarray(y,float),EPS,None); f=np.clip(np.asarray(f,float),EPS,None)
    return y/f - np.log(y/f) - 1.0
SETS = {
 "long_form": ["B1_bow_ridge","B2_tfidf_ridge","B3_lm_linear","B4_lm_features","C1_bert_s1",
   "C2_finbert_s1","C2_finbert_s2","C2_finbert_s3","C2_finbert_s4","C3_roberta_s1","C4_longformer",
   "D1_concat_mlp","D2_gated_fusion"],
 "event_driven": ["B1_bow_ridge","B2_tfidf_ridge","B3_lm_linear","B4_lm_features","C2_finbert_s1","D2_gated_fusion"],
}
tot=0; nonsig=[]
for disc, chs in SETS.items():
    base = preds("A2_har_rv", disc)[KEY+["prediction_realised_vol","label_realised_vol","filing_time_utc"]].rename(columns={"prediction_realised_vol":"pb"})
    print(f"\n=== QLIKE-DM vs A2_har_rv  ({disc}, test) ===")
    for ch in chs:
        c = preds(ch, disc)[KEY+["prediction_realised_vol"]].rename(columns={"prediction_realised_vol":"pc"})
        m = base.merge(c, on=KEY).sort_values("filing_time_utc")
        line=f"  {ch:16s}"
        for h in (5,10,20):
            s=m[m.horizon_days==h]; y=s.label_realised_vol.to_numpy()
            stat,p = dm_test(qlike(y,s.pc.to_numpy()), qlike(y,s.pb.to_numpy()), h=h)
            tot+=1; flag="**" if p<0.01 else ("*" if p<0.05 else "ns")
            if p>=0.05: nonsig.append((disc,ch,h,round(stat,3),round(p,4)))
            line+=f"  h{h}:DM={stat:+6.2f} p={p:.4f}{flag}"
        print(line)
print(f"\nTOTAL tests={tot}  NON-SIGNIFICANT(p>=0.05): {len(nonsig)}")
for x in nonsig: print("  PARITY:",x)
