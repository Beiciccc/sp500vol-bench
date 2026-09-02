import sys, numpy as np, pandas as pd
sys.path.insert(0,"src")
from sp500vol.evaluation.dm_test import dm_test
KEY=["ticker","accession","horizon_days"]; EPS=1e-8
def preds(run,disc):
    p=pd.read_parquet(f"results/runs/{run}_full_{disc}_seed2026/predictions.parquet"); return p[p.split=="test"]
def qlike(y,f):
    y=np.clip(np.asarray(y,float),EPS,None); f=np.clip(np.asarray(f,float),EPS,None); return y/f-np.log(y/f)-1.0
SETS={"long_form":["B1_bow_ridge","B2_tfidf_ridge","B3_lm_linear","B4_lm_features","C1_bert_s1","C2_finbert_s1","C2_finbert_s2","C2_finbert_s3","C2_finbert_s4","C3_roberta_s1","C4_longformer","D1_concat_mlp","D2_gated_fusion"],
      "event_driven":["B1_bow_ridge","B2_tfidf_ridge","B3_lm_linear","B4_lm_features","C2_finbert_s1","D2_gated_fusion"]}
def run(loss):
    rows=[]
    for disc,chs in SETS.items():
        base=preds("A2_har_rv",disc)[KEY+["prediction_realised_vol","label_realised_vol","filing_time_utc"]].rename(columns={"prediction_realised_vol":"pb"})
        for ch in chs:
            c=preds(ch,disc)[KEY+["prediction_realised_vol"]].rename(columns={"prediction_realised_vol":"pc"})
            m=base.merge(c,on=KEY).sort_values("filing_time_utc")
            for h in (5,10,20):
                s=m[m.horizon_days==h]; y=s.label_realised_vol.to_numpy()
                if loss=="se": lc=(s.pc.to_numpy()-y)**2; lb=(s.pb.to_numpy()-y)**2
                else: lc=qlike(y,s.pc.to_numpy()); lb=qlike(y,s.pb.to_numpy())
                stat,p=dm_test(lc,lb,h=h); rows.append([disc,ch,h,stat,p])
    return pd.DataFrame(rows,columns=["disc","model","h","DM","p"])
def holm(ps):
    ps=np.asarray(ps); n=len(ps); order=np.argsort(ps); out=np.empty(n)
    for rank,idx in enumerate(order): out[idx]=ps[idx]*(n-rank)
    # enforce monotonicity
    run=0
    for rank,idx in enumerate(order):
        run=max(run,out[idx]); out[idx]=min(run,1.0)
    return out
def bh(ps):
    ps=np.asarray(ps); n=len(ps); order=np.argsort(ps); out=np.empty(n)
    prev=1.0
    for rank in range(n-1,-1,-1):
        idx=order[rank]; val=ps[idx]*n/(rank+1); prev=min(prev,val); out[idx]=min(prev,1.0)
    return out
md_q=["# DM robustness: QLIKE-loss vs A2_har_rv (test)\n","Positive DM = worse than HAR-RV. n.s. = p>=0.05.\n"]
for loss in ("se","qlike"):
    df=run(loss); df["holm"]=holm(df.p.values); df["bh"]=bh(df.p.values)
    nsig_raw=(df.p<0.05).sum(); nsig_holm=(df.holm<0.05).sum(); nsig_bh=(df.bh<0.05).sum()
    print(f"\n===== loss={loss.upper()}  family n={len(df)} =====")
    print(f"raw sig(p<.05)={nsig_raw}  Holm sig={nsig_holm}  BH sig={nsig_bh}")
    print("NON-significant after RAW:")
    for _,r in df[df.p>=0.05].iterrows(): print(f"   {r.disc:13s} {r.model:16s} h{int(r.h):<2d} DM={r.DM:+.2f} p={r.p:.4f} holm={r.holm:.4f} bh={r.bh:.4f}")
    if loss=="qlike":
        md_q.append("\n| disclosure | model | h | DM | p | Holm | BH |\n|---|---|---|---|---|---|---|")
        for _,r in df.iterrows():
            md_q.append(f"| {r.disc} | {r.model} | {int(r.h)} | {r.DM:+.2f} | {r.p:.4f} | {r.holm:.4f} | {r.bh:.4f} |")
    df.to_csv(f"results/tables/dm_{loss}_all_vs_A2.csv",index=False)
open("results/tables/dm_full_vs_A2_qlike.md","w").write("\n".join(md_q))
print("\nwrote results/tables/dm_full_vs_A2_qlike.md + dm_{se,qlike}_all_vs_A2.csv")
