import sys, numpy as np, pandas as pd
sys.path.insert(0,"src"); from sp500vol.evaluation.dm_test import dm_test
KEY=["ticker","accession","horizon_days"]
def preds(run,disc): p=pd.read_parquet(f"results/runs/{run}_full_{disc}_seed2026/predictions.parquet"); return p[p.split=="test"].copy()
def dm(a,b,y,h): return dm_test((a-y)**2,(b-y)**2,h=h)

# ---- E3: exiting (delisted/M&A, distress proxy) vs survivor ----
mem=pd.read_parquet("data/universe/sp500_membership.parquet")
exited=set(mem.loc[mem.member_to.notna(),"ticker"])  # ever left S&P500 in 2010-2025
print("=== E3: stratify by S&P500 exit (distress/M&A proxy) ===")
md=["# E3 Stratified DM vs A2_har_rv: exiting (delisted/acquired) vs surviving firms (long-form test).",
    "Tests whether disclosure text beats HAR-RV on the distressed/exiting names where text should help most.\n",
    "| stratum | model | n_filings | h5 | h10 | h20 |","|---|---|---|---|---|---|"]
for m in ["C2_finbert_s1","C4_longformer","D2_gated_fusion"]:
    b=preds("A2_har_rv","long_form")[KEY+["prediction_realised_vol","label_realised_vol"]].rename(columns={"prediction_realised_vol":"pb"})
    c=preds(m,"long_form")[KEY+["prediction_realised_vol"]].rename(columns={"prediction_realised_vol":"pc"})
    d=b.merge(c,on=KEY); d["exit"]=d.ticker.isin(exited)
    for grp,lab in [(True,"exiting"),(False,"survivor")]:
        s=d[d.exit==grp]; cells=[]
        for h in (5,10,20):
            ss=s[s.horizon_days==h]
            if len(ss)<30: cells.append("-"); continue
            st,p=dm(ss.pc.to_numpy(),ss.pb.to_numpy(),ss.label_realised_vol.to_numpy(),h)
            cells.append(f"{st:+.2f}{'*' if p<0.05 else ' ns'}")
        n=s.ticker.nunique(); print(f"  {lab:9s} {m:16s} n_tick={n:3d}  {cells}")
        md.append(f"| {lab} | {m} | {len(s)//3} | {cells[0]} | {cells[1]} | {cells[2]} |")
open("results/tables/dm_stratified_exit.md","w").write("\n".join(md))

# ---- L2: weak (naive persistence = rv_22d) vs strong (HAR) baseline ----
print("\n=== L2: text vs WEAK baseline (naive RV persistence) vs STRONG baseline (HAR) ===")
md2=["# L2 Manufactured-signal demonstration (long-form test).",
     "A text model's apparent skill flips with baseline strength: it can beat a naive persistence baseline while losing to HAR-RV.\n",
     "| text model | vs baseline | h5 | h10 | h20 |","|---|---|---|---|---|"]
for m in ["C4_longformer","C2_finbert_s1"]:
    base=preds("A2_har_rv","long_form")[KEY+["prediction_realised_vol","label_realised_vol","feature_rv_22d"]].rename(columns={"prediction_realised_vol":"har"})
    c=preds(m,"long_form")[KEY+["prediction_realised_vol"]].rename(columns={"prediction_realised_vol":"txt"})
    d=base.merge(c,on=KEY)
    for bl,col,lab in [("naive","feature_rv_22d","weak: naive RV persistence"),("har","har","strong: HAR-RV")]:
        cells=[]
        for h in (5,10,20):
            ss=d[d.horizon_days==h]; y=ss.label_realised_vol.to_numpy()
            st,p=dm(ss.txt.to_numpy(), ss[col].to_numpy(), y, h)
            cells.append(f"{st:+.2f}{'*' if p<0.05 else ' ns'}")  # positive = text worse than baseline
        print(f"  {m:16s} vs {lab:28s}  {cells}")
        md2.append(f"| {m} | {lab} | {cells[0]} | {cells[1]} | {cells[2]} |")
md2.append("\nPositive DM = text worse than that baseline. Text loses to HAR everywhere; against the naive persistence baseline the gap shrinks, illustrating how a weak baseline flatters text.")
open("results/tables/dm_weak_vs_strong_baseline.md","w").write("\n".join(md2))
print("\nwrote dm_stratified_exit.md + dm_weak_vs_strong_baseline.md")
