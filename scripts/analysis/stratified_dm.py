import sys, numpy as np, pandas as pd
sys.path.insert(0,"src")
from sp500vol.evaluation.dm_test import dm_test
KEY=["ticker","accession","horizon_days"]
def preds(run,disc):
    p=pd.read_parquet(f"results/runs/{run}_full_{disc}_seed2026/predictions.parquet"); return p[p.split=="test"].copy()
REP={"long_form":["C2_finbert_s1","C4_longformer","D2_gated_fusion"],"event_driven":["C2_finbert_s1","D2_gated_fusion"]}
def dmrow(base,c,sub):
    out={}
    for h in (5,10,20):
        s=sub[sub.horizon_days==h]
        if len(s)<30: out[h]=None; continue
        y=s.label_realised_vol.to_numpy()
        stat,p=dm_test((s.pc.to_numpy()-y)**2,(s.pb.to_numpy()-y)**2,h=h); out[h]=(round(stat,2),round(p,4))
    return out
md=["# Stratified DM vs A2_har_rv (squared-error, test). Positive=worse than HAR-RV.\n"]
for disc,chs in REP.items():
    b=preds("A2_har_rv",disc)[KEY+["prediction_realised_vol","label_realised_vol","filing_time_utc","effective_trading_day"]].rename(columns={"prediction_realised_vol":"pb"})
    b["year"]=pd.to_datetime(b.effective_trading_day).dt.year.replace({2024:"24-25",2025:"24-25"})
    print(f"\n===== {disc} : DM by sub-period =====")
    md.append(f"\n## {disc}\n| model | period | h5 | h10 | h20 |\n|---|---|---|---|---|")
    for ch in chs:
        c=preds(ch,disc)[KEY+["prediction_realised_vol"]].rename(columns={"prediction_realised_vol":"pc"})
        m=b.merge(c,on=KEY).sort_values("filing_time_utc")
        for yr in [2022,2023,"24-25"]:
            sub=m[m.year==yr]; r=dmrow(b,c,sub)
            cell=lambda h:(f"{r[h][0]:+.2f}{'*' if r[h] and r[h][1]<0.05 else ' ns'}" if r[h] else "-")
            line=f"  {ch:16s} {str(yr):6s}  h5 {cell(5):10s} h10 {cell(10):10s} h20 {cell(20):10s}"
            print(line)
            md.append(f"| {ch} | {yr} | {cell(5)} | {cell(10)} | {cell(20)} |")
open("results/tables/dm_stratified.md","w").write("\n".join(md))
print("\nAny text model BETTER (negative DM, p<0.05) in any slice would show as a negative starred cell above.")
print("wrote results/tables/dm_stratified.md")
