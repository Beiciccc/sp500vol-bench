import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0,"src")
import sp500vol.models.fusion.gated_fusion as gf

GATES={}  # horizon -> list of (1-g) arrays
_orig=gf._GatedFusion.forward
_cur={"h":None}
def hooked(self,price,text):
    p=self.price_proj(price); t=self.text_proj(text)
    g=torch.sigmoid(self.gate(torch.cat([p,t],dim=-1)))
    GATES.setdefault(_cur["h"],[]).append((1.0-g).detach().cpu().numpy())  # text-branch weight
    return g*p+(1.0-g)*t
gf._GatedFusion.forward=hooked
m=gf.GatedFusion.load("results/runs/D2_gated_fusion_full_long_form_seed2026/model.pkl")
p=pd.read_parquet("results/runs/D2_gated_fusion_full_long_form_seed2026/predictions.parquet")
test=p[p.split=="test"]
N=300
rows=[]
for h in (5,10,20):
    sub=test[test.horizon_days==h].head(N).copy()
    _cur["h"]=h
    GATES[h]=[]
    _=m.predict(sub)  # triggers hooked forward, fills GATES[h]
    tw=np.concatenate([a.reshape(-1) for a in GATES[h]])  # all (1-g) values across samples x dims
    rows.append([h,len(sub),round(float(tw.mean()),4),round(float(np.median(tw)),4),
                 round(float(np.percentile(tw,25)),4),round(float(np.percentile(tw,75)),4),
                 round(float((tw<0.5).mean()),4)])
    print(f"h{h}: n={len(sub)} text-weight(1-g) mean={tw.mean():.4f} median={np.median(tw):.4f} IQR=[{np.percentile(tw,25):.3f},{np.percentile(tw,75):.3f}] frac(text<0.5 i.e. price-dominant)={ (tw<0.5).mean():.3f}")
df=pd.DataFrame(rows,columns=["horizon","n_sampled","text_wt_mean","text_wt_median","text_wt_q25","text_wt_q75","frac_price_dominant"])
md=["# D2 gated-fusion learned gate: text-branch weight (1-g) on long-form test sample\n",
    "g=sigmoid(gate([price,text])); output = g*price + (1-g)*text. Lower (1-g) => gate down-weights text.\n",
    df.to_markdown(index=False)]
open("results/tables/gate_weight_readout.md","w").write("\n".join(md))
df.to_csv("results/tables/gate_weight_readout.csv",index=False)
print("\nwrote results/tables/gate_weight_readout.md")
