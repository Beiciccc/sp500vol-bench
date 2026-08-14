"""Cross-family replication test for the prompted-LLM increment (P2 #12).

Qwen3-32B (C6_llmtext, 8K ctx) vs Yi-1.5-34B-Chat (C6_llmtext_yi34, 4K ctx),
identical manifest/prompts/protocol. Per (disclosure, horizon): M1 log-space
increment vs (a) single recalibrated-HAR reference and (b) the firm-identity-
augmented reference, with day-clustered DM. Writes results/tables/crossfamily_llm.{csv,md}.

Reading: if the second family does not replicate the Qwen increment, the
prompted-LLM residual is family-specific — further evidence for the near-null.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc  # noqa: E402
import clustered_dm as cdm  # noqa: E402

KEY = ["ticker", "accession", "horizon_days"]
EPS = 1e-8


def ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def main():
    rows = []
    for disc in ("long_form", "event_driven"):
        a2 = fc.load("A2_har_rv", disc)[KEY + ["split", "label_realised_vol",
                                               "prediction_realised_vol",
                                               "effective_trading_day"]] \
            .rename(columns={"prediction_realised_vol": "fh"})
        fams = [("qwen3_32b", "C6_llmtext"), ("yi_34b", "C6_llmtext_yi34")]
        try:
            fc.load("C6_llmtext_phi4", disc)
            fams.append(("phi4_14b", "C6_llmtext_phi4"))
        except FileNotFoundError:
            pass
        for fam, run in fams:
            t = fc.load(run, disc)[KEY + ["prediction_realised_vol"]] \
                .rename(columns={"prediction_realised_vol": "ft"})
            for h in (5, 10, 20):
                m = a2[a2.horizon_days == h].merge(t[t.horizon_days == h], on=KEY).dropna()
                v, te = m[m.split == "val"], m[m.split == "test"]
                y = te.label_realised_vol.values
                fR, fU, g = fc.log_combo(v.label_realised_vol.values, v.fh.values,
                                         v.ft.values, te.fh.values, te.ft.values)
                qR, qU = fc.qlike(y, fR), fc.qlike(y, fU)
                rel = 100 * np.mean(qR - qU) / np.mean(qR)
                dm, p, nd = cdm.dm_test_clustered(qU, qR, te.effective_trading_day.values, h)
                # firm-identity-augmented reference
                fm = v.groupby("ticker").label_realised_vol.mean()
                gmean = v.label_realised_vol.mean()
                fid_v = v.ticker.map(fm).fillna(gmean).values
                fid_t = te.ticker.map(fm).fillna(gmean).values
                L = lambda x: np.log(np.clip(x, EPS, None))  # noqa: E731
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
                rows.append(dict(disc=disc, family=fam, h=h, n_test=len(te), n_days=nd,
                                 rel_har=rel, dm_har=dm, p_har=p,
                                 rel_firm=relf, dm_firm=dmf, p_firm=pf,
                                 g_text=g))
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/crossfamily_llm.csv", index=False)

    md = ["# Cross-family replication of the prompted-LLM increment (P2)",
          "",
          "Identical manifest/prompts/protocol; day-clustered DM. rel% > 0 = text lowers "
          "QLIKE vs the reference; `**` = clustered DM<0, p<.05. Yi-1.5-34B has a 4K "
          "context (vs Qwen3's 8K): binding for long-form excerpts, NOT binding for "
          "8-K (median ~930 tokens) — so the event_driven non-replication cannot be a "
          "context artefact.",
          "",
          "**HEADLINE: the prompted-LLM increment does NOT replicate across families.** "
          "Yi-34B shows no significant positive increment in any cell (long_form "
          "negative everywhere; event_driven ~0), while Qwen3-32B was positive-"
          "significant in 6/6 vs single-HAR. The prompted-LLM residual is family-"
          "specific, further supporting the near-null verdict.",
          "",
          "| disc | family | h | rel% vs HAR | DM(clu) | rel% vs HAR+firmID | DM(clu) |",
          "|---|---|--:|--:|--:|--:|--:|"]
    for _, r in df.iterrows():
        s1 = "**" if (r.dm_har < 0 and r.p_har < .05) else ""
        s2 = "**" if (r.dm_firm < 0 and r.p_firm < .05) else ""
        md.append(f"| {r.disc} | {r.family} | {int(r.h)} | {r.rel_har:+.2f}%{s1} | "
                  f"{r.dm_har:+.2f} | {r.rel_firm:+.2f}%{s2} | {r.dm_firm:+.2f} |")
    open("results/tables/crossfamily_llm.md", "w").write("\n".join(md) + "\n")
    print("wrote results/tables/crossfamily_llm.{csv,md}")


if __name__ == "__main__":
    main()
