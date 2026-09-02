# 70B zero-content date+ticker probe — prereg B2 rider (prereg-rfa v1.3)

Reconciles the Table 6 probe cell (date+ticker reproducing >100% of a fulltext increment on the Qwen arm) with the llama70 replication claim, by running the SAME zero-content probe through the replication family itself. **Descriptive readout — no prereg branch fires; Holm only within blocks** (the llm_contamination.py template convention).

## Disclosures

- **Model / precision**: hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 (weight-only **AWQ-INT4** — the SAME precision as the committed llama70 fulltext runs, so the probe-vs-fulltext comparison is internally consistent; disclosed exactly as the committed runs disclose it). vLLM TP=2, temperature-0 protocol, guided JSON, clip [0.03,3.0], on_missing=rv22 — byte-identical stack to the committed runs.
- **Prompt**: the C6 contamination arm's `c6_datefirm` template VERBATIM (scripts/experiments/e1_llm_forecast/prompt.py): form type + items, filing date, ticker, the line "(No filing text is provided.)" and the identical task text — the SAME fields, NO document text. This is the prompt behind the committed C6_datefirm (Qwen) runs in llm_contamination.md.
- **Seed**: single seed 2026 (the probe is a control readout; the committed fulltext anchor is the 3-seed ensemble — llama70 seed jitter moved rel% by <0.04pp, see crossfamily_llama70_ens.csv — and single-seed fulltext denominators are reported as robustness).
- **Panel**: the identical event_driven panel as the committed crossfamily_llama70 runs (verified by GP2: equal n_test per horizon).
- **>100% convention**: probe-share = probe rel% / committed fulltext rel% per cell; a value ABOVE 100% means the zero-content probe alone reproduces MORE than the fulltext increment in that cell. Shares are quotable only where the denominator is well-identified (fulltext rel% >= 1% AND raw-significant, the llm_contamination stable-denominator rule) — flagged per cell; small denominators inflate shares mechanically.
- **No new Holm family**: Holm within the probe M1 block (6 cells) and within the beyond-identity block (3 cells) only, as the template does; committed anchor cells carry their committed values unchanged (GP1-anchored, not re-derived).

## 1. Probe M1 vs the committed fulltext increment (side by side)

rel% on volatility-unit QLIKE; `**` = clustered DM<0, raw p<.05. fulltext = committed llama70_awq_ens3 (crossfamily_llama70_ens.csv).

| h | probe rel% vs HAR | DM | probe rel% vs HAR+firmID | DM | fulltext rel% vs HAR | fulltext rel% vs HAR+firmID | probe-share vs HAR | probe-share vs firmID | well-identified denom |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| 5 | +0.07% | -1.62 | +0.00% | +0.42 | +1.39% | +0.84% | 5% | 0% | yes |
| 10 | +0.08% | -1.55 | +0.00% | +0.36 | +1.16% | +0.64% | 7% | 0% | yes |
| 20 | +0.22%** | -3.54 | +0.02% | -1.68 | +0.69% | +0.38% | 31% | 6% | NO (do not quote) |

(probe Holm(6) within block: min Holm p vs HAR = 0.002467, vs firmID = 0.4665; single-seed fulltext robustness denominators in the csv: share_har_pct_vs_single / share_firm_pct_vs_single.)

## 2. Text beyond identity — joint reference [1, log fHAR, log f_probe] (+ log f_fulltext)

The same-model identity control for the REPLICATION family: the reference already contains everything llama70 produces from date+ticker alone, so any residual fulltext increment must come from the filing text.

| h | n_test | n_days | QLIKE(R') | QLIKE(U') | rel% | DM(clu) | p raw | p Holm(3) | g_text | retained share of fulltext-vs-HAR rel% |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 5 | 25109 | 996 | 0.1264 | 0.1247 | +1.37%** | -3.17 | 1.55e-03 | 0.0046 | +0.160 | 98% |
| 10 | 25001 | 991 | 0.0882 | 0.0872 | +1.14%** | -2.27 | 2.32e-02 | 0.0465 | +0.139 | 98% |
| 20 | 24732 | 981 | 0.0644 | 0.0640 | +0.65% | -0.75 | 4.51e-01 | 0.4512 | +0.112 | 94% |

## Bottom line (descriptive — feeds the Table 6 reconciliation sentence)

- The zero-content probe carries a positive vs-HAR increment in 3/3 horizons (identity/era memory, zero filing content); its share of the committed fulltext increment is the probe-share column above (>100% possible by convention).
- With the same-model date+ticker forecast INSIDE the reference, the committed llama70-ens fulltext still adds in 2/3 horizons at raw p<.05 (2/3 after within-block Holm(3)), retaining 94%-98% of the uncontrolled fulltext-vs-HAR rel% per cell — this text-beyond-identity readout, not the raw probe-share, is the number the replication claim rests on.
- No prereg branch fires on this table (registered as descriptive; prereg-rfa v1.3 §B2 rider).

## SANITY

- GP1 PASS (G1'' convention): all 6 committed crossfamily_llama70_ens.csv rows (single-seed AND ens3) reproduced on this exact code path to machine precision (rtol 1e-12) on columns ['n_test', 'n_days', 'rel_har', 'dm_har', 'p_har', 'rel_firm', 'dm_firm', 'p_firm', 'g_text']; anchor cells then carried unchanged.
- GP2 PASS: probe cells sit on the identical test panel as the committed llama70 cells (n_test = 25109/25001/24732).
