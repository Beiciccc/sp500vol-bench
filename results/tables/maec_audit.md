# MAEC Audit Master Table (single-shot aggregation; prereg-maec v1.2 §5/§6/§8)

Generated: 2026-07-15 15:06:50 · script `scripts/analysis/maec_audit_table.py` (pure aggregation: all DM/placebo/MDE/Holm/bootstrap values are read from the frozen json, zero recomputation; the only arithmetic = gate/branch logic and the §6.4 unit conversion, see the §1 conversion note).

## 0. Single-shot sources (§6.5 provenance)

| json | generated | merge_dropped | crosscheck max\|Δpred\| |
|---|---|---|---|
| protocol_tfidf_primary.json | 2026-07-15 14:38:04 | 0 | 8.9e-16 |
| protocol_tfidf_shifted.json | 2026-07-15 14:38:06 | 0 | 8.9e-16 |
| protocol_qwen_emb_primary.json | 2026-07-15 14:38:09 | 0 | 8.9e-16 |
| protocol_qwen_emb_shifted.json | 2026-07-15 14:38:11 | 0 | 8.9e-16 |
| protocol_prompted_qwen_primary.json | 2026-07-15 14:38:14 | 0 | 8.9e-16 |
| protocol_prompted_qwen_shifted.json | 2026-07-15 14:38:16 | 0 | 8.9e-16 |
| protocol_identity_probe_primary.json | 2026-07-15 14:38:19 | 0 | 8.9e-16 |
| protocol_identity_probe_shifted.json | 2026-07-15 14:38:22 | 0 | 8.9e-16 |
| published_readings.json | 2026-07-15 14:37:38 | — | — |

- All 8 protocol json files: tag=REAL, placebo seeds 1000–1019 (20, label-shuffle), swap seeds 2000–2004 (5, within-date), embargo_val=False, merge_dropped_rows=0 (including shifted: the text arm is fitted once on primary, §2.3, and the row set is handed over complete).
- Per horizon: n_test=672, 143 call-date clusters, 461 entities, n_val=333; HAC lag L_n = 2/6/14/28 (h=3/7/15/30); STPEV (expanding) test prior coverage 90.9%.
- Share of effectively permuted rows under the within-date swap (test) = 92.1% (call dates with a single call are not swapped, §6.3 disclosure).

## 0b. Decision-ladder source text (prereg §8, verbatim quotation)

> ## 8. Decision ladder and branch commitments (the three branches are hard-coded now, binding on the prose)
> 
> Object of adjudication = family F2 of each headline text arm (identity-controlled residual, 8 cells):
> 
> - **(a) FULLY ABSORBED**: 0/8 cells simultaneously satisfy DM<0, Holm<.05, G4 pass; and
>   identity share (d4/d3) ≥ 100% or the combination increment itself is not significant.
>   → Paper wording: "On MAEC, recalibrating the discarded past-vol baseline and then adding the same-ticker
>   STPEV mean, the published-convention text gain is fully absorbed; call-date-clustered inference (never
>   provided in the literature) confirms, at MDE=X% power, that the representation-layer finding of Yu et al. also holds at the forecasting layer —
>   a third domain, the same measuring instrument." (MDE must be reported side by side, §6.4.)
> - **(b) PARTIALLY ABSORBED**: ≥1/8 cells pass Holm+placebo, and on those horizons
>   identity share ≥ 50%.
>   → "Recalibration + identity absorb X–Y% of the published-convention gain; a bounded, placebo-clean,
>   power-calibrated residual survives — isomorphic to the bounded residual on the SEC panel, its size priced by the panel and its baseline."
> - **(c) SURVIVES**: ≥4/8 cells pass Holm+placebo and identity share < 50%.
>   → "The increment of the text arm on MAEC is not an identity artifact: the problem with this benchmark lies in baseline miscalibration, not in
>   identity; we provide the first clustered significance and power calibration on this benchmark. The representation-layer
>   similarity of Yu et al. does not translate into forecasting redundancy at the combination layer." (honest branch; the fallback framing remains
>   "the size of the shortcut is a property of the panel and its baseline" — the Yelp precedent.)
> - **All remaining combinations**: MIXED, reported truthfully cell by cell, with the wording taking the weakest defensible form.
> - **(d) G1 downgrade branch**: all stand-in arms weak → publish only the baseline-recalibration audit (raw V_past 1.12-type
>   reading vs post-recalibration reading + STPEV pricing), with the repricing claim about the text gain withdrawn in its entirety.
> 
> If the two headline arms are adjudicated into different tiers → report arm by arm, do not merge the wording (Yelp precedent: the identity/content
> split between the prompted and the fitted arm is itself the finding). If the §2.3 alignment sensitivity changes
> the tier → primary governs + disclose the difference (OPEN-2 has been adjudicated as maintained; this sentence is the final version).
>

## 1. published-convention context block (G1/G2, descriptive; §6.2: not entered into Holm, not entered into 'win' wording)

- **G1 PASS**: tfidf 12/12 cells, text standalone beats raw V_past^(n) (pooled 0.3498 vs raw 0.6292); prompted_qwen 4/12 (pooled 0.6419, pooled loses to raw). qwen_emb has no published reading (G1 requires only ≥1 arm; the finality note is transcribed verbatim in the json).
- **G2 PASS**: our raw V_past pooled MSE(v) = 0.6292, vs the Yu et al. reference 1.12, ratio 0.562 ∈ [1/3, 3] (order-of-magnitude gate; per-horizon ratios h3=1.30, h7=0.50, h15=0.28, h30=0.16; the panels differ, so no equivalence gate is applied, and §0-4 forbids stating direct comparability).
- inference_note source text: "published-convention readings are DESCRIPTIVE (§6.2): no clustering, no Holm, never 'win' prose"

**§6.4 conversion inputs for the published-convention gain (per horizon, three annual panels merged row-equal-weight)**:

| arm | h | raw pooled MSE | text pooled MSE | Δ_pub = raw−text | % of raw |
|---|---|---|---|---|---|
| tfidf | 3 | 1.4592 | 0.6847 | +0.7746 | +53.1% |
| tfidf | 7 | 0.5647 | 0.3499 | +0.2149 | +38.0% |
| tfidf | 15 | 0.3186 | 0.2140 | +0.1046 | +32.8% |
| tfidf | 30 | 0.1744 | 0.1507 | +0.0236 | +13.6% |
| prompted_qwen | 3 | 1.4592 | 1.0630 | +0.3962 | +27.2% |
| prompted_qwen | 7 | 0.5647 | 0.7260 | -0.1613 | -28.6% |
| prompted_qwen | 15 | 0.3186 | 0.4779 | -0.1594 | -50.0% |
| prompted_qwen | 30 | 0.1744 | 0.3005 | -0.1262 | -72.4% |

**Conversion note (§6.4, source text "convert to the same units")**: the unit of MDE(ent) is "% of the entity-stage reference MSE_Re" (MDE = (1.96+0.84)·SE_date/MSE_Re·100). Therefore the published-convention text gain (text-alone vs raw V_past^(n), the absolute ΔMSE of the published three annual panels merged row-equal-weight, in v² units) is converted into **G_conv(cell) = 100·Δ_pub(arm,h) / MSE_Re(cell)**, which has the same denominator and the same unit as the MDE(ent) of that cell and is therefore directly comparable. Adjudication: Δ_pub≤0 → that arm at that horizon has no published-convention gain to absorb; MDE(ent) ≤ G_conv → powered, the "fully absorbed" wording is licensed; MDE(ent) > G_conv → the wording is downgraded to "underpowered to rule out". Note: the published panels (2015/2016/2017-18, three panels by year) differ from the audit primary test (2017-05..2018-06) in period and in level — transporting an absolute ΔMSE across panels is an inherent approximation of this conversion, disclosed truthfully; the MSE_Re of the shifted alignment is the value under shifted labels, and the conversion uses the same formula.

## 2. Cell-by-cell F2 (identity-controlled residual, row5) + MDE side by side — the three headline arms

F2 pass = DM<0 and Holm(8)<.05 and G4 (label-shuffle) clean; G4b (within-date swap) is diagnostic, and dirty cells do not enter prose claims (§6.3, Yelp precedent thresholds |mean DM|<2.0, mean p>.05).

### tfidf / primary

| h | ref | row5 rel% | DM | p raw | p Holm(8) | 95% CI (date-block) | G4 | G4b | share d4/d3 | MDE(ent)% | MSE_Re | Δ_pub | G_conv% | §6.4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | r_ar | +3.24% | -0.75 | 0.4536 | 1.0000 | [-3.65, +8.27] | pass | dirty | 114.5% | 8.44% | 0.6825 | +0.7746 | +113.5% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 3 | r_har | +2.59% | -1.22 | 0.2263 | 1.0000 | [-1.67, +6.45] | pass | clean | 89.8% | 6.06% | 0.6453 | +0.7746 | +120.0% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 7 | r_ar | +4.61% | -1.72 | 0.0883 | 0.7066 | [-0.08, +8.27] | pass | dirty | 118.9% | 5.85% | 0.3170 | +0.2149 | +67.8% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 7 | r_har | +3.09% | -1.69 | 0.0924 | 0.7066 | [-0.20, +5.79] | pass | clean | 91.0% | 4.18% | 0.2897 | +0.2149 | +74.2% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 15 | r_ar | +4.13% | -1.22 | 0.2227 | 1.0000 | [-0.80, +8.03] | pass | clean | 111.7% | 6.77% | 0.2034 | +0.1046 | +51.4% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 15 | r_har | +2.77% | -1.14 | 0.2544 | 1.0000 | [-0.76, +5.29] | pass | clean | 90.3% | 4.66% | 0.1881 | +0.1046 | +55.6% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 30 | r_ar | +2.90% | -1.05 | 0.2975 | 1.0000 | [-0.66, +5.75] | pass | clean | 118.7% | 5.00% | 0.1356 | +0.0236 | +17.4% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 30 | r_har | +2.49% | -1.42 | 0.1565 | 0.9392 | [-0.16, +4.53] | pass | clean | 89.4% | 3.19% | 0.1304 | +0.0236 | +18.1% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |

### tfidf / shifted

| h | ref | row5 rel% | DM | p raw | p Holm(8) | 95% CI (date-block) | G4 | G4b | share d4/d3 | MDE(ent)% | MSE_Re | Δ_pub | G_conv% | §6.4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | r_ar | +4.80% | -1.53 | 0.1285 | 0.7145 | [-0.69, +7.23] | pass | clean | 101.1% | 6.87% | 0.6121 | +0.7746 | +126.5% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 3 | r_har | +2.57% | -2.32 | 0.0218 | 0.1742 | [+0.60, +4.34] | pass | clean | 80.2% | 3.35% | 0.5635 | +0.7746 | +137.5% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 7 | r_ar | +2.16% | -1.13 | 0.2609 | 0.9945 | [-1.63, +5.38] | pass | clean | 145.4% | 4.72% | 0.3315 | +0.2149 | +64.8% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 7 | r_har | +1.70% | -2.22 | 0.0283 | 0.1982 | [-0.06, +3.37] | pass | clean | 100.2% | 2.29% | 0.3115 | +0.2149 | +69.0% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 15 | r_ar | +1.71% | -0.20 | 0.8424 | 1.0000 | [-3.42, +3.64] | pass | clean | 119.1% | 4.48% | 0.2069 | +0.1046 | +50.5% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 15 | r_har | +1.16% | -0.55 | 0.5805 | 1.0000 | [-1.60, +2.54] | pass | clean | 126.1% | 2.69% | 0.1994 | +0.1046 | +52.4% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 30 | r_ar | +1.35% | -1.16 | 0.2486 | 0.9945 | [-0.86, +2.09] | pass | clean | 126.2% | 1.48% | 0.1425 | +0.0236 | +16.6% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 30 | r_har | +1.22% | -1.57 | 0.1191 | 0.7145 | [-0.39, +2.01] | pass | clean | 105.7% | 1.35% | 0.1348 | +0.0236 | +17.5% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |

### qwen_emb / primary

| h | ref | row5 rel% | DM | p raw | p Holm(8) | 95% CI (date-block) | G4 | G4b | share d4/d3 | MDE(ent)% | MSE_Re | Δ_pub | G_conv% | §6.4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | r_ar | +4.27% | -1.95 | 0.0536 | 0.3750 | [+0.16, +6.86] | pass | dirty | 112.4% | 5.10% | 0.6825 | +0.7746(proxy: tfidf) | +113.5% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 3 | r_har | +2.24% | -2.23 | 0.0275 | 0.2197 | [+0.34, +4.13] | pass | clean | 113.1% | 2.97% | 0.6453 | +0.7746(proxy: tfidf) | +120.0% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 7 | r_ar | +2.45% | -0.24 | 0.8107 | 1.0000 | [-4.83, +6.16] | pass | dirty | 133.3% | 7.65% | 0.3170 | +0.2149(proxy: tfidf) | +67.8% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 7 | r_har | +1.94% | -0.34 | 0.7315 | 1.0000 | [-3.21, +4.61] | pass | clean | 122.9% | 5.15% | 0.2897 | +0.2149(proxy: tfidf) | +74.2% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 15 | r_ar | +3.75% | -0.30 | 0.7658 | 1.0000 | [-5.35, +7.79] | pass | clean | 110.6% | 9.97% | 0.2034 | +0.1046(proxy: tfidf) | +51.4% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 15 | r_har | +2.68% | -0.19 | 0.8519 | 1.0000 | [-4.10, +5.03] | pass | clean | 94.5% | 7.15% | 0.1881 | +0.1046(proxy: tfidf) | +55.6% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 30 | r_ar | +2.92% | -0.23 | 0.8204 | 1.0000 | [-4.58, +6.01] | pass | clean | 116.9% | 10.07% | 0.1356 | +0.0236(proxy: tfidf) | +17.4% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 30 | r_har | +2.63% | -0.27 | 0.7859 | 1.0000 | [-3.32, +4.89] | pass | clean | 90.0% | 7.48% | 0.1304 | +0.0236(proxy: tfidf) | +18.1% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |

### qwen_emb / shifted

| h | ref | row5 rel% | DM | p raw | p Holm(8) | 95% CI (date-block) | G4 | G4b | share d4/d3 | MDE(ent)% | MSE_Re | Δ_pub | G_conv% | §6.4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | r_ar | +4.81% | -1.70 | 0.0916 | 0.6413 | [-0.40, +5.79] | pass | clean | 101.1% | 5.46% | 0.6121 | +0.7746(proxy: tfidf) | +126.5% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 3 | r_har | +2.58% | -2.33 | 0.0210 | 0.1677 | [+0.53, +4.29] | pass | clean | 81.2% | 3.30% | 0.5635 | +0.7746(proxy: tfidf) | +137.5% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 7 | r_ar | +3.03% | -0.40 | 0.6873 | 1.0000 | [-3.98, +4.74] | pass | clean | 111.2% | 5.82% | 0.3315 | +0.2149(proxy: tfidf) | +64.8% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 7 | r_har | +2.50% | -1.64 | 0.1038 | 0.6413 | [-0.79, +4.58] | pass | clean | 78.7% | 3.79% | 0.3115 | +0.2149(proxy: tfidf) | +69.0% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 15 | r_ar | +2.93% | -0.04 | 0.9651 | 1.0000 | [-5.51, +5.08] | pass | clean | 84.7% | 6.73% | 0.2069 | +0.1046(proxy: tfidf) | +50.5% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 15 | r_har | +2.38% | -0.63 | 0.5302 | 1.0000 | [-2.56, +4.54] | pass | clean | 79.3% | 4.74% | 0.1994 | +0.1046(proxy: tfidf) | +52.4% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 30 | r_ar | +3.06% | -0.90 | 0.3683 | 1.0000 | [-1.77, +3.90] | pass | clean | 84.3% | 3.25% | 0.1425 | +0.0236(proxy: tfidf) | +16.6% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 30 | r_har | +2.78% | -1.18 | 0.2418 | 1.0000 | [-1.19, +3.76] | pass | clean | 67.4% | 2.94% | 0.1348 | +0.0236(proxy: tfidf) | +17.5% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |

### prompted_qwen / primary

| h | ref | row5 rel% | DM | p raw | p Holm(8) | 95% CI (date-block) | G4 | G4b | share d4/d3 | MDE(ent)% | MSE_Re | Δ_pub | G_conv% | §6.4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | r_ar | -0.37% | +0.35 | 0.7234 | 1.0000 | [-8.67, +5.02] | pass | clean | 7890.2% | 10.31% | 0.6825 | +0.3962 | +58.0% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 3 | r_har | -1.17% | +0.77 | 0.4445 | 1.0000 | [-5.30, +1.44] | pass | clean | NaN | 5.25% | 0.6453 | +0.3962 | +61.4% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 7 | r_ar | -2.21% | +0.49 | 0.6266 | 1.0000 | [-11.86, +6.91] | pass | clean | NaN | 12.80% | 0.3170 | -0.1613 | -50.9% | no published gain at this cell (text loses to raw V_past) — nothing to absorb |
| 7 | r_har | -2.22% | +1.06 | 0.2896 | 1.0000 | [-7.45, +2.07] | pass | clean | NaN | 6.52% | 0.2897 | -0.1613 | -55.7% | no published gain at this cell (text loses to raw V_past) — nothing to absorb |
| 15 | r_ar | -3.73% | +0.79 | 0.4330 | 1.0000 | [-9.34, +3.26] | pass | clean | NaN | 9.60% | 0.2034 | -0.1594 | -78.3% | no published gain at this cell (text loses to raw V_past) — nothing to absorb |
| 15 | r_har | -3.37% | +1.47 | 0.1434 | 1.0000 | [-9.37, +0.78] | pass | clean | NaN | 7.07% | 0.1881 | -0.1594 | -84.7% | no published gain at this cell (text loses to raw V_past) — nothing to absorb |
| 30 | r_ar | -0.79% | -0.94 | 0.3482 | 1.0000 | [-2.38, +8.30] | pass | clean | 2807.1% | 5.90% | 0.1356 | -0.1262 | -93.0% | no published gain at this cell (text loses to raw V_past) — nothing to absorb |
| 30 | r_har | +0.02% | -0.73 | 0.4658 | 1.0000 | [-2.31, +5.81] | pass | clean | 3955.3% | 3.81% | 0.1304 | -0.1262 | -96.8% | no published gain at this cell (text loses to raw V_past) — nothing to absorb |

### prompted_qwen / shifted

| h | ref | row5 rel% | DM | p raw | p Holm(8) | 95% CI (date-block) | G4 | G4b | share d4/d3 | MDE(ent)% | MSE_Re | Δ_pub | G_conv% | §6.4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | r_ar | -1.28% | +0.75 | 0.4523 | 0.6180 | [-3.49, +1.70] | pass | clean | NaN | 4.05% | 0.6121 | +0.3962 | +64.7% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 3 | r_har | -0.44% | +1.02 | 0.3090 | 0.6180 | [-1.35, +0.23] | pass | clean | NaN | 1.30% | 0.5635 | +0.3962 | +70.3% | powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed |
| 7 | r_ar | -3.55% | +2.52 | 0.0129 | 0.0773 | [-8.53, -0.72] | pass | clean | NaN | 4.87% | 0.3315 | -0.1613 | -48.6% | no published gain at this cell (text loses to raw V_past) — nothing to absorb |
| 7 | r_har | -1.48% | +2.08 | 0.0391 | 0.1953 | [-3.90, -0.26] | pass | clean | NaN | 2.58% | 0.3115 | -0.1613 | -51.8% | no published gain at this cell (text loses to raw V_past) — nothing to absorb |
| 15 | r_ar | -5.32% **NEG-SIG** | +3.32 | 0.0012 | 0.0093 | [-12.65, -3.17] | pass | clean | NaN | 5.98% | 0.2069 | -0.1594 | -77.0% | no published gain at this cell (text loses to raw V_past) — nothing to absorb |
| 15 | r_har | -3.87% **NEG-SIG** | +2.78 | 0.0062 | 0.0433 | [-9.40, -1.61] | pass | clean | NaN | 5.10% | 0.1994 | -0.1594 | -79.9% | no published gain at this cell (text loses to raw V_past) — nothing to absorb |
| 30 | r_ar | -1.41% | +1.46 | 0.1471 | 0.5884 | [-4.70, +1.34] | pass | clean | NaN | 2.94% | 0.1425 | -0.1262 | -88.6% | no published gain at this cell (text loses to raw V_past) — nothing to absorb |
| 30 | r_har | -1.52% | +1.32 | 0.1897 | 0.5884 | [-4.45, +1.29] | pass | clean | NaN | 3.33% | 0.1348 | -0.1262 | -93.6% | no published gain at this cell (text loses to raw V_past) — nothing to absorb |

## 3. Full cell-by-cell table (row3 combination increment + row5 residual; probe rows are diagnostic)

### tfidf / primary 

| h | ref | row3 rel% | DM | p raw | p Holm | 95% CI | r3 shuffle | r3 swap | row5 rel% | DM | p raw | p Holm | 95% CI | r5 shuffle | r5 swap | share | MDE(AR)% | MDE(ent)% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | r_ar | +13.07% | -3.09 | 0.0024 | 0.0168 | [+5.11,+19.35] | dm+0.33/p0.536 | dm-3.61/p0.001 | +3.24% | -0.75 | 0.4536 | 1.0000 | [-3.65,+8.27] | dm+0.14/p0.594 | dm-0.17/p0.047 | 114.5% | 11.20 | 8.44 |
| 3 | r_har | +5.86% | -2.31 | 0.0224 | 0.1035 | [+1.05,+10.70] | dm+0.31/p0.619 | dm-0.04/p0.027 | +2.59% | -1.22 | 0.2263 | 1.0000 | [-1.67,+6.45] | dm+0.12/p0.626 | dm-0.00/p0.093 | 89.8% | 7.72 | 6.06 |
| 7 | r_ar | +17.52% | -4.38 | 0.0000 | 0.0002 | [+9.68,+23.88] | dm+0.30/p0.549 | dm-3.62/p0.002 | +4.61% | -1.72 | 0.0883 | 0.7066 | [-0.08,+8.27] | dm+0.27/p0.528 | dm-0.84/p0.043 | 118.9% | 9.92 | 5.85 |
| 7 | r_har | +7.29% | -2.75 | 0.0067 | 0.0402 | [+2.50,+12.03] | dm+0.45/p0.497 | dm-1.09/p0.024 | +3.09% | -1.69 | 0.0924 | 0.7066 | [-0.20,+5.79] | dm+0.44/p0.475 | dm-0.13/p0.074 | 91.0% | 6.72 | 4.18 |
| 15 | r_ar | +15.09% | -2.34 | 0.0207 | 0.1035 | [+1.85,+19.25] | dm+0.25/p0.592 | dm-2.52/p0.026 | +4.13% | -1.22 | 0.2227 | 1.0000 | [-0.80,+8.03] | dm+0.20/p0.552 | dm+0.14/p0.605 | 111.7% | 10.48 | 6.77 |
| 15 | r_har | +6.56% | -1.22 | 0.2259 | 0.3387 | [-1.45,+9.69] | dm+0.62/p0.489 | dm-0.39/p0.303 | +2.77% | -1.14 | 0.2544 | 1.0000 | [-0.76,+5.29] | dm+0.67/p0.439 | dm+0.16/p0.610 | 90.3% | 8.04 | 4.66 |
| 30 | r_ar | +9.71% | -1.60 | 0.1129 | 0.3387 | [+1.39,+15.83] | dm+0.25/p0.599 | dm-0.73/p0.175 | +2.90% | -1.05 | 0.2975 | 1.0000 | [-0.66,+5.75] | dm+0.26/p0.501 | dm+0.35/p0.612 | 118.7% | 10.12 | 5.00 |
| 30 | r_har | +5.86% | -1.46 | 0.1456 | 0.3387 | [+0.01,+9.09] | dm+0.43/p0.432 | dm-0.26/p0.381 | +2.49% | -1.42 | 0.1565 | 0.9392 | [-0.16,+4.53] | dm+0.43/p0.373 | dm+0.27/p0.653 | 89.4% | 6.00 | 3.19 |

### tfidf / shifted 

| h | ref | row3 rel% | DM | p raw | p Holm | 95% CI | r3 shuffle | r3 swap | row5 rel% | DM | p raw | p Holm | 95% CI | r5 shuffle | r5 swap | share | MDE(AR)% | MDE(ent)% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | r_ar | +11.82% | -2.66 | 0.0087 | 0.0454 | [+3.65,+15.53] | dm+0.50/p0.508 | dm-2.61/p0.021 | +4.80% | -1.53 | 0.1285 | 0.7145 | [-0.69,+7.23] | dm+0.24/p0.480 | dm-0.63/p0.230 | 101.1% | 11.36 | 6.87 |
| 3 | r_har | +4.09% | -2.93 | 0.0040 | 0.0280 | [+1.66,+6.44] | dm+0.59/p0.443 | dm-0.96/p0.135 | +2.57% | -2.32 | 0.0218 | 0.1742 | [+0.60,+4.34] | dm+0.69/p0.432 | dm-0.75/p0.232 | 80.2% | 4.41 | 3.35 |
| 7 | r_ar | +8.29% | -2.71 | 0.0076 | 0.0454 | [+2.39,+14.96] | dm+0.55/p0.486 | dm-2.66/p0.020 | +2.16% | -1.13 | 0.2609 | 0.9945 | [-1.63,+5.38] | dm+0.64/p0.504 | dm-0.82/p0.244 | 145.4% | 8.85 | 4.72 |
| 7 | r_har | +3.65% | -3.44 | 0.0008 | 0.0060 | [+1.60,+6.87] | dm+0.68/p0.446 | dm+0.27/p0.083 | +1.70% | -2.22 | 0.0283 | 0.1982 | [-0.06,+3.37] | dm+0.68/p0.446 | dm+0.33/p0.167 | 100.2% | 3.75 | 2.29 |
| 15 | r_ar | +6.29% | -1.12 | 0.2640 | 0.4244 | [-4.32,+12.48] | dm+0.16/p0.480 | dm-1.38/p0.186 | +1.71% | -0.20 | 0.8424 | 1.0000 | [-3.42,+3.64] | dm+0.38/p0.513 | dm+0.06/p0.509 | 119.1% | 10.39 | 4.48 |
| 15 | r_har | +2.89% | -1.33 | 0.1851 | 0.4244 | [-1.27,+5.59] | dm+0.49/p0.486 | dm-0.57/p0.555 | +1.16% | -0.55 | 0.5805 | 1.0000 | [-1.60,+2.54] | dm+0.39/p0.515 | dm+0.28/p0.589 | 126.1% | 4.63 | 2.69 |
| 30 | r_ar | +4.97% | -1.48 | 0.1415 | 0.4244 | [-0.85,+7.79] | dm+0.20/p0.494 | dm-0.78/p0.198 | +1.35% | -1.16 | 0.2486 | 0.9945 | [-0.86,+2.09] | dm+0.42/p0.561 | dm+0.27/p0.796 | 126.2% | 5.93 | 1.48 |
| 30 | r_har | +3.31% | -1.97 | 0.0513 | 0.2051 | [+0.32,+5.19] | dm+0.45/p0.459 | dm-0.31/p0.368 | +1.22% | -1.57 | 0.1191 | 0.7145 | [-0.39,+2.01] | dm+0.47/p0.518 | dm+0.36/p0.671 | 105.7% | 3.45 | 1.35 |

### qwen_emb / primary 

| h | ref | row3 rel% | DM | p raw | p Holm | 95% CI | r3 shuffle | r3 swap | row5 rel% | DM | p raw | p Holm | 95% CI | r5 shuffle | r5 swap | share | MDE(AR)% | MDE(ent)% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | r_ar | +13.31% | -4.22 | 0.0000 | 0.0003 | [+7.48,+17.38] | dm+0.44/p0.514 | dm-4.00/p0.000 | +4.27% | -1.95 | 0.0536 | 0.3750 | [+0.16,+6.86] | dm+0.21/p0.593 | dm-1.91/p0.019 | 112.4% | 8.44 | 5.10 |
| 3 | r_har | +4.66% | -3.33 | 0.0011 | 0.0079 | [+2.19,+7.36] | dm+0.30/p0.609 | dm-0.44/p0.070 | +2.24% | -2.23 | 0.0275 | 0.2197 | [+0.34,+4.13] | dm+0.19/p0.613 | dm-0.41/p0.198 | 113.1% | 4.26 | 2.97 |
| 7 | r_ar | +15.63% | -3.02 | 0.0030 | 0.0179 | [+5.24,+21.45] | dm+0.26/p0.551 | dm-3.88/p0.001 | +2.45% | -0.24 | 0.8107 | 1.0000 | [-4.83,+6.16] | dm+0.29/p0.607 | dm-2.05/p0.068 | 133.3% | 11.63 | 7.65 |
| 7 | r_har | +5.40% | -1.44 | 0.1520 | 0.6082 | [-1.28,+9.56] | dm+0.28/p0.562 | dm-1.60/p0.154 | +1.94% | -0.34 | 0.7315 | 1.0000 | [-3.21,+4.61] | dm+0.41/p0.579 | dm-0.83/p0.360 | 122.9% | 7.25 | 5.15 |
| 15 | r_ar | +15.24% | -1.77 | 0.0793 | 0.3963 | [-2.98,+17.93] | dm+0.40/p0.496 | dm-3.83/p0.001 | +3.75% | -0.30 | 0.7658 | 1.0000 | [-5.35,+7.79] | dm+0.25/p0.569 | dm-0.14/p0.416 | 110.6% | 10.86 | 9.97 |
| 15 | r_har | +6.27% | -0.55 | 0.5834 | 1.0000 | [-4.81,+8.93] | dm+0.52/p0.563 | dm-0.32/p0.340 | +2.68% | -0.19 | 0.8519 | 1.0000 | [-4.10,+5.03] | dm+0.65/p0.573 | dm+0.17/p0.742 | 94.5% | 9.53 | 7.15 |
| 30 | r_ar | +9.86% | -0.92 | 0.3576 | 1.0000 | [-2.75,+15.54] | dm+0.12/p0.518 | dm+0.02/p0.076 | +2.92% | -0.23 | 0.8204 | 1.0000 | [-4.58,+6.01] | dm+0.43/p0.580 | dm+0.13/p0.801 | 116.9% | 14.63 | 10.07 |
| 30 | r_har | +5.82% | -0.50 | 0.6149 | 1.0000 | [-3.98,+8.87] | dm+0.38/p0.417 | dm+0.06/p0.522 | +2.63% | -0.27 | 0.7859 | 1.0000 | [-3.32,+4.89] | dm+0.47/p0.466 | dm+0.16/p0.791 | 90.0% | 10.50 | 7.48 |

### qwen_emb / shifted 

| h | ref | row3 rel% | DM | p raw | p Holm | 95% CI | r3 shuffle | r3 swap | row5 rel% | DM | p raw | p Holm | 95% CI | r5 shuffle | r5 swap | share | MDE(AR)% | MDE(ent)% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | r_ar | +11.82% | -3.20 | 0.0017 | 0.0134 | [+4.56,+14.23] | dm+0.94/p0.375 | dm-1.53/p0.004 | +4.81% | -1.70 | 0.0916 | 0.6413 | [-0.40,+5.79] | dm+0.73/p0.373 | dm-0.03/p0.105 | 101.1% | 9.23 | 5.46 |
| 3 | r_har | +4.04% | -3.08 | 0.0024 | 0.0171 | [+1.78,+6.41] | dm+0.90/p0.375 | dm-0.52/p0.245 | +2.58% | -2.33 | 0.0210 | 0.1677 | [+0.53,+4.29] | dm+0.91/p0.381 | dm+0.02/p0.342 | 81.2% | 4.17 | 3.30 |
| 7 | r_ar | +10.85% | -2.20 | 0.0293 | 0.1466 | [+1.95,+16.39] | dm+1.07/p0.319 | dm-2.67/p0.025 | +3.03% | -0.40 | 0.6873 | 1.0000 | [-3.98,+4.74] | dm+0.68/p0.331 | dm-1.55/p0.254 | 111.2% | 11.49 | 5.82 |
| 7 | r_har | +4.65% | -2.63 | 0.0094 | 0.0564 | [+1.46,+8.20] | dm+0.84/p0.347 | dm-0.62/p0.264 | +2.50% | -1.64 | 0.1038 | 0.6413 | [-0.79,+4.58] | dm+0.71/p0.349 | dm-0.55/p0.270 | 78.7% | 5.56 | 3.79 |
| 15 | r_ar | +8.84% | -1.10 | 0.2712 | 0.5089 | [-4.87,+16.07] | dm+0.14/p0.384 | dm-1.78/p0.106 | +2.93% | -0.04 | 0.9651 | 1.0000 | [-5.51,+5.08] | dm+0.17/p0.415 | dm-0.54/p0.522 | 84.7% | 14.09 | 6.73 |
| 15 | r_har | +4.60% | -1.33 | 0.1864 | 0.5089 | [-1.40,+8.20] | dm+0.52/p0.431 | dm-0.57/p0.496 | +2.38% | -0.63 | 0.5302 | 1.0000 | [-2.56,+4.54] | dm+0.42/p0.419 | dm-0.05/p0.654 | 79.3% | 7.01 | 4.74 |
| 30 | r_ar | +7.44% | -1.46 | 0.1460 | 0.5089 | [-0.49,+11.47] | dm+0.03/p0.491 | dm-0.71/p0.132 | +3.06% | -0.90 | 0.3683 | 1.0000 | [-1.77,+3.90] | dm+0.16/p0.539 | dm+0.26/p0.665 | 84.3% | 8.87 | 3.25 |
| 30 | r_har | +5.19% | -1.53 | 0.1272 | 0.5089 | [-0.32,+7.84] | dm+0.15/p0.434 | dm+0.12/p0.450 | +2.78% | -1.18 | 0.2418 | 1.0000 | [-1.19,+3.76] | dm+0.23/p0.476 | dm+0.26/p0.711 | 67.4% | 6.11 | 2.94 |

### prompted_qwen / primary 

| h | ref | row3 rel% | DM | p raw | p Holm | 95% CI | r3 shuffle | r3 swap | row5 rel% | DM | p raw | p Holm | 95% CI | r5 shuffle | r5 swap | share | MDE(AR)% | MDE(ent)% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | r_ar | +0.19% | +0.04 | 0.9666 | 1.0000 | [-7.93,+6.89] | dm+0.52/p0.291 | dm+0.13/p0.745 | -0.37% | +0.35 | 0.7234 | 1.0000 | [-8.67,+5.02] | dm+0.39/p0.283 | dm-0.01/p0.687 | 7890.2% | 10.67 | 10.31 |
| 3 | r_har | -1.42% | +0.89 | 0.3733 | 1.0000 | [-5.15,+1.23] | dm+0.01/p0.297 | dm+0.59/p0.520 | -1.17% | +0.77 | 0.4445 | 1.0000 | [-5.30,+1.44] | dm+0.00/p0.296 | dm+0.60/p0.551 | NaN | 4.83 | 5.25 |
| 7 | r_ar | -0.81% | +0.02 | 0.9818 | 1.0000 | [-9.73,+9.93] | dm+0.44/p0.357 | dm-0.00/p0.580 | -2.21% | +0.49 | 0.6266 | 1.0000 | [-11.86,+6.91] | dm+0.48/p0.388 | dm+0.30/p0.658 | NaN | 13.54 | 12.80 |
| 7 | r_har | -2.38% | +1.12 | 0.2642 | 1.0000 | [-6.77,+1.83] | dm+0.60/p0.376 | dm+0.09/p0.590 | -2.22% | +1.06 | 0.2896 | 1.0000 | [-7.45,+2.07] | dm+0.51/p0.403 | dm+0.14/p0.560 | NaN | 5.88 | 6.52 |
| 15 | r_ar | -3.29% | +0.62 | 0.5363 | 1.0000 | [-11.00,+4.72] | dm+0.44/p0.416 | dm-0.54/p0.250 | -3.73% | +0.79 | 0.4330 | 1.0000 | [-9.34,+3.26] | dm+0.76/p0.316 | dm-0.26/p0.391 | NaN | 10.83 | 9.60 |
| 15 | r_har | -3.88% | +1.67 | 0.0974 | 0.7796 | [-9.99,+0.36] | dm+0.51/p0.357 | dm+0.06/p0.754 | -3.37% | +1.47 | 0.1434 | 1.0000 | [-9.37,+0.78] | dm+0.86/p0.322 | dm+0.13/p0.654 | NaN | 6.70 | 7.07 |
| 30 | r_ar | +0.41% | -1.20 | 0.2334 | 1.0000 | [-1.99,+10.96] | dm+0.58/p0.507 | dm-0.00/p0.059 | -0.79% | -0.94 | 0.3482 | 1.0000 | [-2.38,+8.30] | dm+0.83/p0.403 | dm-0.78/p0.109 | 2807.1% | 7.03 | 5.90 |
| 30 | r_har | +0.13% | -0.75 | 0.4574 | 1.0000 | [-2.53,+6.48] | dm+0.61/p0.430 | dm-0.72/p0.172 | +0.02% | -0.73 | 0.4658 | 1.0000 | [-2.31,+5.81] | dm+0.79/p0.383 | dm-0.57/p0.242 | 3955.3% | 3.77 | 3.81 |

### prompted_qwen / shifted 

| h | ref | row3 rel% | DM | p raw | p Holm | 95% CI | r3 shuffle | r3 swap | row5 rel% | DM | p raw | p Holm | 95% CI | r5 shuffle | r5 swap | share | MDE(AR)% | MDE(ent)% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | r_ar | -0.04% | +0.08 | 0.9403 | 1.0000 | [-2.41,+2.33] | dm+0.52/p0.399 | dm-0.81/p0.429 | -1.28% | +0.75 | 0.4523 | 0.6180 | [-3.49,+1.70] | dm+0.46/p0.425 | dm-0.37/p0.719 | NaN | 3.85 | 4.05 |
| 3 | r_har | -0.27% | +1.03 | 0.3033 | 1.0000 | [-1.03,+0.17] | dm+0.24/p0.564 | dm+0.33/p0.556 | -0.44% | +1.02 | 0.3090 | 0.6180 | [-1.35,+0.23] | dm+0.22/p0.581 | dm+0.39/p0.599 | NaN | 0.98 | 1.30 |
| 7 | r_ar | -2.71% | +2.39 | 0.0179 | 0.1076 | [-7.25,-0.56] | dm+0.86/p0.356 | dm+0.46/p0.593 | -3.55% | +2.52 | 0.0129 | 0.0773 | [-8.53,-0.72] | dm+0.71/p0.410 | dm+0.87/p0.370 | NaN | 4.36 | 4.87 |
| 7 | r_har | -1.29% | +2.07 | 0.0407 | 0.2034 | [-3.50,-0.23] | dm+0.71/p0.478 | dm+0.37/p0.296 | -1.48% | +2.08 | 0.0391 | 0.1953 | [-3.90,-0.26] | dm+0.61/p0.472 | dm+0.89/p0.263 | NaN | 2.41 | 2.58 |
| 15 | r_ar | -4.26% | +3.05 | 0.0028 | 0.0221 | [-11.71,-2.40] | dm+0.57/p0.486 | dm+0.14/p0.601 | -5.32% | +3.32 | 0.0012 | 0.0093 | [-12.65,-3.17] | dm+0.72/p0.423 | dm+0.67/p0.402 | NaN | 6.05 | 5.98 |
| 15 | r_har | -3.51% | +2.75 | 0.0067 | 0.0472 | [-8.90,-1.31] | dm+0.39/p0.519 | dm-0.02/p0.479 | -3.87% | +2.78 | 0.0062 | 0.0433 | [-9.40,-1.61] | dm+0.33/p0.546 | dm+0.65/p0.428 | NaN | 4.83 | 5.10 |
| 30 | r_ar | -0.25% | +0.89 | 0.3735 | 1.0000 | [-3.52,+1.88] | dm+0.15/p0.509 | dm+0.50/p0.473 | -1.41% | +1.46 | 0.1471 | 0.5884 | [-4.70,+1.34] | dm+0.33/p0.504 | dm+0.73/p0.465 | NaN | 3.04 | 2.94 |
| 30 | r_har | -0.83% | +1.04 | 0.2991 | 1.0000 | [-4.01,+1.57] | dm+0.43/p0.517 | dm-0.46/p0.489 | -1.52% | +1.32 | 0.1897 | 0.5884 | [-4.45,+1.29] | dm+0.61/p0.530 | dm+0.77/p0.415 | NaN | 3.44 | 3.33 |

### identity_probe / primary (diagnostic, §6.2: not entered into Holm, not entered into 'win')

| h | ref | row3 rel% | DM | p raw | p Holm | 95% CI | r3 shuffle | r3 swap | row5 rel% | DM | p raw | p Holm | 95% CI | r5 shuffle | r5 swap | share | MDE(AR)% | MDE(ent)% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | r_ar | +8.55% | -1.78 | 0.0767 | — (diagnostic) | [-0.81,+14.79] | dm+0.12/p0.487 | dm-1.65/p0.008 | +3.26% | -0.87 | 0.3877 | — (diagnostic) | [-4.10,+9.16] | dm+0.08/p0.446 | dm-1.20/p0.043 | 175.1% | 12.44 | 10.52 |
| 3 | r_har | +1.88% | -0.81 | 0.4193 | — (diagnostic) | [-2.59,+4.13] | dm+0.07/p0.480 | dm+0.44/p0.454 | +1.27% | -0.70 | 0.4847 | — (diagnostic) | [-2.52,+3.45] | dm+0.19/p0.467 | dm+0.41/p0.489 | 280.2% | 5.72 | 5.00 |
| 7 | r_ar | +5.09% | -1.63 | 0.1048 | — (diagnostic) | [-0.65,+15.53] | dm+0.38/p0.509 | dm-1.47/p0.010 | -0.52% | -0.42 | 0.6718 | — (diagnostic) | [-4.81,+7.52] | dm+0.27/p0.520 | dm-0.91/p0.057 | 409.2% | 11.79 | 8.58 |
| 7 | r_har | +0.93% | -0.48 | 0.6322 | — (diagnostic) | [-3.38,+4.58] | dm+0.14/p0.488 | dm-0.18/p0.344 | +0.05% | -0.10 | 0.9177 | — (diagnostic) | [-3.66,+3.41] | dm+0.13/p0.490 | dm-0.04/p0.436 | 712.6% | 5.20 | 4.63 |
| 15 | r_ar | +1.04% | -1.37 | 0.1714 | — (diagnostic) | [-0.90,+7.23] | dm+0.31/p0.420 | dm-0.86/p0.132 | -0.14% | -0.65 | 0.5184 | — (diagnostic) | [-1.46,+2.88] | dm+0.36/p0.548 | dm-0.09/p0.210 | 1621.6% | 5.46 | 2.54 |
| 15 | r_har | +0.46% | -1.35 | 0.1794 | — (diagnostic) | [-0.48,+1.65] | dm+0.46/p0.492 | dm+0.67/p0.422 | +0.18% | -0.66 | 0.5124 | — (diagnostic) | [-0.55,+1.00] | dm+0.34/p0.572 | dm+0.76/p0.455 | 1301.1% | 0.99 | 0.75 |
| 30 | r_ar | -0.10% | -0.54 | 0.5923 | — (diagnostic) | [-0.30,+0.57] | dm+0.35/p0.443 | dm+0.61/p0.447 | -0.11% | +0.36 | 0.7185 | — (diagnostic) | [-0.29,+0.15] | dm+0.27/p0.533 | dm+0.93/p0.439 | NaN | 0.61 | 0.28 |
| 30 | r_har | -0.05% | +0.68 | 0.4987 | — (diagnostic) | [-0.97,+0.43] | dm+0.40/p0.514 | dm+0.41/p0.688 | +0.07% | +0.01 | 0.9923 | — (diagnostic) | [-0.52,+0.55] | dm+0.43/p0.605 | dm+0.59/p0.663 | NaN | 0.85 | 0.62 |

### identity_probe / shifted (diagnostic, §6.2: not entered into Holm, not entered into 'win')

| h | ref | row3 rel% | DM | p raw | p Holm | 95% CI | r3 shuffle | r3 swap | row5 rel% | DM | p raw | p Holm | 95% CI | r5 shuffle | r5 swap | share | MDE(AR)% | MDE(ent)% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | r_ar | +7.27% | -1.47 | 0.1441 | — (diagnostic) | [-2.17,+12.96] | dm+1.21/p0.314 | dm-3.35/p0.003 | +2.20% | -0.37 | 0.7138 | — (diagnostic) | [-5.49,+6.99] | dm+1.11/p0.323 | dm-2.27/p0.038 | 164.4% | 13.01 | 9.87 |
| 3 | r_har | +1.71% | -0.65 | 0.5157 | — (diagnostic) | [-3.06,+4.96] | dm+1.08/p0.296 | dm-0.60/p0.584 | +1.04% | -0.41 | 0.6823 | — (diagnostic) | [-3.32,+4.12] | dm+1.07/p0.296 | dm-0.49/p0.619 | 191.7% | 6.89 | 6.30 |
| 7 | r_ar | +4.62% | -0.63 | 0.5327 | — (diagnostic) | [-6.07,+10.31] | dm+1.05/p0.328 | dm-1.91/p0.086 | +1.00% | +0.62 | 0.5357 | — (diagnostic) | [-8.23,+3.62] | dm+0.45/p0.329 | dm-0.91/p0.444 | 261.0% | 11.78 | 7.95 |
| 7 | r_har | +1.89% | -0.66 | 0.5120 | — (diagnostic) | [-3.38,+5.31] | dm+0.79/p0.379 | dm-0.21/p0.693 | +1.18% | -0.23 | 0.8192 | — (diagnostic) | [-3.78,+3.93] | dm+0.57/p0.380 | dm+0.05/p0.606 | 193.1% | 6.88 | 5.86 |
| 15 | r_ar | +1.42% | -0.74 | 0.4604 | — (diagnostic) | [-2.99,+6.61] | dm+0.67/p0.408 | dm-0.53/p0.574 | +0.43% | +0.20 | 0.8404 | — (diagnostic) | [-3.17,+2.81] | dm+0.18/p0.500 | dm+0.19/p0.618 | 528.2% | 6.67 | 3.90 |
| 15 | r_har | +1.17% | -1.22 | 0.2253 | — (diagnostic) | [-0.69,+3.10] | dm+0.60/p0.386 | dm+0.01/p0.678 | +0.74% | -0.65 | 0.5174 | — (diagnostic) | [-1.02,+2.23] | dm+0.32/p0.429 | dm+0.58/p0.590 | 311.5% | 2.74 | 2.29 |
| 30 | r_ar | +0.07% | -1.13 | 0.2608 | — (diagnostic) | [-0.28,+1.23] | dm+0.21/p0.490 | dm-0.02/p0.453 | -0.03% | -0.26 | 0.7917 | — (diagnostic) | [-0.34,+0.49] | dm+0.16/p0.550 | dm+0.27/p0.461 | 9313.2% | 0.98 | 0.50 |
| 30 | r_har | -0.00% | +1.48 | 0.1422 | — (diagnostic) | [-0.00,+0.00] | dm+0.34/p0.511 | dm+0.28/p0.640 | +0.00% | -0.78 | 0.4359 | — (diagnostic) | [-0.00,+0.01] | dm+0.21/p0.492 | dm+0.43/p0.657 | NaN | 0.00 | 0.01 |

## 4. §8 branch adjudication (headline arm × alignment; primary governs the paper wording, §2.3)

### tfidf / primary

- **F2 pass(DM<0 & Holm<.05 & G4):0/8**
- F1 combination-increment wins (DM<0 & Holm<.05 & G4): 3/8: h3 r_ar, h7 r_ar, h7 r_har
- Cells with a significant F1 Holm and their identity share: h3 r_ar: 114.5%, h7 r_ar: 118.9%, h7 r_har: 91.0%
- **Cells failing clause (a) (strict cell-by-cell reading)**: h7 r_har (F1 Holm=0.0402, share=91.0%)
- **Triggered branch (strict cell-by-cell reading): MIXED**; under the headline-R-AR reading = (a) FULLY ABSORBED

### tfidf / shifted

- **F2 pass(DM<0 & Holm<.05 & G4):0/8**
- F1 combination-increment wins (DM<0 & Holm<.05 & G4): 4/8: h3 r_ar, h3 r_har, h7 r_ar, h7 r_har
- Cells with a significant F1 Holm and their identity share: h3 r_ar: 101.1%, h3 r_har: 80.2%, h7 r_ar: 145.4%, h7 r_har: 100.2%
- **Cells failing clause (a) (strict cell-by-cell reading)**: h3 r_har (F1 Holm=0.0280, share=80.2%)
- **Triggered branch (strict cell-by-cell reading): MIXED**; under the headline-R-AR reading = (a) FULLY ABSORBED

### qwen_emb / primary

- **F2 pass(DM<0 & Holm<.05 & G4):0/8**
- F1 combination-increment wins (DM<0 & Holm<.05 & G4): 3/8: h3 r_ar, h3 r_har, h7 r_ar
- Cells with a significant F1 Holm and their identity share: h3 r_ar: 112.4%, h3 r_har: 113.1%, h7 r_ar: 133.3%
- **Triggered branch (strict cell-by-cell reading): (a) FULLY ABSORBED** (both readings agree)

### qwen_emb / shifted

- **F2 pass(DM<0 & Holm<.05 & G4):0/8**
- F1 combination-increment wins (DM<0 & Holm<.05 & G4): 2/8: h3 r_ar, h3 r_har
- Cells with a significant F1 Holm and their identity share: h3 r_ar: 101.1%, h3 r_har: 81.2%
- **Cells failing clause (a) (strict cell-by-cell reading)**: h3 r_har (F1 Holm=0.0171, share=81.2%)
- **Triggered branch (strict cell-by-cell reading): MIXED**; under the headline-R-AR reading = (a) FULLY ABSORBED

### prompted_qwen / primary

- **F2 pass(DM<0 & Holm<.05 & G4):0/8**
- F1 combination-increment wins (DM<0 & Holm<.05 & G4): 0/8
- **Triggered branch (strict cell-by-cell reading): (a) FULLY ABSORBED** (both readings agree)

### prompted_qwen / shifted

- **F2 pass(DM<0 & Holm<.05 & G4):0/8**
- **F2 significantly negative (DM>0 & Holm<.05)**: h15 r_ar, h15 r_har — after identity control the text is significantly **harmful**, reported truthfully.
- F1 combination-increment wins (DM<0 & Holm<.05 & G4): 0/8
- Cells with a significant F1 Holm and their identity share: h15 r_ar: NaN% [direction negative], h15 r_har: NaN% [direction negative]
- **Cells failing clause (a) (strict cell-by-cell reading)**: h15 r_ar (F1 Holm=0.0221, share=NaN%, direction negative); h15 r_har (F1 Holm=0.0472, share=NaN%, direction negative)
- **Triggered branch (strict cell-by-cell reading): MIXED** (both readings agree)

## 5. §6.4 MDE discipline adjudication (per arm; null cell = all F2 non-pass cells)

- **tfidf / primary**: powered 8/8, no-gain 0/8, underpowered 0/8 → the "fully absorbed" wording (relative to the published-convention gain) is licensed: all 8 cells have MDE(ent) ≤ G_conv.
- **tfidf / shifted**: powered 8/8, no-gain 0/8, underpowered 0/8 → the "fully absorbed" wording (relative to the published-convention gain) is licensed: all 8 cells have MDE(ent) ≤ G_conv.
- **qwen_emb / primary**: powered 8/8, no-gain 0/8, underpowered 0/8 → the "fully absorbed" wording (relative to the published-convention gain) is licensed: all 8 cells have MDE(ent) ≤ G_conv (qwen_emb has no published reading of its own; G_conv uses the tfidf gain as a PROXY, see §7-5).
- **qwen_emb / shifted**: powered 8/8, no-gain 0/8, underpowered 0/8 → the "fully absorbed" wording (relative to the published-convention gain) is licensed: all 8 cells have MDE(ent) ≤ G_conv (qwen_emb has no published reading of its own; G_conv uses the tfidf gain as a PROXY, see §7-5).
- **prompted_qwen / primary**: powered 2/8, no-gain 6/8, underpowered 0/8 → the cells that do have a published-convention gain (2) are all powered, and the "absorbed" wording is licensed in those cells; in 6 cells this arm's published-convention gain is negative (text loses to raw) — there is nothing to absorb.
- **prompted_qwen / shifted**: powered 2/8, no-gain 6/8, underpowered 0/8 → the cells that do have a published-convention gain (2) are all powered, and the "absorbed" wording is licensed in those cells; in 6 cells this arm's published-convention gain is negative (text loses to raw) — there is nothing to absorb.

Note: the §6.4 downgrade rule takes "the repriced published-convention gain" as its yardstick; the row5 residual observed in each cell is itself generally < MDE(ent) (see the §2 tables), i.e. this design is not sufficient to rule out the **observed residual magnitude** — wherever the prose says "no residual" the MDE must be reported side by side (§6.4, first sentence), and a negative conclusion may only be drawn about "a residual of the magnitude of the published-convention gain".

## 6. identity probe readings (diagnostic, §6.2; OPEN-7: ticker+comnam+date, no transcript)

probe share (pre-declared in prereg §5) = probe combination increment / fulltext (its mirror arm = prompted_qwen) combination increment, i.e. the ratio of the d3 values (same reference, same denominator, hence equal to the ratio of rel%); undefined when the denominator d3≤0 (n/a). The comparison columns for the two fitted arms are context (not pre-declared readings).

### primary

| h | ref | probe row3 rel% | prompted row3 rel% | probe share(vs prompted) | vs tfidf | vs qwen_emb |
|---|---|---|---|---|---|---|
| 3 | r_ar | +8.55% | +0.19% | 4507% | 65% | 64% |
| 3 | r_har | +1.88% | -1.42% | n/a(d3≤0) | 32% | 40% |
| 7 | r_ar | +5.09% | -0.81% | n/a(d3≤0) | 29% | 33% |
| 7 | r_har | +0.93% | -2.38% | n/a(d3≤0) | 13% | 17% |
| 15 | r_ar | +1.04% | -3.29% | n/a(d3≤0) | 7% | 7% |
| 15 | r_har | +0.46% | -3.88% | n/a(d3≤0) | 7% | 7% |
| 30 | r_ar | -0.10% | +0.41% | -24% | -1% | -1% |
| 30 | r_har | -0.05% | +0.13% | -41% | -1% | -1% |

### shifted

| h | ref | probe row3 rel% | prompted row3 rel% | probe share(vs prompted) | vs tfidf | vs qwen_emb |
|---|---|---|---|---|---|---|
| 3 | r_ar | +7.27% | -0.04% | n/a(d3≤0) | 61% | 61% |
| 3 | r_har | +1.71% | -0.27% | n/a(d3≤0) | 42% | 42% |
| 7 | r_ar | +4.62% | -2.71% | n/a(d3≤0) | 56% | 43% |
| 7 | r_har | +1.89% | -1.29% | n/a(d3≤0) | 52% | 41% |
| 15 | r_ar | +1.42% | -4.26% | n/a(d3≤0) | 23% | 16% |
| 15 | r_har | +1.17% | -3.51% | n/a(d3≤0) | 40% | 25% |
| 30 | r_ar | +0.07% | -0.25% | n/a(d3≤0) | 1% | 1% |
| 30 | r_har | -0.00% | -0.83% | n/a(d3≤0) | -0% | -0% |

**Honest probe reading**: under primary, the combination increment of the zero-content probe (r_ar: +8.55/+5.09/+1.04/−0.10%) is at h3/h7/h15 **no lower than, and even exceeds,** that of its mirror fulltext prompted arm (+0.19/−0.81/−3.29/+0.41%) — the share ratio of the prompted arm is meaningless because the denominator d3≈0, so the honest conclusion rests on the paired increments: the gain obtainable from the identity prior alone ≥ the gain after adding the full transcript. For the fitted arms (context columns), the probe reproduces ~64–65% (h3 r_ar) and ~29–33% (h7 r_ar) of their combination increment, decreasing with horizon (see the table).

## 7. Disclosures

1. **Alignment discipline (§2.3)**: primary (strict post-call window) governs the paper wording; differences under shifted (day-0 shift):
   - prompted_qwen/shifted shows **significantly negative** cells at h15 (row5: r_ar Holm=0.0093, r_har Holm=0.0433, DM>0; row3: r_ar Holm=0.0221, r_har Holm=0.0472) — under this alignment the text arm significantly **harms** the combination; the same cells under primary are merely negative and not significant. Reported truthfully; the primary wording is not changed, but the claim qualifiers are written per §2.3.
   - tfidf, qwen_emb: F2 is 0/8 under both alignments and the adjudication structure is unchanged; the cell failing the strict clause (a) moves with the alignment (tfidf: primary h7 r_har → shifted h3 r_har; qwen_emb: none under primary → shifted h3 r_har), i.e. under shifted the two arms move away from / toward the MIXED boundary — under the headline-R-AR reading both arms are (a) under both alignments.
2. **Interpretation of share>100% and NaN**: share = d4/d3 = (MSE_R−MSE_Re)/(MSE_R−MSE_U). >100% = the MSE reduction achieved by the STPEV control alone exceeds the reduction achieved by the text combination (generally 111–145% in the r_ar cells of the two fitted arms); NaN = d3≤0 (the combination increment does not exist, the division is undefined; most prompted cells, probe h30); exploding values (prompted primary h3 r_ar 7890%, h30 r_ar 2807%, h30 r_har 3955%) = artifacts of a denominator d3≈0, not cited as readings.
3. **G4b (within-date swap) dirty cells (mechanical enumeration, Yelp borderline rule)**: under primary, row3_swap is significant/borderline for tfidf (h3 both references, h7 both references, h15 r_ar), qwen_emb (h3 r_ar, h3 r_har p=.070 borderline, h7 r_ar, h15 r_ar) and probe (h3/h7 r_ar) — after the swap permutation the combination still significantly beats the reference, which shows that the row3 combination increment contains a **common date component** (within-date permutation does not destroy call-date-level information); row5_swap dirty: tfidf h3/h7 r_ar (p=.047/.043), qwen_emb h3 r_ar (p=.019), h7 r_ar (|DM|=2.05), probe h3 r_ar, etc. Per §6.3, any cell that is borderline-dirty on the swap does not enter prose claims — this table has no F2 win, and all the r_ar wins of F1 are swap-dirty, so **no cell whatsoever qualifies to enter "win" prose**; the complete flags are in the CSV `gate` column.
4. **Single-shot provenance**: the generated timestamps of the 9 json files are in the §0 table; all were produced in one shot (2026-07-15 14:37–14:38), with force_rerun_reason null throughout; the protocol reference predictions agree row by row with the fitted half-run (max|Δpred|<1e-8, the prediction-level gate introduced after the 2026-07-15 bug fix).
5. **The §6.4 comparison gap for qwen_emb (PROXY disclosure)**: the published scorer was run only for tfidf and prompted_qwen (G1 requires only ≥1 arm to pass the gate), so qwen_emb has no published-convention gain of its own; its G_conv uses **the Δ_pub of tfidf as a proxy**. Directionality: the tfidf gain is the largest of the three arms, so the proxy is biased upward → the "powered" conclusion is **not conservative** for this arm; the tightest cell is h30 r_ar (primary: MDE(ent)=10.07% vs G_conv≈17.4%, margin ≈1.7×) — if the true published-convention gain of qwen_emb is below ~58% of that of tfidf, this cell flips to underpowered. This gap is written truthfully into the Limitations.
6. **Disclosure of how clause (a) is read**: "identity share ≥100% or the combination increment itself not significant" is read **strictly cell by cell** (requiring share≥100% in every cell where F1-Holm is significant), aligning with the "on those horizons" construction of (b); the headline-R-AR reading (the share clause looks only at the OPEN-3 headline reference) is reported alongside separately, and the places where the two disagree are given in §4 — the wording takes the weakest defensible form per the §8 MIXED clause. Significance = Holm(8) two-sided (direction listed separately).
7. **oracle injection (power-side evidence)**: the disclosure sentence transcribed verbatim — "ORACLE injection — s uses test labels BY DESIGN; power calibration only, never citable as forecast performance"; the adaptive target was detected in 62/64 cells in the AR stage; cells not detected: prompted_qwen/primary h7 r_ar (kappa calibration did not converge, converged=False, achieved=-1131% — a mechanical artifact of κ/g exploding when g_text≈0, disclosed truthfully); prompted_qwen/primary h15 r_ar (kappa calibration did not converge, converged=False, achieved=-1617% — a mechanical artifact of κ/g exploding when g_text≈0, disclosed truthfully); the numbers serve power calibration only and enter no statement.
8. **Holm values for the probe**: the protocol json also computes p_holm8 mechanically for the probe, but per §6.2 the probe is a diagnostic row, so this table never displays or invokes its Holm (the §3 tables use "— (diagnostic)" as a placeholder), and the gate column in the CSV is always diagnostic.
