# ROW-2 — delta-text (B2d) + firm-demeaned-target (C2dm) arms through the M1 protocol

## RESTATED vs BEFORE

BEFORE = the ORIGINAL level-representation counterparts (B2_tfidf_ridge long_form; C2_finbert_s1 long_form + event_driven), committed in m1_ensemble_primary.csv (seed2026 `s26_*` columns — seed-matched to the new arms) and firm_identity_control.csv (canonical firm-mean reference of maximal_reference_firm_control.py). RESTATED = the same 9 (disc x h) cells with the text model made WITHIN-FIRM BY CONSTRUCTION: delta TF-IDF features (Lazy-Prices lineage; first filing of each (cik, form) sequence excluded) or a firm-demeaned training objective (log RV minus train-window firm mean; predictions restored to level units; seed 2026 reduced form). Combiner weights val-fit test-frozen; day-clustered DM; 5-seed label-shuffle placebo.

| quantity | BEFORE (level representations) | RESTATED (within-firm arms) |
|---|---|---|
| genuine vs single recalibrated HAR (Holm, placebo-gated) | 6/9 (s26 basis) | **6/9** |
| survives the firm-identity reference (Holm) | 0/9 | **0/9** |
| HAR-genuine cell composition | long_form/B2_tfidf_ridge/h5, long_form/B2_tfidf_ridge/h10, long_form/B2_tfidf_ridge/h20, long_form/C2_finbert_s1/h10, event_driven/C2_finbert_s1/h5, event_driven/C2_finbert_s1/h10 | long_form/C2dm_finbert_s1/h5, long_form/C2dm_finbert_s1/h10, long_form/C2dm_finbert_s1/h20, event_driven/C2dm_finbert_s1/h5, event_driven/C2dm_finbert_s1/h10, event_driven/C2dm_finbert_s1/h20 |

**Pre-declared multiplicity:** two Holm families for the new arms, each the 9 cells of this table — FAMILY-H (clustered DM p vs recalibrated HAR) and FAMILY-F (clustered DM p vs the firm-identity-augmented reference). Original columns keep their committed 69-cell-family Holm values (marked `holm69`). `genuine` = clustered DM<0, Holm<.05, |mean placebo DM|<2.


## Standalone (text-alone) test accuracy — new arm vs original counterpart
QLIKE is VOL-unit q(y, f) throughout (M1 convention; committed metrics.json stores VARIANCE-unit q(y^2, f^2) — cross-checked in SANITY below).
| disc | new arm | h | n | QLIKE new | R2 new | QLIKE orig | R2 orig |
|---|---|--:|--:|--:|--:|--:|--:|
| long_form | B2d_tfidf_delta | 5 | 7848 | 0.1775 | -0.109 | 0.2085 | -0.122 |
| long_form | B2d_tfidf_delta | 10 | 7830 | 0.1461 | -0.116 | 0.1646 | -0.105 |
| long_form | B2d_tfidf_delta | 20 | 7799 | 0.1151 | -0.157 | 0.1284 | -0.120 |
| long_form | C2dm_finbert_s1 | 5 | 7951 | 0.1973 | -0.140 | 0.1676 | -0.048 |
| long_form | C2dm_finbert_s1 | 10 | 7933 | 0.1236 | -0.027 | 0.1647 | -0.108 |
| long_form | C2dm_finbert_s1 | 20 | 7902 | 0.1549 | -0.266 | 0.1663 | -0.401 |
| event_driven | C2dm_finbert_s1 | 5 | 25109 | 0.2133 | -0.135 | 0.1719 | -0.024 |
| event_driven | C2dm_finbert_s1 | 10 | 25001 | 0.2129 | -0.254 | 0.1935 | -0.203 |
| event_driven | C2dm_finbert_s1 | 20 | 24732 | 0.1623 | -0.213 | 0.1480 | -0.182 |

## M1 vs single recalibrated HAR (log-space combiner, day-clustered DM, FAMILY-H Holm)
| disc | new arm | h | n_test | n_days | rel% NEW | cluDM | Holm | placebo | genuine | rel% ORIG(s26,holm69) | cluDM | genuine(s26) | rel% ORIG same-rows | cluDM |
|---|---|--:|--:|--:|--:|--:|--:|--:|---|--:|--:|---|--:|--:|
| long_form | B2d_tfidf_delta | 5 | 7848 | 808 | -0.16 | +2.11 | 0.036 | -1.04 | no | +3.33 (0.000) | -5.39 | YES | +3.32 | -5.26 |
| long_form | B2d_tfidf_delta | 10 | 7830 | 802 | -0.22 | +2.52 | 0.036 | +0.77 | no | +3.48 (0.000) | -8.89 | YES | +3.55 | -9.13 |
| long_form | B2d_tfidf_delta | 20 | 7799 | 793 | -1.46 | +2.39 | 0.036 | +0.05 | no | +5.92 (0.000) | -9.04 | YES | +6.02 | -8.95 |
| long_form | C2dm_finbert_s1 | 5 | 7951 | 809 | +1.14 | -4.54 | 0.000 | -0.48 | YES | +0.56 (1.000) | -0.84 | no | +0.56 | -0.84 |
| long_form | C2dm_finbert_s1 | 10 | 7933 | 803 | +0.54 | -3.95 | 0.000 | +0.86 | YES | +4.56 (0.000) | -6.46 | YES | +4.56 | -6.46 |
| long_form | C2dm_finbert_s1 | 20 | 7902 | 794 | +9.62 | -5.70 | 0.000 | -1.75 | YES | -0.08 (1.000) | +0.88 | no | -0.08 | +0.88 |
| event_driven | C2dm_finbert_s1 | 5 | 25109 | 996 | +1.12 | -3.39 | 0.003 | +1.01 | YES | +2.14 (0.000) | -4.95 | YES | +2.14 | -4.95 |
| event_driven | C2dm_finbert_s1 | 10 | 25001 | 991 | +1.46 | -4.56 | 0.000 | +0.57 | YES | +2.10 (0.000) | -5.52 | YES | +2.10 | -5.52 |
| event_driven | C2dm_finbert_s1 | 20 | 24732 | 981 | +1.86 | -4.67 | 0.000 | +0.72 | YES | +0.92 (1.000) | -0.50 | no | +0.92 | -0.50 |

## M1 vs FIRM-IDENTITY-augmented reference (the DA-CRITICAL test; FAMILY-F Holm)
Reference = exp OLS[1, log fHAR, log firm-mean-val-RV] on the 5-price-model panel (committed canonical spec). A within-firm-by-construction model that still cannot beat this reference carries no filing-specific increment.
| disc | new arm | h | n_test | rel% NEW | cluDM | Holm | placebo | genuine | rel% ORIG(holm69) | cluDM | survives(orig) | rel% ORIG same-rows | cluDM | verdict |
|---|---|--:|--:|--:|--:|--:|--:|---|--:|--:|---|--:|--:|---|
| long_form | B2d_tfidf_delta | 5 | 7450 | +0.21 | +0.97 | 0.440 | +0.04 | no | -0.61 (0.002) | +4.19 | no | -0.71 | +4.04 | **HURTS vs HAR** |
| long_form | B2d_tfidf_delta | 10 | 7068 | +0.11 | +1.23 | 0.440 | +0.21 | no | -3.89 (0.000) | +7.02 | no | -3.99 | +8.05 | **HURTS vs HAR** |
| long_form | B2d_tfidf_delta | 20 | 6999 | -0.60 | +1.61 | 0.324 | +0.86 | no | -8.09 (0.000) | +8.15 | no | -8.23 | +8.01 | **HURTS vs HAR** |
| long_form | C2dm_finbert_s1 | 5 | 7550 | -0.69 | +3.72 | 0.001 | -0.22 | no | -0.18 (1.000) | +1.89 | no | -0.18 | +1.89 | **adds vs HAR; HURTS vs firm ref** |
| long_form | C2dm_finbert_s1 | 10 | 7167 | -2.21 | +2.36 | 0.074 | +0.08 | no | -0.60 (0.002) | +4.09 | no | -0.60 | +4.09 | **adds vs HAR only (absorbed by firm ref)** |
| long_form | C2dm_finbert_s1 | 20 | 7097 | +2.52 | -3.88 | 0.001 | -2.27 | no | -0.92 (1.000) | +0.17 | no | -0.92 | +0.17 | **beats firm ref but FAILS placebo (artifact)** |
| event_driven | C2dm_finbert_s1 | 5 | 23855 | -0.49 | +3.65 | 0.001 | +0.69 | no | -0.28 (0.000) | +4.88 | no | -0.28 | +4.88 | **adds vs HAR; HURTS vs firm ref** |
| event_driven | C2dm_finbert_s1 | 10 | 22785 | -1.32 | +5.25 | 0.000 | +0.05 | no | -1.22 (0.000) | +6.33 | no | -1.22 | +6.33 | **adds vs HAR; HURTS vs firm ref** |
| event_driven | C2dm_finbert_s1 | 20 | 22318 | -2.33 | +5.00 | 0.000 | +0.84 | no | +0.58 (1.000) | -0.81 | no | +0.58 | -0.81 | **adds vs HAR; HURTS vs firm ref** |

## SANITY
- **GATE-A (HAR reference):** the 9 original-counterpart cells re-run through this script's pipeline reproduce m1_ensemble_primary.csv `s26_*` columns (qlike_R/U, g_log, clustered DM/p, placebo) to max|diff| = 2.22e-16 — PASS.
- **GATE-B (firm reference):** the same cells reproduce firm_identity_control.csv (n_test exact; qlike_Rfirm/Ufirm, g_text, rel%, clustered DM/p) to max|diff| = 2.22e-16 — PASS.
- Overall gate max|diff| = 2.22e-16 (< 1e-09); gate cells = 9.
- Standalone cross-check vs each new run's committed metrics.json (which stores VARIANCE-unit QLIKE q(y^2,f^2); the tables above use the M1 vol-unit convention): max|dQLIKE_var| = 0.00e+00, max|dR2| = 0.00e+00 (informational).
- B2d panel excludes first-of-sequence filings (test rows 7799-7848 vs original 7902-7951); the same-rows columns re-run the ORIGINAL text on the reduced panel so row exclusion cannot explain new-vs-original differences. C2dm predictions are level-unit (config demeaning.mechanics); C2dm is seed-2026 only (reduced form, disclosed).


## HONEST bottom line

- **The null DEEPENS.** Making the text model within-firm by construction — delta TF-IDF features or a firm-demeaned training objective — recovers NO filing-specific increment that survives the firm-identity reference (0/9 cells genuine under the pre-declared criterion: cluDM<0 AND FAMILY-F Holm<.05 AND |placebo|<2; 1 cell(s) clear Holm but fail the placebo — reported verbatim below; vs HAR alone: 6/9 genuine).
- CAVEAT (report verbatim): long_form/C2dm_finbert_s1/h20 DOES beat the firm reference on paper (+2.52%, cluDM -3.88, Holm 0.001) but FAILS the label-shuffle placebo (mean placebo DM -2.27, |.|>=2): permuted text 'improves' the reference too, so the gain is a combination artifact of the demeaned forecast's marginal distribution, not filing-specific information.
- B2d delta TF-IDF (long_form): rel% vs HAR -1.46..-0.16 (orig B2 same-rows +3.32..+6.02); vs firm ref -0.60..+0.21%.
- C2dm demeaned FinBERT: long_form rel% vs firm ref -2.21..+2.52%; event_driven -2.33..-0.49%.
- The HAR-genuine count is unchanged in TOTAL (6/9 -> 6/9) but the COMPOSITION flips: B2d delta text LOSES all of level-B2's HAR-genuine cells (it significantly HURTS vs recalibrated HAR, FAMILY-H Holm<.05) while demeaned C2dm adds vs HAR in all 6 of its cells — yet every such gain is absorbed by (or reverses under) the firm-identity reference, exactly like the level originals (0/9 orig survivors).