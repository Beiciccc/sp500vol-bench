# C-anon — entity-anonymisation arms (prereg-ea-v1.0): identity share, bound -> point estimate

Committed before any statistic; scored once on 2026-07-16T05:15:15+00:00. identity_share_anon = 1 - masked/unmasked M1 increment (log-space combiner, val-fit test-frozen, day-clustered DM — the crossfamily_llama70.py block verbatim). Holm(6) per arm (3 horizons x 2 references), the pre-declared family; share=n/a cells stay in the family carrying their DM test. `**` = masked increment Holm-significantly POSITIVE (rel% > 0 and Holm p<.05 — the v1.4 direction convention).

## Gates

- S-A PASS: all 6 unmasked vs-HAR cells reproduce forecast_combination_grid.csv at machine precision (rtol 1e-12).
- S-B PASS: unmasked C6 cells reproduce crossfamily_llm.csv (qwen3_32b, event_driven) on all M1 columns at machine precision.
- G1 c6: FAIL — exact 111458/117407 predictions, max|diff| 1.200e-01. DEVIATION RECORDED: GPU-arm bit-identity deviations, v1.1 registered route: c6 = bf16 TP=2 batch-inference nondeterminism on regeneration (exact-match 94.93%, max|diff| 0.12, committed weights, both arms identical env); c2 = GPU retraining nondeterminism under identical recipe/seed (exact 0, max|diff| 0.99; 26% cross-seed spread documented). Ctrl and masked arms of each pair produced in the identical July box environment => within-pair contrast unaffected.
- G1 c2: FAIL — exact 0/333192 predictions, max|diff| 9.901e-01. DEVIATION RECORDED: GPU-arm bit-identity deviations, v1.1 registered route: c6 = bf16 TP=2 batch-inference nondeterminism on regeneration (exact-match 94.93%, max|diff| 0.12, committed weights, both arms identical env); c2 = GPU retraining nondeterminism under identical recipe/seed (exact 0, max|diff| 0.99; 26% cross-seed spread documented). Ctrl and masked arms of each pair produced in the identical July box environment => within-pair contrast unaffected.
- G1 b2: FAIL — **ARM EXITED** (v1.4 CPU rule, no deviation path). Official box G1 (g1_control_b2_boxvenv.json): exact 0/333192, max|diff| 1.926e-01. No share, no tests; full diagnostic chain in Disclosures.
- G2 masking: 112528 docs, 100.0% with >=1 mask, mean masked-char fraction 9.48%; leak rates: own-ticker 0.00%, own-name-token 0.17% (audit sample: results/anon/mask_audit_sample.md).
- G3 truncation: masked prompt_chars median 4664 vs committed 4938; parse_ok 100.0% vs 100.0%.

## Table — masked vs unmasked M1 increment and identity share

| arm | h | vs HAR: unmask rel% | masked rel% | DM | Holm p | share | vs HAR+firmID: unmask rel% | masked rel% | DM | Holm p | share | interval bound | swap retention |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| c6 | 5 | +1.214 | +0.597** | -3.91 | 0.0005965 | +0.51 | +0.448 | +0.334** | -3.43 | 0.003175 | +0.25 | +0.63 | +0.42 |
| c6 | 10 | +0.998 | +0.438 | -2.39 | 0.06771 | +0.56 | +0.253 | +0.224 | -2.22 | 0.07933 | +0.12 | +0.75 | +0.18 |
| c6 | 20 | +0.658 | +0.192 | -1.11 | 0.481 | +0.71 | +0.196 | +0.080 | -1.17 | 0.481 | +0.59 | +0.70 | +0.10 |
| c2 | 5 | +2.135 | +2.083** | -3.81 | 0.0004502 | +0.02 | -0.314 | +0.129** | -4.46 | 3.586e-05 | n/a | +1.15 | +0.52 |
| c2 | 10 | +2.104 | +1.031** | -5.90 | 2.93e-08 | +0.51 | -1.223 | -1.975 | +5.79 | 4.76e-08 | n/a | +1.58 | +0.41 |
| c2 | 20 | +0.920 | +1.277 | -1.68 | 0.09424 | -0.39 | +0.658 | -0.349 | +3.07 | 0.004395 | +1.53 | +0.28 | -2.69 |
| b2 | — | g1-fail: exact-match 0.0000, max|diff| 1.926e-01 (g1_control_b2_boxvenv.json) | — | — | — | — | — | — | — | — | — | — | — |

## Pre-registered branches (per arm)

- **c6**: (c) otherwise (incl. median in (0.50,0.75) or >=2/3 firmID cells Holm-significantly positive = absorption broken): median share 0.56, Holm-significantly-positive masked cells 1/3 vs HAR, 1/3 vs firm-ID — mixed, reported cell by cell (Holm-significantly-positive masked cells: 1/3 vs HAR, 1/3 vs firm-ID).
- **c2**: (b) firm-stable content: defined-cell share median 0.02 <= 0.5 AND the masked increment is still absorbed by the firm-ID reference (<=1/3 firmID cells Holm-significantly positive) — title wording softens (committed) (Holm-significantly-positive masked cells: 2/3 vs HAR, 1/3 vs firm-ID).
- **b2**: g1-fail — arm exited at G1; contributes no defined share cell and no test (see Gates/Disclosures).

## Aggregate point estimates

- identity share^anon, median across the 6 HAR cells: **+0.51** (per arm: {'c6': 0.5609884320586759, 'c2': 0.024397746320116154}).
- median across the 6 firm-ID cells: +0.42.
- triangulation: reference-interval bound median +0.72; matched-swap retention median +0.29 (share and 1-retention should bracket/agree if the interval logic is sound).

## Disclosures

- Branch adjudication follows the prereg-ea v1.4 quantified conditions (registered BEFORE either channel's table fired; shared by both channels): (a) <=> defined-cell share median >= 0.75 AND 0/3 masked HAR cells Holm-significantly positive; (b) <=> median <= 0.50 AND masked increment still absorbed by firmID (<=1/3 firmID cells Holm-significantly positive); (c) <=> otherwise (incl. median in (0.50,0.75) or >=2/3 firmID cells Holm-significantly positive = absorption broken). 'Holm-significantly positive' = masked rel% > 0 AND Holm p < .05. The registered quantities themselves (per-cell shares, Holm p) are reported unconditionally above.
- share is undefined (n/a) where the unmasked increment is <= 0 — no clipping, no exclusion from the table.
- **b2 (B2_tfidf_ridge) EXITED at G1**: arm exited at G1 (v1.4 CPU rule, no deviation path). Official G1 = the box control (g1_control_b2_boxvenv.json; the registered arm executed on the box): exact-match 0, max|diff| 1.926e-01. Diagnosis: the committed June (env x cache) pair is unreconstructible (env.json has no package versions, pip_freeze_hash=null; scipy 1.18 does not exist on py3.11 so the June box scipy was <=1.17.1) and the text-store lineage has drifted — the box-control and local-control TF-IDF fits agree on CV alphas (1.0 at h=5/10/20) yet carry DIFFERENT vocabulary SETS with idf max|diff| 7.5, so the two text caches are different lineages and the committed fit's lineage is a third, unreconstructible state. Local reproduction attempts bottom out at max|diff| 1.303e-03 (sklearn 1.8.0/py3.11.15; numpy 2.4.6 and 2.3.5 deviations bit-identical -> numpy ruled out). No share, no tests; see Disclosures.
- b2 G1 reproduction attempts, side by side — box sklearn1.9.0/py3.12 box-cache (OFFICIAL): exact 0/333192, max|diff| 1.926e-01 [g1_control_b2_boxvenv.json]; local sklearn1.8.0/py3.11.15 numpy2.4.6: exact 0/333192, max|diff| 1.303e-03 [g1_control_b2_local_numpy246.json]; local trial venv numpy2.3.5: exact 0/333192, max|diff| 1.303e-03 [g1_control_b2_local_numpy235.json]. Vocabulary drift in one sentence: the box-control and local-control model.pkl fits share identical CV alphas yet hold DIFFERENT vocabulary sets with idf max|diff| 7.5 — text-store lineage drift, not a numeric-library effect. The same-lineage box ctrl/masked pair lives in the NON-REGISTERED annex (anon_annex_samelineage.{csv,md}), outside this table and outside branch adjudication.
- Scope (registered): event_driven only; long_form is a stretch goal, scored separately (anon_arm_lf.{csv,md}, --channel lf); C5x not run (GPU budget), pre-registered as not-done.
- Single-shot: this file is written once; re-running the script refuses while it exists.
