# C-anon-lf — entity-anonymisation, LONG-FORM stretch, B2-only (prereg-ea v1.4 §C-anon: long-form stretch): identity share, bound -> point estimate

Committed before any statistic; scored once on 2026-07-16T07:39:17+00:00. identity_share_anon = 1 - masked/unmasked M1 increment (log-space combiner, val-fit test-frozen, day-clustered DM — the crossfamily_llama70.py block verbatim). Holm(6) per arm (3 horizons x 2 references), the pre-declared family; share=n/a cells stay in the family carrying their DM test. `**` = masked increment Holm-significantly POSITIVE (rel% > 0 and Holm p<.05 — the v1.4 direction convention).

## Gates

- S-A n/a: no executed arm in this channel (B2-lf exited at G1; C5/C2/C6-lf not-executed) — no unmasked cell was recomputed.
- S-B n/a: the crossfamily anchor is C6/event_driven-only and C6-lf is not an arm in this channel (see Disclosures).
- G1 b2: FAIL — **ARM EXITED** (v1.4 CPU rule, no deviation path). Official box G1 (g1_control_b2_lf_boxvenv.json): exact 0/94237, max|diff| 2.846e-01. No share, no tests; full diagnostic chain in Disclosures.
- G1 c5: not-executed — arm excluded pre-statistic (see Not-executed arms).
- Triangulation source (E-lf document-swap): retention column n/a — swap_longform_meta.json absent — E-lf gate status unknown (v1.4: missing status counts as not passed); cells absent from swap_longform.csv also render n/a.
- G2 masking: 31601 docs, 100.0% with >=1 mask, mean masked-char fraction 5.24%; leak rates: own-ticker 0.00%, own-name-token 0.31% (audit sample: results/anon/mask_audit_sample_lf.md).
- G3 (LF form): not-executed — no GPU arm in this channel (B2-only; C5-lf excluded pre-statistic), so there is no frozen artefact whose hash invariance could be gated; no C6 arm -> no truncation stats either.

## Table — masked vs unmasked M1 increment and identity share

| arm | h | vs HAR: unmask rel% | masked rel% | DM | Holm p | share | vs HAR+firmID: unmask rel% | masked rel% | DM | Holm p | share | interval bound | E-lf doc-swap retention |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| c5 | 5 | n/a (not-executed) | — | — | — | — | — | — | — | — | — | — | — |
| c5 | 10 | n/a (not-executed) | — | — | — | — | — | — | — | — | — | — | — |
| c5 | 20 | n/a (not-executed) | — | — | — | — | — | — | — | — | — | — | — |
| b2 | — | g1-fail: exact-match 0.0000, max|diff| 2.846e-01 (g1_control_b2_lf_boxvenv.json) | — | — | — | — | — | — | — | — | — | — | — |

## Not-executed arms

- **c5** (C5_qwen3): excluded pre-statistic (prereg-ea v1.4): HAR side constructively n/a — committed m1_multiseed.csv long_form C5_qwen3 seed-2026 rel_impr_pct = -1.0347/-3.1346/-6.6467 (primary, same single-seed basis as this scorer; deployable_combiner FIXED mean rel% -0.85/-2.48/-5.97 corroborates, 3-seed basis), all negative, so every HAR share cell is empty before execution; firmID side — no committed table carries that increment, so constructive n/a is NOT invoked: excluded on GPU budget/scope (as C6-lf), and its execution could not change the branch adjudication (the share median takes HAR cells only)

## Pre-registered branches (per arm; stretch-channel readout — the ED table carries the binding adjudication)

- **b2**: g1-fail — arm exited at G1; contributes no defined share cell and no test (see Gates/Disclosures).
- **c5**: not-executed (see Not-executed arms).
- **channel adjudication: undefined** — no executed arm and hence no defined share cell in this channel (all arms g1-failed or not-executed); the registered branch conditions have no domain to act on.

## Aggregate point estimates

- none: no executed arm in this channel — no share cell, no aggregate point estimate (see Gates and Not-executed arms).

## Disclosures

- Branch adjudication follows the prereg-ea v1.4 quantified conditions (registered BEFORE either channel's table fired; shared by both channels): (a) <=> defined-cell share median >= 0.75 AND 0/3 masked HAR cells Holm-significantly positive; (b) <=> median <= 0.50 AND masked increment still absorbed by firmID (<=1/3 firmID cells Holm-significantly positive); (c) <=> otherwise (incl. median in (0.50,0.75) or >=2/3 firmID cells Holm-significantly positive = absorption broken). 'Holm-significantly positive' = masked rel% > 0 AND Holm p < .05. The registered quantities themselves (per-cell shares, Holm p) are reported unconditionally above.
- share is undefined (n/a) where the unmasked increment is <= 0 — no clipping, no exclusion from the table.
- **b2 (B2_tfidf_ridge) EXITED at G1**: arm exited at G1 (v1.4 CPU rule: 1e-8 full-panel reproduction, no deviation path). Official G1 = the box control (g1_control_b2_lf_boxvenv.json; same box venv/cache lineage as the ED control): exact-match 0, max|diff| far above 1e-8 (row g1 columns). Same root cause as the ED exit — committed June (env x cache) pair unreconstructible + text-store lineage drift (vocabulary-set/idf evidence in the ED table's b2 diagnostics). No share, no tests.
- Channel: the registered STRETCH ("long-form as stretch, only if ED completes"), trimmed to B2-ONLY before any LF statistic; the event_driven table (anon_arm.{csv,md}) is the registered primary and carries the branch adjudication that binds the paper's wording — this table triangulates it on the long-form panel.
- Registered executed arm: B2-lf ONLY (masked RETRAIN under the fixed committed recipe; CPU classical model — no neural text model in this channel). It EXITED at G1 under the v1.4 CPU rule (see Gates + the b2 disclosure above), so the channel closes with ZERO executed arms: this table documents the exit and the pre-statistic exclusions; it reports no masked statistic.
- LF degeneracy (v1.4, registered): B2-lf's UNMASKED increment is already fully absorbed by the firm-identity reference — committed firm_identity_control.csv long_form B2 rel_impr_pct_firm = -0.615/-3.892/-8.089% (h=5/10/20), all non-positive — so the firmID-side shares are constructively n/a before execution, the share median draws on the 3 HAR cells only, and branch (b)'s absorption clause is expected-true and weakly discriminating in this channel; the paper wording is softened accordingly.
- C5-lf NOT run (prereg-ea v1.4, decided before any LF statistic; rows above marked not-executed). HAR side: constructively n/a — primary citation committed m1_multiseed.csv long_form C5_qwen3 seed-2026 rel_impr_pct = -1.0347/-3.1346/-6.6467 (the same single-seed basis this scorer uses); deployable_combiner FIXED mean rel% -0.85/-2.48/-5.97 corroborates (3-seed basis); all negative, so every HAR share cell is empty before execution. firmID side: no committed table carries that increment, so constructive n/a is NOT invoked — excluded on GPU budget/scope (as C6-lf), and its execution could not change the branch adjudication (the share median takes HAR cells only).
- C2-lf NOT run (decided before any LF statistic): two fixed-recipe FinBERT trainings (~20-30 GPU-h) for marginal defensive value; the ED C2 arm already covers the fine-tuned lineage, and the E-lf C2 arm is registered artefact-lost.
- C6-lf NOT run: the committed UNMASKED C6 long-form run exists (genuine LLM output over all 11,907 long-form val+test filings), so a masked C6-lf arm is well-defined; it was excluded on GPU budget/scope (another Qwen3-32B bf16 TP=2 block), NOT because the baseline is missing. The LF channel therefore lacks the prompted-LLM lineage.
- Triangulation column (v1.4 operationalisation): the E-lf B2 arm's same-horizon DOCUMENT-swap retention from committed swap_longform.csv, reported ONLY where the cell exists AND the E-lf gates G1-G3 are verified PASS in swap_longform_meta.json (missing status counts as not passed -> n/a, reason in the Gates section). The firm-ref swap variant exists only for the ED forecast-level swap, hence n/a here.
- Single-shot: this file is written once; re-running the script refuses while it exists.
