**NON-PREREGISTERED EXPLORATORY — SAME-LINEAGE B2 ANONYMISATION SENSITIVITY ANNEX. NOT PART OF anon_arm.{csv,md}; NO BRANCH ADJUDICATION; CITABLE IN THE MAIN TEXT ONLY AS A SENSITIVITY REMARK.**

# Same-lineage B2 ctrl/masked contrast (exploratory annex)

The registered B2 anon arms exited at G1 — the box control fails the committed anchor (env unreconstructible + text-store lineage drift; see anon_arm.{csv,md} b2 disclosures). This annex reports the box's OWN ctrl/masked pair: both legs share one venv and one text cache, so the contrast is internally self-consistent, but neither leg ties to the committed estimand — hence exploratory, descriptive p only (no Holm), no branch mapping.

| channel | h | vs HAR: ctrl rel% | masked rel% | share | DM | descr. p | vs HAR+firmID: ctrl rel% | masked rel% | share | DM | descr. p |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ed | 5 | +1.211 | +1.180 | +0.03 | -2.73 | 0.006512 | +0.421 | +0.496 | -0.18 | -3.64 | 0.0002892 |
| ed | 10 | +1.335 | +1.187 | +0.11 | -2.02 | 0.04368 | +0.307 | +0.344 | -0.12 | -4.03 | 5.928e-05 |
| ed | 20 | +1.810 | +1.612 | +0.11 | -2.40 | 0.01659 | +0.105 | +0.066 | +0.37 | -4.90 | 1.143e-06 |
| lf | 5 | +1.833 | +3.445 | -0.88 | -5.92 | 4.849e-09 | -1.728 | -1.002 | n/a | +5.26 | 1.832e-07 |
| lf | 10 | +1.651 | +3.679 | -1.23 | -8.98 | 1.969e-18 | -3.964 | -3.947 | n/a | +7.65 | 5.813e-14 |
| lf | 20 | +3.466 | +5.952 | -0.72 | -9.44 | 3.925e-20 | -6.021 | -6.938 | n/a | +7.77 | 2.518e-14 |

## Disclosures

- SAME-LINEAGE, SELF-CONSISTENT: ctrl and masked were produced in the SAME box venv on the SAME box text cache, so the ctrl/masked contrast is internally coherent (identical vectoriser lineage on both legs).
- BUT ANCHOR-FAILED: the same box ctrl does NOT reproduce the committed June predictions (official g1_control_b2*_boxvenv.json: exact-match 0; per-channel numbers in the csv's g1_official_* columns). Cause per the closed diagnosis: the committed (env x cache) pair is unreconstructible (env.json without package versions) and the text-store lineage drifted (different fitted vocabulary sets, idf max|diff| 7.5). The ctrl leg is therefore NOT the committed unmasked increment, and shares here are NOT the registered identity-share estimand.
- Statistics are DESCRIPTIVE: day-clustered DM p values are reported raw, deliberately outside any Holm family; nothing here enters branch adjudication or the registered tables.
- File hygiene: regenerated on each run (channels are merged by row replacement); no single-shot discipline applies to this annex.

Generated 2026-07-16T05:51:23+00:00 by scripts/analysis/anon_annex_samelineage.py.
