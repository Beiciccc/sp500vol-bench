# Pre-declared Holm families — enumeration (release documentation)

R11/R12 reviewers asked for a verifiable enumeration of every multiple-testing family.
Each family below is Holm-corrected WITHIN itself; families are never pooled. "Declared" cites
the pre-registration tag or the committed table whose header fixed the family before scoring.

| # | family | size | members | declared in | carried by |
|---|---|---|---|---|---|
| F1 | Standalone leaderboard | 180 | challenger-vs-A2 squared-error DM, models×disclosures×horizons | protocol (FACTS §1/§3) | dm_pairwise_clustered.csv |
| F2 | Standalone variance-unit | 153 | text/fusion subset of F1 under variance-unit QLIKE | variance_unit_standalone180.csv header | same |
| F3 | M1 primary (69-cell) | 69 | combination cells, vol-unit, seed-ensemble | protocol; FACTS §4 | m1_ensemble_primary.csv |
| F4 | M1 variance-unit | 69 | F3 under variance-unit QLIKE | FACTS §4 | m1_variance_unit |
| F5 | Firm-identity reference | 69 | F3 vs identity-augmented reference; battery = 4 specs, each its own 69-family | FACTS §5 | firm_identity_control/_ensemble.csv |
| F6 | Maximal pool | 69 | F3 vs 5-model pool reference | FACTS §5 | maximal_reference_ensemble.md |
| F7 | Residual symmetric family | 12 | pre-declared 12-cell two-sided family for the 8-K residual | prereg (FACTS §13b) | omnibus/residual tables |
| F8 | Earnings-window control | 3 | C6 ED h∈{5,10,20} vs firmID+Item-2.02 reference | prereg-rfa (FACTS §13a) | itemcode_control.csv |
| F9 | Deployable FIXED | 75 | pooled DM per cell, fixed val-frozen scheme | declared pre-inspection in deployable_combiner.md | deployable_combiner.csv |
| F10 | Deployable EXPANDING | 75 | pooled DM per cell, expanding scheme | same | same |
| F11 | Anonymisation, per arm | 6 | 3 horizons × 2 references, per executed arm (c6, c2) | prereg-ea v1.0/v1.4 | anon_arm.csv |
| F12 | Range-based PK / GK | 69 each | cascade rungs under Parkinson / Garman-Klass labels, per rung per proxy | prereg-cd v1.0 | rangebased_cascade.csv |
| F13 | Public-variant panels | as F1/F3 | panels B/C replicate the F1/F3 family structures | prereg-h v1.0 | public_variant_cascade.csv |
| F14 | MAEC audit | 24 | 4 horizons × arms × 2 references, clustered | prereg-maec | maec_audit.md |
| F15 | Yelp cascade | per-gate | month-clustered DM; label-shuffle placebo primary | second-domain plan (G1–G5) | yelp_cascade.md |

Notes: (i) placebo gates (label-shuffle, |DM|<2) are applied AFTER Holm within the same family —
"genuine" = DM<0 ∧ Holm<.05 ∧ placebo-clean (FACTS §1). (ii) Not-executed / retired arms carry no
tests and never enter a family (prereg-ea v1.4/v1.5). (iii) The n/a rule affects share estimands
only, never family membership (v1.4 clarification).
