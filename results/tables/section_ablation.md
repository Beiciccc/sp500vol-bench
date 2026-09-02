# E2 — Section ablation (long_form, B2 TF-IDF+Ridge recipe, seed 2026)

Which 10-K/10-Q sections carry the incremental text signal over a recalibrated
HAR-RV (A2)? Four TF-IDF(1-2gram, 5k features, sublinear)+per-horizon-Ridge
variants trained on a single section (or the complement) of each filing,
replicating the archived B2_tfidf_ridge recipe exactly; splits joined from A2.

**sections_json finding:** values are section TEXT, not offsets — but the dataset
parser takes the FIRST label match, which in filings with a table of contents is
the TOC line (median 24–96 chars; only ~10–17% substantive). Sections were
therefore re-extracted from the cached full text with a TOC-skip rule (first
label-match span ≥ 500 chars, else last match). Where sections_json DID
capture substantive text (>2000 chars), the re-extraction recovers it:

- item1a: 4578/4578 (100.0%) of substantive sections_json heads recovered
- item7: 3392/3392 (100.0%) of substantive sections_json heads recovered
- item7a: 724/724 (100.0%) of substantive sections_json heads recovered

## Sanity: full-text B2 recipe re-run vs archived B2 (test QLIKE, variance-unit)

| horizon | archived B2 | re-run | diff |
|---|---|---|---|
| 5 | 1.2371 | 1.2371 | +0.00% |
| 10 | 1.0522 | 1.0522 | +0.00% |
| 20 | 0.7083 | 0.7084 | +0.01% |

## Results

Standalone = test QLIKE (variance-unit, as metrics.json). M1 = leakage-free
log-space combination vs recalibrated HAR (fit on val, frozen on test);
rel improve % = QLIKE(f_R) → QLIKE(f_U) reduction (vol-unit QLIKE, fc.qlike);
DM on fc.qlike losses, h=horizon, NEGATIVE stat = text adds (same convention
as scripts/analysis/forecast_combination.py).

| model_id        | section                                                        |   horizon_days |   n_test |   frac_nonempty_all |   frac_nonempty_10K |   frac_nonempty_10Q |   qlike_test_var |   qlike_vs_B2_pct |   m1_qlike_fR |   m1_qlike_fU |   m1_rel_improve_pct |   m1_g_text |   m1_dm_stat |   m1_dm_p |
|:----------------|:---------------------------------------------------------------|---------------:|---------:|--------------------:|--------------------:|--------------------:|-----------------:|------------------:|--------------:|--------------:|---------------------:|------------:|-------------:|----------:|
| B2_tfidf_ridge  | full text (archived B2 run)                                    |              5 |     7951 |              1      |              1      |              1      |           1.2371 |              0    |       0.12087 |       0.11684 |                3.331 |      0.2368 |      -11.338 |         0 |
| B2_tfidf_ridge  | full text (archived B2 run)                                    |             10 |     7933 |              1      |              1      |              1      |           1.0522 |              0    |       0.08729 |       0.08425 |                3.482 |      0.17   |      -15.299 |         0 |
| B2_tfidf_ridge  | full text (archived B2 run)                                    |             20 |     7902 |              1      |              1      |              1      |           0.7083 |              0    |       0.07008 |       0.06593 |                5.92  |      0.2044 |      -19.617 |         0 |
| B2sec_fullrepro | B2 recipe re-run on full text (sanity, memory-safe vocab path) |              5 |     7951 |              1      |              1      |              1      |           1.2371 |             -0    |       0.12087 |       0.11684 |                3.331 |      0.2368 |      -11.338 |         0 |
| B2sec_fullrepro | B2 recipe re-run on full text (sanity, memory-safe vocab path) |             10 |     7933 |              1      |              1      |              1      |           1.0522 |             -0    |       0.08729 |       0.08425 |                3.484 |      0.1701 |      -15.299 |         0 |
| B2sec_fullrepro | B2 recipe re-run on full text (sanity, memory-safe vocab path) |             20 |     7902 |              1      |              1      |              1      |           0.7084 |              0.01 |       0.07008 |       0.06594 |                5.908 |      0.204  |      -19.62  |         0 |
| B2sec_item1a    | Risk Factors only (10-K Item 1A / 10-Q Part II Item 1A)        |              5 |     7951 |              0.9496 |              0.9892 |              0.9364 |           1.4123 |             14.16 |       0.12087 |       0.11921 |                1.372 |      0.1617 |       -7.772 |         0 |
| B2sec_item1a    | Risk Factors only (10-K Item 1A / 10-Q Part II Item 1A)        |             10 |     7933 |              0.9496 |              0.9892 |              0.9364 |           1.179  |             12.05 |       0.08729 |       0.08625 |                1.199 |      0.0971 |      -10.755 |         0 |
| B2sec_item1a    | Risk Factors only (10-K Item 1A / 10-Q Part II Item 1A)        |             20 |     7902 |              0.9496 |              0.9892 |              0.9364 |           0.7337 |              3.58 |       0.07008 |       0.06882 |                1.808 |      0.1002 |      -14.267 |         0 |
| B2sec_item7     | MD&A only (10-K Item 7 / 10-Q Part I Item 2)                   |              5 |     7951 |              0.9958 |              0.995  |              0.996  |           1.3922 |             12.53 |       0.12087 |       0.11814 |                2.262 |      0.1431 |      -12.336 |         0 |
| B2sec_item7     | MD&A only (10-K Item 7 / 10-Q Part I Item 2)                   |             10 |     7933 |              0.9958 |              0.995  |              0.996  |           1.1955 |             13.62 |       0.08729 |       0.08551 |                2.038 |      0.0936 |      -15.564 |         0 |
| B2sec_item7     | MD&A only (10-K Item 7 / 10-Q Part I Item 2)                   |             20 |     7902 |              0.9958 |              0.995  |              0.996  |           0.7708 |              8.82 |       0.07008 |       0.06784 |                3.205 |      0.1041 |      -20.168 |         0 |
| B2sec_item7a    | Market risk only (10-K Item 7A; empty for 10-Q)                |              5 |     7951 |              0.2461 |              0.9844 |              0      |           1.3686 |             10.63 |       0.12087 |       0.12258 |               -1.415 |     -0.3173 |        8.41  |         0 |
| B2sec_item7a    | Market risk only (10-K Item 7A; empty for 10-Q)                |             10 |     7933 |              0.2461 |              0.9844 |              0      |           1.1561 |              9.87 |       0.08729 |       0.08914 |               -2.112 |     -0.5011 |        7.048 |         0 |
| B2sec_item7a    | Market risk only (10-K Item 7A; empty for 10-Q)                |             20 |     7902 |              0.2461 |              0.9844 |              0      |           0.7574 |              6.93 |       0.07008 |       0.07263 |               -3.631 |     -0.6866 |        6.459 |         0 |
| B2sec_rest      | Full text MINUS Item 1A + MD&A + Item 7A spans                 |              5 |     7951 |              1      |              1      |              1      |           1.184  |             -4.29 |       0.12087 |       0.11836 |                2.078 |      0.1831 |       -9.147 |         0 |
| B2sec_rest      | Full text MINUS Item 1A + MD&A + Item 7A spans                 |             10 |     7933 |              1      |              1      |              1      |           1.0255 |             -2.53 |       0.08729 |       0.08598 |                1.509 |      0.111  |      -10.319 |         0 |
| B2sec_rest      | Full text MINUS Item 1A + MD&A + Item 7A spans                 |             20 |     7902 |              1      |              1      |              1      |           0.6947 |             -1.91 |       0.07008 |       0.06818 |                2.719 |      0.1798 |      -11.314 |         0 |

## Conclusions

1. **Sanity passed exactly**: the memory-safe re-implementation of the B2 recipe
   reproduces the archived B2 long_form run to within rounding (+0.00/+0.00/+0.01%
   test QLIKE) — far inside the ~5% tolerance — validating the whole pipeline.
2. **MD&A (Item 7 / 10-Q Part I Item 2) carries the largest single-section
   increment**: M1 rel QLIKE improvement 2.26/2.04/3.21% (h=5/10/20), DM −12.3 to
   −20.2 (p < 1e-5), roughly two-thirds of the full-text increment (3.33/3.48/5.92%).
3. **Risk Factors (Item 1A) add a smaller but still significant increment**
   (1.37/1.20/1.81%, DM −7.8 to −14.3).
4. **Item 7A (market risk) is not a useful standalone signal in this design**: it
   exists only for 10-Ks (24.6% of long_form filings; empty for all 10-Qs), and its
   combination weight goes negative (g −0.32 to −0.69) with significantly WORSE
   combined QLIKE (−1.4 to −3.6%, DM +6.5 to +8.4). Interpret as an artifact of the
   75%-empty-text design rather than evidence about Item 7A content per se.
5. **The signal is distributed, not localised**: the complement ("rest", i.e. full
   text minus 1A/7/7A) still carries a significant 1.5–2.7% increment and is even
   slightly BETTER than full text standalone (−1.9 to −4.3% QLIKE). No single
   section recovers the full-text increment; MD&A is the strongest single carrier,
   and MD&A + Risk Factors + remainder are complementary.

All m1_dm_p values shown as 0 are p < 1e-5 (rounded to 5 decimals).
Run dirs: results/runs/B2sec_{item1a,item7,item7a,rest}_full_long_form_seed2026/.
Reproduce: .venv/bin/python scripts/experiments/section_ablation.py --stage all
