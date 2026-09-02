# Pre-registered analysis D — cross-cell OMNIBUS joint test + power calibration (69-cell M1 primary family)

> Pre-registered in `configs/prereg_residual_family_audit.md` §D (tag `prereg-rfa-v1.1`) BEFORE computation. All branches committed to the paper regardless of direction.

> **ORACLE INJECTION — POWER CALIBRATION, NOT A FORECAST.** The power section injects the oracle firm-orthogonal signal of `signal_injection_power.py` (within-firm demeaned test log-residual of f_R; the one declared exception to the no-look-ahead rule). Never citable as forecasting performance.

## Disclosures

- **Cells**: the exact 69-cell primary family of `results/tables/m1_ensemble_primary.csv` — `forecast_combination.SETS` (long_form: 15 challenger arms, event_driven: 8) x horizons (5, 10, 20), panel construction byte-identical to `m1_ensemble_primary.py` (min-row filters included).
- **Basis**: seed-ensemble — per-observation mean of `prediction_realised_vol` across seeds 2026/2027/2028 for 3-seed C/D arms (`m1_ensemble_primary.ensemble_text`, inner join on ticker/accession/horizon); A/B, C6_llmtext, D4_llmfused single-run. Reference = the single recalibrated HAR (A2), log-space combiner fit on validation only, frozen on test (`forecast_combination.log_combo`).
- **Statistic**: per cell c and test day t (effective_trading_day, calendar-normalised), d_c(t) = within-day mean QLIKE(f_R) - within-day mean QLIKE(f_U) (vol-unit QLIKE `forecast_combination.qlike`; positive = challenger helps; computed as daily_mean(QLIKE_R) - daily_mean(QLIKE_U), bit-for-bit the negation of the daily differential inside the committed `dm_test_clustered`). Pooled series D(t) = unweighted mean of d_c(t) over cells present on day t; days enter with equal weight.
- **HAC spec**: `sp500vol.evaluation.dm_test.dm_test` reused verbatim on (D, 0) with h = 20: Newey-West/Bartlett HAC lag = max(h)-1 = 19 trading days, Harvey-Leybourne-Newbold small-sample factor (h=20), Student-t(n_days-1) reference, two-sided p. Sign convention: omnibus t > 0 = text helps (the negation of the per-cell table's DM sign).
- **Subfamilies (pre-declared)**: long_form (45 cells), event_driven (24 cells), all 69; Holm(3) across the three p-values (`forecast_combination.holm`).
- **Injection**: definition reused EXACTLY from `signal_injection_power.py` — s = within-firm demeaned test log-residual of f_R (verbatim construction; max within-firm |mean s| = 1.5e-16 < 1e-12), per-cell bisection of kappa = g1*delta (`calibrate_kappa`, tolerance 0.02pp) so the realised test rel-QLIKE improvement of f_U over f_R hits the target level in EVERY cell simultaneously (cells above the target get kappa < 0: signal removed down to the target). Grid {0.1, 0.2, 0.3, 0.5, 1.0}% per prereg. Note the pre-existing 0.02pp tolerance is +-20% relative at the 0.1% level.
- **Replications**: N = 100 per level (target met; runtime allowed it). The pre-registered injection is deterministic, so the detection RATE comes from day-block moving-bootstrap replications (blocks of 20 consecutive days, circular; the exact index mechanics of `clustered_dm.mbb_ci_daily`; seeds spawned from 2026) of the injected pooled daily series. Pooling is a per-day operation, so resampling day blocks of D(t) is numerically identical to resampling day blocks of the injected 69-cell panel and recomputing the omnibus. Rejection = t > 0 AND two-sided p < 0.05 (sign requirement matches the committed 'detected' criterion). The deterministic one-shot omnibus per level is tabulated alongside.
- **Secondary (SPA/MCS)**: pre-registered as report-only; out of scope of this script.

## SANITY — per-cell day series reproduce the committed per-cell DM

Gate over ALL 69 cells (abort on failure): committed code path (`dm_test_clustered` on per-observation losses) vs `results/tables/m1_ensemble_primary.csv` — max |dDM| = 8.88e-16, max |dp| = 9.71e-17, n_days mismatches = 0; MY per-cell day series d_c(t) aggregated back to a per-cell DM (`dm_test(-d_c, 0, h)`) vs that code path — max |dDM| = 0.00e+00, max |dp| = 0.00e+00. **PASS**.

5 pre-declared sample cells:

| cell | committed DM / p / n_days | recomputed (committed path) | from MY day series | |dDM| vs CSV | |dDM| series vs path |
|---|---|---|---|---|---|
| long_form/B2_tfidf_ridge/h10 | -8.892544328521 / 3.901005e-18 / 803 | -8.892544328521 / 3.901005e-18 / 803 | -8.892544328521 / 3.901005e-18 | 0.00e+00 | 0.00e+00 |
| long_form/C2_finbert_s1/h5 | -4.532482283094 / 6.710430e-06 / 809 | -4.532482283094 / 6.710430e-06 / 809 | -4.532482283094 / 6.710430e-06 | 0.00e+00 | 0.00e+00 |
| long_form/D2_gated_fusion/h20 | +5.560568741332 / 3.675984e-08 / 794 | +5.560568741332 / 3.675984e-08 / 794 | +5.560568741332 / 3.675984e-08 | 0.00e+00 | 0.00e+00 |
| event_driven/B4_lm_features/h20 | -2.010255968836 / 4.467772e-02 / 981 | -2.010255968836 / 4.467772e-02 / 981 | -2.010255968836 / 4.467772e-02 | 0.00e+00 | 0.00e+00 |
| event_driven/C6_llmtext/h5 | -5.043983940743 / 5.416952e-07 / 996 | -5.043983940743 / 5.416952e-07 / 996 | -5.043983940743 / 5.416952e-07 | 0.00e+00 | 0.00e+00 |

## Omnibus results (pre-declared subfamilies, Holm(3))

| subfamily | n_cells | n_days | mean cells/day | mean D(t) (daily QLIKE diff) | approx rel% | t | p (two-sided) | Holm(3) p | reject (Holm, 5%) |
|---|---|---|---|---|---|---|---|---|---|
| long_form | 45 | 809 | 44.6 | +0.001100 | +1.19% | +6.90 | 1.06e-11 | 2.12e-11 | YES |
| event_driven | 24 | 996 | 23.8 | +0.000508 | +0.55% | +4.57 | 5.61e-06 | 5.61e-06 | YES |
| all_69 | 69 | 996 | 60.1 | +0.000783 | +0.84% | +8.29 | 3.76e-16 | 1.13e-15 | YES |

(approx rel% = 100 * mean D(t) / unweighted mean of the member cells' QLIKE(f_R) — descriptive scale only.)

## Power calibration — all-69 omnibus recovery of an injected firm-orthogonal signal

| injected level (realised rel-QLIKE, every cell) | converged | max |calib miss| (pp) | kappa<0 cells | mean D(t) | one-shot t | one-shot p | one-shot reject | MBB reject rate (N=100) |
|---|---|---|---|---|---|---|---|---|
| 0.1% | 69/69 | 0.0187 | 51 | +0.000089 | +0.84 | 4.00e-01 | no | 0.10 |
| 0.2% | 69/69 | 0.0188 | 45 | +0.000178 | +1.72 | 8.59e-02 | no | 0.46 |
| 0.3% | 69/69 | 0.0197 | 40 | +0.000266 | +2.63 | 8.77e-03 | YES | 0.68 |
| 0.5% | 69/69 | 0.0200 | 37 | +0.000442 | +4.51 | 7.36e-06 | YES | 1.00 |
| 1.0% | 69/69 | 0.0196 | 32 | +0.000881 | +9.41 | 3.20e-20 | YES | 1.00 |

**80%-power MDE**: empirical 0.375% (interpolated); analytic supplement 0.302% (one-shot t is ~linear in the level, slope 9.3 per %; MDE = 2.80/slope — normal approximation, reported because the pre-registered grid may saturate at its smallest level; values below 0.1% are extrapolations beneath the grid).

## Pre-registered language ladder (copied from §D) — fired branch

> - omnibus does not reject and MDE ≤ 0.3% → "a power-endorsed bound";
> - omnibus rejects → written consistently with the detectable≠attributable≠bankable trichotomy (what is detected is a systematic cross-cell micro-increment; attribution and bankability are unchanged);
> - insufficient power → report the MDE truthfully; do not escalate the wording.

**Fired branch**: REJECT -> written consistently with the detectable != attributable != bankable trichotomy: what is detected is a systematic cross-cell micro-increment; attribution and bankability claims unchanged.

## Caveats

1. The omnibus pools DEPENDENT cells (shared filings, shared reference, overlapping horizons); the day-clustered HAC(19)+HLN inference treats the pooled series as one time series, which is exactly the pre-registered design — the cross-cell mean gains power from averaging noise, not from pretending cells are independent.
2. The MBB rejection rate estimates power at the injected effect size given the empirical day-to-day distribution of the pooled differential; HAC autocovariances beyond the 20-day block joins are broken, a standard, slightly anti-conservative approximation disclosed here.
3. Oracle injection: never citable as achievable forecast gains (see signal_injection_power.md).
