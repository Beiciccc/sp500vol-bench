# Pre-registration: quantitative audit of the MAEC earnings-call volatility benchmark (Experiment A)

Date: 2026-07-15. Status: **FROZEN (prereg-maec-v1.0)**. All 9 OPEN items were adjudicated by the lead on
2026-07-15 (rulings inlined in the relevant sections, the original [OPEN] numbering retained for traceability);
this file is committed and tagged before any MAEC label construction, any price statistic, any model run.
As of this draft, only the metadata of `maec_manifest.parquet` (row counts by date/character count,
see the §3 disclosure) and `ACQUISITION_NOTES.md` have been read; transcripts, prices and any labels remain untouched.

Commitment (same as prereg-rfa-v1.2): **the results of all branches enter the paper regardless of direction;
the definition, unit, family structure, split or arm must not be chosen conditional on results.** Revisions must be
committed to file before the corresponding statistic is computed, with the date and reason stated (the revision-log
format follows the prereg-rfa G5 precedent).

## 0. Positioning, prior work and prohibited-claims clause (reframe mandate, must read)

**Prior work (motivation, not competitor)**: Yu, Liu & He, "Same Company, Same Signal"
(Findings of ACL 2025, arXiv:2412.18029; suggested bib key `yu2025samecompany`, declared with
`% NEWBIB:` per the FACTS §9 rule) have shown on EC (MDRM) / MAEC-15/16 that the mean of a
ticker's historical post-earnings volatility ("STPEV") matches all text models at the level of
representational similarity (their MSE 0.257 vs Gemini 0.258). **What they lack**: (a) any recalibrated
past-vol reference (their raw V_past MSE 1.12, discarded); (b) the combination question (whether text
adds an increment over a calibrated price/history baseline); (c) any significance test (no p-values,
no clustering -- and call dates are heavily clustered); (d) a bracketing/attribution ladder;
(e) economic significance.

**The sole positioning of this experiment**: a **quantified repricing with formal inference**
of Yu et al.'s representation-level finding -- porting this paper's protocol (log-space combiner
fitted on val and frozen on test, recalibrated AR reference, entity-mean control, call-date-clustered DM,
placebo, MDE power calibration) to a new domain for the third time (first SEC 8-K, second Yelp),
with STPEV included and priced as **one arm inside the ladder**.

**Prohibited-claims clause (binding on the prose)**:
1. **Must not** sell A as discovery -- the paper must not claim first discovery of the
   identity effect in earnings-call benchmarks; Yu et al.'s priority is cited as motivation in a dedicated passage throughout all of A's prose.
2. **Must not** revive the split-rule comparison (entity-disjoint / group-wise CV): Action-A was
   falsified and rolled back by this project's own pre-registered falsifier (FACTS §12, DO-NOT-REVIVE).
   The entity-disjointness question is **entirely out of scope for this experiment**.
3. **Must not** write cross-domain "the same shortcut" (withdrawn 2026-07-14); the only permitted framing:
   **"the size of the shortcut is a property of the panel and its baseline, not a constant"** (Yelp precedent).
4. MDRM cite-only (structurally blocked: 5-part split zip bundling audio, see §9); no numerical claim
   about MDRM may be made. Must not claim direct comparability with Yu et al.'s numbers (their panel is a
   MAEC-15/16 subset, we use the complete 3,443-call release; see the order-of-magnitude gate in G2).

## 1. Frozen inputs (already on disk, integrity-checked, untouched by any label/statistic at the time this file is frozen)

- Transcripts: `/path/to/data-root/second-domain/earnings_calls/MAEC/MAEC_Dataset/`
  (3,443 `YYYYMMDD_TICKER` directories, each containing `text.txt`; clone commit
  `65a109f5b1a8cb4c96e8337b749ce3db41f2c210`; license CC BY-SA 4.0, copy at
  `LICENSE_MAEC_CC-BY-SA-4.0.txt`; cite Li et al., CIKM 2020).
- Manifest: `/path/to/data-root/second-domain/earnings_calls/maec_manifest.parquet`
  (3,443 rows; call_id, ticker, call_date, n_chars, n_sentences, path;
  1,213 tickers; 2015-02-25..2018-06-21; (ticker, call_date) has no duplicates).
- Prices (700 non-S&P500 tickers):
  `/path/to/data-root/second-domain/earnings_calls/crsp_sp1500_daily_2014_2019.parquet`
  (958,071 rows, 706 (ticker, permno) pairs, OHLC+DlyRet+facpr, 2014-01-02..2019-06-28,
  union trading calendar 1,382 days).
- Prices (remaining 513 tickers): the existing `/path/to/data-root/sp500vol-data/market/full_ohlcv.parquet`.
- Ticker→PERMNO point-in-time map: `ticker_permno_map.parquet` (700/700 resolved, 0 unresolved;
  6 ambiguous tickers flagged, handling rule in §3.3).
- Coverage table: `maec_price_coverage_by_ticker.csv` (44/700 tickers with <90% coverage inside the window,
  all mid-period IPOs/delistings, as expected, see §9).

## 2. Task definition (labels, estimand, horizon, alignment)

### 2.1 Estimand (the Eq.-1 convention native to MAEC/MDRM, the convention of the audited object)

For call i (ticker→permno p, call date τ), take the CRSP total return r_t = DlyRet(p, t)
(**simple return, dividends and splits included** -- facpr/ex-adjustment is already built into DlyRet,
no further adjustment from close prices is needed; this is consistent with the adjusted-price convention
of the audited benchmark). The n-day realised volatility label:

    v_[a,b] = ln( sqrt( (1/n) · Σ_{t∈window} (r_t − r̄_window)² ) )

that is, **log daily volatility** (demeaned RMS of daily returns). Labels, predictions and losses are all in
v (log-vol) units -- the MSE of the audited literature (their 0.257 / 1.12) is exactly in this unit; in this unit
the log-space combiner degenerates to a linear OLS on v (algebraically identical to the exp/log port of
yelp_protocol, and simpler). Robustness note (zero cost, reported side by side only): the
log-return variant with r_t = ln(1+DlyRet).

### 2.2 Horizons

n ∈ {3, 7, 15, 30} trading days (the published-convention family of the MDRM/MAEC line). All four horizons
enter the Holm family of §6; **the arms auditing published claims take these four published-convention horizons as primary**.
[OPEN-8 ADJUDICATED: kept inside the family] n=30 stays in the Holm family (no post hoc removal); its HAC lag /
number-of-clusters ratio is disclosed at freeze time, and the confirmatory reading for the n=30 cell is the
date-block bootstrap CI of §6.4 (pre-declared).

### 2.3 Date alignment (MAEC has no timestamps -- day-0 ambiguity, pre-declared)

MAEC contains no before/afterMarket timestamp for the call. Yu et al. report that 64–69% of calls are
before the open. Let t_0 = the last trading day ≤ call_date, t_1 = the first trading day > call_date.

- **PRIMARY alignment**: label window = {t_1, …, t_n} (strictly post-call, no day-0 leakage under any
  before/after-market scenario; cost: one fewer reaction day counted for the before-open majority).
- **SENSITIVITY arm (whole window shifted by one day)**: label window = {t_0, …, t_{n−1}} (day-0 included,
  matching the before-open majority). This arm is **zero GPU**: all text-arm predictions are unchanged, only the
  labels and the val refit of the combiner change; run the §5 ladder once in full, and report only whether the verdict branch changes.
- If primary and sensitivity disagree on the verdict branch (§8): **primary governs the paper's wording**,
  and the sensitivity difference is disclosed faithfully and used to qualify the claim. [OPEN-2 ADJUDICATED: keep the strictly post-call window as primary]
  Zero leakage is the DNA of this protocol (a port of the effective-trading-day discipline on the SEC side).
  [v1.1 correction following the OPEN-1 verification] The published convention aligns to day-1-start (unadjusted), in the same direction as PRIMARY;
  day-0-inclusive is retained only as the shifted sensitivity arm.

### 2.4 past-vol features (raw material of the reference arms)

Same estimand as the label, with all windows **ending at t_0** (primary alignment) or t_0−1 (sensitivity
alignment, to avoid overlapping the day-0 label window):
- V_past^(n): matched window {t_0−n+1, …, t_0} -- i.e. the raw baseline that Yu et al. discarded;
- HAR-style three windows: V_past^(5), V_past^(22), V_past^(66).

## 3. Sample construction and exclusion rules (all fixed before label construction)

The counts below come from the manifest metadata and ACQUISITION_NOTES (quoted as already-acquired facts, not new statistics):

### 3.1 stub transcript exclusion

The 32 records with n_chars < 100 (mostly 11-character shells) are **excluded from all arms without exception**
(deleted from text and non-text arms alike, guaranteeing identical row sets across arms), with per-split counts
disclosed in the build script output (under the split of §4, 2 of them fall in test). There are 95 records with
< 500 characters in total: the 100–500 band is **retained**, and its count (63) is disclosed under Limitations.
[OPEN-9 ADJUDICATED: 100] The exclusion rule takes the minimum (n=32); the 100–500 character band is retained with its count disclosed,
and differences in content quality are absorbed by the arms themselves.

### 3.2 price coverage exclusion (judged independently per horizon, counts disclosed item by item)

- The label window must be 100% complete (DlyRet present for all n trading days); any missing value → that
  (call, n) row is deleted, with the count disclosed. permno delisted inside the label window (CRSP last date
  falls in the window) → deleted and counted (no delisting-return imputation, disclosed under Limitations).
- past windows: the V_past^(n) matched window requires ≥80% of days present, computed over the days present;
  below that → that (call, n) row is deleted. The same rule applies to the three HAR windows.
- The number of rows passing all gates per horizon, and the number of deletions contributed by the 44
  low-coverage tickers, go into the build report.

### 3.3 ambiguous tickers (6, permno-keyed, pre-declared rule)

- Dual share classes (GEF 83233/83264, HVT 10294/41217, WSO 46068/66376): for each call take
  **the share class with the higher median daily turnover (|DlyClose|×DlyVol) inside the window**; the rule is
  executed once at build time, and the selection outcome is disclosed together with the median turnover of both lines. [OPEN-11 ADJUDICATED: turnover-median tie-break confirmed] The selection outcome is disclosed together with the medians of both share classes.
- Ticker reuse within the window (ENR, FLOW, TIVO): disambiguated point-in-time by call_date ∈
  [SecInfoStartDt, SecInfoEndDt]; falling into two windows or none → that call is deleted and counted.
- Assertion: exactly one permno per call after disambiguation (G5).

### 3.4 Keys and merge discipline

Row key = (permno, call_date, horizon); (ticker, call_date) has been verified to have no duplicates. All arms are
inner-merged on the same row set; after the merge the label is asserted row by row to be identical (same as yelp_protocol).

## 4. Splits (dates pinned; entity-disjointness out of scope, see §0-2)

**PRIMARY = chronological split on call_date, 70/10/20 (by call count), with boundary dates pinned**:
- train: call_date ≤ **2017-02-23** (2,436 calls)
- val: 2017-02-24 .. **2017-05-09** (333 calls)
- test: 2017-05-10 .. 2018-06-21 (674 calls, 143 distinct call dates, 463 tickers)
The boundaries come from the 70%/80% quantile dates of the manifest row counts (see the §0 disclosure; labels untouched).
Assertions: max(train date) < min(val date) < min(test date); the lower bounds of val ≥ 100 rows and
test ≥ 30 rows (MIN_VAL/MIN_TEST, same as Yelp) are checked per horizon.
Boundary-overlap disclosure: calls within the last h−1 trading days of val have outcome windows reaching into
the test period -- counts reported, with `--embargo-val` as a robustness lever (same as yelp_protocol).

**published-convention arm (only to reproduce the published-style readings under audit) [v1.1 revision, per the OPEN-1 literature verification]**:
split = the **three per-year panels with chronological 7:1:2 inside each panel** of MAEC (CIKM 2020) Table 5
(2015: train ≤2015-10-22 / val ≤2015-10-28 / test ≤2015-12-17; 2016: ≤2016-08-03 /
≤2016-08-12 / ≤2016-11-15; 2017–18: ≤2017-11-07 / ≤2018-02-15 / ≤2018-06-21;
fitted independently year by year, consistent with the original's "different models for different years"); alignment = day-1-start
unadjusted (the footnote to Yu et al. Table 3 states explicitly that their scoring does not adjust for beforeAfterMarket). Readings follow the published style:
raw V_past^(n) (not recalibrated) vs each text arm standalone, MSE(v), no clustered inference.
The 80/20 split of the MDRM line is not adopted (MDRM cite-only).
[OPEN-1 VERIFICATION COMPLETE, 2026-07-15 (v1.1)] Verification conclusions: the published convention = three per-year panels at 7:1:2
(pinned above per Table 5); the published alignment = **day-1-start unadjusted** (same direction as this pre-registration's PRIMARY,
so the sentence in §2.3 that "day-0-inclusive belongs to the published-convention arm" is void, and day-0-inclusive is
retained only as the shifted sensitivity); the label formula / horizons / units agree with §2.1–2.2; the "days" ambiguity
(MDRM writes calendar days, MAEC does not specify, Yu et al. use trading days) is executed as trading days and disclosed.
The primary split/alignment is unchanged. The revision occurred before any scoring of this arm (panel construction also never touched this arm's split).

## 5. Arm ladder (each arm vs the recalibrated past-vol reference; combination weights always fitted on val and frozen on test)

Combiner = OLS on v (§2.1; = the identical port of yelp_protocol `log_ols_frozen`),
with predictions clipped to v ∈ [ln 1e-4, 0], corresponding to σ_daily ∈ [1e-4, 1.0].

**References (two, both entering the Holm family, mirroring the dual-reference precedent of prereg-rfa)**:
- **R-AR (matched window)**: f_R = OLS[1, V_past^(n)] -- the recalibrated version of the raw baseline
  discarded by the audited literature, the main reference of the audit narrative ("the baseline they threw away, what is it worth after calibration");
- **R-HAR**: f_R' = OLS[1, V_past^(5), V_past^(22), V_past^(66)] -- the strong-reference end of this paper's
  reference-interval discipline.
[OPEN-3 ADJUDICATED: R-AR is the headline reference] The audit narrative takes priority ("the baseline they discarded,
what is it worth after calibration"); R-HAR is the conservative end within the family, always reported side by side in the same table.

**STPEV entity-mean control (an arm inside the ladder, the object of Yu et al.)**:
STPEV_i(τ) = the mean of that ticker's historical call labels **already realised as at τ** (counting only
prior calls with τ′+n trading days ≤ τ, i.e. whose label window falls entirely before τ; point-in-time, expanding).
Rows with no prior call → fall back to the train+val global mean; coverage disclosed per horizon (the manifest
counts give a lower bound: 75/674 rows in test have a ticker with no call at all in train+val, 11.1%;
the expanding definition also counts realised calls earlier within the test period, so the actual coverage is as disclosed at build). Control reference: f_Re = OLS[1, V_past·, STPEV]
(one version for each of the two references); the zero-text STPEV-only row is reported in the same table (descriptive).
[OPEN-4 ADJUDICATED: point-in-time expanding is primary] (also closer to the original meaning of STPEV in Yu et al.);
the Yelp-port fixed-mean definition is a robustness row, and both are computed.

**Text arms**:
1. **TF-IDF ridge** (fitted): word 1–2 gram, min_df=5, max_features 50,000,
   sublinear-tf; ridge α ∈ {1e-2 … 1e3} (log grid) fitted on train, selected on val;
   target = v.
2. **Prompted LLM** (Qwen3-32B-AWQ, single GPU, temperature 0, single seed -- same definition as the paper's C6
   primary and disclosed as such): reuse the prompt/guided-JSON/clip/retry machinery of
   `scripts/experiments/e1_llm_forecast`; the ask = annualised volatility % over the next n trading days,
   clipped to [3, 300]%, converted as σ_daily = (ann%/100)/√252 → v̂ = ln σ_daily. Transcript truncation:
   the head 12,000 tokens (max-model-len 16,384; median transcript ≈ 10.8k characters, truncation
   trigger count disclosed). The full prompt is written into
   `scripts/experiments/second_domain/maec_prompt.py` before freezing and frozen with the tag.
   [OPEN-12 ADJUDICATED: head-only] Truncation trigger count disclosed (median transcript ≈2.7k tokens, expected to be rare).
3. **Zero-content identity probe** (the mirror of the prompted arm, corresponding to the paper's date+ticker probe
   and Yelp's name+city probe): the prompt gives only ticker + call date, **no transcript**,
   everything else word for word identical to arm 2. The probe is a diagnostic row (§6.2), and its "share of the
   fulltext combination increment reproduced" is a pre-declared reported quantity. [OPEN-7 ADJUDICATED: ticker + CRSP comnam company name + call date] The probe's job is to maximise
   the elicitation of the identity prior -- the ticker alone of a small-cap S&P 1500 name may fail to elicit the prior, and a weak probe would
   underestimate the identity share and flatter the text. The payload difference from the SEC probe (date+ticker) and the Yelp probe
   (name+city+categories+month) is disclosed in the paper.
4. [OPEN-5 ADJUDICATED: included] **frozen-embedding ridge** (Qwen3-Emb-8B mean-pool +
   ridge, ridge grid the same as TF-IDF, selected on val) -- mirrors the three-arm Yelp design and fills
   the fitted-representation cell. Its F1/F2 Holm families are established isomorphically per §6.2 (see that section).
5. [OPEN-6 ADJUDICATED: not included] A second prompted family does not enter this audit (single family; the
   cross-family robustness question is carried by B1 on the SEC side); if it is added in future, it must be pre-registered by revision.

Readings per arm (for each reference X ∈ {R-AR, R-HAR}):
- text-alone: recalibrated text standalone vs f_X (descriptive + raw DM);
- **combination increment (row-3 analogue)**: f_U = OLS[1, V_past·, v_text] vs f_X;
- STPEV control (row-4 analogue): f_Xe vs f_X (descriptive + raw DM);
- **identity-controlled residual (row-5 analogue)**: f_Ue = OLS[1, V_past·, STPEV,
  v_text] vs f_Xe;
- identity share: (the share of the combination increment absorbed by the STPEV control) = d4/d3 (the same-named
  quantity in yelp_protocol), probe share = probe combination increment / fulltext combination increment.

## 6. Statistics and inference (pre-declared, executed once, all entering the tables)

### 6.1 Clustered DM

- **PRIMARY: call-date-clustered DM** (the day → call-date port): loss differences are first averaged with
  equal weight by call date (143 date clusters in test in total), DM is run on the date series, HAC lag
  **L_n = the maximum, over the frozen test date grid, of the number of subsequent distinct
  call dates falling within n−1 trading days after any given date** (computed once from the manifest date grid and
  disclosed with the freeze -- it depends only on date metadata, not on labels), HLN small-sample correction, t(#dates−1).
- **Co-primary robustness: two-way CGM over date × ticker** (the port of yelp_protocol `dm_test_2way`,
  entity = permno), reported side by side in the same table.
- The date grid is non-contiguous (clustered by earnings season): the contiguity assertion of `monthly_mean` is
  relaxed on the date grid to "ordered, deduplicated", the L_n definition already covers overlap on the real grid,
  and the L_n/#dates ratio at n=30 is disclosed.

### 6.2 Holm families (pre-declared; once frozen, the family is neither extended nor reduced)

For each headline text arm a ∈ {TF-IDF, frozen-embedding (Qwen3-Emb-8B), prompted-Qwen}
(OPEN-5 included, OPEN-6 excluded; three arms, isomorphic family structure):
- **Family F1(a) "combination increment"**: 4 horizons × 2 references = **Holm(8)**;
- **Family F2(a) "identity-controlled residual"**: isomorphic **Holm(8)**.
The probe, text-alone, STPEV-only and published-convention readings are all descriptive /
diagnostic rows (raw p or no p), **not entering Holm and not permitted into any "win" statement**.

### 6.3 placebo gate (any cell entering a "win" statement must pass)

- **PRIMARY: label-shuffle** (text predictions permuted across all rows, the same permutation on val+test,
  weights refitted on val), 20 seeds (1000–1019, same as Yelp);
- **Diagnostic: within-date text-swap** (permutation within the same call date; single-call dates are not
  swapped, and the fraction of effectively permuted rows is disclosed), 5 seeds (2000–2004).
The verdict follows the Yelp precedent: shuffle is the G4 main gate, swap is the G4b diagnostic; a marginally dirty swap
(as in Yelp h=3) → that cell does not enter prose claims.

### 6.4 Power and CI

- **Analytical MDE (80% power)**: (1.96+0.84)·SE_date/MSE_ref·100, with SE_date from the HAC(L_n) variance of the
  date-mean loss differences; reported per stage (AR stage / entity stage) × horizon.
- **oracle signal-injection** (the yelp_protocol row-1 port, s = within-permno
  demeaned test residual, entity-orthogonal; targets {0.5, 1.0, 2.0}% + adaptive
  max(2, 1.5·MDE, real+1)): a mechanism-detection gate; the disclosure sentence is reproduced verbatim ("ORACLE injection —
  power calibration only, never citable as forecast performance").
- **CI**: every residual rel% that enters the prose is accompanied by a date-block moving bootstrap CI
  (block = 5 call dates, 2,000 draws, seed 2026; the Yelp/omnibus
  block-bootstrap precedent degenerates on a grid of 6 test months, so date blocks replace month blocks, disclosed).
- **Wording discipline for nulls**: any "absorbed / no residual" conclusion must be reported alongside the MDE;
  if the MDE exceeds the repriced published-style increment (converted to the same unit), the wording is downgraded to
  "underpowered to rule out" and must not be written as a clean zero.

### 6.5 Single-shot discipline

`maec_protocol.py` (the yelp_protocol port, entity=permno, cluster=call-date,
combiner = OLS in v-space) is **run only once** for each (arm × alignment) combination; all numbers, whatever their direction, go into
`results/second_domain/maec/protocol_<arm>.json` +
`results/tables/maec_audit.{csv,md}` + FACTS.md. Reruns are permitted only for script bugs, and the diff and the reason
must be recorded in the revision log of this file.

## 7. Sanity gates (any failure aborts, no tables produced; G numbering follows family convention)

- **G1 (sign reproduction of the published-style reading, a precondition of the audit)**: under the
  published-convention arm, at least one text arm's text-alone MSE(v) is below raw V_past^(n) (mirroring the
  0.257 vs 1.12 ordering of Yu et al.). If **all** text arms fail: the stand-in arms are too weak to carry the
  "repricing the text increment" claim → the audit scope is downgraded to "baseline recalibration audit only" (§8 branch D),
  and arms must not be swapped nor prompts tuned for a retry mid-course.
- **G2 (order-of-magnitude gate)**: our raw V_past MSE(v) is of the same order of magnitude as the 1.12 reported by Yu et al.
  (ratio ∈ [1/3, 3]; the panels differ -- their MAEC-15/16 subset, our full 3,443 --
  so only an order-of-magnitude gate is applied, not an equality gate). Failure → check label construction (§2) before anything else, and no table may be produced while the fault stands.
- **G3 (leakage assertions)**: the split boundary assertions (§4); the code assertion that combiner weights are
  val-fit and test-frozen; the STPEV point-in-time assertion (the window end of every contributing label ≤ τ of the current call);
  the mean-zero assertion for the within-entity injected signal (same as yelp_protocol).
- **G4/G4b**: the placebo gate (§6.3).
- **G5 (keys and merges)**: exactly one permno per call after disambiguation; identical row sets across arms;
  the label identical row by row after the merge; no duplicates in (permno, call_date, horizon).
- **G6 (exclusion audit)**: the stub exclusion is exactly the pre-declared rule (n_chars<100, 32 in total),
  with per-split counts and each exclusion count of §3.2/§3.3 disclosed item by item and reconciled in total
  (3,443 − exclusions = the row count of each arm).

## 8. Verdict ladder and branch commitments (three branches written down now, binding on the prose)

Object of the verdict = family F2 of each headline text arm (identity-controlled residual, 8 cells):

- **(a) FULLY ABSORBED**: 0/8 cells simultaneously satisfy DM<0, Holm<.05 and passing G4; and
  identity share (d4/d3) ≥ 100% or the combination increment itself is not significant.
  → Paper wording: "On MAEC, once the discarded past-vol baseline is recalibrated and the same-ticker
  STPEV mean is added, the published-style text increment is fully absorbed; call-date-clustered inference (never
  provided in the literature) confirms at a power of MDE=X% that Yu et al.'s representation-level finding also holds at the prediction level --
  a third domain, the same measuring instrument." (The MDE must be reported alongside, §6.4.)
- **(b) PARTIALLY ABSORBED**: ≥1/8 cells pass Holm+placebo, and identity share ≥ 50% at those horizons.
  → "Recalibration + identity absorb X–Y% of the published-style increment; a bounded, placebo-clean,
  power-calibrated residual survives -- isomorphic to the bounded residual of the SEC panel, its size priced by the panel and its baseline."
- **(c) SURVIVES**: ≥4/8 cells pass Holm+placebo and identity share < 50%.
  → "On MAEC the increment of the text arms is not an identity artefact: the problem with this benchmark is
  baseline miscalibration rather than identity; we provide the first clustered significance and power calibration on this benchmark. Yu et al.'s representation-level
  similarity does not translate into predictive redundancy at the combination level." (The honest branch; the fallback framing remains
  "the size of the shortcut is a property of the panel and its baseline" -- the Yelp precedent, fully compatible with the existing manuscript framing.)
- **All other combinations**: MIXED, reported faithfully cell by cell, with the wording taking the weakest defensible form.
- **(d) G1 downgrade branch**: all stand-in arms weak → publish only the baseline recalibration audit (raw V_past 1.12-type
  readings vs post-recalibration readings + STPEV pricing), and withdraw the repricing claim about the text increment entirely.

The two headline arms landing in different branches → report arm by arm, do not merge the wording (Yelp precedent: the
identity/content split between the prompted and fitted arms is itself a finding). The alignment sensitivity of §2.3 changing
the branch → primary governs + the difference is disclosed (OPEN-2 was adjudicated to keep it; this sentence is the final version).

## 9. Pre-disclosed deviations and limitations (the draft for the paper's Limitations)

1. **A text-only audit of a multimodal benchmark**: the 59 GB of audio features were not acquired (per the acquisition decision);
   justification: Yu et al.'s analysis is likewise mainly on the text side, and the AMA-LSTM precedent shows the
   increment of the audio channel to be marginal and unstable. The object of the audit is the "text increment" claim, not the full multimodal stack.
2. **MDRM cite-only**: the full data is structurally blocked by a 5-part split zip bundling audio
   (ACQUISITION_NOTES §2); no license, not redistributed.
3. **No timestamps**: the day-0 ambiguity is handled as pre-declared in §2.3 (primary + shifted sensitivity).
4. **44/700 tickers with <90% window coverage** (mid-period IPOs/delistings): excluded row by row per the §3.2 rule and
   counted; not a gap in the data, but a listing-window effect.
5. **Small panel** (3,443 calls, test 674 rows / 143 date clusters): power is priced openly as an MDE
   (§6.4); the HAC lag at n=30 is large relative to the number of clusters (OPEN-8: kept inside the family, with the bootstrap CI as the confirmatory reading).
6. **Single seed for the prompted arm** (temperature 0, same definition as the C6 primary); if "multi-seed"
   is done, the wording must follow FACTS §13c: reproducibility jitter, and must not be called a stochastic-decoding
   ensemble.
7. **Contamination disclosure**: MAEC 2015–2018 sits deep inside the LLM pre-training window; the probe arm (§5-3) exists
   precisely to price the memorization/identity prior, and the paper discloses this under the established llm_contamination
   framing from the SEC side.
8. While drafting this DRAFT the date/character counts of the manifest were read (the §0 and §4 boundaries come from there);
   labels, prices and transcript content were not read. This disclosure is retained with the frozen version.

## 10. Compute and implementation

- Hardware: a single 4×A100-40GB node; Qwen3-32B-AWQ on a single GPU (vLLM offline batch, word-for-word the same protocol as
  e1_llm_forecast); 3,443 calls ≈ a few GPU hours, low cost. The fitted arms run on
  CPU (obeying the local ≤half-the-cores rule; on the box, saturated up to the cgroup quota).
- New scripts (committed before freezing): `scripts/experiments/second_domain/maec_build_panel.py`
  (labels + features + exclusion audit), `maec_baseline_text.py` (the R-AR/R-HAR/TF-IDF arms),
  `maec_prompt.py` (prompt freeze), `maec_protocol.py` (the yelp_protocol port).
  Outputs: `results/second_domain/maec/` + `results/tables/maec_audit.{csv,md}`.
- Boundary with the experiments in flight: this experiment reads no test split of the SEC panel; B1 (Mistral) and
  the single-shot HPO evaluation are each governed by their own pre-registration; every statistic in A touches only the §1 frozen inputs.

## 11. Revision log

- **v1.2 (2026-07-15)**: two build-time clarifications/revisions, both prior to any arm scoring (the
  panel-construction gates have been run, zero protocol/arm statistics computed): (1) **return-source clarification** --
  the full_ohlcv pinned in §1 has no return column, and its adj_close was verified to be unadjusted DlyClose (over 548,223 overlapping days
  1.36% differ by >1bp; on the XRX reverse-split day the difference is 301%), so per the §2.1 estimand it is replaced by
  `market/crsp/market_returns.parquet` from the same ingest
  (log1p(DlyRet) exactly recovered via expm1, isomorphic on both sides); (2) **filling the membership-period gap on the S&P500 side** --
  the S&P500 cache covers only the index-membership window, causing ~307-311 calls (all on the S&P500 side, including
  122 calls of 49 tickers with zero rows) to be deleted by the §3.2 price gate; this is an artefact of the cache structure rather than data being
  unavailable, so the frozen input `crsp_sp500side_gapfill_2014_2019.parquet` is added (extracted from the local CRSP full-universe raw zip, by the same mechanism as
  the 700-ticker extraction), the panel is rebuilt from three sources, and the exclusion reconciliation is updated with the build_report.
- **v1.1 (2026-07-15)**: the OPEN-1 literature verification is complete (CIKM 2020 Table 5 + Yu et al. Table 1/3 +
  MDRM §6.2 verbatim citations, report kept in the session archive). Revisions: the published-convention arm split changed to three per-year
  panels at 7:1:2 (dates pinned per Table 5), and the alignment changed to day-1-start unadjusted; the corresponding sentence in §2.3 corrected.
  Evidence: at the time of the revision the MAEC panel was still being built and the published-convention arm had no statistics of any kind; primary unchanged.
