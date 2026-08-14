# M1 — the audit cascade on Kogan et al. (2009)'s OWN 10-K corpus

*Pre-registered: `configs/prereg_kogan_corpus.md`, tag `prereg-kc-v1.0`. Single-shot. Generated 2026-07-17 13:25:48 by `scripts/analysis/kogan_corpus_audit.py` in 3.0 min, local CPU only.*

## The prereg, quoted

> **L0 published convention**: Kogan's convention — text features (TF-IDF) + `logvol.-12` as control, regressed on `logvol.+12`, **naive obs-level inference**, their annual OOS split (train ≤ y, test = y+1). Reading = the MSE improvement rate of the text arm vs the `logvol.-12`-only arm (the quantity they report).
> **L1 recalibrated baseline**: the baseline is changed to a **recalibrated** `logvol.-12` (OLS intercept + slope, fitted on the training years and frozen on the test year). **L2 firm-identity reference**: the reference additionally includes the **training-period mean log volatility of the same CIK** (zero text terms). **L3 clustered inference**: clustered by **filing date** (the shock-sharing unit), HAC + HLN, replacing the naive obs-t. **L4 Holm (pre-declared family)**: Holm within the family of L3's per-year p-values. **L5 conjunction**: survival requires L1∧L2∧L4 to hold simultaneously. **placebo**: label permutation (5 seeds), |DM|<2 as the gate.

## The census — three corpora, not a sample

The prereg's scope correction, verbatim: *"the 'N independent published results' demanded literally by the internal adversarial dry-run do not exist in this field"* — the public evidence base for disclosure-text → volatility is **three corpora**, and HTML / NumHTML / VolTAGE / KeFVP / ECHO-GL are all built on MDRM (per each repo's own README), so auditing *N models* would be *N models on one corpus* = pseudo-replication with a fake denominator. Hence a **census**:

| # | corpus | status | authority |
|---|---|---|---|
| 1 | MDRM / earnings-call | **cite-only, not obtainable** — text+audio bundled in split volumes, no licence, no longer redistributed | `prereg_maec_audit.md` §9 (prior ruling; unchanged by this experiment) |
| 2 | MAEC | **audited** — cascade run end-to-end | FACTS §13g |
| 3 | **Kogan 10-K corpus** | **audited — THIS TABLE** | prereg-kc-v1.0 |

Distinct from the committed `kogan_dissolve.md`, which ports Kogan's evaluation *design* onto **our** panel. This runs **our cascade on their corpus** — the "reproduce a published positive, then re-price it under the protocol" audit.

## G-K0 — provenance and integrity

Source `http://www.cs.cmu.edu/~ark/10K/data/` (Version 1.0, 2009-03-31; addendum 2009-09-18). **Data is NOT redistributed — only the pipeline ships** (`kogan_corpus_fetch.py` re-fetches it). 45 files, 364.4 MB, fetched 2026-07-17 11:25:43 UTC; every file re-hashed at audit time and matched against the fetch manifest.

| file | bytes | SHA-256 |
|---|---|---|
| `README.txt` | 2,043 | `8c890c452701bcc9cc36615423277dcd170f4f9ef2392f13a62c762e87757627` |
| `1996.meta.txt` | 186,262 | `b558581c6f039ac18b804791d0b2437b61910ae6825289774b62ef541f0a72dd` |
| `1996.tok.tgz` | 7,831,787 | `2f84ca525f76a4dc8f142980567a879896508bf38f2e635c758b602b51321ecf` |
| `1996.logvol.+12.txt` | 58,971 | `3c51c952e803950da460259ee8e4118fa176ea48ad4a653436ba5cd3fbb79268` |
| `1996.logvol.-12.txt` | 58,993 | `81a5df3022d03bc760425c05950b80a45de473a499664dd522bcc2b5428b6d36` |
| `1997.meta.txt` | 300,133 | `98a4660e4c8199238fc3082e0336fd7c7f9a9cd1502e4359e49523a5ff55bd11` |
| `1997.tok.tgz` | 13,257,003 | `0b1bd3e87242913d03e6a5f9cb166d4c22764e222a4251e34c5a815c083b7461` |
| `1997.logvol.+12.txt` | 94,664 | `9cfda730b4d89499c2d38e42b3b4effdb9319848e9ca21082da8338dbd9c029f` |
| `1997.logvol.-12.txt` | 94,649 | `b88141cb3aadcfb315d7cf32fbbc8891740a4eae49523131d0242d91eaf0ede1` |
| `1998.meta.txt` | 327,813 | `40996056fdc47175b4e3bfd8761d66abd639612f729a5639a2b66b00b5960764` |
| `1998.tok.tgz` | 16,986,490 | `b88976c223f486fb313741913a0089dff13cc4a1b85fccc6bc1789d8d19444bc` |
| `1998.logvol.+12.txt` | 103,135 | `7e6d559c7a8dcc0dfc0b81157523158d9598d40dcca7c88ad805a10d4474e340` |
| `1998.logvol.-12.txt` | 103,123 | `c84705afcef1d671047cef9a2387e53fbddf136ffef9aa886e0bb177cacc85cb` |
| `1999.meta.txt` | 335,775 | `15afa197668fcf7a9fab658c458ac8266f2d8e6e692aa7d34ff60597d27d07c6` |
| `1999.tok.tgz` | 21,142,531 | `3ad1f4f110b127dfc453497689dceb54cbf62520badbe5f9c7e014d98f13152d` |
| `1999.logvol.+12.txt` | 105,733 | `97085e5038d65c2c459b8509b87c87cb0c3106ada432bd8df7257bcf092840d0` |
| `1999.logvol.-12.txt` | 105,741 | `afd5739800a2280e1c779dc995433f3b823836557fc9e178a094ae0a40c7586e` |
| `2000.meta.txt` | 322,467 | `a78d19acbb298c9035e97d6d7bc12c43861ddf8d5951a377de59efda52a1066b` |
| `2000.tok.tgz` | 19,320,835 | `a93542a26fd27f4e66e0490e78bc9bdbc457dc9f4ca89739474a3d3c0f9346e4` |
| `2000.logvol.+12.txt` | 101,597 | `30620b00294a1411411ae3ab680a751eaa02b8ea5315cbd3c87c6ae60ebcc05a` |
| `2000.logvol.-12.txt` | 101,571 | `f74578ed04890bd29a3c72d00a87d96c9287d30e7acfe0b811b146ad119a7f6f` |
| `2001.meta.txt` | 345,576 | `9bf67e77f23e1b82a138583f3a8f66095d2c17ebc76848cf02dbbd201b32c9f8` |
| `2001.tok.tgz` | 21,926,827 | `e48174da09f5dacecfbf58c2575f9a749006a510625bb696a3f5cf9d8d742deb` |
| `2001.logvol.+12.txt` | 109,215 | `6b320a6f0c19fa036762f75c217cc163962bd09b0924bd1ae5d0e7a2074bf3c5` |
| `2001.logvol.-12.txt` | 109,165 | `30ce18274786c7b20cddf133e1b7ca6eefd104248e1e744d68b3b2096bd82ae1` |
| `2002.meta.txt` | 383,048 | `e90ce4195ed0d42f3aa6d5deb6ee8bbfc37fd2d6e6b313e31d1c2625c2c3ec56` |
| `2002.tok.tgz` | 33,028,464 | `faa4aa292ff11993a689a5169f0cc1457ef9f029d0b8316718b55926fdb4b4a4` |
| `2002.logvol.+12.txt` | 124,925 | `43d691d94072af09b9d84055cdf83aed56ce65d663e8ffab05c8d05ed79b4bb6` |
| `2002.logvol.-12.txt` | 124,913 | `0f2f8fdd1d988232e280c36ed9c4292317380309b470355323492f950a49f4cc` |
| `2003.meta.txt` | 486,058 | `7b38e6c504cff8490ee5b0ff49e65fc74ce757759503997b7794b91cc49efc13` |
| `2003.tok.tgz` | 51,142,325 | `346b78eca28924c16fb8b6f180d54db2c825e4074e4840d1550f5fb094036776` |
| `2003.logvol.+12.txt` | 158,473 | `6e2bfbed298d3f6a07401c76c658075c1e6c6d9e914b6d935d64ea49e39178f3` |
| `2003.logvol.-12.txt` | 158,516 | `60955317c52c513440c7c18172646f7f0e9e592e42eb44d2d0f1e7c660af80ca` |
| `2004.meta.txt` | 478,977 | `9134224e1db27440cff675b334c579dbb8540b1324c3b157962905ed7988277f` |
| `2004.tok.tgz` | 56,678,116 | `f91daafc0067ccf772fb60dae811b5cbc6471c20a412d3bc783961caa95deb51` |
| `2004.logvol.+12.txt` | 156,205 | `86c80ea41683cb1c61b3abea00c23d81c68f943ac198e2bd890390fadc8ef8db` |
| `2004.logvol.-12.txt` | 156,196 | `c8e24dbb16d43cec1f79c55f99e3ba3e63339d6867021a6fc0c6f0c18e2200d6` |
| `2005.meta.txt` | 467,160 | `0b04071255d6e4c47c5b0e89b1f266fc8ec8f056668013c4095068c2135c20c1` |
| `2005.tok.tgz` | 60,979,960 | `77d7f80a51beb5369dffcc498d59298dfbaadeed477e3b230b67f23c486ed504` |
| `2005.logvol.+12.txt` | 152,498 | `868a2e70e85f1ddc1f2449b4e5de37705ff45be2a2c2ea7c8082627c082ed8a4` |
| `2005.logvol.-12.txt` | 152,437 | `f0c0c19cdde939046249163f1d15d00178f754da726980091d0c004c9cd78412` |
| `2006.meta.txt` | 444,679 | `6428caf976fae87ce1f89f761f70d788b78a28ad8c8a5fd92ef026dc99f9dbd9` |
| `2006.tok.tgz` | 55,412,900 | `b417fe73b8284f83613130e8989d577daef19408dc9bd6c91ebf765e10126ad8` |
| `2006.logvol.+12.txt` | 145,153 | `8dc6307345c9ba56af4a0471cf0d4d61e8bd6971160b911bfcf44d7d609ee060` |
| `2006.logvol.-12.txt` | 145,197 | `d443771caa00a8f649248149b4b71849eb2d3217d914d736492f8bca57b8ad96` |

### Row / key-space consistency (asserted, hard)

Per year (hard assertions): `len(meta) == len(logvol.+12) == len(logvol.-12) == n(tok members)`; meta / +12 / −12 / tok agree as **key multisets**; every duplicated key is an **exact** duplicate in every column; no null log-volatility; no key spans two years. Our row counts vs **Kogan et al. (2009) Table 1** ("documents" column):

| year | our rows | Kogan Table 1 | match | unique keys | exact dup rows | unique CIKs |
|---|---|---|---|---|---|---|
| 1996 | 1,408 | 1,408 | YES | 1,406 | 2 | 1,402 |
| 1997 | 2,260 | 2,260 | YES | 2,260 | 0 | 2,249 |
| 1998 | 2,462 | 2,462 | YES | 2,461 | 1 | 2,456 |
| 1999 | 2,524 | 2,524 | YES | 2,524 | 0 | 2,515 |
| 2000 | 2,425 | 2,425 | YES | 2,424 | 1 | 2,419 |
| 2001 | 2,596 | 2,596 | YES | 2,596 | 0 | 2,586 |
| 2002 | 2,846 | 2,846 | YES | 2,845 | 1 | 2,835 |
| 2003 | 3,612 | 3,612 | YES | 3,611 | 1 | 3,605 |
| 2004 | 3,559 | 3,559 | YES | 3,558 | 1 | 3,546 |
| 2005 | 3,474 | 3,474 | YES | 3,474 | 0 | 3,458 |
| 2006 | 3,308 | 3,308 | YES | 3,306 | 2 | 3,299 |
| **total** | **30,474** | **30,474** (sum of their own column) | YES | 30,465 | 9 | 7,207 |

**Exact duplicate records (9 rows in 30,474) are KEPT, deliberately.** The corpus ships 9 records that repeat a key with byte-identical date, URL, company, CIK and *both* log volatilities, and with the `.mda` member repeated inside the tarball. They are not dropped because **the per-year counts that include them are exactly the counts Kogan et al. publish** — their reading is computed over these rows too, so dropping them would silently change the denominator this audit has to match. They are carried as exact duplicates (asserted), share one text row, and at 0.03% of the corpus cannot move any reading materially.

**Every year matches Kogan's published per-year document count exactly** — the corpus we audit is the corpus they report. One published-table arithmetic note (theirs, not ours): Table 1's *total* row reads 26,806 documents, but its own per-year column sums to **30,474**. Their *words* column does sum to the published 247.7M total, and 247.7M/26,806 = 9,240 = their published words/doc, so the 26,806 total is internally consistent with the words/doc cell and inconsistent with the document column. This does not touch the reproduction: their Table 2 micro-average is reproduced exactly from the **per-year** counts (see G-K1), which are the counts our download matches.

## G-K1 — does L0 reproduce the published positive?

### What we compared against (the published number, located and cited)

**Kogan, Levin, Routledge, Sagi & Smith (2009), "Predicting Risk from Financial Reports with Regression", NAACL-HLT 2009, Table 2 (p. 5)** (`http://www.cs.cmu.edu/~nasmith/papers/kogan+levin+routledge+sagi+smith.naacl09.pdf`, SHA-256 `9538e0e07ee36588a2bd478cf41b6b7e47e7c33d8a6da0a5cced1ab805230cd2`). Table 2 reports MSE of log-volatility on test-year predictions:

| Table 2 row | 2001 | 2002 | 2003 | 2004 | 2005 | 2006 | micro-ave |
|---|---|---|---|---|---|---|---|
| `v^(-12)` (baseline) | 0.1747 | 0.1600 | 0.1873 | 0.1442 | 0.1365 | 0.1463 | **0.1576** |
| TFIDF (text only) | 0.2033 | 0.2118 | 0.2178 | 0.1660 | 0.1544 | 0.1599 | 0.1842 |
| **TFIDF+** (text + `v^(-12)`) | 0.1919 | 0.1618 | 0.1965 | *0.1246 | *0.1276 | *0.1403 | **\*0.1557** |

So **the published gain we test against is +1.21%** (0.1576 → 0.1557 micro-averaged MSE; `*` = significant vs baseline under their permutation test, p<0.05). TFIDF+ is the row commensurable with our L0 (their only TF-IDF text+`v^(-12)` model). Their best combined row (LOG1P+ with bigrams, 0.1538) is a +2.41% gain, so the published effect sits in the **1–2.5%** band.

*Verification that we read the table correctly:* their `micro-ave` is the count-weighted pooled MSE over test years, and recomputing it from their own Table 1 per-year counts (2001–06: 2596, 2846, 3612, 3559, 3474, 3308; n=19,395) reproduces **0.15761 → 0.1576** for the baseline and **0.15567 → 0.1557** for TFIDF+ **exactly**. That fixes both the estimand and the aggregation rule we must match, from their numbers alone.

### Our L0 reading

| arm | convention | MSE(`logvol.-12`-only) | MSE(text+control) | gain |
|---|---|---|---|---|
| **L0_pub** (the G-K1 comparator) | Kogan's published split: train = the 5 years preceding, test = 2001..2006, count-weighted micro-average | 0.1576 | 0.1426 | **+9.54%** |
| published (Table 2) | *ibid.* | 0.1576 | 0.1557 | **+1.21%** |

**G-K1: PASS** — sign agrees (both positive), order of magnitude agrees (ours +9.54% vs published +1.21%; ratio 7.91x, gate = within 10x). **Stated plainly:** the gate operationalises the prereg's *same order of magnitude* as "within a factor of 10" (|log10 ratio| < 1); at 7.9x our reading is inside that bound but **not comfortably** — both are single-digit-percent positives, yet ours is the larger. Under the stricter "same power of ten" reading (1.21% is 10^0, 9.54% is 10^1) the gate would not pass and branch (c) would fire. The per-year table below is the evidence that this is one effect reproduced at a different magnitude rather than a different effect. 
**Nothing was tuned to make this match**: the estimator, feature recipe, vocabulary size, alpha grid and the x10 control scaling are the committed `kogan_dissolve.py` constants, fixed before the corpus was downloaded; the published number was located and written into the script as a constant before the ladder ran.

### Per-year, against their Table 2 — the structure, not just the average

A single micro-average is a weak reproduction check, so the same comparison is made year by year (published gain = their `v^(-12)` row vs their `TFIDF+` row):

| test year | published gain % | our L0 gain % | sign agrees |
|---|---|---|---|
| 2001 | -9.85 | -3.01 | YES |
| 2002 | -1.12 | +3.92 | no |
| 2003 | -4.91 | +13.72 | no |
| 2004 | +13.59 | +20.22 | YES |
| 2005 | +6.52 | +10.21 | YES |
| 2006 | +4.10 | +8.74 | YES |

**4/6 test years agree in sign**, and the *shape* of their result is reproduced: 2001 is the worst year for text under both readings (both negative), 2004 the best under both, and text is positive in every post-Sarbanes-Oxley year (2004–06) under both — the very pattern their §6.3 builds the SOX argument on. Our per-year gains sit uniformly **above** theirs (by roughly 5–18pp), which is what a better-conditioned text arm predicts (sublinear + L2-normalised TF-IDF and a CV-tuned ridge vs their raw TF×IDF and fixed SVR hyper-parameters, Disclosure 4) — i.e. we reproduce a **stronger** apparent positive than published, not a weaker one. That direction matters for this audit: the rung being re-priced downstream is, if anything, more favourable to text than the published one.

## A prereg imprecision, recorded — and resolved by reporting BOTH arms

The prereg calls `train ≤ y, test = y+1` *"their annual OOS split"* (**their** annual OOS split). **It is not theirs.** Kogan et al. §6 state: *"We used as training examples all reports from the five-year period preceding the test year (so six experiments on six different training and test sets are shown in the figure)"*, with test years 2001–2006 (Table 2); their Table 4 varies that window over 1, 2 and 5 years and **never** uses an expanding one. The prereg's *rule* is unambiguous; only its *attribution* is wrong. Running only the prereg rule would leave G-K1 unanswerable as specified (an expanding-window reading compared against a rolling-window published number); running only the published rule would violate the binding prereg. **Both were therefore declared in the script before any statistic and both are reported unconditionally** — no arm is selected on its outcome:

- **`L0_prereg`** — the prereg's literal rule (expanding, train ≤ y, test = y+1, y = 1996..2005). **This is the binding rung that feeds L1–L5.**
- **`L0_pub`** — Kogan's actual published convention (5-year rolling, test 2001–06, micro-averaged). **The G-K1 comparator**, because it is the only arm commensurable with their Table 2.

## G-K2 — no look-ahead from L1 on

Asserted per split (hard): `max(train_years) < test_year`; `max(train filing date) < min(test filing date)`; the TF-IDF vocabulary **and** idf are fit on training-year documents only and frozen on the test year; the price control's standardisation uses training-year mean/sd only; the L1 recalibration (intercept+slope) and the L2 reference (incl. the per-CIK mean) are fit on training years only and frozen on the test year.

**L0's look-ahead convention is reproduced deliberately and labelled** (see Disclosures): L0's baseline is `logvol.-12` used *directly* as the prediction with no fit — Kogan's own Table 2 baseline row — which is exactly the rung L1 then repairs.

## G-K3 — CIK coverage and cross-year firm recurrence

- **7,207 unique CIKs** across **30,474 filings** (1996–2006).
- Mean **4.21** years per CIK (median 4, max 11).
- **95.2%** of filings come from a CIK that appears in more than one year — the precondition for the L2 firm-identity reference.
- 20.1% of CIKs appear in exactly one year; 40.2% appear in 5+ years.
- Per-split coverage (share of test rows whose CIK was seen in training, i.e. gets a real per-CIK mean rather than the global fallback) is in the CSV column `cik_train_coverage_test`; the range over the binding `L0_prereg` splits is **43.2%–92.1%**.

## L2 — a SECOND prereg ambiguity, and why it decides the branch

The prereg says the L2 reference *"additionally gets the same-CIK **training-period mean** log volatility (zero-text term)"*. It does not say whether a **training** row's own label may enter its own CIK mean. That silence is not cosmetic — it flips the fired branch, so both readings are computed and reported, and neither is chosen on its outcome:

- **`incl` (literal)** — the per-CIK training mean, self-inclusive. A training row's feature then **contains its own label**.
- **`loo` (leave-one-out)** — the same mean with the row's own label removed (singleton-CIK rows fall back to the global training mean; 5.2%–99.1% of training rows across the binding splits). **PRIMARY.**

Test rows are **identical** under both readings: a test row's own label can never enter a mean taken over training years (G-K2 holds either way). The fork is purely about the training fit.

**Why `loo` is primary — a declared structural reason, not a result.** L2 is specified to *strengthen* the reference (it is the firm-identity control the text must beat). Under `incl` it does the opposite: the fitted coefficient on the per-CIK mean is driven toward 1.0 because the feature partly **is** the label, so the reference overfits and its **test** MSE lands *worse than L1's* — which mechanically *inflates* the text's measured gain and can manufacture a survival. A rung that weakens the reference cannot be the rung the prereg describes. The committed template agrees: `maec_protocol.py`'s entity-mean control (STPEV) is a **point-in-time expanding prior-label mean** built with `shift(1)` — i.e. the current row's label is excluded by construction, and the self-inclusive fixed mean is demoted to a robustness block.

| test year | β on CIK-mean (`incl`) | β on CIK-mean (`loo`) | MSE L1 | MSE L2 `incl` | MSE L2 `loo` | `incl` weakens ref? |
|---|---|---|---|---|---|---|
| 1997 | +1.000 | +0.307 | 0.1272 | 0.2637 | 0.1419 | **YES** |
| 1998 | +1.014 | +0.125 | 0.2298 | 0.3019 | 0.2292 | **YES** |
| 1999 | +1.008 | +0.139 | 0.1465 | 0.2452 | 0.1456 | **YES** |
| 2000 | +0.951 | +0.207 | 0.1513 | 0.2195 | 0.1500 | **YES** |
| 2001 | +0.908 | +0.247 | 0.1922 | 0.2183 | 0.1789 | **YES** |
| 2002 | +0.907 | +0.287 | 0.1615 | 0.2173 | 0.1629 | **YES** |
| 2003 | +0.884 | +0.290 | 0.1978 | 0.2441 | 0.1975 | **YES** |
| 2004 | +0.863 | +0.287 | 0.1375 | 0.2273 | 0.1500 | **YES** |
| 2005 | +0.797 | +0.283 | 0.1308 | 0.1985 | 0.1385 | **YES** |
| 2006 | +0.742 | +0.274 | 0.1426 | 0.1989 | 0.1485 | **YES** |

Across the 10 binding splits the self-inclusive reference is worse than L1's in **10/10**, while the leave-one-out reference is better than L1's in **5/10** — i.e. only `loo` behaves like the control the rung is meant to be. The cleanest demonstration is the `test 1997` split (training = 1996 alone): 99.1% of training rows are the only filing their CIK has, so the "firm mean" *is* that row's label, and the fitted β is **+1.000**.

**Branch under each reading:** `loo` (primary) → **(b)** with 3/10 placebo-gated survivors (6/10 ungated); `incl` (literal) → **(a)** with 0/10 placebo-gated (8/10 ungated).

**The two readings fire DIFFERENT branches** — the prereg as written does not determine the finding, and the authors must rule on the wording before the paper commits.

## THE LADDER — `L0_prereg` (binding), per test year

`gain_pct` = % reduction in log-volatility MSE of the text arm vs that rung's reference. Sign convention: naive obs t **positive** = text better; DM **negative** = text better.

L2–L5 are shown under the **primary (leave-one-out)** L2 reading; the literal self-inclusive reading is tabulated in the L2 section above.

| rung | test year | n | gain % | stat | value | p | verdict |
|---|---|---|---|---|---|---|---|
| L0 | 1997 | 2,260 | +3.10 | naive obs t | +1.91 | 0.0560 | **null** |
| L0 | 1998 | 2,462 | +7.73 | naive obs t | +12.24 | 1.84e-33 | **text adds** |
| L0 | 1999 | 2,524 | +6.07 | naive obs t | +3.87 | 1.12e-04 | **text adds** |
| L0 | 2000 | 2,425 | +9.69 | naive obs t | +9.22 | 6.45e-20 | **text adds** |
| L0 | 2001 | 2,596 | -3.01 | naive obs t | -3.02 | 0.0026 | **text HURTS** |
| L0 | 2002 | 2,846 | +4.72 | naive obs t | +4.36 | 1.32e-05 | **text adds** |
| L0 | 2003 | 3,612 | +14.70 | naive obs t | +12.95 | 1.59e-37 | **text adds** |
| L0 | 2004 | 3,559 | +20.74 | naive obs t | +13.38 | 7.35e-40 | **text adds** |
| L0 | 2005 | 3,474 | +11.69 | naive obs t | +8.14 | 5.42e-16 | **text adds** |
| L0 | 2006 | 3,308 | +8.68 | naive obs t | +6.33 | 2.81e-10 | **text adds** |
| L1 | 1997 | 2,260 | +1.14 | naive obs t | +0.89 | 0.3713 | **null** |
| L1 | 1998 | 2,462 | +4.65 | naive obs t | +11.27 | 9.21e-29 | **text adds** |
| L1 | 1999 | 2,524 | +0.88 | naive obs t | +0.89 | 0.3745 | **null** |
| L1 | 2000 | 2,425 | +6.41 | naive obs t | +7.06 | 2.12e-12 | **text adds** |
| L1 | 2001 | 2,596 | +6.38 | naive obs t | +6.90 | 6.45e-12 | **text adds** |
| L1 | 2002 | 2,846 | +5.63 | naive obs t | +6.72 | 2.23e-11 | **text adds** |
| L1 | 2003 | 3,612 | +19.25 | naive obs t | +23.52 | 6.21e-114 | **text adds** |
| L1 | 2004 | 3,559 | +16.89 | naive obs t | +12.47 | 5.64e-35 | **text adds** |
| L1 | 2005 | 3,474 | +7.89 | naive obs t | +6.05 | 1.60e-09 | **text adds** |
| L1 | 2006 | 3,308 | +6.28 | naive obs t | +4.91 | 9.72e-07 | **text adds** |
| L2 | 1997 | 2,260 | +11.40 | naive obs t | +6.89 | 7.31e-12 | **text adds** |
| L2 | 1998 | 2,462 | +4.40 | naive obs t | +8.97 | 5.62e-19 | **text adds** |
| L2 | 1999 | 2,524 | +0.22 | naive obs t | +0.19 | 0.8502 | **null** |
| L2 | 2000 | 2,425 | +5.60 | naive obs t | +5.87 | 5.00e-09 | **text adds** |
| L2 | 2001 | 2,596 | -0.61 | naive obs t | -0.55 | 0.5827 | **null** |
| L2 | 2002 | 2,846 | +6.44 | naive obs t | +6.34 | 2.70e-10 | **text adds** |
| L2 | 2003 | 3,612 | +19.12 | naive obs t | +22.23 | 9.53e-103 | **text adds** |
| L2 | 2004 | 3,559 | +23.82 | naive obs t | +15.43 | 5.10e-52 | **text adds** |
| L2 | 2005 | 3,474 | +12.96 | naive obs t | +8.58 | 1.37e-17 | **text adds** |
| L2 | 2006 | 3,308 | +10.04 | naive obs t | +6.60 | 4.75e-11 | **text adds** |
| L3 | 1997 | 2,260 | +11.40 | date-clustered DM | -2.50 | 0.0131 | **text adds** |
| L3 | 1998 | 2,462 | +4.40 | date-clustered DM | -4.05 | 7.17e-05 | **text adds** |
| L3 | 1999 | 2,524 | +0.22 | date-clustered DM | +0.99 | 0.3241 | **null** |
| L3 | 2000 | 2,425 | +5.60 | date-clustered DM | -2.80 | 0.0056 | **text adds** |
| L3 | 2001 | 2,596 | -0.61 | date-clustered DM | -0.09 | 0.9248 | **null** |
| L3 | 2002 | 2,846 | +6.44 | date-clustered DM | -3.00 | 0.0030 | **text adds** |
| L3 | 2003 | 3,612 | +19.12 | date-clustered DM | -8.10 | 3.38e-14 | **text adds** |
| L3 | 2004 | 3,559 | +23.82 | date-clustered DM | -6.91 | 5.34e-11 | **text adds** |
| L3 | 2005 | 3,474 | +12.96 | date-clustered DM | -2.72 | 0.0071 | **text adds** |
| L3 | 2006 | 3,308 | +10.04 | date-clustered DM | -2.17 | 0.0315 | **text adds** |
| L4 | 1997 | 2,260 | +11.40 | date-clustered DM + Holm | -2.50 | 0.0523 | **null** |
| L4 | 1998 | 2,462 | +4.40 | date-clustered DM + Holm | -4.05 | 5.74e-04 | **text adds** |
| L4 | 1999 | 2,524 | +0.22 | date-clustered DM + Holm | +0.99 | 0.6482 | **null** |
| L4 | 2000 | 2,425 | +5.60 | date-clustered DM + Holm | -2.80 | 0.0337 | **text adds** |
| L4 | 2001 | 2,596 | -0.61 | date-clustered DM + Holm | -0.09 | 0.9248 | **null** |
| L4 | 2002 | 2,846 | +6.44 | date-clustered DM + Holm | -3.00 | 0.0208 | **text adds** |
| L4 | 2003 | 3,612 | +19.12 | date-clustered DM + Holm | -8.10 | 3.38e-13 | **text adds** |
| L4 | 2004 | 3,559 | +23.82 | date-clustered DM + Holm | -6.91 | 4.80e-10 | **text adds** |
| L4 | 2005 | 3,474 | +12.96 | date-clustered DM + Holm | -2.72 | 0.0355 | **text adds** |
| L4 | 2006 | 3,308 | +10.04 | date-clustered DM + Holm | -2.17 | 0.0944 | **null** |
| L5 | 1997 | 2,260 | +11.40 | conjunction L1&L2&L4 | -2.50 | 0.0523 | **does not survive** |
| L5 | 1998 | 2,462 | +4.40 | conjunction L1&L2&L4 | -4.05 | 5.74e-04 | **SURVIVES** |
| L5 | 1999 | 2,524 | +0.22 | conjunction L1&L2&L4 | +0.99 | 0.6482 | **does not survive** |
| L5 | 2000 | 2,425 | +5.60 | conjunction L1&L2&L4 | -2.80 | 0.0337 | **SURVIVES** |
| L5 | 2001 | 2,596 | -0.61 | conjunction L1&L2&L4 | -0.09 | 0.9248 | **does not survive** |
| L5 | 2002 | 2,846 | +6.44 | conjunction L1&L2&L4 | -3.00 | 0.0208 | **SURVIVES** |
| L5 | 2003 | 3,612 | +19.12 | conjunction L1&L2&L4 | -8.10 | 3.38e-13 | **SURVIVES** |
| L5 | 2004 | 3,559 | +23.82 | conjunction L1&L2&L4 | -6.91 | 4.80e-10 | **SURVIVES** |
| L5 | 2005 | 3,474 | +12.96 | conjunction L1&L2&L4 | -2.72 | 0.0355 | **SURVIVES** |
| L5 | 2006 | 3,308 | +10.04 | conjunction L1&L2&L4 | -2.17 | 0.0944 | **does not survive** |

## THE LADDER — `L0_pub` (published convention), per test year

| rung | test year | n | gain % | stat | value | p | verdict |
|---|---|---|---|---|---|---|---|
| L0 | 2001 | 2,596 | -3.01 | naive obs t | -3.02 | 0.0026 | **text HURTS** |
| L0 | 2002 | 2,846 | +3.92 | naive obs t | +3.55 | 3.86e-04 | **text adds** |
| L0 | 2003 | 3,612 | +13.72 | naive obs t | +12.00 | 1.46e-32 | **text adds** |
| L0 | 2004 | 3,559 | +20.22 | naive obs t | +13.27 | 2.82e-39 | **text adds** |
| L0 | 2005 | 3,474 | +10.21 | naive obs t | +7.39 | 1.88e-13 | **text adds** |
| L0 | 2006 | 3,308 | +8.74 | naive obs t | +7.03 | 2.57e-12 | **text adds** |
| L1 | 2001 | 2,596 | +6.38 | naive obs t | +6.90 | 6.45e-12 | **text adds** |
| L1 | 2002 | 2,846 | +5.72 | naive obs t | +6.69 | 2.67e-11 | **text adds** |
| L1 | 2003 | 3,612 | +20.76 | naive obs t | +25.63 | 2.99e-133 | **text adds** |
| L1 | 2004 | 3,559 | +9.58 | naive obs t | +8.28 | 1.71e-16 | **text adds** |
| L1 | 2005 | 3,474 | +3.87 | naive obs t | +4.25 | 2.20e-05 | **text adds** |
| L1 | 2006 | 3,308 | +4.81 | naive obs t | +6.19 | 6.91e-10 | **text adds** |
| L2 | 2001 | 2,596 | -0.61 | naive obs t | -0.55 | 0.5827 | **null** |
| L2 | 2002 | 2,846 | +6.43 | naive obs t | +6.33 | 2.77e-10 | **text adds** |
| L2 | 2003 | 3,612 | +22.07 | naive obs t | +24.60 | 1.24e-123 | **text adds** |
| L2 | 2004 | 3,559 | +15.71 | naive obs t | +11.57 | 1.99e-30 | **text adds** |
| L2 | 2005 | 3,474 | +4.21 | naive obs t | +3.87 | 1.13e-04 | **text adds** |
| L2 | 2006 | 3,308 | +3.80 | naive obs t | +3.94 | 8.28e-05 | **text adds** |
| L3 | 2001 | 2,596 | -0.61 | date-clustered DM | -0.09 | 0.9248 | **null** |
| L3 | 2002 | 2,846 | +6.43 | date-clustered DM | -3.24 | 0.0014 | **text adds** |
| L3 | 2003 | 3,612 | +22.07 | date-clustered DM | -8.63 | 1.10e-15 | **text adds** |
| L3 | 2004 | 3,559 | +15.71 | date-clustered DM | -5.24 | 3.79e-07 | **text adds** |
| L3 | 2005 | 3,474 | +4.21 | date-clustered DM | -1.87 | 0.0635 | **null** |
| L3 | 2006 | 3,308 | +3.80 | date-clustered DM | -1.32 | 0.1889 | **null** |
| L4 | 2001 | 2,596 | -0.61 | date-clustered DM + Holm | -0.09 | 0.9248 | **null** |
| L4 | 2002 | 2,846 | +6.43 | date-clustered DM + Holm | -3.24 | 0.0056 | **text adds** |
| L4 | 2003 | 3,612 | +22.07 | date-clustered DM + Holm | -8.63 | 6.58e-15 | **text adds** |
| L4 | 2004 | 3,559 | +15.71 | date-clustered DM + Holm | -5.24 | 1.90e-06 | **text adds** |
| L4 | 2005 | 3,474 | +4.21 | date-clustered DM + Holm | -1.87 | 0.1904 | **null** |
| L4 | 2006 | 3,308 | +3.80 | date-clustered DM + Holm | -1.32 | 0.3777 | **null** |
| L5 | 2001 | 2,596 | -0.61 | conjunction L1&L2&L4 | -0.09 | 0.9248 | **does not survive** |
| L5 | 2002 | 2,846 | +6.43 | conjunction L1&L2&L4 | -3.24 | 0.0056 | **SURVIVES** |
| L5 | 2003 | 3,612 | +22.07 | conjunction L1&L2&L4 | -8.63 | 6.58e-15 | **SURVIVES** |
| L5 | 2004 | 3,559 | +15.71 | conjunction L1&L2&L4 | -5.24 | 1.90e-06 | **SURVIVES** |
| L5 | 2005 | 3,474 | +4.21 | conjunction L1&L2&L4 | -1.87 | 0.1904 | **does not survive** |
| L5 | 2006 | 3,308 | +3.80 | conjunction L1&L2&L4 | -1.32 | 0.3777 | **does not survive** |

### Micro-averaged rung summary (`L0_pub`, Kogan's aggregation)

| rung | reference | MSE(ref) | MSE(text) | gain % |
|---|---|---|---|---|
| L0 | RAW `logvol.-12` (their baseline) | 0.1576 | 0.1426 | **+9.54** |
| L1 | recalibrated `logvol.-12` | 0.1577 | 0.1426 | **+9.62** |
| L2 loo | recalibrated `logvol.-12` + same-CIK train mean (**primary**, leave-one-out) | 0.1583 | 0.1426 | **+9.93** |
| L2 incl | recalibrated `logvol.-12` + same-CIK train mean (literal, self-inclusive) | 0.2207 | 0.1426 | **+35.39** |

## Placebo — label shuffle (5 seeds), |DM| < 2 gate **— THE LOAD-BEARING RESULT**

The text rows are permuted (price control and label kept aligned — the committed `maec_protocol.run_placebo` convention), the arm is refit, and the L3-stage filing-date-clustered DM is recomputed against the same L2 reference. Alpha is frozen at each split's real-data CV choice (disclosed: the placebo interrogates the signal, not the tuning). DM **negative = the arm beats the reference**.

| test year | REAL L3 DM | placebo mean DM | placebo max abs DM | gate (<2) | real vs placebo |
|---|---|---|---|---|---|
| 1997 | -2.50 | -2.08 | 2.87 | **FAIL** | *not separated* |
| 1998 | -4.05 | +1.12 | 1.64 | PASS | **separated** |
| 1999 | +0.99 | +1.87 | 2.70 | **FAIL** | no real edge |
| 2000 | -2.80 | +0.44 | 0.98 | PASS | **separated** |
| 2001 | -0.09 | +5.19 | 6.16 | **FAIL** | no real edge |
| 2002 | -3.00 | -1.39 | 3.05 | **FAIL** | **separated** |
| 2003 | -8.10 | -0.37 | 1.89 | PASS | **separated** |
| 2004 | -6.91 | -2.82 | 3.91 | **FAIL** | **separated** |
| 2005 | -2.72 | -2.38 | 2.99 | **FAIL** | *not separated* |
| 2006 | -2.17 | -0.59 | 1.48 | PASS | **separated** |

**Only 4/10 splits pass the |DM|<2 placebo gate, and 13/50 individual placebo draws show the SHUFFLED-text arm *significantly beating* the L2 reference (DM < −2).** This is the most important number in the table and it is reported first because it bounds everything below it.

**Why the placebo is not null here, and what that means.** Unlike `maec_protocol`, where the text enters a *combiner on top of the reference* (so permuting it collapses the arm back onto the reference and DM→0 by construction), Kogan's convention makes the arm a **single joint model** — one ridge on `[TF-IDF | scaled logvol.-12]`. Permuting the text therefore does **not** reduce the arm to the reference: it leaves a ridge on `[noise | logvol.-12]`, which still carries the real price control and is a structurally different estimator from the L2 reference (`OLS[1, logvol.-12, CIK-mean]`). So a non-zero placebo DM is partly a **form** difference, not text signal — the ported gate cannot separate the two, and it is the *comparison of real vs placebo DM*, not the gate alone, that carries the inference.

Read that way the years split sharply:

- **Genuine text signal** — 1998 (real −4.05 vs placebo +1.12), 2000 (−2.80 vs +0.44), 2003 (−8.10 vs −0.37) and 2006 (−2.17 vs −0.59): the real arm is far more negative than shuffled text ever gets, and the gate passes. (2006 clears the placebo but not L4's Holm-adjusted p, so it is not an L5 survivor.)
- **Not attributable to text** — 1997 (real −2.50 vs placebo −2.08) and 2005 (−2.72 vs −2.38): **shuffled text very nearly reproduces the whole edge**, so these years' apparent gains are the arm-vs-reference form difference, not disclosure content.
- **No edge to explain** — 1999 (real +0.99) and 2001 (real −0.09): the real arm does not beat the reference at all, and 2001's placebo runs strongly the *other* way (+5.19).

Consequently the L5 conjunction is reported **twice**: ungated (6/10) and **placebo-gated** (3/10) — the latter is the honest survivor count, and the one the branch is decided on.

## L3 inference — the HAC lag, disclosed in full

The prereg fixes the clustering unit (**filing date**) and the estimator (**HAC + HLN**) but not the lag. Two lags are reported for every split; neither is chosen on its outcome:

- **`lag_strict_overlap`** — the `maec_protocol.hac_lag_L` port: the number of later distinct test filing dates whose **12-month label window still overlaps**. With a 12-month label and a 12-month test year this is ≈ `n_dates − 1` **by construction**, so the HLN factor collapses to ≤ 0 and the test is *undefined*. That is not a bug — it is the honest statement that **a single test year of 12-month-forward labels contains ≈ one effective observation**.
- **`lag_primary_nw`** — the Newey–West rule of thumb `floor(4*(n/100)^(2/9))` on the filing-date grid. **This is the primary**, chosen by a rule declared before the numbers: of the two candidates it is the **more permissive**, i.e. the one **most favourable to text**. Any death under it therefore cannot be a lag artefact, and any survival is reported on text's best terms.

| test year | n dates | lag (NW, primary) | DM | p | lag (strict overlap) | DM | p |
|---|---|---|---|---|---|---|---|
| 1997 | 216 | 4 | -2.50 | 0.0131 | 215 | n/a (h≈n) | n/a |
| 1998 | 209 | 4 | -4.05 | 7.17e-05 | 208 | n/a (h≈n) | n/a |
| 1999 | 204 | 4 | +0.99 | 0.3241 | 203 | n/a (h≈n) | n/a |
| 2000 | 203 | 4 | -2.80 | 0.0056 | 202 | n/a (h≈n) | n/a |
| 2001 | 202 | 4 | -0.09 | 0.9248 | 201 | n/a (h≈n) | n/a |
| 2002 | 223 | 4 | -3.00 | 0.0030 | 222 | n/a (h≈n) | n/a |
| 2003 | 227 | 4 | -8.10 | 3.38e-14 | 226 | n/a (h≈n) | n/a |
| 2004 | 219 | 4 | -6.91 | 5.34e-11 | 218 | n/a (h≈n) | n/a |
| 2005 | 221 | 4 | -2.72 | 0.0071 | 220 | n/a (h≈n) | n/a |
| 2006 | 205 | 4 | -2.17 | 0.0315 | 204 | n/a (h≈n) | n/a |

## THE FIRED BRANCH

**(b) text SURVIVES the full cascade** — *registered consequence:* "**the protocol certified a genuine published positive result**: this is exactly the **real-world positive control** repeatedly demanded by the internal adversarial dry-run (proving the protocol does not only kill); report it faithfully and accordingly soften the universalising wording of 'near-null' — `the size of the shortcut is a property of the panel and the baseline, not a constant` (directly absorbed by the existing FACTS §11/§13g framework); **this is good news for the paper, write it as such**."

*Why this branch:* G-K1 PASS (L0 reproduces the published positive in sign and order of magnitude) and **3/10** `L0_prereg` test years survive the L5 conjunction (L1 ∧ L2 ∧ L4) **and** clear the label-shuffle placebo gate under the primary leave-one-out L2 reading (6/10 survive L5 before the gate) — placebo-clean surviving years: [1998, 2000, 2003].

**Robustness of the branch across the two open choices.** The four combinations give: L2 `loo` gated **(b)** / ungated **(b)**; L2 `incl` gated **(a)** / ungated **(b)**.  **They do not all agree, and the exception is diagnostic rather than substantive.** Under the *primary* `loo` reading the branch is stable — (b) both gated and ungated. The only cell that differs is `incl`+gate → (a), and it differs for a degenerate reason: the self-inclusive reference is so damaged that **shuffled text beats it in all 10 splits** (placebo mean DM −2.1 to −9.0), so *nothing* clears the placebo gate and the survivor count collapses to 0. That is the placebo independently detecting the broken control — it is not evidence that text fails. A reader should therefore not read `incl`+gate as a genuine reproduce-then-dissolve.

**But read the size honestly.** The positive control is real yet *narrow*: 3/10 years, and the placebo shows that in the non-surviving and in two of the L5-surviving years shuffled text reproduces much or all of the apparent edge. The defensible claim is *"the protocol certifies text on Kogan's corpus in a minority of test years, concentrated in 1998/2000/2003"* — **not** "text survives on Kogan's corpus".

## Disclosures

1. **L0's look-ahead / naive convention is reproduced deliberately.** L0's baseline is `logvol.-12` used directly as the forecast with **no fit at all** — Kogan's own Table 2 baseline row — and L0/L1/L2 inference is **naive obs-level t**, treating ~2.5k same-year filings with 12-month-overlapping labels as independent. Both are the defects being reproduced, not our protocol. From L1 on every weight is training-year-fit and frozen (G-K2); from L3 on inference is filing-date-clustered.
2. **Data is not redistributed.** Only `kogan_corpus_fetch.py` and this script ship. The corpus is public at `http://www.cs.cmu.edu/~ark/10K/` with no licence terms beyond a citation request, which we honour in the text and here: Kogan, Levin, Routledge, Sagi & Smith, *Predicting Risk from Financial Reports with Regression*, NAACL-HLT 2009. Every file's SHA-256 is above, so the exact bytes we read are verifiable without us hosting them.
3. **The text is the tokenised MD&A section**, not the whole 10-K: the corpus's own README defines `yyyy.tok.tgz` as "the tokenized MD&A sections", and the paper (§4) filters to Section 7/7A on purpose. Kogan's Table 2 is computed on this same text, so the comparison is like-for-like.
4. **Estimator and feature deviations from their exact spec** (each is the committed `kogan_dissolve.py` recipe, fixed before any number here was seen, and none was adjusted afterwards): ridge with alpha by 5-fold CV over `[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]` in place of their SVR (SVM^light, linear kernel, eps=0.1, C=1/mean(h·h)); TF-IDF 1–2gram, top-5k by train term frequency, sublinear tf, L2 norm, in place of their `(1/|d|)·freq × log(N/df)` over the full training vocabulary; the control enters the text arm standardised and scaled ×10 so its ridge penalty is ~1/100 of a text feature's ≈ the unpenalised control their design implies. These are why G-K1 is a **sign + order-of-magnitude** gate, exactly as pre-registered, and not an exact-value gate.
5. **Vocabulary pruning guard.** Term counters are pruned within 256-doc chunks (hapaxes) and per year (count < 5) for memory. Both prunings are asserted irrelevant to top-5k selection per split: the smallest surviving top-5k count must exceed both the year-prune bound (5 × #train years) and the chunk-prune bound (#chunks, the max count a term can lose). Margins are in `_work/results.json` (`vocab_guard`).
6. **Two prereg ambiguities were found and neither was resolved by choosing.** (i) the OOS split attribution (§"A prereg imprecision"), and (ii) L2's self-inclusion (§"L2 — a SECOND prereg ambiguity"). Both are reported under every reading, with the primary fixed by a structural argument declared in the script before execution. (i) does not change the branch. (ii) **does**: `loo` fires (b) while `incl` fires (a) once the placebo gate is applied — see "Robustness of the branch" for why `incl`'s outcome is a symptom of its broken control rather than a finding. **The prereg wording therefore needs a ruling before this section is written up**, and no `incl` L2 number should be quoted.
6b. **The placebo gate is not a formality here — it fails in 6/10 splits** and is the reason the survivor count is reported as 3/10 rather than 6/10. Because Kogan's convention makes the arm a joint model rather than a combiner over the reference, the ported |DM|<2 gate conflates text signal with the arm-vs-reference form difference; the real-vs-placebo DM comparison (see that section) is what the inference actually rests on. A protocol-level note for the panel: the gate's ported form is mis-calibrated for joint arms and should be re-specified for them.
7. **Single-shot.** The script refuses to overwrite these tables without `--force-rerun --reason`; this run WAS a force-rerun, reason: "BUG-FIX (pre-tabulation, final): (1) the label-shuffle placebo is pre-registered as a GATE ('|DM|<2 as the gate') but L5 was reporting survivors WITHOUT applying it, while the gate in fact fails in 6/10 splits and 13/50 placebo draws show the SHUFFLED-text arm significantly beating the L2 reference; L5 now carries both an ungated and a placebo-gated survivor count and the branch is decided on the gated count. (2) the L2 self-inclusion fork: literal reading puts a training row's OWN label in its own feature (fitted beta ~1.0 in 10/10 splits), so its reference is WORSE than L1's in 10/10; leave-one-out (committed maec STPEV shift(1) precedent) added as PRIMARY, BOTH reported, nothing selected on outcome. (3) corrected three self-contradictions in the generated prose: the 'branch robust under every combination' claim (it is not - incl+gate fires (a) because shuffled text beats that broken reference in all 10 splits, a degenerate outcome now explained), the stale 'neither ambiguity changes the branch' disclosure, and a placebo column that labelled a year 'separated' when its real DM was -0.09 (no edge). All such claims are now value-driven. No statistic retuned; L0/L1 reproduce earlier runs bit-for-bit.". The two L0 arms and the two L2 readings were all declared before execution precisely so that none could be picked after the fact.
8. **Compute.** Local CPU only, BLAS threads pinned to 1 and joblib capped at 4 workers on a shared machine; no GPU, no `/Volumes/Z`, total runtime 3.0 min.
