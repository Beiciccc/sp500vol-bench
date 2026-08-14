# Label-noise budget for the close-to-close RV label

Two nearly-unbiased estimators of the same latent integrated variance
over the same window identify the label's estimator noise without any
high-frequency data: close-to-close (the label) and Garman--Klass
(same daily bars, disjoint information). Within-firm, test era
(windows starting 2022-01-01 onward), non-overlapping windows.

| h | windows | Var(log CC) | signal | noise | noise share | R2 ceiling | median MDE (%) |
|---|---|---|---|---|---|---|---|
| 5 | 100,485 | 0.916 | 0.414 | 0.502 | 54.8% | 0.452 | 0.76 |
| 10 | 49,748 | 0.578 | 0.325 | 0.253 | 43.8% | 0.562 | 0.82 |
| 20 | 24,617 | 0.397 | 0.258 | 0.139 | 34.9% | 0.651 | 1.22 |

`noise share` = 1 - Cov(log CC, log GK)/Var(log CC). `R2 ceiling` is the
highest R^2 any forecast can attain against this label even with perfect
knowledge of the latent variance. `attenuation` (in the CSV) is the factor
by which a true proportional loss reduction shrinks when scored against
the noisy label: an increment of size delta in true-RV units is measured
as roughly `attenuation` x delta here.

## Scope and direction of the bias

Rows price the INTRADAY component (open-to-close vs GK, target-matched).
Roughly 34% of the close-to-close label's variance is the
overnight return, which GK cannot see; scoring the label directly against
GK charges mix shifts to noise and inflates the share (carried as
`noise_share_cc_mismatched`). The overnight half is estimated from one
return per day and is not identified here.

The identity also assumes the two errors are uncorrelated. They
are not: both read the same price path, so a large intraday move
inflates both. A simulated diffusion with known latent variance gives
Corr(u, v) = +0.26 at h=5, under which this identity returns a
0.454 noise share where the truth is 0.507. A positive
Cov(u, v) makes the covariance overstate the signal, so the true noise
share of the intraday component is **at least** what is tabulated.
for a paper reporting a null. `bias_probe()` in this script reproduces
the simulation.
