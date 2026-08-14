# Weight-window sensitivity for Eq. (1)

Does fitting the combination weights on the COVID validation block push
the text coefficient toward zero? Validation (2020--2021) splits into a
COVID half (2020) and a calm half (2021); refitting on each and freezing
to the IDENTICAL test rows isolates the regime, since both halves are the
same block under the same protocol. Models are never retrained.

## The identifying contrast (calm half vs COVID half)

- cells: **69** (the full grid)
- calm-fit minus COVID-fit increment: mean **+0.433pp**, median **+0.064pp**
- calm higher in **37/69** cells
- paired t = **+1.12**, p = **0.268**

So the direction the reviewer anticipated is present -- a calm-window fit
does credit text slightly more -- but it is small and not significant, and
the mean increment is negative under BOTH halves
(calm -0.087pp, COVID -0.519pp).

## Each arm against the committed full-validation fit

| arm | n | mean shift | sign flips down / up | Spearman |
|---|---|---|---|---|
| calm_2021 | 69 | -0.07pp | 20 / 14 | -0.073 |
| covid_2020 | 69 | -0.50pp | 9 / 6 | +0.800 |
| alt_2018_19 | 57 | -19.84pp | 27 / 8 | -0.308 |

`alt_2018_19` refits on the tail of the TRAINING era. It is reported for
completeness only and must not be read as a regime test: those rows are
in-sample for every text model, so its large negative shift reflects
fitting g to memorised predictions, not a property of the regime.
Note also that the committed validation block is the early-stopping set
for the neural arms, so no arm here is fully out-of-sample for them;
what the contrast holds fixed is the protocol, not model exposure.

