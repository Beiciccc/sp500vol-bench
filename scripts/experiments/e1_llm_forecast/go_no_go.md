# E1 go/no-go gate — after `run_inference.py run --pilot 500`

Decision rule: start the full 102k-generation run ONLY if **all four gates pass for
both variants** on the 500-filing pilot (first 500 TEST filings, stratified by form).
Evaluate per variant (`c6_text`, `d4_fused`).

## Gates

| # | Criterion | Threshold | Rationale |
|---|---|---|---|
| 1 | Parse-success rate | **>= 95%** | JSON discipline; below this the retry/fill machinery distorts the signal |
| 2 | Clipped-rate (outside [0.03, 3.0] annualized) | **<= 5%** | LLM garbage guard; high clip rate = the model doesn't understand the unit |
| 3 | Pilot test QLIKE (vol-unit) vs `A1_hv` naive baseline on the SAME filings | **QLIKE_llm < QLIKE_A1** | If a zero-shot LLM can't beat naive historical vol, the full run is wasted GPU money |
| 4 | Spearman(prediction, label) | **> 0.15** | Minimal cross-sectional ranking skill |

Notes:
* Gate 3 uses VOL-unit qlike `y/f - log(y/f) - 1` (same convention as
  `forecast_combination.qlike`); the metric is computed over all 3 horizons pooled,
  joined to `A1_hv_full_combined_seed2026` test predictions by `(text_path, horizon_days)`.
* Gates 3-4 are deliberately soft (A1_hv, not A2 HAR): the C6/D4 headline claim is made
  by the M1 combiner, not by standalone accuracy. The pilot only needs to show the
  output is not noise.

## Evaluation command (run on the Mac; needs results/runs/A1_hv_*)

```bash
.venv/bin/python scripts/experiments/e1_llm_forecast/postprocess.py pilot-eval \
    --raw-dir <pilot raw dir> --manifest results/e1_llm_forecast/manifest_valtest.parquet
```

Prints per-variant JSON with each gate boolean and a final `"GO"` flag.

## Equivalent standalone snippet

```python
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

RAW, MAN = "<pilot raw dir>", "results/e1_llm_forecast/manifest_valtest.parquet"
raw = pd.concat([pd.read_parquet(p) for p in sorted(Path(RAW).glob("part-*.parquet"))])
a1 = pd.read_parquet("results/runs/A1_hv_full_combined_seed2026/predictions.parquet")
a1 = a1[a1.split == "test"][["text_path", "horizon_days", "label_realised_vol",
                             "prediction_realised_vol"]].rename(columns={"prediction_realised_vol": "f_a1"})
qlike = lambda y, f: (np.clip(y,1e-8,None)/np.clip(f,1e-8,None)
                      - np.log(np.clip(y,1e-8,None)/np.clip(f,1e-8,None)) - 1)
for variant, g in raw.groupby("variant"):
    parse_rate = g.parse_ok.mean()
    long = g.melt(id_vars="text_path", value_vars=["vol_5d","vol_10d","vol_20d"],
                  var_name="h", value_name="f_llm").dropna()
    long["horizon_days"] = long.h.map({"vol_5d":5,"vol_10d":10,"vol_20d":20})
    clip_rate = ((long.f_llm < 0.03) | (long.f_llm > 3.0)).mean()
    j = long.merge(a1, on=["text_path","horizon_days"])
    j["f_llm"] = j.f_llm.clip(0.03, 3.0)
    ql_llm, ql_a1 = qlike(j.label_realised_vol, j.f_llm).mean(), qlike(j.label_realised_vol, j.f_a1).mean()
    rho = stats.spearmanr(j.f_llm, j.label_realised_vol).statistic
    print(variant, dict(parse=round(parse_rate,3), clip=round(clip_rate,3),
                        ql_llm=round(ql_llm,4), ql_a1=round(ql_a1,4), rho=round(rho,3),
                        GO=(parse_rate>=.95) and (clip_rate<=.05) and (ql_llm<ql_a1) and (rho>.15)))
```

## If a gate fails — prompt iteration playbook (re-pilot after each change; ~30-60 min each)

* **Gate 1 fails (parse)**: (a) confirm guided JSON decoding is active (runner logs a
  warning if not) — if inactive, upgrade vLLM; (b) tighten `SYSTEM_PROMPT` in
  `prompt.py`; (c) raise `--max-tokens` to 200 (model may be running out mid-JSON);
  (d) check thinking-mode is disabled (raw_output starting with `<think>` = bug).
* **Gate 2 fails (clip)**: the unit convention isn't landing — add one worked example
  to the task text ("a typical calm large-cap is ~0.20") or switch the ask to
  percent-and-divide server-side.
* **Gate 3/4 fail for d4_fused**: serious — with HAR lags in-context the model merely
  needs to echo an anchor. Check the lags render correctly in the prompt; try the 14B
  fallback for comparison; if both fail, the model is unusable — stop, no full run.
* **Gate 3/4 fail for c6_text only**: expected-ish (text-alone loses to price models in
  this project too — see M1 memory). Judgement call: if `d4_fused` passes everything and
  `c6_text` passes gates 1-2 and rho > 0.05, proceed with BOTH (the M1 combiner, not
  standalone QLIKE, is the endpoint); document the waiver in the run notes. If
  c6_text rho <= 0.05, iterate the excerpt policy (e.g. 8-K-only pilot vs 10-K-only
  pilot to localise the failure) before deciding.
* Any iteration = new `--out-dir` for the pilot (do not mix prompt versions in one raw dir).
