# A100-40G Batch Probe

Device: NVIDIA A100-SXM4-40GB.
Headroom rule: largest passing batch with torch peak allocated <= 36.0 GB.
Batch kind: `encode` = frozen-LLM encode_batch_size (C5_*/D3_*); `train` = training step batch.

| model | safe_batch@40G | kind | torch_peak_GB | nvidia_smi_peak_GB | step_s | first_oom |
|---|---:|---|---:|---:|---:|---:|
| C1_bert_s1 |  | train |  |  |  |  |
| C2_finbert_s1 |  | train |  |  |  |  |
| C3_roberta_s1 |  | train |  |  |  |  |
| C4_longformer |  | train |  |  |  |  |
| D1_concat_mlp |  | train |  |  |  |  |
| D2_gated_fusion |  | train |  |  |  |  |
| C2_finbert_s2 |  | train |  |  |  |  |
| C2_finbert_s3 | 6 | train | 29.095 | 30.251 | 0.462967 |  |
| C2_finbert_s4 | 6 | train | 29.2 | 30.013 | 0.465651 | 8 |
| C5_qwen3 |  | encode |  |  |  |  |
| C5_gteqwen2 |  | encode |  |  |  |  |
| C5_e5mistral |  | encode |  |  |  |  |
| D3_qwen3 |  | encode |  |  |  |  |
| D3_gteqwen2 |  | encode |  |  |  |  |
| D3_e5mistral |  | encode |  |  |  |  |
