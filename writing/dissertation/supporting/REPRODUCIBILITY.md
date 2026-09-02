# Release manifest

We release the auditable, licence-safe core of SP500Vol-Bench inside the
code-and-data archive lodged with this dissertation, and a public repository
carries the redistributable part of the same material: (i) an **accession index** of all 144,129 benchmark filings
(accession number, CIK, ticker, form, filing timestamp, effective trading day,
and train/validation/test split), which doubles as the exact SEC EDGAR pull
list; (ii) the **914 survivorship-free membership intervals** and the
**point-in-time PERMNO->CIK link windows**, including the
audited successor-CIK overrides that recover filings from acquired or delisted
firms; (iii) the **chronological split definition** (reproducible from the
effective trading day alone); (iv) **SHA-256 configuration fingerprints**
for **240 production runs** spanning the price, text, and fusion
blocks, including the frozen-embedding, input-parity, and prompted-LLM arms
and the contamination controls, **together with the 240 `config.json`
preimages themselves** (`run_configs/`), so the fingerprint check is
executable rather than merely asserted: `config_fingerprints.py
--verify-preimages` re-hashes every released preimage and matches it against
the released index, using only files inside the package. That index carries
*two* digests per run: 38 of the 240 configs name an absolute path on the
authors' machine (a local model checkout, a data root) that had to be
neutralised before release, so 38 released preimages differ from their original
bytes; each row's `sanitised_fields` records exactly which fields moved. Rather than quietly reconcile the two, each row publishes the original
fingerprint, the fingerprint of the file we actually ship, and the exact
dotted key paths that were rewritten; the remaining 202 preimages hash
identically to their original fingerprint. 233 digests are distinct; 7 pairs
collide, each pair being one cross-family probe's `_full_combined_` and
`_full_event_driven_` run (Mistral-24B, Phi-4-14B, Llama-3.1-70B and their
seed replicates). The collision is exact and expected: those two runs *do*
share a byte-identical configuration, because the combined/event-driven
distinction is a downstream row-subset selection over one generation pass
rather than a configuration field. The fingerprint therefore identifies the
configuration, not the run; and (v) a datasheet. We also ship the
**test suite** (33 test modules, 186 tests, all passing), which includes the
byte-identity checks that the pre-tokenisation caches (both the flat and the
chunked long-document variant) reproduce their uncached reference exactly. Disclosure text is fetched from EDGAR by accession,
so no SEC content needs redistributing.

We also release the **aggregate evidence tables** behind every claim:
day-clustered Diebold--Mariano and forecast-encompassing statistics with
Holm-corrected p-values and placebo outcomes; the seed-ensemble primary 69-cell
grid; the firm-identity specification battery; the maximal-pool robustness and
control-intersection tables; and the contamination, cross-family, and
elicitation-sensitivity tables. These are summary statistics with no per-row
CRSP-derived value, so a reader can audit every reported test against them.
Because the prompted-LLM pipeline consumes only SEC filing text, its
**prompt templates, decoding configurations** (temperature 0, JSON
output), **and complete raw generations** are releasable in full and **ship
inside this package**: `raw_generations/` holds 608,221 generations across 1,221
shards: the C6 primary, the D4 fused arm, the paraphrase elicitation repeats,
every cross-family probe with its seed replicates, and the date+ticker
contamination probe for the 70B rider, each row carrying the model's verbatim
output string alongside the parsed horizons. The shipped `README.md` gives the
exact variant x model inventory, including the two arms whose per-row
generations did not survive the compute node's decommissioning (their
aggregate evidence ships in full) and the eight rows withheld because the model
emitted non-English text inside its own reasoning. No CRSP content enters any prompt or output; the single path column
is rewritten data-root-relative and is the join key into `accession_index.csv`.

The realised-volatility labels, RV features, and model predictions are derived
from CRSP/WRDS data, whose licence prohibits redistribution, and are therefore
**withheld**; we release the full **regeneration pipeline**
(ingestion, point-in-time alignment, labelling, training, and evaluation
scripts, with pinned configurations) so any holder of a CRSP/WRDS subscription
can reconstruct them exactly. Per-row loss-differential series would be
releasable only in a non-invertible form (loss differentials alone, without
predictions or labels) and only with written confirmation from the
institutional CRSP licensing administrator; absent that, the aggregate
statistics serve as the reproducibility artefact, which already suffices to
audit every claim. We follow CRSP and WRDS attribution requirements throughout
the released materials.

## Compute

All production training and inference ran on a single rented Linux node
(4× NVIDIA A100 40GB; kernel 6.8.0-90-generic). Two pre-production exceptions
are on the record: batch-size feasibility probes ran on consumer RTX hardware
(the probe tables ship in `aggregate_results/`), and the elicitation-fairness pilot (P6) ran later on a single A100-80GB card, which is
why it runs at TP=1 against the TP=2 committed anchor, an external hardware
constraint, recorded in the P6 amendment before the pilot rather than a
results-driven change. Interpreter versions across the 203 runs whose
metadata records one: Python 3.11.15 (175 runs), 3.12.3 (27), 3.11.8 (1).
One environment freeze survives, `ENVIRONMENT_inference.txt` (224
packages), and we label its scope exactly: it pins the *prompted-LLM
inference* environment (offline vLLM) as captured on the node's final day,
not the stack that trained the neural text arms, which predates our
environment-logging regime (Limitations); `pyproject.toml` gives those arms'
declared constraints rather than a resolved pin. Prompted-LLM
inference (C6, D4, cross-family probes) used offline vLLM on the same node,
with pilot budgets on the order of a few GPU-hours per arm as
recorded in the analysis records (P5 §10, P6). Classical arms
(A-block, B-block) and every audit-stage recomputation (DM/Holm families,
combination weights, placebos, power calibration) are CPU-only; local analysis
respected a ≤50%-of-cores convention. The frozen specifications behind the audit arms are pinned by git tag in the
project's version-controlled archive, released with the public repository. That is an integrity claim over the authors' own history,
not an independent third-party timestamp.
