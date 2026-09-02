# External pre-registration timestamp — HPO challenger arm

**Purpose.** A git tag proves internal consistency, not temporal priority: the author holds
both the clock and the prior, and a tag can be force-pushed. A reviewer said so explicitly
("hashed configs prove internal consistency, not temporal priority"). This directory carries
an *external*, author-independent timestamp instead.

## What is stamped

`hpo_prereg_v1.1.tar` bundles the complete pre-registration as of the stamp:

| File | Role |
|---|---|
| `configs/hpo_arm.yaml` | v1.0 space/rules + **Amendment 1** (2026-07-14) |
| `results/HPO_ARM_SPEC.md` | design rationale, backfire plan, degradation ladder |
| `scripts/experiments/hpo/asha_hpo.py` | the harness (test firewall, ASHA, selection) |
| `scripts/experiments/hpo/qlike_loss.py` | the QLIKE-aligned objective |

SHA-256: `2c9a1c9b0b774ef1f330725a39049a57da14e4e7f9a4423e1335fa5f022f5b59`
(also in `hpo_prereg_v1.1.sha256`)

## The timestamp

`hpo_prereg_v1.1.tar.ots` is an [OpenTimestamps](https://opentimestamps.org) proof,
submitted 2026-07-14 to three independent calendars (`b.pool.opentimestamps.org`,
`a.pool.eternitywall.com`, `ots.btc.catallaxy.com`) and anchored in the Bitcoin
blockchain. It is verifiable by anyone, needs no account, and reveals no identity —
so it can be checked without disclosing who produced it.

Verify with:

    ots verify results/prereg/hpo_prereg_v1.1.tar.ots

(Bitcoin attestation lands within ~a few hours of submission; `ots upgrade` fetches it.
Until then the calendar attestations already establish the submission time.)

## What this does and does not prove

**Proves:** the bundled space, selection rules, gates, Holm families, and harness existed
in exactly this form at the stamped time — which is **before any test row was ever scored**
(at stamp time the search stage was ~40% complete: T1a and T1c selected on val-select only;
T4 and the T3 family in flight; zero test evaluations run).

**Does not prove:** that no earlier, unstamped version existed. The honest claim is
temporal priority over the *test evaluation*, not over the authors' own thinking. The
pilot arm's prior test exposure is disclosed separately in `hpo_arm.yaml:pilot_disclosure`.

## Amendment discipline

Amendments are appended to `hpo_arm.yaml:amendments` with a date, the defect they remedy,
and an explicit `before_any_test_evaluation` flag — never by silent edit.

**The stamped copies and the working copies diverge, and here is exactly how.**
Both `configs/hpo_arm.yaml` and `results/HPO_ARM_SPEC.md` were written partly in
Chinese. The dissertation's examiners do not read Chinese, so on 2026-09-01 the
working copies were translated into English. **The stamped archive was not
touched**: `shasum -a 256 -c hpo_prereg_v1.1.sha256` still returns OK, and the
OpenTimestamps proof still commits to that digest, so nothing that was frozen
has moved.

What this means for a reader:

- The archive holds the registration **as it was frozen**, in the language it
  was written in. It is the authority, and it is the copy to read if the
  question is what was registered.
- The working copies hold the **same registration in English**, with one
  further change made on 2026-09-02 and recorded here: two sentences in
  `HPO_ARM_SPEC.md` that referred to the companion manuscript's review
  process were removed, one parenthetical about the OSF registration's
  anonymity and one closing remark on how a branch would read to that
  manuscript's reviewers. Neither touched a registered field, threshold,
  gate, branch, consequence or number; the (C) branch's registered
  consequence is exactly as it was. `configs/hpo_arm.yaml` parses to the same YAML
  in both copies except for the prose field `pilot_disclosure`; every design
  field is identical.
- So `diff` between the two will be large for `HPO_ARM_SPEC.md` and four lines
  for `hpo_arm.yaml`, and all of it is language. Each translation was checked
  line by line against the original by a reader who had not produced it, with
  the numbers compared as multisets.

The tagged Chinese originals also remain in the working repository, pinned by
the pre-registration tags listed in `release/tag_manifest.txt`. Amendment 1 adds
D3 and the C5 head family, which v1.0 omitted from *both* `tasks` and `exclusions`; that
omission mattered because D3/gte-Qwen2 is the strongest archived long-form text challenger,
so v1.0 would have tuned only the weak challengers. Each amendment gets its own stamp.
