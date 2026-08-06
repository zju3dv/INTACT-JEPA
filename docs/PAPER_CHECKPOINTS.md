# Paper Checkpoint Matrix

## Scope

The public [`paper-e5-goal-v1`](https://huggingface.co/INTACT-JEPA/INTACT/tree/paper-e5-goal-v1)
model revision contains all six controlled Math-SDPA shared-encoder E5 cells from the
main paper table. Every cell includes training seeds `0`, `42`, and `3072`;
every seed has one shared encoder and four task-specific Forward/action-head
shards. The complete bundle therefore contains `6 x 3 x 4 = 72` checkpoints.

| Cell ID | Paper name | Native interface | PushT | Cube | Reacher | TwoRoom | Macro |
|---|---|---|---:|---:|---:|---:|---:|
| `lewm` | LeWM | CEM 300x30 | 74.56 | 67.33 | 83.11 | 39.67 | 66.17 |
| `inverse_only` | Inverse only | Direct | 35.00 | 61.89 | 61.00 | 68.78 | 56.67 |
| `waypoint_intent` | Waypoint intent only | Direct | 58.89 | 100.00 | 65.89 | 72.11 | 74.22 |
| `goal_intent` | Goal intent only | Direct | 70.67 | 100.00 | 82.33 | 69.44 | 80.61 |
| `waypoint_intact` | Waypoint INTACT | Direct | 71.22 | 99.00 | 58.22 | 77.22 | 76.42 |
| `goal_intact` | Goal-displacement INTACT | Direct | **80.22** | **99.56** | **95.67** | **82.11** | **89.39** |

Values are official SR percentages averaged over three training seeds. Full
sample standard deviations and every asset/shard hash are recorded in
[`PAPER_E5_MATRIX_MANIFEST.json`](../checkpoints/PAPER_E5_MATRIX_MANIFEST.json).

### Headline cell by training seed

| Train seed | PushT | Cube | Reacher | TwoRoom | Macro |
|---:|---:|---:|---:|---:|---:|
| 0 | 79.33 | 100.00 | 93.67 | 86.67 | 89.92 |
| 42 | 81.67 | 100.00 | 96.33 | 81.00 | 89.75 |
| 3072 | 79.67 | 98.67 | 97.00 | 78.67 | 88.50 |
| **Mean +/- sample std** | **80.22 +/- 1.26** | **99.56 +/- 0.77** | **95.67 +/- 1.76** | **82.11 +/- 4.11** | **89.39 +/- 0.77** |

These are Goal-displacement INTACT official Direct SR values. Each training-seed cell averages 100
episodes for evaluation seeds `0`, `1`, and `42`. CLEAR-LeWM scores are a
separate audit and are not mixed into this table.

## Download

Download the required assets from the pinned
[`paper-e5-goal-v1`](https://huggingface.co/INTACT-JEPA/INTACT/tree/paper-e5-goal-v1)
revision. Verify every downloaded archive and `.pt` shard against the SHA-256
values in the machine-readable source of truth:
[`checkpoints/PAPER_E5_MATRIX_MANIFEST.json`](../checkpoints/PAPER_E5_MATRIX_MANIFEST.json).
The smaller headline manifest is
[`checkpoints/PAPER_E5_GOAL_MANIFEST.json`](../checkpoints/PAPER_E5_GOAL_MANIFEST.json).

## Evaluate

The current repository-root launchers evaluate checkpoints trained with the
corrected 7/7 previous-action contract. They must not be used for this frozen
paper matrix. This merge retains `paper_runtime/` for checkpoint compatibility
but intentionally removes the legacy paper download and evaluation wrappers.
Reproducing a published matrix cell therefore requires an adapter that loads
the frozen runtime and exactly matches the evaluator fingerprint below.

## Compatibility Boundary

Do not load these weights through the clean runtime at repository root. The
paper matrix uses the legacy five-slot Actor checkpoint grammar, while the
clean runtime removes its constant-zero relation slot and uses the four-slot
Fig. 1 grammar. Both train local and goal likelihoods together, but their Actor
parameter shapes remain incompatible.
The exact audited runtime is pinned under [`paper_runtime/`](../paper_runtime/README.md).
Keep it byte-for-byte unchanged when reproducing reported numbers.

The recorded evaluator fingerprint is
`1e475a069338bdad6c9e6a38b8ea4e2d5557ecc4cff88844e5efee79121a9fc7`.
