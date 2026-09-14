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
| `inverse_only` | Inverse only | Direct | 36.11 | 67.56 | 90.56 | 78.56 | 68.19 |
| `waypoint_intent` | Waypoint intent only | Direct | 58.89 | 100.00 | 65.89 | 72.11 | 74.22 |
| `goal_intent` | Goal intent only | Direct | 81.78 | 100.00 | 88.67 | 69.22 | 84.92 |
| `waypoint_intact` | Waypoint INTACT | Direct | 71.22 | 99.00 | 58.22 | 77.22 | 76.42 |
| `goal_intact` | Goal-displacement INTACT | Direct | **86.11** | **100.00** | **97.22** | **81.56** | **91.22** |

Values are official SR percentages averaged over three training seeds. Full
sample standard deviations and every asset/shard hash are recorded in
[`PAPER_E5_MATRIX_MANIFEST.json`](../checkpoints/PAPER_E5_MATRIX_MANIFEST.json).

### Headline cell by training seed

| Train seed | PushT | Cube | Reacher | TwoRoom | Macro |
|---:|---:|---:|---:|---:|---:|
| 0 | 85.00 | 100.00 | 96.33 | 85.33 | 91.67 |
| 42 | 86.67 | 100.00 | 97.33 | 81.33 | 91.33 |
| 3072 | 86.67 | 100.00 | 98.00 | 78.00 | 90.67 |
| **Mean +/- sample std** | **86.11 +/- 0.96** | **100.00 +/- 0.00** | **97.22 +/- 0.84** | **81.56 +/- 3.67** | **91.22 +/- 0.51** |

These are Goal-displacement INTACT official Direct SR values. Each training-seed cell averages 100
episodes for evaluation seeds `0`, `1`, and `42`. CLEAR-LeWM scores are a
separate audit and are not mixed into this table.

## Download

Download and verify one headline training seed without authentication:

```bash
bash scripts/download_paper_checkpoints.sh 0
```

This backward-compatible form downloads the `goal_intact` cell. Select another
cell, all seeds, or all 72 checkpoints with:

```bash
bash scripts/download_paper_checkpoints.sh inverse_only 42
bash scripts/download_paper_checkpoints.sh waypoint_intact all
bash scripts/download_paper_checkpoints.sh matrix all /path/to/stable-wm-cache
```

Every archive and every `.pt` shard is checked against SHA256 before use. The
machine-readable source of truth is
[`checkpoints/PAPER_E5_MATRIX_MANIFEST.json`](../checkpoints/PAPER_E5_MATRIX_MANIFEST.json).
The default host is `INTACT-JEPA/INTACT`; mirrors can be selected with
`INTACT_HF_REPO`, `INTACT_CHECKPOINT_REVISION`, or
`INTACT_CHECKPOINT_BASE_URL`. Downloads use resumable `curl` by default; set
`INTACT_USE_ARIA2=1` only on a host where Hugging Face redirects have been
validated with `aria2c`.

## Evaluate

```bash
bash scripts/eval_paper_direct.sh pusht 0 0 100
bash scripts/eval_paper_direct.sh cube 42 1 100
bash scripts/eval_paper_matrix.sh waypoint_intent pusht 0 0 100
bash scripts/eval_paper_matrix.sh lewm reacher 3072 42 100
```

The matrix wrapper takes `CELL TASK TRAIN_SEED EVAL_SEED NUM_EVAL`. It
automatically uses CEM 300x30 for LeWM and zero-search Direct for the other
five cells.

## Compatibility Boundary

Do not load these weights through the clean runtime at repository root. The
paper matrix uses the legacy five-slot Actor checkpoint grammar, while the
clean runtime removes its constant-zero relation slot and uses the four-slot
Fig. 1 grammar. Both train local and goal likelihoods together, but their Actor
parameter shapes remain incompatible.
The exact audited runtime is pinned under [`paper_runtime/`](../paper_runtime/README.md).
The evaluation wrapper prepends that runtime to `PYTHONPATH`, enables the
recorded deterministic Math-SDPA policy, and then calls stable-worldmodel.

The current result revision uses `official_expert_continuation_v1`: a sampled
continuation receives its causal expert action prefix, while only a true
episode boundary is left-padded with normalized raw-zero actions.
