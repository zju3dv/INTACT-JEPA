# Audited Results

This document records the headline values currently used by the INTACT project
page. It is a summary, not a substitute for the complete paper tables or raw
evaluation manifests.

## History-Free E1 Ablation

The released history-free actor removes previous action from both training and
evaluation. With full task data, one epoch, training seeds `0`, `42`, and
`3072`, evaluation seeds `0`, `1`, and `42`, and 100 episodes per cell, it
reaches the following Official Direct results:

| Released interface | PushT | Cube | Reacher | TwoRoom | Macro |
|---|---:|---:|---:|---:|---:|
| No previous action | 85.33 +/- 0.58 | 99.44 +/- 0.38 | 93.44 +/- 0.38 | 98.78 +/- 0.69 | **94.25 +/- 0.08** |

The checkpoint manifests and complete evaluation rows are available in
[`INTACT-no-previous-action`](https://huggingface.co/INTACT-JEPA/INTACT/tree/main/INTACT-no-previous-action).
This is a separate ablation cohort; the headline task-specific INTACT results
are reported below.

## Task-Specific Models

Protocol: full task data, one end-to-end epoch, training seeds `3072`, `3073`,
and `3074` per task, evaluation seeds `0`, `1`, and `42` per checkpoint, and
100 episodes per evaluation seed.
Success rates are percentages on the official LeWM protocol.
The task-specific Guarded A row uses `H=5`, `RH=5`, 128 samples x 3 rounds,
raw-action `sigma=0.25`, top-k 16, and the same causal action history as Direct.

| Training coordinate | Inference | Sequences | PushT | Cube | Reacher | TwoRoom | Macro |
|---|---|---:|---:|---:|---:|---:|---:|
| Waypoint INTACT | Direct | 0 | 77.67 +/- 0.88 | 99.89 +/- 0.19 | 88.11 +/- 0.69 | 98.00 +/- 0.33 | 90.92 +/- 0.30 |
| Goal-displacement INTACT | **Direct** | **0** | 87.44 +/- 1.26 | 100.00 +/- 0.00 | 97.33 +/- 0.00 | 97.67 +/- 1.20 | **95.61 +/- 0.59** |
| Goal-displacement INTACT | Pure CEM 300x30 | 9,000 | 88.44 +/- 1.17 | 68.44 +/- 0.77 | 83.67 +/- 0.67 | 82.89 +/- 0.84 | 80.86 +/- 0.51 |
| Goal-displacement INTACT | Actor-on CEM 300x30 | 9,000 | 93.89 +/- 1.58 | 97.67 +/- 0.67 | 88.89 +/- 1.71 | 98.00 +/- 1.45 | 94.61 +/- 0.82 |
| Goal-displacement INTACT | **Guarded A 128x3** | **384** | 91.56 +/- 0.69 | 99.67 +/- 0.33 | 97.56 +/- 0.51 | 97.56 +/- 1.26 | **96.58 +/- 0.44** |

Interpretation:

1. Goal displacement is the strongest final Direct coordinate.
2. Broad actor-on CEM spends 9,000 sequences yet trails zero-search Direct.
3. Small local verification improves Direct while preserving the learned plan.
4. Pure CEM is intentionally retained as an actor-disabled representation
   control, not as INTACT's native deployment interface.

## Shared Encoder

Protocol: one visual encoder is jointly updated by four task processes;
forward and action heads remain task-specific. Values are means and sample
standard deviations over three training seeds, each evaluated on seeds
`0`, `1`, and `42` with 100 episodes.

| Training cell | Evaluation | PushT | Cube | Reacher | TwoRoom | Macro |
|---|---|---:|---:|---:|---:|---:|
| LeWM | CEM 300x30 | 74.56 +/- 3.67 | 67.33 +/- 1.86 | 83.11 +/- 0.96 | 39.67 +/- 8.39 | 66.17 +/- 2.67 |
| Goal intent only | Direct | 81.78 +/- 0.96 | 100.00 +/- 0.00 | 88.67 +/- 0.33 | 69.22 +/- 7.04 | 84.92 +/- 1.86 |
| Goal-displacement INTACT | **Direct** | 86.11 +/- 0.96 | 100.00 +/- 0.00 | 97.22 +/- 0.84 | 81.56 +/- 3.67 | **91.22 +/- 0.51** |
| Goal-displacement INTACT | Actor-disabled CEM | 78.11 +/- 1.17 | 70.33 +/- 3.00 | 80.33 +/- 2.03 | 51.56 +/- 4.53 | 70.08 +/- 1.13 |
| Goal-displacement INTACT | Guarded A 128x3 | 85.44 +/- 2.78 | 99.22 +/- 0.38 | 97.00 +/- 0.88 | 80.44 +/- 3.36 | 90.53 +/- 0.32 |

The shared-encoder Guarded A row is a separate E5 cohort; its 90.53% macro is
not the task-specific one-epoch Guarded A result of 96.58% above.
Its training seeds are `0`, `42`, and `3072`. The complete 72-cell Guarded A
identity, including per-seed SR and checkpoint SHA-256 values, is frozen in
[`results/guarded_a_official.json`](../results/guarded_a_official.json).

Adding the attached physical call to matched goal-intent-only training improves
the Direct macro by 6.31 points. The actor-disabled comparison improves LeWM by
3.91 points overall, showing that representation and direct readout effects are
related but distinct.

## Theory-Linked Diagnostics

Across 15 eligible goal-displacement E1-E5 shared-encoder checkpoints:

| Diagnostic | Pearson correlation with Direct SR | Intended role |
|---|---:|---|
| Predicted-expert action-family kNN overlap | **0.968** | local family-neighborhood agreement |
| Predicted-expert linear CKA | **0.988** | global centered family geometry |
| Pointwise action R2 | **0.983** | single-action recoverability |

Effective rank is reported as a capacity diagnostic, not a quality score. In a
controlled E5 comparison, rank 93.87 accompanies 74.22% SR while rank 89.26
accompanies 91.22% SR. Higher latent spread alone is therefore not a semantic
certificate.

## Scope

- Published external rows retain their original training and evaluation
  protocols and are landscape context, not paired controls.
- Official LeWM SR and CLEAR-LeWM SR answer different evaluation questions and
  are never pooled.
- `+/-` denotes sample standard deviation over training seeds after averaging
  each checkpoint's evaluation seeds.
- Candidate counts are sequences per planning solve, not environment steps.
