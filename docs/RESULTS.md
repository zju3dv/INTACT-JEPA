# Audited Results

This document records the headline values currently used by the INTACT project
page. It is a summary, not a substitute for the complete paper tables or raw
evaluation manifests.

## Task-Specific Models

Protocol: full task data, one end-to-end epoch, three training seeds per task,
three evaluation seeds per checkpoint, and 100 episodes per evaluation seed.
Success rates are percentages on the official LeWM protocol.

| Training coordinate | Inference | Sequences | PushT | Cube | Reacher | TwoRoom | Macro |
|---|---|---:|---:|---:|---:|---:|---:|
| Waypoint INTACT | Direct | 0 | 77.67 +/- 0.88 | 99.89 +/- 0.19 | 88.11 +/- 0.69 | 98.00 +/- 0.33 | 90.92 +/- 0.30 |
| Goal-displacement INTACT | **Direct** | **0** | 85.78 +/- 1.54 | 100.00 +/- 0.00 | 97.67 +/- 0.00 | 97.89 +/- 1.26 | **95.33 +/- 0.58** |
| Goal-displacement INTACT | Pure CEM 300x30 | 9,000 | 88.44 +/- 1.17 | 68.44 +/- 0.77 | 83.67 +/- 0.67 | 82.89 +/- 0.84 | 80.86 +/- 0.51 |
| Goal-displacement INTACT | Actor-on CEM 300x30 | 9,000 | 93.56 +/- 0.96 | 96.89 +/- 0.19 | 86.67 +/- 0.88 | 98.00 +/- 1.15 | 93.78 +/- 0.77 |
| Goal-displacement INTACT | **Guarded A 128x3** | **384** | 92.22 +/- 0.69 | 99.78 +/- 0.19 | 97.44 +/- 0.77 | 98.00 +/- 1.15 | **96.86 +/- 0.38** |

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
| Goal intent only | Direct | 70.67 +/- 1.76 | 100.00 +/- 0.00 | 82.33 +/- 3.18 | 69.44 +/- 6.75 | 80.61 +/- 2.11 |
| Goal-displacement INTACT | **Direct** | 80.22 +/- 1.26 | 99.56 +/- 0.77 | 95.67 +/- 1.76 | 82.11 +/- 4.11 | **89.39 +/- 0.77** |
| Goal-displacement INTACT | Actor-disabled CEM | 78.11 +/- 1.17 | 70.33 +/- 3.00 | 80.33 +/- 2.03 | 51.56 +/- 4.53 | 70.08 +/- 1.13 |
| Goal-displacement INTACT | Guarded A 128x3 | 85.00 +/- 2.85 | 99.00 +/- 0.00 | 97.89 +/- 0.69 | 80.00 +/- 4.41 | 90.47 +/- 0.84 |

Adding the attached physical call to matched goal-intent-only training improves
the Direct macro by 8.78 points. The actor-disabled comparison improves LeWM by
3.91 points overall, showing that representation and direct readout effects are
related but distinct.

## Theory-Linked Diagnostics

Across 45 eligible E1-E5 shared-encoder checkpoints:

| Diagnostic | Pearson correlation with Direct SR | Intended role |
|---|---:|---|
| Predicted-expert action-family kNN overlap | **0.954** | local family-neighborhood agreement |
| Predicted-expert linear CKA | **0.897** | global centered family geometry |
| Pointwise action R2 | 0.815 | single-action recoverability |

Effective rank is reported as a capacity diagnostic, not a quality score. In a
controlled E5 comparison, rank 93.87 accompanies 74.22% SR while rank 89.26
accompanies 89.39% SR. Higher latent spread alone is therefore not a semantic
certificate.

## Scope

- Published external rows retain their original training and evaluation
  protocols and are landscape context, not paired controls.
- Official LeWM SR and CLEAR-LeWM SR answer different evaluation questions and
  are never pooled.
- `+/-` denotes sample standard deviation over training seeds after averaging
  each checkpoint's evaluation seeds.
- Candidate counts are sequences per planning solve, not environment steps.
