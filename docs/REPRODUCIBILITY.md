# Reproducibility Contract

INTACT results are released only when the model, data, planner, and evaluator
identities are all explicit. This document is the contract that released code,
configurations, checkpoints, and result manifests must satisfy.

## Required Identity Fields

Every reported cell must record:

- task and dataset identifier;
- dataset revision, split, and episode-subset hash;
- training objective and all loss weights;
- model architecture and parameter-sharing pattern;
- optimizer, learning rate, batch size, precision, and epoch;
- training seed and checkpoint SHA256;
- inference interface and planner hyperparameters;
- evaluator version, evaluation seed, and episode count;
- software revision and environment lockfile hash.

## Headline Protocol

| Field | Task-specific result | Shared-encoder E5 result |
|---|---|---|
| Domains | PushT, Cube, Reacher, TwoRoom | same four domains |
| Data | full official task data | full official task data |
| Encoder | one per task | one shared encoder |
| Control heads | task-specific | task-specific |
| Training seeds | 3 per task | 3 joint runs |
| Evaluation seeds | `0`, `1`, `42` | `0`, `1`, `42` |
| Episodes | 100 per checkpoint/seed | 100 per checkpoint/seed |
| Primary interface | Direct | Direct |
| Verification | Guarded A 128x3 | Guarded A 128x3 |
| Representation control | actor-disabled CEM | actor-disabled CEM |

## Inference Budgets

- **Direct:** zero candidate sequences and no terminal latent-cost call.
- **Guarded A:** 128 samples for 3 CEM iterations, 384 candidate sequences per
  planning solve, centered on the Direct plan.
- **Matched broad CEM:** 300 samples for 30 iterations, 9,000 candidate
  sequences per planning solve.

Planner-side latency excludes environment rendering, observation transport,
and unrelated simulator overhead. It must not be presented as end-to-end robot
latency.

## Statistical Reporting

1. Average evaluation seeds within each independently trained checkpoint.
2. Report the mean and sample standard deviation across training seeds.
3. Keep task-specific and shared-encoder cohorts separate.
4. Keep official and CLEAR-LeWM evaluators separate.
5. Mark unavailable or intentionally stopped cells with `--`, never `0`.
6. Label post-selection studies and historical pilots; do not mix them into
   controlled objective rankings.

## Release Gates

Before a configuration or checkpoint is published, verify:

- [ ] exact objective and gradient routing match the paper definition;
- [ ] training/evaluation seeds and episode counts match the manifest;
- [ ] model hash matches every referenced result row;
- [ ] no internal path, host, token, or private dataset URI is present;
- [ ] third-party source and dataset licenses permit redistribution;
- [ ] commands run from a clean environment using documented dependencies;
- [ ] aggregate tables regenerate from machine-readable per-episode results;

## Public Implementation Artifacts

The public source release contains:

```text
config/           exact task-specific and shared-encoder recipes
*.py              model, losses, Direct controller, and local verifier
scripts/          train, evaluate, aggregate, and audit entry points
checkpoints/      paper checkpoint manifests (weights are separate assets)
paper_runtime/    frozen compatibility runtime for paper checkpoints
requirements-*.lock
                  resolved environment lock
```

Model weights are publicly hosted at
[`INTACT-JEPA/INTACT`](https://huggingface.co/INTACT-JEPA/INTACT/tree/paper-e5-goal-v1).
The downloader pins that revision and verifies both the archive checksums and
the extracted checkpoint hashes. The status table in the root README is
authoritative.
