# Frozen Paper Evaluation Runtime

This directory is the minimal compatibility layer for the six-cell checkpoint
matrix published under the `paper-e5-goal-v1` release. It is intentionally
separate from the current training and evaluation runtime at repository root.

The controlled matrix includes the legacy waypoint actor grammar and the later
`InverseTransitionActor(feature_layout="delta_condition_product")` goal
grammar. The current root runtime uses the zero-free four-slot
`IntentActionActor` and paired local/goal likelihoods with a different
parameter layout, so loading paper checkpoints through the root runtime is not
supported.

The five evaluator files listed in `RUNTIME_SHA256SUMS` exactly match the
fingerprint recorded by the controlled paper evaluation. Do not edit them when
reproducing reported numbers. Use `scripts/eval_paper_matrix.sh` or the
headline-only `scripts/eval_paper_direct.sh` rather than invoking this directory
manually.
