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

The evaluator files listed in `RUNTIME_SHA256SUMS` exactly match the fingerprint
recorded by the controlled paper evaluation. Do not edit them when reproducing
reported numbers. The current repository intentionally omits the legacy paper
launchers; this directory is retained as read-only compatibility code for an
external adapter that reproduces the recorded evaluator contract.
