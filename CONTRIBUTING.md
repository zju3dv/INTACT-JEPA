# Contributing

INTACT is a public research project. Contributions are welcome through focused
issues and pull requests.

## Before Opening a Pull Request

1. Open or reference an issue that states the scientific or engineering goal.
2. Keep generated results, source changes, and manuscript edits in separate
   commits when practical.
3. Add or update tests for behavior-changing code.
4. Record seeds, checkpoint hashes, evaluator version, and planner budgets for
   every experimental claim.
5. Run the repository hygiene checks and inspect the diff for internal paths,
   hostnames, tokens, or private data references.
6. For code changes, also run `ruff check .`, `bash -n scripts/*.sh`, and the
   smallest relevant smoke or configuration check described in the README.

## Experimental Claims

An aggregate success rate is not sufficient on its own. A result contribution
should include a machine-readable manifest, per-run provenance, aggregation
command, and a short statement distinguishing planned, pilot, and controlled
evidence.

Use the experiment-report issue template for new evaluation cells. Do not edit
headline README values before the corresponding manifest has been reviewed.

## Style

- Prefer small, reviewable changes.
- Keep research names and inference modes consistent with `docs/METHOD.md`.
- Treat `Direct`, `Guarded A`, and actor-disabled `Pure CEM` as distinct
  interfaces.
- Never pool official LeWM and CLEAR-LeWM scores.
- Do not commit datasets, local caches, logs, or untracked checkpoints.

## Review

At least one project maintainer should review changes to objectives, evaluation
rules, headline tables, release metadata, or third-party attribution.
