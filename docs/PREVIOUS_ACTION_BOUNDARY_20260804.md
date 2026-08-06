# Previous-Action Boundary Correction (2026-08-04)

## Frozen baseline

The superseded paper-reproduction implementation is preserved at commit
`494dc282700134f354a6870bf5ba3c9800d71001` and tag
`paper-reproduction-20260804`. It used normalized zero as the first
previous-action token of every sampled clip. That token denotes the dataset mean
action, not a raw neutral command. The current `junhan` branch contains the
corrected contract below.

## Corrected contract

No source dataset is rewritten. A boundary-aware wrapper adds an aligned
`previous_action` tensor to each clip:

- An interior clip loads the `frameskip` real primitive actions immediately before
  the clip start.
- A clip beginning within the first `frameskip` steps of an episode left-pads only
  its unavailable history with raw zero commands.
- Raw zero padding is transformed by action statistics fitted exclusively on real,
  finite demonstration actions.
- Every later previous chunk is exactly the preceding normalized target chunk.

The collector rotates its reset placeholder to the final row of each episode.
That terminal action belongs to the eighth, post-goal chunk and is not one of the
seven supervised transitions. The action transform maps this missing raw value
to zero before z-scoring so loaded tensors remain finite; the normalization
statistics still use only real expert actions.

For action mean `mu` and standard deviation `sigma`, one neutral primitive token is
`(0 - mu) / sigma`. With `frameskip=5`, the initial previous chunk contains five
copies of that per-dimension vector.

## Evaluation contract

At environment reset, stable-worldmodel represents the unavailable previous action
as `NaN`. The corrected evaluator replaces this value with raw zero before applying
the training-compatible (`ddof=1`) action z-score. After control begins, the policy
uses actions actually executed by the controller.

The model and training path reject any non-finite value that remains after this
explicit raw-coordinate boundary handling.

## Compatibility

This correction changes the training distribution and initial Direct inference
condition. Existing paper checkpoints remain reproducible only with the frozen
baseline. Corrected checkpoints must be retrained and evaluated as a separate
protocol; results must not be mixed across the two contracts.
