# INTACT Training Protocol

The repository exposes pure LeWM, inverse-only, goal-only, waypoint, and
goal-displacement objectives for both single-task and shared-encoder training.
See [`TRAIN_EVAL_MATRIX_CN.md`](TRAIN_EVAL_MATRIX_CN.md) for the exact loss and
evaluation matrix. The equations below describe the full INTACT variants.

## Shared action law

The same `IntentActionActor` is called for both supported condition families:

```text
physical:   (z[t], z[t+1] - z[t], a[t-1]) -> a[t]
deployment: (z[t], stopgrad(z[7]) - z[t], a[t-1]) -> a[t]
```

Both calls use `[z, m, z * m, A(a_prev)]` and a proper diagonal-Gaussian NLL.
They are separate likelihood terms with shared parameters, not a stochastic
route and not an MSE between intent vectors.

## Window indexing

For eight encoded frames and seven aligned action chunks:

| Term | Current indices | Endpoint | Target actions |
|---|---|---|---|
| Forward | `0..6` | next frame | `0..6` |
| Physical NLL | `0..6` | attached `z[t+1]` | `0..6` |
| Goal NLL | `0..6` | detached `z[7]` | `0..6` |

Physical and Goal supervision both start at index `0` and use the same
boundary-aware previous-action contract. At a true episode boundary, missing
history is raw zero before z-scoring; at an interior clip boundary, both calls
receive the real preceding demonstrated chunk.

## Fixed paper settings

| Setting | Single-task | Multi-task |
|---|---:|---:|
| Epochs | 1 | 5 |
| Batch | 256 | 256 per task/rank |
| Learning rate | `5e-4` | `5e-4` |
| LR schedule | 1% warmup + cosine to zero | constant |
| Weight decay | `1e-3` | `1e-3` |
| Forward weight | 1.0 | 1.0 |
| Physical weight | 0.1 | 0.1 |
| Goal weight | 0.05 | 0.05 |
| SIGReg | 0.02 | 0.03 |
| SIGReg projections | 1024 | 1024 |
| Attention backend | Math-SDPA | Math-SDPA |
| Training seeds | 0/42/3072 | 0/42/3072 |
| Evaluation seeds | 0/1/42 | 0/1/42 |

"Full data" means `data_fraction=1.0` before the canonical LeWM 90/10
train-validation split. It does not mean training on the validation partition.

## Multi-task ownership

The four ranks own PushT, Cube, Reacher, and TwoRoom respectively. Only the
ViT-Tiny/14 encoder and two-layer projector are shared. Their gradients are
averaged and their buffers synchronized at every optimizer step. Predictor,
action encoder, prediction projector, and action operator are task-specific,
which preserves each task's native action coordinates.

The epoch length is the maximum of the four task loaders. A shorter loader is
cycled so that every synchronized update has one batch from every task. Each
epoch records the per-task loader sizes and the shared-state SHA256.
