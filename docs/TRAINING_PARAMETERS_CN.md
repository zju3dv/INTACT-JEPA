# INTACT 规范训练参数手册

本文档用于组员逐项校对当前 `junhan` 分支。若实验命令没有显式覆盖参数，应以这里列出的当前代码默认值为准。

五种 objective 及其完整评估组合以
[TRAIN_EVAL_MATRIX_CN.md](TRAIN_EVAL_MATRIX_CN.md) 为准。本文件进一步记录主方法
`displacement` 的底层参数。

## 1. 单任务与多任务总表

| 参数 | 单任务 | 四任务联合训练 |
|---|---:|---:|
| 入口 | `train.py` / `scripts/train_single.sh` | `train_multitask.py` / `scripts/train_multitask.sh` |
| 任务数 | 每次 1 个 | 4 个 rank，每 rank 1 个任务 |
| epochs | `1` | `5` |
| batch size | `256` | 每任务/rank `256`，同步 step 总 batch 为 `1024` |
| 训练数据 | 官方 full data 后做 90/10 split | 每任务官方 full data 后做 90/10 split，仅使用 train split |
| optimizer | AdamW | fused AdamW |
| learning rate | `5e-4` | `5e-4` |
| weight decay | `1e-3` | `1e-3` |
| scheduler | 1% linear warmup + cosine decay | **无 scheduler，固定 lr** |
| precision | Lightning `bf16-mixed` | `torch.autocast(..., bfloat16)` |
| gradient clip | Lightning global norm `1.0` | shared 参数与 task head 参数分别 clip norm `1.0` |
| SIGReg weight | `0.02` | `0.03` |
| SIGReg projections | `1024` | `1024` |
| SIGReg knots | `17` | `17` |
| SDPA | deterministic Math-SDPA | deterministic Math-SDPA |
| 保存频率 | 每 epoch | 每 epoch、每任务一个 shard |
| validation | 10% split，最多 1 个 val batch | 不运行 validation loop |
| W&B | 默认关闭 | 无 W&B |
| 训练 seeds | `0/42/3072` | `0/42/3072` |
| 评测 seeds | `0/1/42` | `0/1/42` |

注意：单任务和多任务的优化器参数相同，但优化轨迹不同。单任务 scheduler 由 stable-pretraining 的手动优化循环在每个 batch 后 step；多任务代码直接调用 fused AdamW，训练期间不改变学习率。

## 2. 数据与窗口

| 参数 | 当前值 |
|---|---:|
| image size | `224 x 224` |
| latent dim | `192` |
| sampled frames | `8`，即 `z_0 ... z_7` |
| supervised transitions | `7`，即 `0 ... 6` |
| frameskip / primitive actions per chunk | `5` |
| Forward history size | `3` |
| train split | `0.9` |
| validation split | `0.1` |
| training shuffle | 开启 |
| train `drop_last` | `true` |
| loader workers | `6` |
| persistent workers | `true` |
| prefetch factor | `3` |
| pin memory | `true` |

`full data` 指先使用完整公开数据集，再按固定 seed 做 90/10 train-validation clip split；不是把 validation 部分也送入优化器。

四任务数据映射：

| Task | 训练数据名 | 数据配置 |
|---|---|---|
| PushT | `pusht_expert_train.lance` | `config/train/data/pusht.yaml` |
| Cube | `ogbench/cube_single_expert.h5` | `config/train/data/ogb.yaml` |
| Reacher | `reacher.h5` | `config/train/data/dmc.yaml` |
| TwoRoom | `tworoom.h5` | `config/train/data/tworoom.yaml` |

## 3. Loss 与时间索引

总目标为：

```text
L = 1.0 * L_forward
  + lambda_sig * L_SIGReg
  + 0.1 * L_local_NLL
  + 0.05 * L_goal_NLL
```

其中单任务 `lambda_sig=0.02`，多任务 `lambda_sig=0.03`。

| 项 | current latent | condition / target | action target | 数量 |
|---|---|---|---|---:|
| Forward MSE | `z_0 ... z_6` | 预测相邻 `z_1 ... z_7` | `A_0 ... A_6` | 7 |
| Local NLL | `z_0 ... z_6` | `m_t=z_{t+1}-z_t`，successor 保持 attached | `A_0 ... A_6` | 7 |
| Goal NLL | `z_0 ... z_6` | `m_t=sg(z_7)-z_t` | `A_0 ... A_6` | 7 |

Local 和 Goal 调用同一个 `IntentActionActor`，只是两次独立 NLL；不是两个 actor，也没有强迫 local intent 与 goal intent 数值相等的 loss。

规范 intent 是 `goal_displacement`。匹配 waypoint ablation 使用：

```text
m_t = (sg(z_7) - z_t) / remaining_steps
```

## 4. Previous-action 边界

每个 actor 条件均显式使用 `A(a_{t-1})`：

- episode 中间 clip 的第一项使用真实前一 action chunk；
- episode 起点缺少的 primitive action 使用 raw zero 左填充；
- raw zero 先进入真实专家动作统计量的 z-score，不使用 normalized zero；
- mean/std 仅由有限的真实专家动作计算，标准差使用 `ddof=1`；
- eval reset 与训练使用同一 initial previous-action 语义。

详细定义见 [PREVIOUS_ACTION_BOUNDARY_20260804.md](PREVIOUS_ACTION_BOUNDARY_20260804.md)。

## 5. 模型结构

### Shared visual stack

| 模块 | 参数 |
|---|---|
| Encoder | ViT-Tiny/14，image 224，hidden 192，随机初始化，`pretrained=false`，`use_mask_token=false` |
| Projector | `192 -> 2048 -> 192`，BatchNorm1d + GELU |

### Task-specific stack

| 模块 | 参数 |
|---|---|
| Forward Predictor | history 3，depth 6，heads 16，dim-head 64，MLP 2048，dropout 0.1 |
| Action Embedder | 输入维度 = `5 x primitive_action_dim`，输出 192 |
| Prediction Projector | `192 -> 2048 -> 192`，BatchNorm1d + GELU |
| INTACT Actor | four-slot，hidden 1024，depth 3，dropout 0 |
| Action distribution | diagonal Gaussian，`log_std` clamp 到 `[-5, 2]` |
| Latent prediction | `predict_residual=false` |

Four-slot 输入为：

```text
[z_t, m_t, z_t * m_t, A(a_{t-1})]
```

## 6. 单任务规范

```bash
bash scripts/train_single.sh TASK SEED displacement
```

- 默认配置 seed 为 `3072`；规范训练 seed 统一为 `0/42/3072`。
- `train_single.sh` 会拒绝规范集合以外的 seed。
- 每个 checkpoint 固定在 evaluation seeds `0/1/42` 上分别评测。
- `trainer.devices=1`，`accelerator=gpu`。
- `gradient_clip_val=1.0`。
- `limit_val_batches=1`，`num_sanity_val_steps=0`。
- checkpoint 保存到 `$STABLEWM_HOME/checkpoints/<output_model_name>/`。
- 每次运行保存解析后的 `train_config.yaml` 与 `run_metadata.json`。

单任务 scheduler 精确定义：

- `LinearWarmupCosineAnnealingLR`；
- warmup steps = `max(1, 1% * estimated_stepping_batches)`；
- warmup start lr = `0`；
- peak/base lr = `5e-4`；
- cosine minimum lr = `0`；
- stable-pretraining 手动优化循环在每个 optimizer step 后调用 scheduler。

## 7. 四任务联合训练规范

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/train_multitask.sh SEED displacement OUTPUT_CACHE RUN_DIR
```

固定 rank 所有权：

| Global rank | Task |
|---:|---|
| 0 | PushT |
| 1 | Cube |
| 2 | Reacher |
| 3 | TwoRoom |

共享边界：

- **共享并同步**：Encoder、Projector；
- **任务专属**：Forward Predictor、Action Embedder、Prediction Projector、INTACT Actor；
- 每 step 对共享参数梯度做 all-reduce mean；
- 每 step 平均 floating buffers，非 floating buffers 从 rank 0 broadcast；
- 每个 epoch 校验四 rank 的共享 state SHA256 完全一致。

数据步数：

- 每个任务建立自己的 90% train loader；
- epoch steps 取四个 loader 的最大值；
- 较短 loader 耗尽后从头继续，保证每个同步 step 四个任务都提供一个 batch；
- 因此一个联合 epoch 不是简单的“每个任务只遍历一次”。

随机性：

- CLI 只接受 seed `0/42/3072`；
- 模型与 split 使用训练 seed；
- 各 rank shuffle generator 使用 `seed + 1000 * rank`；
- 初始共享状态从 rank 0 broadcast。

输出：

- 每个 epoch 输出四个可独立评测的 task shard；
- `run_identity.json` 记录数据、loss、优化器和 previous-action contract；
- `training_metadata_epoch_<N>.json` 记录 checkpoint 级训练元数据；
- `run_dir/epochs/epoch_<N>/complete.json` 是该 epoch 四 rank 完成标志。

## 8. 允许覆盖的参数

单任务使用 Hydra，可在命令后追加覆盖，例如：

```bash
python train.py --config-name=intact_goal \
  data=pusht seed=3072 \
  trainer.max_epochs=1 \
  loader.batch_size=256 \
  optimizer.lr=5e-4 \
  loss.sigreg.weight=0.02
```

多任务 CLI 支持：

```text
--variant --seed --epochs --batch-size --num-workers
--lr --weight-decay --gradient-clip --log-every
--output-prefix --output-cache --run-dir
```

任何偏离本手册的覆盖都必须保存在输出 metadata 中，并在结果表中单独命名，不能与规范 checkpoint 混合聚合。
