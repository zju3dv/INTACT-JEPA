# INTACT 训练与评估规范矩阵

本文档定义 `junhan` 分支的正式实验单元。所有变体均使用 8 帧、7 个 transition
的数据窗口；四种含 actor 的变体统一使用 **four-slot actor，local/goal 均从
index 0 开始（7/7）**。纯 LeWM 使用相同 forward 窗口，但不包含 actor。

## 1. 五种训练变体

| CLI 变体 | Forward | SIGReg | Local inverse | Goal | Goal 形式 | Actor |
|---|---:|---:|---:|---:|---|---|
| `lewm` | 1.0 | 开启 | 0 | 0 | 无 | 无 |
| `inverse` | 1.0 | 开启 | 0.1 x 7 | 0 | 无 | four-slot |
| `goal_only` | 1.0 | 开启 | 0 | 0.05 x 7 | `sg(z7)-zt` | four-slot |
| `waypoint` | 1.0 | 开启 | 0.1 x 7 | 0.05 x 7 | `(sg(z7)-zt)/remaining` | four-slot |
| `displacement` | 1.0 | 开启 | 0.1 x 7 | 0.05 x 7 | `sg(z7)-zt` | four-slot |

`displacement` 是论文主方法。`goal_only` 与 `displacement` 的 goal 条件相同，
区别是前者不施加 local inverse NLL。`lewm` 配置中 `intent_actor=null`，因此是
真正的 LeWM/SIGReg 对照，不包含随机 action head。

four-slot actor 输入统一为：

```text
[z_t, m_t, z_t * m_t, A(a_{t-1})]
```

## 2. 训练命令

单任务，每个任务分别训练，默认 1 epoch：

```bash
bash scripts/train_single.sh TASK SEED VARIANT
```

例如：

```bash
bash scripts/train_single.sh pusht 3072 displacement
bash scripts/train_single.sh pusht 3072 inverse
bash scripts/train_single.sh pusht 3072 lewm
```

四任务共享 encoder/projector、任务专属小头，默认 5 epochs：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_multitask.sh SEED VARIANT OUTPUT_CACHE RUN_DIR
```

五种 `VARIANT` 均为：

```text
lewm | inverse | goal_only | waypoint | displacement
```

规范训练 seeds 为 `0/42/3072`。

## 3. 单任务与多任务超参数

| 参数 | 单任务 | 四任务联合 |
|---|---:|---:|
| epochs | 1 | 5 |
| batch size | 256 | 每任务 256，同步总 batch 1024 |
| optimizer | AdamW | fused AdamW |
| base/peak learning rate | `5e-4` | `5e-4` |
| learning-rate schedule | 1% warmup + cosine decay 到 0 | 无，固定 `5e-4` |
| weight decay | `1e-3` | `1e-3` |
| gradient clip | global norm 1.0 | shared/head 分别 norm 1.0 |
| SIGReg weight | `0.02` | `0.03` |
| precision | bf16-mixed | bf16 autocast |
| validation | 10% split，最多 1 batch | 无 validation loop |

所以两者的**标称学习率相同**，都是 `5e-4`；差别在轨迹：单任务从 warmup
升到 `5e-4` 后余弦下降，多任务的每个 optimizer step 都保持 `5e-4`。多任务还
有更大的同步总 batch，因此不能把两者看作完全相同的优化设置。

## 4. 评估矩阵

| 训练变体 | Direct | Pure-CEM 300x30 | Actor-CEM 300x30 | Guarded A 128x3 |
|---|---:|---:|---:|---:|
| `lewm` | 不适用 | **规范主结果** | 不适用 | 不适用 |
| `inverse` | 支持 | 支持 | 支持 | 支持 |
| `goal_only` | 支持 | 支持 | 支持 | 支持 |
| `waypoint` | 支持 | 支持 | 支持 | 支持 |
| `displacement` | 支持 | 支持 | 支持 | 支持 |

Official evaluator：

```bash
bash scripts/eval_official.sh MODE TASK CHECKPOINT SEED 100
```

CLEAR-LeWM v0.5.1 Moderate：

```bash
bash scripts/eval_clear_v051.sh MODE TASK CHECKPOINT DATASET \
  CLEAR_ROOT UPSTREAM_ROOT SEED OUTPUT_JSON
```

其中 `MODE` 为 `direct`、`pure_cem`、`actor_cem` 或 `guarded_a`。规范评估
seeds 为 `0/1/42`，每个 seed 100 episodes。

Pure-CEM 使用显式 zero-mean 初始分布，绝不调用 actor。Actor-CEM 使用同一
300x30 搜索预算，但以 actor 计划作为初始均值。Direct 只执行 actor，不进行候选
搜索。Guarded A 先生成 Direct 计划，再围绕它执行 `128x3`、初始标准差 `0.25`
的局部验证，只评估 384 条候选序列；每轮保留 Direct reference 和全局 best，并
重评分最终均值。LeWM 没有 actor，因此只允许 Pure-CEM；给它报告 Direct、
Actor-CEM 或 Guarded A 都没有算法意义。
