# 当前规范配置：四槽 7/7

当前 `junhan` 分支的默认训练协议固定为：

```text
feature_layout = four_slot
local_start = 0
goal_start = 0
local indices = 0..6
goal indices = 0..6
terminal goal = z_7
actor input = [z_t, m_t, z_t * m_t, A(a_{t-1})]
```

训练 seed 为 `0/42/3072`；每个 checkpoint 在评测 seed `0/1/42` 上各运行
100 episodes。以下结果来自修复 previous-action 边界后的同一组三训练 seed、
三评测 seed checkpoint。

| Evaluator | PushT | Cube | Reacher | TwoRoom | Macro |
|---|---:|---:|---:|---:|---:|
| Official Direct | 82.11±1.64 | 100.00±0.00 | 95.89±0.69 | 97.33±2.65 | 93.83±1.13 |
| CLEAR-LeWM v0.5.1 Moderate Direct | 86.00±1.20 | 100.00±0.00 | 49.56±0.51 | 97.00±3.28 | 83.14±1.23 |

`±` 为三个训练 seed 各自先平均三个评测 seed 后的样本标准差。完整 216 条
seed-level 结果分别位于 `official_direct/` 与 `clear_v051_moderate/`。

