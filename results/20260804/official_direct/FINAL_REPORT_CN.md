# Previous-Action 修复与窗口协议终极结果

## 实验范围

- 任务：PushT、Cube、Reacher、TwoRoom。
- 结构：五槽 `[z,m,0,z*m,A(a_prev)]` 与匹配四槽 `[z,m,z*m,A(a_prev)]`。
- 协议：7/5、7/7、6/6。
- 每个 cell：3 个训练 seed（0/42/3072），每个 checkpoint 使用 3 个评估 seed（0/1/42），每个 seed 100 episode。
- 总量：72 个 checkpoint、216 个 Official Direct 评估、21,600 episode。
- 训练：1 epoch、batch size 256、AdamW、lr `5e-4`、SIGReg `0.02`、Math-SDPA。
- 动作监督：同一个共享对角高斯 actor 分别接受 local physical intent 与 goal-displacement intent，产生两个 Gaussian NLL；不是两个参数独立 actor。
- previous action：episode 起点采用 raw-zero 后 z-score；中间窗口第一项采用真实前一 action block；eval reset 使用相同语义。

## Official Direct SR

数值为 9 个 SR 的均值；`±` 为三个训练 seed 各自对三个评估 seed 取平均后的样本标准差。

| 输入 | 协议 | PushT | Cube | Reacher | TwoRoom | Macro |
|---|---|---:|---:|---:|---:|---:|
| 五槽 | 7/5 | 83.33±0.67 | 100.00±0.00 | 97.89±0.19 | 98.78±0.51 | 95.00±0.22 |
| 五槽 | 7/7 | 83.89±0.51 | 100.00±0.00 | 96.00±0.33 | 99.11±0.51 | 94.75±0.17 |
| **五槽** | **6/6** | **84.33±0.33** | **100.00±0.00** | **97.11±0.38** | **99.00±0.00** | **95.11±0.05** |
| 四槽 | 7/5 | 82.44±1.90 | 100.00±0.00 | 97.00±0.00 | 96.89±1.26 | 94.08±0.75 |
| 四槽 | 7/7 | 82.11±1.64 | 100.00±0.00 | 95.89±0.69 | 97.33±2.65 | 93.83±1.13 |
| 四槽 | 6/6 | 82.00±0.58 | 100.00±0.00 | 96.33±1.20 | 94.78±6.74 | 93.28±1.54 |

## 主要结论

1. 五槽 6/6 的 Macro 最高，为 `95.11±0.05%`；但五槽三协议只相差 `0.36pp`，窗口选择对整体结果并不敏感。
2. 五槽相对匹配四槽在 7/5、7/7、6/6 上分别提高 `+0.92/+0.92/+1.83pp` Macro，并显著降低跨训练 seed 方差。
3. 常数零槽不携带信息，也不增加理论函数类。当前优势应解释为单 epoch 下由 fan-in、初始化轨迹和优化条件带来的经验偏置，而不是额外语义。
4. Cube 六个配置全部为 100%，说明 Official Cube 在本实验区间已经饱和，不能用于区分协议。
5. 四槽 6/6 的 TwoRoom seed 0 只有 87.00%，而 seed 42/3072 均为 98.67%；其 `6.74pp` 标准差是四槽 6/6 Macro 较低的主要原因。

## 与旧错误 7/5 五槽对照

旧版本把每个采样窗口第一项的 previous action 错置为 normalized zero（即数据集平均动作），而不是中间窗口的真实前序动作；eval reset 也没有使用规范的 normalized raw-zero。

| Official Direct | 旧错误版 | 当前修复版 | 变化 |
|---|---:|---:|---:|
| PushT | 85.78±1.54 | 83.33±0.67 | -2.45pp |
| Cube | 100.00±0.00 | 100.00±0.00 | 0.00pp |
| Reacher | 97.67±0.00 | 97.89±0.19 | +0.22pp |
| TwoRoom | 97.89±1.26 | 98.78±0.51 | +0.89pp |
| Macro | 95.33±0.58 | 95.00±0.22 | -0.33pp |

论文中若保持原定 7/5 协议，应将 headline 更新为修复后的 `95.00±0.22%`。若改用 sweep 最佳 6/6，则可报告 `95.11±0.05%`，但必须明确它是窗口选择实验后的配置，不能把它伪装成原 7/5 的直接修复结果。

## 证据文件

- `RESULTS_CN.md`：24 个 task cell 与 6 个 Macro。
- `all_216_seed_results.csv`：全部 216 条 seed-level SR。
- `train_seed_means.csv`：每个 checkpoint 的三评估 seed 均值。
- `results_summary.json`：机器可读完整聚合。
- `COMPLETION_AUDIT.json`：严格完成审计，状态为 `PASS`。
- `../collected/raw_sidecars/`：216 份原始 Official sidecar。
- `../collected/checkpoint_metadata/`：72 份训练配置与运行 metadata。
