<div align="center">
  <img src="assets/intact-wordmark.png" width="610" alt="INTACT">
  <h2>面向无搜索世界模型的同构意图到动作学习</h2>
  <p><strong>让世界模型在训练时就学会回答部署时真正收到的控制查询。</strong></p>
  <p>
    Junhan Sun<sup>1,4</sup> &nbsp;&middot;&nbsp;
    Hao Zhao<sup>2,4,&dagger;</sup> &nbsp;&middot;&nbsp;
    Guofeng Zhang<sup>1,3,&dagger;</sup>
  </p>
  <p>
    <sup>1</sup>浙江大学 CAD&amp;CG 国家重点实验室<br>
    <sup>2</sup>清华大学智能产业研究院（AIR）<br>
    <sup>3</sup>InSpatio &nbsp;&middot;&nbsp; <sup>4</sup>RoboParty Lab<br>
    <sup>&dagger;</sup>通讯作者
  </p>
  <p>
    <a href="https://arxiv.org/abs/2607.26056">论文</a> &nbsp;&middot;&nbsp;
    <a href="https://zju3dv.github.io/INTACT-JEPA/">项目主页</a> &nbsp;&middot;&nbsp;
    <a href="docs/METHOD.md">方法说明</a> &nbsp;&middot;&nbsp;
    <a href="#代码与复现">代码与复现</a> &nbsp;&middot;&nbsp;
    <a href="#主要结果">主要结果</a> &nbsp;&middot;&nbsp;
    <a href="docs/REPRODUCIBILITY.md">复现说明</a> &nbsp;&middot;&nbsp;
    <a href="https://zju3dv.github.io/INTACT-JEPA/community/"><strong>World Model Community / 世界模型交流群</strong></a> &nbsp;&middot;&nbsp;
    <a href="README.md">English</a>
  </p>
</div>

## 最新动态

- **[2026-09-14] 评测与模型更新：** 修复带 Actor 评测中 previous-action 被重复写入历史缓冲的问题。在修正后的因果历史协议下，单任务 E1 INTACT 的 Official Direct Macro SR 为 **95.61 +/- 0.59%**，可选 Guarded A 达到 **96.58 +/- 0.44%**。同时将[无历史 INTACT checkpoint 与结果](https://huggingface.co/INTACT-JEPA/INTACT/tree/main/INTACT-no-previous-action)作为独立消融发布。
- **[2026-08-06] 代码开源：** 发布训练与评测代码、单任务与共享 Encoder 配置、复现工具和模型文档。
- **[2026-07-28] 项目发布：** 发布[论文](https://arxiv.org/abs/2607.26056)、[项目主页](https://zju3dv.github.io/INTACT-JEPA/)与项目视频。

<p align="center">
  <img src="assets/intact-teaser.png" width="100%" alt="INTACT 方法与结果概览">
</p>

<p align="center">
  <strong>好的表征能够完整保留真正重要的信息。</strong><br>
  <strong>INTACT 做到了这一点，并将 LeWM 变成了更强的世界模型。</strong>
</p>

## 为什么叫 INTACT？

我们提出 **INTACT**（**IN**tent-To-**ACT**ion）：一个将带动作标注的无奖励
轨迹转化为可部署意图到动作接口的端到端 JEPA。这个名字同时描述了方法的结构
与它所保留的信息：

- **Predictor graph 之间的同构。** Local 与 goal motion-intent 使用相同的
  四槽输入语法和共享参数。
- **受支持意图族之间的同构。** 两类意图通过共享 Predictor 所诱导的动作律
  语义对应，而不是要求 latent 逐点相等。
- **从 RGB 证据到 latent intent 的完整传递。** 端到端动作梯度保留动作有效
  的视觉信息，同时抑制与运动意图无关的干扰。
- **从意图族到动作律族的完整传递。** 共享 Predictor 将训练支撑集上的族对应
  关系一直保留到直接动作读出。

## 动机

世界模型能够预测“执行某个动作会发生什么”，但部署时仍常依赖 CEM 或 MPPI，
从大量候选动作中搜索“现在应该怎么做”。这种范式让训练与推理相互割裂，二者
之间没有学出的语义对应关系，因而当前世界模型更像是“预测器 + 动作搜索器”，
而不是“意图 + 动作”的自洽模型。INTACT 将这个缺失的意图到动作接口直接纳入
端到端 JEPA 学习。

## 代码与复现

训练/评估源码、单任务与四任务共享训练配置、统一的
`scripts/eval.sh`（Direct/CEM/Guarded A）、Official/CLEAR-LeWM v0.8 判分适配、
checkpoint 清单和 CUDA 12.4 依赖锁已经
公开。完整的 72 个 paper checkpoint 已在
[Hugging Face `INTACT-JEPA/INTACT`](https://huggingface.co/INTACT-JEPA/INTACT/tree/paper-e5-goal-v1)
公开，下载脚本会固定版本并逐项校验 SHA-256。

```bash
git clone https://github.com/zju3dv/INTACT-JEPA.git
cd INTACT-JEPA
bash scripts/install.sh cu124
source .venv/bin/activate
cp .env.example .env                 # 填写本机数据与输出目录
source scripts/fleet_env.sh
"$INTACT_PYTHON" scripts/verify_install.py --require-cuda
"$INTACT_PYTHON" scripts/verify_data.py

# 单任务真实路径 smoke
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh goal pusht \
  --smoke --run-name smoke_goal_pusht

# 四任务共享 encoder smoke（每个任务一张 GPU）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_multitask.sh \
  --smoke --run-name smoke_multitask_goal
```

完整的数据布局、正式训练、Official/CLEAR 评估和 paper checkpoint 命令见英文
主页的 [Implementation and Reproduction](README.md#implementation-and-reproduction)、
[安装文档](docs/INSTALL.md)与 [checkpoint 映射](docs/PAPER_CHECKPOINTS.md)。
详细发布边界见 [`docs/RELEASE.md`](docs/RELEASE.md)。

标准评测统一使用因果 continuation history：采样起点 `t` 首次输入
`rows[t-5:t]`，仅在真实 episode 起点前做 raw-zero 左填充；之后持续移入控制器
实际执行的动作。`row[t]` 和目标动作不会作为 actor 输入。Official 与 CLEAR
只是不同 benchmark 判分协议，不是不同的 history 模式。

### 当前结果与模型

| 设置 | 推理方式 | Macro SR | 模型 / 详情 |
|---|---|---:|---|
| 单任务 E1 INTACT | Direct，零搜索 | **95.61 +/- 0.59** | [`INTACT`](https://huggingface.co/INTACT-JEPA/INTACT/tree/main/INTACT) |
| 单任务 E1 INTACT | Guarded A 128x3 | **96.58 +/- 0.44** | [审计结果](docs/RESULTS.md#task-specific-models) |
| 四任务共享 Encoder E5 | Direct，零搜索 | **91.22 +/- 0.51** | [`INTACT-unified`](https://huggingface.co/INTACT-JEPA/INTACT/tree/main/INTACT-unified) |
| 无历史 E1 消融 | Direct，零搜索 | **94.25 +/- 0.08** | [`INTACT-no-previous-action`](https://huggingface.co/INTACT-JEPA/INTACT/tree/main/INTACT-no-previous-action) |

无历史结果是独立消融，不应与可选 Guarded A 结果混为一谈。各任务数值、方差
和评测协议见[审计结果](docs/RESULTS.md)。

## 核心思想：一种输入形式，两种意图实例

INTACT 只有一种 Predictor 输入形式。对任意意图实例 $m_t$：

$$
x_t(m_t)=\big[z_t,m_t,z_t\odot m_t,A(a_{t-1})\big],
\qquad
G_\eta\left(x_t(m_t)\right)=p_\eta(a_t\mid x_t(m_t)).
$$

其中 $m_t$ 有两种取值：


$$
m_t^{\mathrm{local}}=z_{t+1}-z_t,\qquad
m_t^{\mathrm{goal}}=\mathrm{sg}(z_g)-z_t.
$$

**Local intent** 使用真实 successor，负责把可实现的物理变化与真实动作
$a_t$ 锚定；**goal intent** 负责提出行动前可获得的部署意图。二者来自同一条
示范并共享正确动作，但每个监督条件仍是独立三元组 $(z_t,m_t,a_t)$：一个
endpoint、一个 NLL。共享 Predictor 分别计算

$$
\mathcal L_{\mathrm{I2A}}
=\lambda_{\mathrm{local}}[-\log p_\eta(a_t\mid x_t(m_t^{\mathrm{local}}))]
+\lambda_{\mathrm{goal}}[-\log p_\eta(a_t\mid x_t(m_t^{\mathrm{goal}}))].
$$

INTACT 不直接最小化 local 与 goal endpoint 或 displacement 之间的距离。
它在固定状态下按所诱导的专家动作律建立**条件动作等价类**：在任务定义的合理
误差范围内，预测动作 $\hat a_t^{(1)}$、$\hat a_t^{(2)}$ 与示范动作 $a_t$
可以属于同一个动作等价邻域。相较于强迫逐点动作完全相等，这种分布式映射对小
预测误差更鲁棒，并能减轻闭环 drift 的累积；Forward JEPA 则继续保留世界预测
所需的丰富信息。

## 主要结果

<p align="center">
  <img src="assets/direct-control-results.png" width="100%" alt="单 epoch 直接控制与局部验证成功率">
</p>

- 单任务端到端训练仅 **1 epoch**，零搜索 Direct 达到四任务 **95.61%** macro SR。
- 可选 Guarded A 使用 `H=5`、`RH=5`、128x3、raw-action `sigma=0.25` 和
  top-k 16；其 **384** 条采样候选达到 **96.58%** macro SR，相比 CEM
  300x30 的 9,000 条候选减少 **23.44 倍**。额外一次确定性的最终均值复评分
  单独记录，不计入采样候选数。
- Direct planner-side latency 为 **3.9-4.8 ms**。
- 四任务共享一个视觉 encoder 时，E5 Direct 达到 **91.22%** macro SR；匹配的
  shared LeWM + CEM 300x30 为 **66.17%**。
- 15 个 goal-displacement checkpoint 上，predicted-expert action-family kNN 与
  Direct SR 的相关性为 **r=0.968**；linear CKA 为 **r=0.988**，逐点动作
  $R^2$ 为 **r=0.983**。

<p align="center">
  <img src="assets/shared-encoder-results.png" width="100%" alt="四任务共享编码器的成功率对比">
</p>

各任务精确数值、方差和协议说明见[审计结果](docs/RESULTS.md)。

联系邮箱：<luoliibaqi4747@gmail.com>
