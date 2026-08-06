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

<p align="center">
  <img src="assets/intact-teaser.png" width="100%" alt="INTACT 方法与结果概览">
</p>

<p align="center">
  <a href="https://zju3dv.github.io/INTACT-JEPA/community/">
    <img src="assets/intact-manifesto.svg" width="100%" alt="强大的信息传递机制能够确保重要信息不被遗漏。INTACT 正是实现了这一目标，将 LeWM 提升为一个更强大的世界模型。欢迎提交 Issue、PR，并加入我们的 Community。">
  </a>
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

训练/评估源码、单任务与四任务共享训练配置、Direct/Pure-CEM/Actor-CEM
接口、Official/CLEAR-LeWM 评估入口、checkpoint 清单和 CUDA 12.4 依赖锁已经
公开。完整的 72 个 paper checkpoint 已在
[Hugging Face `INTACT-JEPA/INTACT`](https://huggingface.co/INTACT-JEPA/INTACT/tree/paper-e5-goal-v1)
公开，下载脚本会固定版本并逐项校验 SHA-256。

```bash
git clone https://github.com/zju3dv/INTACT-JEPA.git
cd INTACT-JEPA
bash scripts/install.sh cu124
source .venv/bin/activate
cp .env.example .env                 # 设置 STABLEWM_HOME 和 LOCAL_DATASET_DIR

# 单任务预检查与训练
python scripts/preflight_check.py train-single \
  --task pusht --train-seed 3072
CUDA_VISIBLE_DEVICES=0 bash scripts/train_single.sh \
  pusht 3072 displacement

# 四任务共享 encoder 预检查与训练（每个任务一张 GPU）
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/preflight_check.py train-multitask --train-seed 3072
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_multitask.sh \
  3072 displacement "$STABLEWM_HOME/checkpoints" \
  outputs/intact_multitask_goal_s3072_e5
```

单任务命令中，将 `pusht` 替换为 `cube`、`reacher` 或 `tworoom` 即可训练
其他任务；将 `displacement` 替换为 `waypoint` 可运行匹配的坐标意图控制。

当前根目录实现采用修正后的 previous-action 边界和 7/7 监督协议：physical
与 goal 均从 index 0 开始覆盖七个 transition。episode 中间窗口使用真实前一
action chunk；episode 起点只对不可用的历史使用 raw zero，再执行 z-score。
精确定义与旧 checkpoint 的兼容边界见
[`docs/PREVIOUS_ACTION_BOUNDARY_20260804.md`](docs/PREVIOUS_ACTION_BOUNDARY_20260804.md)。
已发布的旧 paper checkpoint 只能使用隔离的 `paper_runtime/`，不能通过当前
根目录运行时直接加载。

完整的数据布局、正式训练、Official/CLEAR 评估和 paper checkpoint 命令见英文
主页的 [Implementation and Reproduction](README.md#implementation-and-reproduction)、
[安装文档](docs/INSTALL.md)与 [checkpoint 映射](docs/PAPER_CHECKPOINTS.md)。
详细发布边界见 [`docs/RELEASE.md`](docs/RELEASE.md)。

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

- 单任务端到端训练仅 **1 epoch**，零搜索 Direct 达到四任务 **95.33%** macro SR。
- 可选 Guarded A 仅评估 **384** 条候选序列，达到 **96.86%** macro SR；相比
  CEM 300x30 的 9,000 条候选减少 **23.44 倍**。
- Direct planner-side latency 为 **2.9-5.5 ms**。
- 四任务共享一个视觉 encoder 时，E5 Direct 达到 **89.39%** macro SR；匹配的
  shared LeWM + CEM 300x30 为 **66.17%**。
- 45 个 checkpoint 上，predicted-expert action-family kNN 与 Direct SR 的相关性
  为 **r=0.954**，高于逐点动作 $R^2$ 的 **r=0.815**。

<p align="center">
  <img src="assets/shared-encoder-results.png" width="100%" alt="四任务共享编码器的成功率对比">
</p>

各任务精确数值、方差和协议说明见[审计结果](docs/RESULTS.md)。

问题咨询：<luoliibaqi4747@gmail.com> · [欢迎提交 Issue](https://github.com/zju3dv/INTACT-JEPA/issues)
