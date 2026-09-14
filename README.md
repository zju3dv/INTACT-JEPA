<div align="center">
  <img src="assets/intact-wordmark.png" width="610" alt="INTACT: intent to action">

  <h1>INTACT: Isomorphic Intent-to-Action Learning for Search-Free World Models</h1>

  <p><strong>Train a world model to answer the control query it will receive at deployment.</strong></p>
  <p>End-to-end JEPA world modeling for goal-conditioned robot control without broad action search.</p>

  <p>
    <a href="https://github.com/DavidSunok">Junhan Sun</a><sup>1,4</sup>
    &nbsp;&middot;&nbsp; Hao Zhao<sup>2,4,&dagger;</sup>
    &nbsp;&middot;&nbsp; Guofeng Zhang<sup>1,3,&dagger;</sup>
  </p>
  <p>
    <sup>1</sup>State Key Laboratory of CAD&amp;CG, Zhejiang University<br>
    <sup>2</sup>Institute for AI Industry Research (AIR), Tsinghua University<br>
    <sup>3</sup>InSpatio &nbsp;&middot;&nbsp; <sup>4</sup>RoboParty Lab<br>
    <sup>&dagger;</sup>Corresponding authors
  </p>

  <p>
    <a href="https://arxiv.org/abs/2607.26056"><img src="https://img.shields.io/badge/Paper-arXiv%3A2607.26056-B31B1B?style=flat-square" alt="Paper on arXiv"></a>
    <a href="https://zju3dv.github.io/INTACT-JEPA/"><img src="https://img.shields.io/badge/Project_Page-Live-198F7A?style=flat-square" alt="Live project page"></a>
    <a href="#implementation-and-reproduction"><img src="https://img.shields.io/badge/Code-Available-3B82B8?style=flat-square" alt="Code available"></a>
    <a href="https://huggingface.co/INTACT-JEPA/INTACT"><img src="https://img.shields.io/badge/Models-Hugging_Face-7A5AF8?style=flat-square" alt="Models on Hugging Face"></a>
  </p>
  <p>
    <a href="https://zju3dv.github.io/INTACT-JEPA/community/"><img src="https://img.shields.io/badge/Community-Join_the_World_Model_Discussion-F59E0B?style=for-the-badge" alt="Join the bilingual INTACT World Model Community"></a>
  </p>
  <p>
    <a href="#why-intact">Why INTACT?</a> &nbsp;&middot;&nbsp;
    <a href="#installation">Installation</a> &nbsp;&middot;&nbsp;
    <a href="#training">Training</a> &nbsp;&middot;&nbsp;
    <a href="#evaluation">Evaluation</a> &nbsp;&middot;&nbsp;
    <a href="docs/METHOD.md">Method Notes</a> &nbsp;&middot;&nbsp;
    <a href="#results">Results</a> &nbsp;&middot;&nbsp;
    <a href="README_CN.md">中文</a>
  </p>
</div>

## News

- **[2026-09-14] Evaluation and model update:** Fixed duplicate previous-action insertion during actor-backed evaluation. Under the corrected causal-history protocol, task-specific E1 INTACT reaches **95.61 +/- 0.59%** Official Direct macro SR and **96.58 +/- 0.44%** with optional Guarded A. We also released [history-free INTACT checkpoints and results](https://huggingface.co/INTACT-JEPA/INTACT/tree/main/INTACT-no-previous-action) as a separate ablation.
- **[2026-08-06] Code release:** Open-sourced training and evaluation code, task-specific and shared-encoder configurations, reproducibility tools, and model documentation.
- **[2026-07-28] Project release:** Released the [paper](https://arxiv.org/abs/2607.26056), [project page](https://zju3dv.github.io/INTACT-JEPA/), and project film.

## Project Film

<p align="center">
  <a href="https://zju3dv.github.io/INTACT-JEPA/#project-film">
    <img src="docs/assets/intact-hero-film-poster.jpg" width="100%" alt="Watch the INTACT project film">
  </a>
</p>
<p align="center">
  <a href="docs/assets/intact-project-film-share.mp4">Download the share edition with project QR</a>
  &nbsp;&middot;&nbsp;
  <a href="https://zju3dv.github.io/INTACT-JEPA/#project-film">Open the interactive project page</a>
</p>

<p align="center">
  <img src="assets/intact-teaser.png" width="100%" alt="INTACT v31 teaser: matched control, action-aligned representation, and search-free inference">
</p>

<p align="center">
  <strong>A strong representation keeps the information that matters intact.</strong><br>
  <strong>INTACT does exactly that, turning LeWM into a stronger world model.</strong>
</p>

## Why INTACT?

We introduce **INTACT** (**IN**tent-To-**ACT**ion), an end-to-end JEPA that
turns action-labeled, reward-free trajectories into a deployable
intent-to-action interface. The name captures both the structure we impose and
the information we preserve:

- **Isomorphic between predictor graphs.** Local and goal motion-intent calls
  use the same four-slot input grammar and the same parameters.
- **Isomorphic between supported families.** Local and goal intent families
  correspond through the action-law semantics induced by that shared
  predictor, not through pointwise latent equality.
- **Intact from RGB evidence to latent intent.** End-to-end action gradients
  retain action-effective visual information while suppressing nuisance that
  is unrelated to motion intent.
- **Intact from intent families to action-law families.** The shared predictor
  preserves the supported family correspondence all the way to direct action
  readout.

## TL;DR

Forward world models answer **"what will happen if I execute this action?"**
Yet deployment asks the inverse question: **"which action realizes this
intent?"** CEM and MPPI answer it by numerically searching over many candidates,
leaving training and inference without a learned semantic correspondence. The
result is often a *predictor plus an action searcher*, rather than a
self-consistent intent-action model.

**INTACT learns that missing correspondence end to end.** One conditional
operator maps both observed physical change and deployable goal intent to an
action law. Its conditional mean is a zero-search controller; sampling is
retained only for diversity or optional local verification. No frozen encoder,
extra policy-training stage, or globally linear latent dynamics is required.

<p align="center">
  <strong>1 epoch</strong> training &nbsp;&middot;&nbsp;
  <strong>95.61%</strong> Direct macro &nbsp;&middot;&nbsp;
  <strong>0</strong> search &nbsp;&middot;&nbsp;
  <strong>3.9-4.8 ms</strong> latency
</p>
<p align="center">
  <strong>91.22%</strong> Shared E5 Direct &nbsp;&middot;&nbsp;
  <strong>96.58%</strong> Guarded &nbsp;&middot;&nbsp;
  <strong>23.44x</strong> fewer candidates
</p>

## Current Results and Models

| Setting | Inference | Macro SR | Model / details |
|---|---|---:|---|
| Task-specific E1 INTACT | Direct, zero search | **95.61 +/- 0.59** | [`INTACT`](https://huggingface.co/INTACT-JEPA/INTACT/tree/main/INTACT) |
| Task-specific E1 INTACT | Guarded A 128x3 | **96.58 +/- 0.44** | [Audited results](docs/RESULTS.md#task-specific-models) |
| Shared-encoder E5 INTACT | Direct, zero search | **91.22 +/- 0.51** | [`INTACT-unified`](https://huggingface.co/INTACT-JEPA/INTACT/tree/main/INTACT-unified) |
| History-free E1 ablation | Direct, zero search | **94.25 +/- 0.08** | [`INTACT-no-previous-action`](https://huggingface.co/INTACT-JEPA/INTACT/tree/main/INTACT-no-previous-action) |

The history-free row is a separate ablation and should not be confused with
the optional Guarded A result. Exact task values, variances, and evaluation
contracts are recorded in [Audited Results](docs/RESULTS.md).

## Implementation and Reproduction

The released implementation includes task-specific training, four-task
shared-encoder training, Direct/CEM/Guarded-A inference, Official LeWM and
CLEAR-LeWM v0.8 scoring adapters, checkpoint manifests, and a
checkpoint-compatible paper evaluation runtime. Smoke mode exercises the real
data, model, optimizer, and checkpoint path; it is a code-path check, not an
accuracy claim.

### Installation

The locked CUDA environment was validated on Ubuntu 22.04, Python 3.10,
PyTorch 2.6.0, and CUDA 12.4:

```bash
git clone https://github.com/zju3dv/INTACT-JEPA.git
cd INTACT-JEPA
bash scripts/install.sh cu124
source .venv/bin/activate
cp .env.example .env
# Edit .env with local dataset and output roots.
source scripts/fleet_env.sh
"$INTACT_PYTHON" scripts/verify_install.py --require-cuda
```

Use `bash scripts/install.sh cpu` only for import and configuration checks on
a machine without an NVIDIA GPU. Full training and evaluation require CUDA.
See [Installation](docs/INSTALL.md) for system packages, manual setup, data
conversion, and first-run diagnostics.

### Data

INTACT uses the official LeWM datasets. Existing data are reused in place and
are never downloaded implicitly by the training scripts. Configure this layout
through `.env`:

```text
$LOCAL_DATASET_DIR/
└── datasets/
    ├── pusht_expert_train.lance
    ├── ogbench/cube_single_expert.h5
    ├── reacher.h5
    └── tworoom.h5
```

The public PushT archive is HDF5. Convert it once to the Lance training layout
after placing `pusht_expert_train.h5` in `datasets/`:

```bash
"$INTACT_PYTHON" -m stable_worldmodel.cli convert \
  pusht_expert_train pusht_expert_train.lance \
  --source-format hdf5 --dest-format lance
"$INTACT_PYTHON" scripts/verify_data.py
bash scripts/check_fleet.sh
```

### Smoke tests

```bash
# One task, one GPU, one optimizer step.
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh goal pusht \
  --smoke --run-name smoke_goal_pusht

# Shared encoder, one task per GPU, one synchronized optimizer step.
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_multitask.sh \
  --smoke --run-name smoke_multitask_goal
```

Run directories are immutable by default: an existing log or non-empty output
directory causes an actionable failure instead of silent overwriting.

### Training

Training always uses Math SDPA.

#### **Single-Task Training**

The task-specific paper setting trains one model per task for one full-data
epoch:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh goal pusht \
  --run-name intact_goal_pusht_s3072_e1 \
  seed=3072 trainer.max_epochs=1
```

Replace `pusht` with `cube`, `reacher`, or `tworoom`, and replace `goal` with
`waypoint` for the matched coordinate control.

Each effective eight-frame window trains all seven physical and all seven goal
transitions. On every index, the same demonstrated action is evaluated under
the attached successor condition and the detached deployment-goal condition.
An interior clip uses its true preceding action block; only unavailable history
before a real episode boundary is raw-zero padded and then normalized.

#### **Joint Multi-Task Training**

The four-task E5 setting uses the same seven-local/seven-goal objective in four processes, one task
per GPU, with one shared encoder/projector and task-specific Forward/action
heads:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_multitask.sh \
  --run-name intact_multitask_goal_s3072_e5
```

Its released default is fused AdamW, constant `lr=5e-4`, weight decay `1e-3`,
SIGReg `0.03`, five epochs, Math SDPA, and batch 256 per task. Do not silently
reduce the batch when reporting a paper-protocol reproduction.

Both launchers print all artifact paths and emit machine-readable progress at
step 1, every 100 steps, and the final step:

```text
TRAIN_PROGRESS={"epoch": 1, "global_step": 100, "loss": ..., "lr": ..., "eta_seconds": ...}
MULTITASK_PROGRESS={"task": "pusht", "rank": 0, "global_step": 100, ...}
```

### Evaluation

INTACT has one policy-evaluation entrypoint. `MODE` selects the controller; it
does not select a different history protocol:

| Mode | Actor | Search | Role |
|---|---|---|---|
| `direct` | yes | none | Native search-free controller |
| `cem` | no | broad CEM | Actor-disabled representation/control baseline |
| `guarded_a` | yes | local CEM 128x3 | H5/RH5, sigma=0.25, top-k 16 around Direct |

```bash
bash scripts/eval.sh direct pusht \
  intact_goal_pusht_s3072_e1/weights_epoch_1.pt 42 100
bash scripts/eval.sh cem pusht \
  intact_goal_pusht_s3072_e1/weights_epoch_1.pt 42 100
bash scripts/eval.sh guarded_a pusht \
  intact_goal_pusht_s3072_e1/weights_epoch_1.pt 42 100
```

All actor-backed modes use the same causal continuation contract. At a sampled
start step `t`, the initial actor input is `rows[t-5:t]`; missing rows before
the true episode boundary are raw-zero padded. After reset, the history shifts
in actions actually executed by the controller. The current dataset action
`row[t]` and target action are never exposed.

The command above uses Official LeWM scoring. CLEAR-LeWM v0.8 is a different
benchmark protocol, not a different INTACT history mode. Install its pinned
environment and invoke the scoring adapter directly:

```bash
export CLEAR_LEWM_ROOT=/path/to/CLEAR-LeWM-v0.8
export CLEAR_LEWM_PYTHON="$CLEAR_LEWM_ROOT/.venv/bin/python"
MANIFEST="$CLEAR_LEWM_ROOT/manifests/v0.8/pusht/moderate-seed42-n100.json"
CHECKPOINT=intact_goal_pusht_s3072_e1/weights_epoch_1.pt

"$CLEAR_LEWM_PYTHON" clear_eval.py \
  --clear-root "$CLEAR_LEWM_ROOT" \
  --manifest "$MANIFEST" \
  --policy "$CHECKPOINT" \
  --output results/pusht-clear-direct.json \
  --mode direct
```

Use `--mode pure_cem` or `--mode guarded_a` for the corresponding CLEAR run.
Official and CLEAR results must be reported separately because their reset
manifests and success criteria differ.

### Paper checkpoints

The checked-in manifests define six controlled shared-encoder E5 cells, three
training seeds, and four task shards per seed (72 checkpoints). The immutable
`paper-e5-goal-v1` model revision is publicly hosted at
[`INTACT-JEPA/INTACT`](https://huggingface.co/INTACT-JEPA/INTACT/tree/paper-e5-goal-v1).
These commands anonymously download and verify one cell or the full matrix:

```bash
bash scripts/download_paper_checkpoints.sh waypoint_intact all
bash scripts/download_paper_checkpoints.sh matrix all
bash scripts/eval_paper_matrix.sh goal_intact pusht 3072 42 100
```

Paper checkpoints use the bundled compatibility runtime in `paper_runtime/`.
Exact cell mappings, expected paper scores, hash verification, and
compatibility boundaries are documented in
[Paper Checkpoints](docs/PAPER_CHECKPOINTS.md).

### Reproduction record

For every reported run, retain the source commit, resolved configuration,
dataset and protocol, task, training/evaluation seeds, epoch, batch per task,
inference mode, SDPA backend, checkpoint hash, completion metadata, and
machine-readable evaluator output. The launchers record these fields where
available. The complete reporting requirements remain normative in the
[Reproducibility Contract](docs/REPRODUCIBILITY.md).

## Core Insight: One Input Grammar, Two Intent Instances

INTACT has one predictor input form. For either intent instance $m_t$, it uses

$$
x_t(m_t)=\big[z_t,m_t,z_t\odot m_t,A(a_{t-1})\big],
\qquad
G_\eta\left(x_t(m_t)\right)
=p_\eta(a_t\mid x_t(m_t)).
$$

Only the value and gradient role of $m_t$ change:

$$
m_t^{\mathrm{local}}=z_{t+1}-z_t,
\qquad
m_t^{\mathrm{goal}}=\mathrm{sg}(z_g)-z_t.
$$

The **local instance** uses a realized successor to ground which physical
change produced the demonstrated action. The **goal instance** presents the
same operator with the intent available before acting. Both come from the same
demonstration and are supervised against the same correct $a_t$, but each
supervised conditional remains one triplet $(z_t,m_t,a_t)$ with one endpoint
and one proper NLL:

$$
\mathcal L_{\mathrm{I2A}}
=\lambda_{\mathrm{local}}[-\log p_\eta(a_t\mid x_t(m_t^{\mathrm{local}}))]
+\lambda_{\mathrm{goal}}[-\log p_\eta(a_t\mid x_t(m_t^{\mathrm{goal}}))].
$$

There is **no direct loss between the two endpoints or their latent
displacements**. Instead, the shared likelihood creates a conditional action
quotient: at a fixed state, intents are equivalent when they induce the same
expert action law. Under a task-appropriate tolerance, nearby predictions
$\hat a_t^{(1)}$, $\hat a_t^{(2)}$, and the demonstrated $a_t$ can therefore
belong to the same action-equivalence neighborhood. This distributional view
makes direct control less sensitive to small prediction errors and helps limit
closed-loop drift, while the forward JEPA keeps the richer world information
needed for prediction.

## Method

The physical successor remains attached to ground reachability; the future
goal is a stop-gradient deployment anchor. INTACT aligns the two supported
condition families through the actions they induce, **without matching their
endpoints, imposing globally linear latent dynamics, freezing the encoder, or
adding a phase-2 controller**. The goal likelihood alone is a goal-conditioned
imitation objective; full INTACT is its shared, end-to-end coupling with the
attached physical likelihood and forward JEPA.

INTACT retains the forward JEPA and adds one action-law predictor with a matched
input grammar for physical and deployable intents. The key construction is not
two unrelated auxiliary losses: both calls share parameters and a proper action
likelihood, while their upstream gradient routes remain deliberately asymmetric.

| Component | Role |
|---|---|
| **Forward predictor** | Preserves latent dynamics, contacts, topology, and visual information needed for rollout. |
| **Physical intent call** | Uses the observed successor with attached gradients to preserve action-recoverable change. |
| **Goal intent call** | Uses a detached future goal to train the same interface on a condition available before acting. |
| **Matched interaction** | Pairs first-order intent $m_t$ with the state-intent feature $z_t\odot m_t$. |
| **Direct controller** | Emits an action chunk with no candidate search or terminal latent-cost call. |
| **Guarded local verification** | Optionally refines the coherent Direct plan with a small local CEM budget. |

The four-domain model shares one visual encoder and keeps lightweight,
task-specific forward/action heads:

<p align="center">
  <img src="assets/shared-encoder-method.png" width="100%" alt="INTACT inference and deployment pipeline">
</p>

See [Method Notes](docs/METHOD.md) for the statistical construction, gradient
contract, and the distinction from inverse dynamics, goal-conditioned behavior
cloning, and post-hoc sampling priors.

## Results

### Task-Specific, One Epoch

The matched task-specific setting trains three models per task. Each checkpoint
is evaluated with three seeds and 100 episodes per seed on the official LeWM
protocol.

<p align="center">
  <img src="assets/direct-control-results.png" width="100%" alt="One-epoch direct control and local verification results">
</p>

One epoch of goal-displacement INTACT reaches **95.61 +/- 0.59%** Direct macro
SR with no candidate search. The same checkpoints reach **96.58 +/- 0.44%**
with Guarded A (`H=5`, `RH=5`, 128x3, raw-action `sigma=0.25`, top-k 16).
Its 384 sampled candidates are distinct from the one deterministic final-mean
rescore. Published LeWM numbers use its separate
10-epoch CEM protocol and serve only as landscape context, not paired
significance controls. Exact task values and the full matched inference matrix
remain available in [Audited Results](docs/RESULTS.md).

### One Shared Encoder

At epoch 5, Goal-displacement INTACT reaches **91.22 +/- 0.51%** Direct macro SR
with one encoder shared across all four visual domains. The matched shared LeWM
baseline reaches **66.17 +/- 2.67%** with CEM 300x30. With all INTACT action heads
disabled, pure-CEM macro still rises to **70.08 +/- 1.13%**, separating
representation shaping from direct action readout.

<p align="center">
  <img src="assets/shared-encoder-results.png" width="100%" alt="Controlled shared-encoder success rates across four tasks">
</p>

## Theory Meets Measurement

At fixed state $z$, INTACT treats two endpoint conditions as equivalent when
they induce the same expert action law:

$$
y\sim_z y' \iff p_E^\star(a\mid z,y)=p_E^\star(a\mid z,y').
$$

This conditional action quotient predicts that control should track the
relation between predicted and expert action-law families, not task clustering
or latent rank alone. Across 15 eligible goal-displacement checkpoints,
predicted-expert kNN overlap correlates with Direct SR at **r = 0.968** and
linear CKA at **r = 0.988**; pointwise action $R^2$ reaches **r = 0.983**.

<p align="center">
  <img src="assets/action-family-alignment.png" width="100%" alt="Action-family alignment diagnostics and their correlation with direct control">
</p>

## Community

Join the **INTACT World Model Community** for focused discussion around JEPA,
LeWM, INTACT, representation learning, and efficient robot control. New results,
reproductions, critiques, collaborations, and early ideas are all welcome.

<p align="center">
  <a href="https://zju3dv.github.io/INTACT-JEPA/community/"><strong>Open the permanent Community page</strong></a><br>
  The current WeChat invitation is maintained behind this stable link.
</p>

## Citation

An arXiv preprint is available at [arXiv:2607.26056](https://arxiv.org/abs/2607.26056).
Please prefer the machine-readable [`CITATION.cff`](CITATION.cff) record.

```bibtex
@misc{sun2026intact,
  title         = {INTACT: Isomorphic Intent-to-Action Learning for Search-Free World Models},
  author        = {Sun, Junhan and Zhao, Hao and Zhang, Guofeng},
  year          = {2026},
  eprint        = {2607.26056},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2607.26056}
}
```

## Acknowledgements

INTACT builds on the LeWM and stable-worldmodel research ecosystem. Evaluation
is reported on the official LeWM protocol and, where explicitly marked, the
separate [CLEAR-LeWM](https://davidsunok.github.io/CLEAR-LeWM/) evaluator.
See [NOTICE.md](NOTICE.md) for provenance and licensing boundaries.

<p align="center">
  Questions: <a href="mailto:luoliibaqi4747@gmail.com">luoliibaqi4747@gmail.com</a>
</p>
