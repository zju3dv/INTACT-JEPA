# Installation and Data Setup

The corrected runtime is validated on Ubuntu 22.04, Python 3.10, PyTorch 2.6,
and CUDA 12.4. Full training and evaluation require a compatible NVIDIA GPU.
The CPU environment is intended only for imports, configuration checks, and
unit tests.

## System Packages

On a minimal Ubuntu machine, install the headless rendering libraries once:

```bash
sudo apt-get update
sudo apt-get install -y \
  git ffmpeg zstd libegl1 libgl1 libglfw3 libglew2.2 libosmesa6
```

## Isolated Python Environment

From the repository root:

```bash
bash scripts/install.sh cu124
source .venv/bin/activate
```

For CPU-only checks:

```bash
bash scripts/install.sh cpu
source .venv/bin/activate
```

The installer uses `uv` when available and otherwise uses standard `venv` and
`pip`. It installs the explicit PyTorch wheel and `requirements-dev.txt`, runs
`pip check`, and executes the test suite. The checked-in
`requirements-cu124.lock` documents the earlier public-release environment;
it is retained for compatibility records but is not consumed by the corrected
release installer.

## Local Paths

Copy the template and edit the ignored `.env` file:

```bash
cp .env.example .env
```

Set both cache variables to the same cache root unless the local installation
requires separate locations:

```bash
export STABLEWM_HOME=/path/to/stable-wm-cache
export LOCAL_DATASET_DIR=$STABLEWM_HOME
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Shell launchers inherit these variables. They never download datasets or
checkpoints implicitly.

## Datasets

Download the four public archives from the
[LeWM Hugging Face collection](https://huggingface.co/collections/quentinll/lewm):

| Task | Dataset repository | Published archive |
|---|---|---|
| PushT | [`quentinll/lewm-pusht`](https://huggingface.co/datasets/quentinll/lewm-pusht) | `pusht_expert_train.h5.zst` |
| Cube | [`quentinll/lewm-cube`](https://huggingface.co/datasets/quentinll/lewm-cube) | `cube_single_expert.tar.zst` |
| Reacher | [`quentinll/lewm-reacher`](https://huggingface.co/datasets/quentinll/lewm-reacher) | `reacher.tar.zst` |
| TwoRoom | [`quentinll/lewm-tworooms`](https://huggingface.co/datasets/quentinll/lewm-tworooms) | `tworoom.tar.zst` |

Extract them into this layout:

```text
$STABLEWM_HOME/
`-- datasets/
    |-- pusht_expert_train.h5
    |-- ogbench/
    |   `-- cube_single_expert.h5
    |-- reacher.h5
    `-- tworoom.h5
```

All extracted datasets are HDF5 at this stage. The canonical PushT training
configuration uses Lance because its shuffled pixel clips benefit from compact,
batched random access; Official and CLEAR evaluation still consume the HDF5
file. Convert only PushT once:

```bash
python -m stable_worldmodel.cli convert \
  pusht_expert_train pusht_expert_train.lance \
  --source-format hdf5 --dest-format lance
```

The final training layout is:

```text
$STABLEWM_HOME/
`-- datasets/
    |-- pusht_expert_train.lance/
    |-- ogbench/cube_single_expert.h5
    |-- reacher.h5
    `-- tworoom.h5
```

The four-task layout measured for the public release occupies roughly 230 GB.
PushT conversion temporarily needs roughly 45 GB more while both formats are
present. Verify the available space on the target filesystem.

## Preflight Verification

The preflight command checks the source fingerprint, canonical hyperparameters,
environment, CUDA visibility, dataset paths, checkpoints when applicable, and
the corrected previous-action tests.

```bash
python scripts/preflight_check.py train-single \
  --task pusht --train-seed 3072

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/preflight_check.py train-multitask --train-seed 3072
```

The training and evaluation launchers run the matching preflight automatically.
Use `INTACT_SKIP_PREFLIGHT=1` only after the same check has completed separately.

## First Training Runs

Single-task training uses one full-data epoch:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_single.sh \
  pusht 3072 displacement
```

Four-task shared-encoder training uses exactly four visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_multitask.sh \
  3072 displacement "$STABLEWM_HOME/checkpoints" \
  outputs/intact_multitask_goal_s3072_e5
```

Canonical training seeds are `0`, `42`, and `3072`. See
[`TRAINING_PROTOCOL.md`](TRAINING_PROTOCOL.md) for the 7/7 objective and
[`PREVIOUS_ACTION_BOUNDARY_20260804.md`](PREVIOUS_ACTION_BOUNDARY_20260804.md)
for the corrected boundary contract.

## Expected Layout After Setup

After following the installation, dataset, and first-run instructions, the
repository and external cache should have the following structure. Generated
weights stay under `STABLEWM_HOME`; source files remain inside the repository.

```text
INTACT-JEPA/
|-- .env
|-- .venv/
|-- checkpoints/                  # checked-in manifests only
|-- config/
|-- docs/
|-- outputs/                      # shared-run logs and metadata, after launch
|-- scripts/
|-- train.py
`-- train_multitask.py

$STABLEWM_HOME/
|-- datasets/
|   |-- pusht_expert_train.h5     # Official/CLEAR evaluation
|   |-- pusht_expert_train.lance/ # canonical PushT training
|   |-- ogbench/
|   |   `-- cube_single_expert.h5
|   |-- reacher.h5
|   `-- tworoom.h5
`-- checkpoints/
    |-- displacement_pusht_s3072/ # created by the single-task example
    `-- intact_multitask_goal_*/   # task shards created by shared training
```

Checkpoint archives downloaded from Hugging Face may also be extracted under
`$STABLEWM_HOME/checkpoints/`; preserve the paths recorded in the checked-in
manifests when reproducing paper results.
