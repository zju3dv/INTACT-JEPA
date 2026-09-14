# Installation and data setup

The release environment was validated on Ubuntu 22.04, Python 3.10, PyTorch
2.6.0, and CUDA 12.4. The PyTorch wheel bundles its CUDA runtime; a separate
CUDA toolkit is not required, but the machine needs an NVIDIA driver compatible
with CUDA 12.4.

## System packages

On a minimal Ubuntu machine, install the headless rendering libraries once:

```bash
sudo apt-get update
sudo apt-get install -y \
  git ffmpeg zstd libegl1 libgl1 libglfw3 libglew2.2 libosmesa6
```

## Isolated Python environment

From the repository root:

```bash
bash scripts/install.sh cu124
source .venv/bin/activate
```

For import and configuration checks on a machine without an NVIDIA GPU:

```bash
bash scripts/install.sh cpu
source .venv/bin/activate
```

The CPU environment is not intended for full training or evaluation. The
installer creates `.venv`, installs the explicit PyTorch wheel, consumes the
fully resolved `requirements-cu124.lock`, runs `pip check`, and checks the
four environment registrations. `requirements.txt` remains the readable
direct-dependency contract and is used for the CPU-only check environment. If
`uv` is unavailable, the installer falls back to Python's standard `venv`
and `pip`.

## Local paths

Copy the template and edit only the local, ignored `.env` file:

```bash
cp .env.example .env
```

The variables have separate responsibilities:

- `INTACT_PYTHON`: Python executable used by all shell launchers.
- `LOCAL_DATASET_DIR`: cache root containing the `datasets/` directory.
- `INTACT_OUTPUT_HOME`: output root for logs, checkpoints, and result files.
- `MUJOCO_GL`: headless rendering backend; `egl` is the tested GPU setting.

Values exported in the current shell take precedence over `.env`. Load and
inspect the resolved configuration with:

```bash
source scripts/fleet_env.sh
printf 'Python: %s\nData: %s\nOutput: %s\n' \
  "$INTACT_PYTHON" "$LOCAL_DATASET_DIR" "$INTACT_OUTPUT_HOME"
```

## Datasets

Download the four public archives from the
[LeWM Hugging Face collection](https://huggingface.co/collections/quentinll/lewm):

| Task | Dataset repository | Published archive |
|---|---|---|
| PushT | `quentinll/lewm-pusht` | `pusht_expert_train.h5.zst` |
| Cube | `quentinll/lewm-cube` | `cube_single_expert.tar.zst` |
| Reacher | `quentinll/lewm-reacher` | `reacher.tar.zst` |
| TwoRoom | `quentinll/lewm-tworooms` | `tworoom.tar.zst` |

Extract them into this layout:

```text
$LOCAL_DATASET_DIR/
└── datasets/
    ├── pusht_expert_train.h5
    ├── ogbench/
    │   └── cube_single_expert.h5
    ├── reacher.h5
    └── tworoom.h5
```

PushT training uses Lance. With `fleet_env.sh` loaded, convert the published
HDF5 file once:

```bash
"$INTACT_PYTHON" -m stable_worldmodel.cli convert \
  pusht_expert_train pusht_expert_train.lance \
  --source-format hdf5 --dest-format lance
```

The final training layout is:

```text
$LOCAL_DATASET_DIR/
└── datasets/
    ├── pusht_expert_train.lance/
    ├── ogbench/cube_single_expert.h5
    ├── reacher.h5
    └── tworoom.h5
```

The four-task layout measured in the release environment occupies roughly
230 GB. PushT conversion temporarily needs roughly 45 GB more while both
formats are present. These figures are planning estimates, not preflight
checks; verify available space on your filesystem.

## Verification

The first command checks imports, pinned package visibility, environment
registration, and CUDA. The second checks paths only; it does not checksum or
fully read the datasets. The final command combines the checks expected by the
training launchers.

```bash
"$INTACT_PYTHON" scripts/verify_install.py --require-cuda
"$INTACT_PYTHON" scripts/verify_data.py
bash scripts/check_fleet.sh
```

## First smoke run

Smoke mode executes the real data, model, optimizer, and checkpoint path on one
small batch. Use a unique run name because output directories are immutable.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh goal pusht \
  --smoke --run-name smoke_goal_pusht
```

For the four-task path, provide exactly four visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_multitask.sh \
  --smoke --run-name smoke_multitask_goal
```

After smoke validation, use the formal commands and paper settings in the
main [README](../README.md#training).

## Manual equivalent

If the installer cannot be used, its CUDA path is equivalent to:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-cu124.lock \
  --extra-index-url https://download.pytorch.org/whl/cu124
python -m pip check
python scripts/verify_install.py --require-cuda
```

Do not install a second unpinned PyTorch or torchvision build afterward. A
CPU/CUDA wheel mismatch is a common cause of missing operators during import.
