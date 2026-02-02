## Veritas RL (LGC‑V2 + PRINT) — training + environments

This repo is a lightweight, practical wrapper around the [`train/verl/`](./train/verl) submodule (VERL RLHF/RL training library), plus:

- **`environments/`**: single-turn verifiable environments (currently `lgc-v2`, `trace`)
- **`data_processing/`**: dataset generation, evaluation, and conversion scripts
- **`train/`**: training tools, scripts, and VERL integration
- **`evaluate/`**: model evaluation tools (deploy model to Chutes, then evaluate on affine environments)

---

## Quickstart

### 1. Clone (with submodules, VERL v0.5.x)

```bash
git clone https://github.com/emglab01/veritas-rl.git
cd veritas-rl
git submodule update --init --recursive
```

### 2. Set Up Python Environment

**Data processing and evaluation scripts run directly on your host machine** (not in Docker).

**Quick Setup with uv:**

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup
chmod +x ./setup_uv.sh

./setup_uv.sh

# Configure environment
cp env.example .env
# Edit .env with your API keys
```

See [docs/setup_uv.md](docs/setup_uv.md) for detailed instructions.

### 3. Docker Setup (For Training)

**Training runs inside Docker container**:

```bash
cd train
./docker_run_verl.sh
```

This starts a GPU-enabled container. Training commands run inside the container.

- start a GPU-enabled container with the appropriate VERL image
- mount the repo at `/workspace/veritas-rl`
- set working directory to `/workspace/veritas-rl`
- drop you into a shell inside the container

**Inside the Docker container**, run training commands (see training section below).

**Note:** Docker is only needed for training. Data processing and evaluation run on your host machine.

---

## Dataset Generation

Generate datasets with problems and answers from LGC-V2 or Trace environments. You can generate datasets for specific task types, add reasoning using OpenAI, and evaluate the generated datasets to verify correctness.

For detailed documentation and command examples, see [docs/dataset_generation.md](docs/dataset_generation.md).

---

## Convert to Parquet

Convert JSONL datasets to VERL-compatible parquet format for training. Supports both RL format (with ground truth) and SFT format (with response field). The converter automatically detects response fields and preserves metadata.

For detailed options, schema information, and command examples, see [docs/dataset_generation.md](docs/dataset_generation.md).

---

## Training

Training is handled by the VERL submodule with wrapper scripts. The project supports two training approaches:

### SFT (Supervised Fine-Tuning)

SFT training fine-tunes models on pre-generated datasets in parquet format. The training process:
- Uses pre-generated parquet datasets (`train/parquet/train.parquet` and `val.parquet`)
- Fine-tunes the Qwen3-4B-Instruct model on prompt-response pairs
- Requires 1× H200 GPUs (140GB+)
- Logs metrics to WandB and saves checkpoints automatically
- Training scripts handle model downloading, configuration, and checkpoint management

### RL Training (GRPO)

RL training uses reinforcement learning with the GRPO (Group Relative Policy Optimization) algorithm:
- Uses dynamic environment-based datasets that generate challenges on-the-fly during training
- Evaluates model responses using verifiable reward functions from the environments
- No pre-generated dataset needed - challenges are generated dynamically
- Requires ≥4× H200 GPUs for optimal performance
- Custom reward functions route to the correct environment (LGC-V2 or Trace) based on task type

### How to run

**Prerequisites:** Set `HF_TOKEN` (e.g. in `.env` at repo root). Optionally set `WANDB_API_KEY` for online logging.

1. **Start the training container** (from repo root):
   ```bash
   cd train
   ./docker_run_verl.sh
   ```
   You will be dropped into a shell inside the container at `/workspace/veritas-rl`.

2. **Run one of the trainers** (from inside the container at `/workspace/veritas-rl`):

   | Trainer | Command | Notes |
   |--------|--------|--------|
   | **SFT** | `bash train/scripts/lgc-v2-SFT-trainer.sh` | Needs `train/parquet/train.parquet` and `val.parquet` (see [Dataset Generation](#dataset-generation) and [Convert to Parquet](#convert-to-parquet)). |
   | **RL (LGC-V2 GRPO)** | `bash train/scripts/lgc-v2-RL-GRPO-trainer.sh` | No dataset prep; dynamic env generates tasks on-the-fly. |
   | **OpenSpiel PPO** | `bash train/scripts/openspiel-ppo-trainer.sh` | No dataset prep; requires OpenSpiel (`import pyspiel`) installed in the container. Fixed tasks per case (e.g. board sizes) via adapter config. |

3. **Optional overrides** (append to the script command or use env var for game_types). Examples:
   ```bash
   # OpenSpiel: train only on hex, go, chess (set GAME_TYPES as param)
   GAME_TYPES="hex,go,chess" bash train/scripts/openspiel-ppo-trainer.sh

   # OpenSpiel: 200 tasks per case, 5 random rollout steps
   bash train/scripts/openspiel-ppo-trainer.sh \
     data.custom_cls.config.adapter_config.num_tasks_per_case=200 \
     data.custom_cls.config.adapter_config.max_random_steps=5
   ```

4. **Monitor:** Checkpoints go to `train/artifacts/RL/checkpoints/` (RL) or `train/artifacts/checkpoints/` (SFT). Logs under `train/outputs/` and `train/artifacts/wandb/`.

### OpenSpiel PPO — Read me & examples

**Read me:** OpenSpiel PPO trains a policy on board/card games (Hex, Go, Chess, etc.) with no dataset prep. Tasks are generated on-the-fly; you can restrict to specific games with `GAME_TYPES` or train on all. Requires OpenSpiel (`import pyspiel`) and `HF_TOKEN`. Run inside the VERL Docker container.

**Examples** (run from repo root or from inside the container at `/workspace/veritas-rl`):

```bash
# All games (default)
bash train/scripts/openspiel-ppo-trainer.sh

# Only Hex, Go, and Chess (all board sizes for each)
GAME_TYPES="hex,go,chess" bash train/scripts/openspiel-ppo-trainer.sh

# Only Hex and Othello
GAME_TYPES="hex,othello" bash train/scripts/openspiel-ppo-trainer.sh

# More tasks per case (200) and 5 random rollout steps
bash train/scripts/openspiel-ppo-trainer.sh \
  data.custom_cls.config.adapter_config.num_tasks_per_case=200 \
  data.custom_cls.config.adapter_config.max_random_steps=5

# Combine: only Go and Chess, 100 tasks per case
GAME_TYPES="go,chess" bash train/scripts/openspiel-ppo-trainer.sh
```

**Game names** you can put in `GAME_TYPES` (OpenSpiel short names): e.g. `hex`, `go`, `chess`, `checkers`, `othello`, `breakthrough`, `dots_and_boxes`, `clobber`, `leduc_poker`, `gin_rummy`, `goofspiel`, `liars_dice`, `pig`, `backgammon`, `blackjack`, and others.

**Default:** If you don’t set `cases` or `game_configs`, the adapter uses **FOCUS_CASES** (7 games: clobber, gin_rummy, goofspiel, hex, leduc_poker, liars_dice, othello) with their config variants. To use all 21 games with variants, set `cases=AVAILABLE_CASES` in adapter config. See `train/tools/openspiel_adapter.py` (`FOCUS_CASES`, `AVAILABLE_CASES`).

### Training Workflow (overview)

1. **Prepare Data**: For SFT only — generate and convert to parquet. For RL/OpenSpiel — no static dataset; dynamic env generates tasks.
2. **Start Docker**: `./train/docker_run_verl.sh`
3. **Run script**: See [How to run](#how-to-run) above.
4. **Checkpoints**: Saved automatically at configured intervals.

All training runs inside the Docker container. For detailed training docs, scripts, and WandB setup, see [docs/training.md](docs/training.md).

---

## Model Evaluation

Evaluate trained models on affine environments. The workflow involves deploying your model to Chutes (model serving platform) and then running evaluation scripts against the deployed model. Supports multiple evaluation methods including task IDs from file, single task evaluation, and range-based evaluation.

For detailed evaluation documentation, deployment instructions, and command examples, see [docs/evaluate_model.md](docs/evaluate_model.md).

---

## Environments

- **LGC-V2**: Deterministic logic puzzle generation with verification
- **Trace**: "Predict stdout" task generation

For environment details, see [docs/environments.md](docs/environments.md).

---

## Documentation

Detailed documentation is available in the [`docs/`](docs/) folder:

- [Dataset Generation](docs/dataset_generation.md) - Generate, evaluate, and convert datasets
- [Training](docs/training.md) - SFT and RL training guides
- [Model Evaluation](docs/evaluate_model.md) - Evaluation workflows
- [Environments](docs/environments.md) - Environment implementations
- [Setup Guide](docs/setup_uv.md) - Detailed setup instructions

---

## Security

- Use environment variables for API keys and tokens
- Never hardcode secrets in scripts
- See `env.example` for required environment variables


