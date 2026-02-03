## Training recipes

This repo uses the `train/verl/` submodule for all training and provides thin,
project-specific wrappers and scripts.

> All training commands below are intended to be run **inside** the Docker
> container started by running, from the repo root:
> 
> ```bash
> cd train
> ./docker_run_verl.sh
> ```
> 
> Inside the container, the working directory is `/workspace/veritas-rl`.

---

### SFT (parquet-based, Qwen3‑4B‑Instruct)
```bash
. train/scripts/lgc-v2-SFT-trainer.sh
```

The recommended way to run SFT is via the helper script:

- **Script**: `train/scripts/lgc-v2-SFT-trainer.sh`
- **Model**: `rhuanmatias/trained-top-sn120`
- **Data**: `train/parquet/train.parquet`, `train/parquet/val.parquet`
- **GPUs**: `--nproc_per_node=2` (adjustable via `gpu_count` in the script)
- **Recommended hardware**: **at least 2× A100 GPUs** (1600GB or larger) for the provided settings

High-level flow:

1. **Prepare parquet SFT data** on the host (see `docs/dataset_generation.md`):
   - For SFT format:
     - `train/parquet/train.parquet`
     - `train/parquet/val.parquet`
2. **Start the training container** from repo root:
   - `./train/docker_run_verl.sh`
3. **Inside the container**, from `/workspace/veritas-rl`, run:
   - `bash train/scripts/lgc-v2-SFT-trainer.sh`

What the script does:

- Loads `.env` from repo root (if present) and requires:
  - **`HF_TOKEN`** (for model download)
  - Optional: **`WANDB_API_KEY`** (for online logging) or `WANDB_MODE=offline`
- Installs `flashinfer-python` / `flashinfer-cubin` for better performance.
- Downloads the base model:
  - `rhuanmatias/trained-top-sn120` → `train/models/rhuanmatias/trained-top-sn120`
- Points the FSDP SFT trainer at:
  - `data.train_files=train/parquet/train.parquet`
  - `data.val_files=train/parquet/val.parquet`
  - `data.prompt_key=prompt`
  - `data.response_key=response`
- Uses key training hyperparameters (edit in the script as needed):
  - `train_batch_size=32`
  - `data.micro_batch_size_per_gpu=8`
  - `data.max_length=2048`
  - `model.fsdp_config.model_dtype=bf16`
  - `optim.lr=2e-6`, `optim.betas=[0.9, 0.98]`, `optim.weight_decay=0.05`
  - `trainer.total_epochs=10`
- Configures WandB:
  - `WANDB_PROJECT=Qwen3-4B-Instruct`
  - `WANDB_EXP=${pure_agent_model_name}-${TIME}-batch_size-...`
  - `WANDB_DIR=train/artifacts/wandb/Qwen3-4B-Instruct-<task_name>-<DATE>/` (WandB logs saved here)
- Configures checkpoints:
  - `trainer.default_local_dir=train/artifacts/checkpoints/${WANDB_EXP}`
  - `trainer.save_freq=50` (save every 50 global steps)
  - `trainer.test_freq=50` (validate every 50 global steps)

Under the hood, the script ultimately runs:

- `torchrun --nnodes=1 --nproc_per_node=${gpu_count} -m verl.trainer.fsdp_sft_trainer ...`

Checkpoints are written by `FSDPSFTTrainer.save_checkpoint` under:

- `train/artifacts/checkpoints/${WANDB_EXP}/global_step_<N>/`

**Training artifacts location:**
- All training outputs are organized under `train/`:
  - **Hydra outputs**: `train/outputs/` - Training run logs and configs (created by Hydra, format: `YYYY-MM-DD/HH-MM-SS/`)
  - **WandB logs**: `train/artifacts/wandb/`
  - **Model checkpoints**: `train/artifacts/checkpoints/` (SFT) or `train/artifacts/RL/checkpoints/` (RL)
- These directories are automatically ignored by `.gitignore`
- Hydra outputs are configured to save under `train/outputs/` to keep training artifacts organized in the `train/` folder

If you prefer to keep SFT parquet separate from RL parquet, generate to a different folder
like `train/parquet_sft/` and edit `dataset_path` in `lgc-v2-SFT-trainer.sh` and the
`data.{train,val}_files` overrides accordingly.

---

### GRPO (RL) with verifiable reward (dynamic env dataset)

```bash
. train/scripts/lgc-v2-GRPO-trainer.sh
```

For RL with GRPO and verifiable rewards, use:

- `python -m verl.trainer.main_ppo algorithm.adv_estimator=grpo`

Key parts for this repo (see `train/scripts/lgc-v2-RL-GRPO-trainer.sh`):

- **Dynamic dataset**:
  - `data.custom_cls.path=tools/dataset-manager.py` (relative to `train/` directory)
  - `data.custom_cls.name=LGCV2DynamicDataset` (back-compat alias; actual class is `DynamicEnvDataset`)
- **Custom reward**:
  - `custom_reward_function.path=tools/reward-manager.py` (relative to `train/` directory)
  - `custom_reward_function.name=compute_lgc_v2_score` (alias of `compute_score`, both work)

The RL script follows the same pattern as SFT:

1. Start Docker via `./train/docker_run_verl.sh`.
2. Run `bash train/scripts/lgc-v2-RL-GRPO-trainer.sh` inside the container.

**RL training artifacts:**
- WandB logs: `train/artifacts/wandb/`
- Model checkpoints: `train/artifacts/RL/checkpoints/`

**Recommended hardware for RL**: for realistic GRPO runs on these environments
with the provided configs, plan for **at least 4× H200 GPUs** (or comparable)
to get reasonable throughput and headroom.

#### OpenSpiel (fixed tasks per case)

**How to use `openspiel-ppo-trainer.sh`**

1. **Prerequisites**
   - **OpenSpiel**: The script installs it if missing (`pip install open_spiel`). If that fails (e.g. no PyPI build for your platform), install from source: [OpenSpiel installation](https://github.com/google-deepmind/open_spiel#installation).
   - **HF_TOKEN** set (e.g. in `.env`) for model download.
   - Training is intended to run **inside the VERL Docker container**: from repo root run `cd train && ./docker_run_verl.sh`, then inside the container run the script.

2. **Run the trainer** (from repo root, or from inside the container at `/workspace/veritas-rl`):
   ```bash
   bash train/scripts/openspiel-ppo-trainer.sh
   ```

3. **Optional overrides** (pass at the end of the command):
   - **`game_types`** — list of game short names or full case strings (e.g. `[hex, go, chess]`). If set, only these games/cases are trained on; if not set, **all games** are used. A short name like `hex` matches all hex cases (e.g. `hex(board_size=5)`, `hex(board_size=7)`). You can set it **as a script parameter**: `GAME_TYPES="hex,go,chess" bash train/scripts/openspiel-ppo-trainer.sh` (comma-separated; empty = all games).
   - **`cases`** — list of game load strings, one per "case" (e.g. board 2×2, 3×3, … 5×5). Each case gets a fixed number of tasks. Example: hex 5×5, 7×7, 9×9 and go 9×9, 13×13. Set via Hydra as a list (see your config format) or use **`game_list`** + **`game_configs`** to build cases automatically (one case per (game, config)).
   - **`num_tasks_per_case`** (default 100) — fixed number of tasks per case. Total fixed tasks = len(cases) × num_tasks_per_case.
   - **`max_random_steps`** — random rollout steps from root before presenting the state.
   - **`reward_mode`** — `outcome` (play out and use return) or `legal` (1 if legal else 0).
   - **`seed`** — base seed for reproducibility.

   Example (more tasks per case, random rollout; use `+` prefix for adapter_config overrides):
   ```bash
   bash train/scripts/openspiel-ppo-trainer.sh \
     +data.custom_cls.config.adapter_config.num_tasks_per_case=200 \
     +data.custom_cls.config.adapter_config.max_random_steps=5
   ```

4. **LoRA is default** (only train adapter parameters; base stays frozen):
   - VERL LoRA uses **strategy=fsdp/fsdp2** and **rollout.name=vllm** with HuggingFace PEFT. By default only LoRA adapters are trained; the base model is frozen so training a **new environment does not change** the base or previously trained adapters (per-env adapters can be saved and swapped).
   - To full fine-tune (train all params): `ENABLE_LORA=0 bash train/scripts/openspiel-ppo-trainer.sh`
   - Default: `lora_rank=32`, `lora_alpha=32`, `target_modules=all-linear`, `load_format=safetensors`, `use_shm=True`, and learning rate `6e-5`. Override via Hydra if needed (e.g. `+actor_rollout_ref.model.lora_rank=64`). Strategy: `FSDP_STRATEGY=fsdp2` to use fsdp2.
   - **Resume from adapter** (multi-stage): `LORA_ADAPTER_PATH=/path/to/adapter bash train/scripts/openspiel-ppo-trainer.sh` (directory must contain `adapter_model.safetensors` and `adapter_config.json`).
   - **Large model or limited GPU**: `LAYERED_SUMMON=1` enables per-layer sync of LoRA to vLLM to reduce peak memory (recommended for 70B+ or GPU < 48GB).
   - After training, run `merge-ppo.sh`; if the checkpoint is LoRA-only, the merger may write `TARGET_DIR/lora_adapter/`. To get a single full model, merge the adapter into the base with `train/tools/merge_adapter_into_base.py` (e.g. `--base_model` = your base, `--adapter_model` = the merged output or `TARGET_DIR/lora_adapter`, `--output_dir` = where to save the full model).

**Fixed tasks per case (e.g. board 2×2, 3×3, … 5×5)**

A **case** is one (game + config), e.g. hex 5×5, hex 7×7, go 9×9, breakthrough 6×6, dots_and_boxes 3×3, 4×4, 5×5.

- You provide a list **`cases`** of OpenSpiel game load strings (e.g. `["hex(board_size=5)", "hex(board_size=7)", "go(go_board_size=9)"]`), or use **`game_list`** + **`game_configs`** to build one case per (game, config).
- You set **`num_tasks_per_case`** (default 100). **Total fixed tasks** = len(cases) × num_tasks_per_case. Same task_id always yields the same state (seed = base_seed + task_index).
- In `train/tools/openspiel_adapter.py`, **`DEFAULT_CASES_BOARD_SIZES`** is an example list (hex 5/7/9, go 9/13, breakthrough 6×6/8×8, dots_and_boxes 3×3/4×4/5×5, clobber 5×5/6×6/7×7, othello, chess, checkers). Use it as a template or pass your own **`cases`** in adapter config.

The adapter lives in `train/tools/openspiel_adapter.py` and is registered as env `openspiel`. Reward is routed by `extra_info["env"]` to the OpenSpiel adapter automatically.

#### Merging PPO checkpoints

After training, merge the actor checkpoint to a single HuggingFace model. The merged model keeps its **own** `config.json` (from the checkpoint); it is not overwritten with the base/agent model config.

- **Scripts**: `train/scripts/merge.sh` and `train/scripts/merge-ppo.sh` merge the PPO actor to a single HuggingFace model. Use `CHECKPOINT_BASE` when you have an experiment subdir:  
  `CHECKPOINT_BASE=train/artifacts/RL/checkpoints/openspiel-20250202-grpo bash train/scripts/merge.sh`
- **LoRA checkpoints** (default): The merger may write a LoRA adapter under `TARGET_DIR/lora_adapter/`. Use `train/tools/merge_adapter_into_base.py` to merge that adapter into the base and save a full model.

#### ValueError: "There is no module or parameter named 'block' in Qwen3ForCausalLM"

This occurs when vLLM (used for rollout) loads a **Qwen3-based** model whose checkpoint uses different layer names than vLLM expects (e.g. HuggingFace uses `model.layers`, vLLM may look for `block`). It is a vLLM ↔ Qwen3 architecture/version mismatch.

- **Workarounds:**  
  1. Use a **Qwen2-based** base model instead: set `AGENT_MODEL_REPO_ID` to a Qwen2 model (e.g. `Qwen/Qwen2.5-4B-Instruct` or your 4B Qwen2 checkpoint) in `.env` or the environment.  
  2. Use a VERL/Docker image and vLLM version that explicitly support your Qwen3 checkpoint (check VERL/vLLM release notes and issues for Qwen3 support).  
  3. If you must use this Qwen3 model, the fix is in the vLLM submodule (e.g. `train/verl` or the container’s vLLM); align layer names in the loader with the checkpoint or upgrade vLLM once support is added.

---

## WandB Setup and Monitoring

### Initial Setup

1. **Create a WandB account** (if you don't have one):
   - Go to [https://wandb.ai](https://wandb.ai) and sign up
   - Create a new account or log in

2. **Get your API key**:
   - Navigate to [https://wandb.ai/authorize](https://wandb.ai/authorize)
   - Copy your API key

3. **Set up API key** (choose one method):
   
   **Option A: Environment variable (recommended)**
   ```bash
   export WANDB_API_KEY=your_api_key_here
   ```
   
   **Option B: Add to `.env` file** (from repo root):
   ```bash
   # Add to .env file
   WANDB_API_KEY=your_api_key_here
   ```
   
   The training scripts automatically load `.env` if present.

### Online vs Offline Mode

**Online mode (default):**
- Logs are synced to WandB cloud in real-time
- View metrics on [wandb.ai](https://wandb.ai) dashboard
- Requires `WANDB_API_KEY` to be set
- Requires internet connection

**Offline mode:**
- Logs are saved locally only (in `train/artifacts/wandb/`)
- No internet connection required
- Can sync later with `wandb sync`
- Set before training:
  ```bash
  export WANDB_MODE=offline
  ```

### WandB Configuration in Training Scripts

Both SFT and RL training scripts configure WandB automatically:

- **Project name**: `Qwen3-4B-Instruct` (configurable via `WANDB_PROJECT`)
- **Experiment name**: Auto-generated with timestamp and model info
- **Log directory**: `train/artifacts/wandb/<WANDB_DIR>/`
- **Logging**: Console + WandB (configurable via `trainer.logger`)

### Monitoring Training

#### 1. View Runs in WandB Dashboard

Once training starts, you can view runs at:
- **URL**: `https://wandb.ai/<your-username>/Qwen3-4B-Instruct`
- The dashboard shows all runs for the project

#### 2. Key Metrics Logged

**SFT Training:**
- `train/loss` - Training loss
- `train/learning_rate` - Current learning rate
- `val/loss` - Validation loss (if validation enabled)
- `train/epoch` - Current epoch
- `train/global_step` - Global training step
- `train/throughput` - Samples per second
- `system/gpu_utilization` - GPU utilization (if available)
- `system/memory_usage` - Memory usage

**RL/GRPO Training:**
- `train/policy_loss` - Policy loss
- `train/value_loss` - Value function loss
- `train/kl_loss` - KL divergence loss
- `train/reward` - Average reward
- `train/advantage` - Advantage estimates
- `train/entropy` - Policy entropy
- `train/learning_rate` - Current learning rate
- `train/global_step` - Global training step
- `rollout/throughput` - Rollout throughput
- `system/gpu_utilization` - GPU utilization

#### 3. Real-time Monitoring

**In the terminal:**
- Training scripts output progress bars and key metrics to console
- Look for lines like: `Training Progress: 10%|████      | 1000/10000 [05:23<48:27, 3.10it/s]`

**In WandB dashboard:**
- **Runs page**: See all runs, compare experiments
- **Charts**: Real-time plots of metrics
- **System**: GPU/CPU usage, memory consumption
- **Logs**: Training logs and output
- **Files**: Saved artifacts and checkpoints

#### 4. Comparing Runs

1. Go to your project page: `https://wandb.ai/<username>/Qwen3-4B-Instruct`
2. Select multiple runs (checkbox)
3. Click "Compare" to see side-by-side metrics
4. Use filters to find specific runs (by name, tags, etc.)

### Troubleshooting WandB

#### Issue: "WANDB_API_KEY not set" warning

**Solution:**
```bash
# Set API key
export WANDB_API_KEY=your_key_here

# Or add to .env file
echo "WANDB_API_KEY=your_key_here" >> .env
```

#### Issue: WandB login fails in Docker

**Solution:**
- Make sure `WANDB_API_KEY` is set before starting Docker
- Or use `WANDB_MODE=offline` for local-only logging
- Check that the container has network access (for online mode)

#### Issue: Can't see runs in dashboard

**Check:**
1. Verify API key is correct: `echo $WANDB_API_KEY`
2. Check internet connection (for online mode)
3. Verify project name matches: `echo $WANDB_PROJECT`
4. Check local logs: `ls -la train/artifacts/wandb/`

#### Issue: Offline runs not syncing

**Solution:**
```bash
# Sync offline runs to WandB cloud
wandb sync train/artifacts/wandb/<run-directory>

# Or sync all offline runs
wandb sync train/artifacts/wandb/
```

#### Issue: Too many runs cluttering dashboard

**Solution:**
- Use tags to organize runs (add in training script or via WandB UI)
- Delete old runs via WandB dashboard
- Use project groups to organize experiments

### Advanced WandB Usage

#### Custom Metrics

You can add custom metrics in your reward function or dataset:

```python
# In reward-manager.py or dataset-manager.py
import wandb

# Log custom metric
wandb.log({"custom/reward_breakdown": reward_value})
```

#### Hyperparameter Tracking

WandB automatically tracks:
- All Hydra config parameters
- Environment variables (if configured)
- System information

View in dashboard under "Config" tab for each run.

#### Artifact Management

WandB can track model checkpoints as artifacts:

```python
# In training script (if needed)
wandb.log_artifact("path/to/checkpoint", type="model")
```

#### Resume Interrupted Training

If training is interrupted, WandB can help resume:
1. Check the last logged `global_step` in WandB
2. Use that step to resume from checkpoint
3. WandB will continue the same run if `resume=True` is set

### Best Practices

1. **Use descriptive experiment names**: The scripts auto-generate names, but you can customize them
2. **Tag your runs**: Add tags for easy filtering (e.g., "baseline", "experiment-v2")
3. **Monitor system metrics**: Keep an eye on GPU utilization and memory
4. **Save checkpoints frequently**: Use `trainer.save_freq` to balance storage vs. recovery
5. **Compare systematically**: Use WandB's compare feature to evaluate hyperparameter changes
6. **Clean up old runs**: Periodically archive or delete old experiments to keep dashboard clean

### Local WandB Directory Structure

```
train/artifacts/wandb/
├── Qwen3-4B-Instruct-<date>/
│   └── <run-id>/
│       ├── files/
│       │   ├── wandb-metadata.json
│       │   ├── wandb-summary.json
│       │   └── output.log
│       └── logs/
│           └── debug.log
```

All WandB data is stored locally in `train/artifacts/wandb/` and automatically ignored by `.gitignore`.

