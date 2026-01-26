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
- **Model**: `Qwen/Qwen3Guard-Gen-8B`
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
  - `Qwen/Qwen3Guard-Gen-8B` → `train/models/Qwen/Qwen3Guard-Gen-8B`
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

