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

SFT supports **multiple train parquet files** and **.env-driven params**. In `.env` (or the environment) you can set: `AGENT_MODEL_REPO_ID`, `SFT_TRAIN_FILES` (comma-separated), `SFT_VAL_FILES`, `SFT_PARQUET_DIR`, `WANDB_PROJECT`, `WANDB_EXP`, `SFT_TRAIN_BATCH_SIZE`, `SFT_MAX_LENGTH`, `SFT_TOTAL_EPOCHS`, `GPU_COUNT`, `SFT_SAVE_DIR`, `SFT_LR`, **`SFT_PROMPT_KEY`** and **`SFT_RESPONSE_KEY`** (parquet column names for prompt and response), **`SFT_LORA_RANK`**, **`SFT_LORA_ALPHA`**, **`SFT_LORA_TARGET_MODULES`** (LoRA mode: set `SFT_LORA_RANK=16` and `SFT_LORA_ALPHA=32` to train only low-rank adapters and reduce overfitting), **`SFT_MAX_CKPT_TO_KEEP`** (keep only the last N checkpoints to save disk; e.g. `SFT_MAX_CKPT_TO_KEEP=3`; unset = keep all), etc. See the header of `train/scripts/lgc-v2-SFT-trainer.sh` for the full list. **SWE-synth style** parquet (e.g. `instance_id`, `messages`, `model_patch`, `run_name`): use `SFT_PROMPT_KEY=problem_statement` (or a column that holds the prompt string) and `SFT_RESPONSE_KEY=model_patch`; `instance_id` and `run_name` may be present but are not used for the loss. If your prompt is in a list column like `messages`, add a string column (e.g. formatted problem statement) and point `SFT_PROMPT_KEY` at it.

**SFT with SWE-Synth Moatless SFT Trajectories (HuggingFace):** The dataset [swesynth/SWE-Synth_Moatless-SFT-Trajectories](https://huggingface.co/datasets/swesynth/SWE-Synth_Moatless-SFT-Trajectories) has columns `instance_id`, `messages` (list of chat dicts), `model_patch`, and `run_name`. To train on it without manual conversion, set **`SFT_HF_DATASET`**; the script will run `data_processing/convert_swe_synth_sft.py` to build parquet with `prompt` (from `messages`) and `response` (from `model_patch`), then train. Example (from repo root):

```bash
export SFT_HF_DATASET=swesynth/SWE-Synth_Moatless-SFT-Trajectories
export SFT_VAL_RATIO=0.1
bash train/scripts/lgc-v2-SFT-trainer.sh
```

Optional: `SFT_VAL_RATIO` (default `0.1`) controls the validation fraction; `SFT_PARQUET_DIR` (default `train/parquet`) is where the converted parquet files are written. To convert only (no training), run:

```bash
python data_processing/convert_swe_synth_sft.py \
  --dataset swesynth/SWE-Synth_Moatless-SFT-Trajectories \
  --output-dir train/parquet --val-ratio 0.1
```

Then set `SFT_TRAIN_FILES=train/parquet/swe_synth_sft_train.parquet`, `SFT_VAL_FILES=train/parquet/swe_synth_sft_val.parquet`, `SFT_PROMPT_KEY=prompt`, `SFT_RESPONSE_KEY=response` and run the SFT script as usual. The converter truncates prompt and response by character length (`--max-chars-prompt`, `--max-chars-response`; when invoked from the script, derived from `SFT_MAX_LENGTH`) so tokenized sequences stay under the model limit and the trainer does not hit tokenizer length warnings.

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

#### SWE-bench / mini-swe-agent

[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) is a minimal AI software-engineering agent that scores >74% on [SWE-bench verified](https://www.swebench.com/). This adapter uses **SWE-bench** datasets and the mini-swe-agent ecosystem: you load **tasks** (problem statements) from HuggingFace or cache; your **fixer** model produces a patch; **evaluation** uses mini-extra swebench and [sb-cli](https://www.swebench.com/) (or in-process when mini-swe-agent / swe-rex expose an API).

**1. Data source**

Tasks come from SWE-bench–style datasets on HuggingFace (same format mini-swe-agent uses):

- **Default:** [princeton-nlp/SWE-bench_Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) — 300 test instances.
- **Alternative:** [princeton-nlp/SWE-bench_Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) for verified subset.

By default the adapter loads **princeton-nlp/SWE-bench_Lite**, split **test**. Optional: **cache_dir** with `task_0.json`, … (SWE-bench schema) and **use_cache_only** to read only from disk.

**2. Cache format (optional local cache)**

If you use **cache_dir**, each file is **`task_{task_id}.json`** with SWE-bench row shape:

- **instance_id**, **repo**, **base_commit**, **problem_statement**, **FAIL_TO_PASS**, **PASS_TO_PASS**, **environment_setup_commit**, **version**, etc.

**3. Training**

1. **Install:** `pip install datasets mini-swe-agent` (optional: `mini-swe-agent[full]` for swe-rex).
2. **Optional:** Set **task_id_range** (e.g. `[0, 99]`) or **task_ids**; **SWE_SYNTH_DATASET_NAME** (e.g. `princeton-nlp/SWE-bench_Verified`).
3. **Run:** `bash train/scripts/swe-synth-ppo-trainer.sh`

**4. Reward and evaluation**

Reward is **1.0** if the model’s patch passes required tests, **0.0** otherwise. In-process reward uses mini-swe-agent / swe-rex when available; otherwise use:

- **Batch inference:** `mini-extra swebench --model <your-model> --subset verified --split test --workers 4`
- **Evaluation:** `sb-cli submit swe-bench_verified test --predictions_path preds.json --run_id <id>` (see [SWE-bench evaluation](https://swebench.com/SWE-bench/guides/evaluation/)).

**5. Adapter config summary**

- **dataset_name**: HuggingFace dataset (default `princeton-nlp/SWE-bench_Lite`).
- **split**: Dataset split (default `test`).
- **task_ids** or **task_id_range**: which task indices to use (default first 100).
- **cache_dir**: optional; local dir with `task_*.json` (SWE-bench schema).
- **use_cache_only**: if true and cache_dir set, load only from cache (no HF).

The adapter is in `train/tools/swe_synth_adapter.py` and is registered as env `swe-synth` (or `swe_synth` / `swesynth`).

#### Merging PPO checkpoints

After training, merge the actor checkpoint to a single HuggingFace model. The merged model keeps its **own** `config.json` (from the checkpoint); it is not overwritten with the base/agent model config.

- **Scripts**: `train/scripts/merge.sh` and `train/scripts/merge-ppo.sh` merge the PPO actor to a single HuggingFace model. Use `CHECKPOINT_BASE` when you have an experiment subdir:  
  `CHECKPOINT_BASE=train/artifacts/RL/checkpoints/openspiel-20250202-grpo bash train/scripts/merge.sh`
- **LoRA checkpoints** (default): The merger may write a LoRA adapter under `TARGET_DIR/lora_adapter/`. Use `train/tools/merge_adapter_into_base.py` to merge that adapter into the base and save a full model.

#### Merged model is 4.4B instead of 4B / pretrained envs don't work with new model

If you train with LoRA on a 4B base and the **merged** model becomes **4.4B** (e.g. `lm_head.weight` appears as an extra or resized layer in the second shard), or **pretrained envs** (other games/tasks) perform poorly with the new model, the cause is usually LoRA being applied to **lm_head** (and sometimes **embed_tokens**). That can change output/vocab behavior and break compatibility with the base model.

**Fix:** The OpenSpiel and SWE-synth trainer scripts now **exclude** `lm_head` and `embed_tokens` from LoRA by default (`exclude_modules=[lm_head,embed_tokens]`). The merged model then stays the same size as the base (4B) and remains compatible with pretrained envs.

- **Retrain** with the updated script (no extra env vars needed; exclude is default).
- To **include** lm_head in LoRA again (e.g. for task-specific output tuning): run with `LORA_EXCLUDE_MODULES="" bash train/scripts/openspiel-ppo-trainer.sh`.
- To exclude only lm_head: `LORA_EXCLUDE_MODULES="[lm_head]" bash train/scripts/openspiel-ppo-trainer.sh`.

#### ValueError: "There is no module or parameter named 'block' in Qwen3ForCausalLM"

This occurs when **vLLM** (used for rollout) loads a Qwen3-based model whose checkpoint uses different layer names than vLLM expects. Alternatives to load the same checkpoints:

- **1. Use SGLang for rollout (different loader):** Use the VERL image that includes SGLang (e.g. in `train/docker_run_verl.sh` set `IMAGE="verlai/verl:app-verl0.6-transformers4.56.1-sglang0.5.2-mcore0.13.0-te2.2"` or the latest sglang image), then run with `ROLLOUT_BACKEND=sglang bash train/scripts/openspiel-ppo-trainer.sh`. SGLang may load your Qwen3 checkpoint correctly.
- **2. Upgrade vLLM:** Use a VERL Docker image with vLLM ≥ 0.8.5 (or whatever version adds support for your Qwen3 layout). Keep `ROLLOUT_BACKEND=vllm` (default).
- **3. Use a Qwen2 base:** Set `AGENT_MODEL_REPO_ID=Qwen/Qwen2.5-4B-Instruct` (or another Qwen2 model) in `.env` so rollout uses a vLLM-compatible model until your stack supports Qwen3.
- **4. Patch vLLM:** In the container or `train/verl` submodule, fix the Qwen3 loader so it maps `block` to the checkpoint’s layer names (e.g. `model.layers`).

#### ImportError: flash_attn undefined symbol (e.g. `_ZN3c104cuda9SetDeviceE...`)

This is an **ABI mismatch**: the installed `flash_attn` wheel was built for a different PyTorch or CUDA version than the one in your container. VERL/SGLang (or their dependencies) load flash_attn and fail.

- **Fix inside the container:** Reinstall flash-attn so it matches the current PyTorch and CUDA:
  1. Check versions: `python3 -c "import torch; print(torch.__version__, torch.version.cuda)"` and `nvidia-smi`.
  2. Uninstall and reinstall: `pip uninstall -y flash-attn && pip install flash-attn --no-build-isolation` (builds for your env; can be slow). Or use a [matching prebuilt wheel](https://flashattn.dev/#finder) for your PyTorch/CUDA/Python.
  3. If you recently ran `pip install sglang[all]`, that may have pulled an incompatible flash-attn; reinstalling flash-attn as above after sglang often fixes it.
- **Alternative:** Use a VERL Docker image whose flash-attn, PyTorch, and CUDA are already aligned (e.g. avoid mixing pip-installed sglang with an image built for a different stack).

#### ImportError: cannot import name 'ReleaseMemoryOccupationReqInput' from 'sglang.srt.managers.tokenizer_manager'

An **older VERL submodule** imports this from sglang; newer sglang removed or renamed it. Fix by **updating the VERL submodule** to the latest main (which no longer uses that import):

```bash
# From repo root, on the host (not inside Docker)
git submodule update --init --recursive train/verl
cd train/verl && git fetch origin && git checkout origin/main && cd ../..
git add train/verl && git commit -m "chore: update VERL submodule for SGLang compatibility"  # optional
```

Then rebuild or restart the container so it uses the updated `train/verl` code. If the container mounts the repo at `/workspace/veritas-rl`, run the `git submodule update` and `git checkout` above on the host (where the repo is mounted from), then run the trainer again inside the container.

#### IndexError: tuple index out of range (sglang/srt/patch_torch.py during update_weights)

This occurs during **weight sync** from the training process to the SGLang rollout engine: SGLang’s `patch_torch.py` modifies a tuple during tensor serialization and hits an index that doesn’t exist (often due to a **sglang ↔ PyTorch version mismatch** or a bug in a specific sglang patch).

- **Try in the container:**  
  1. Pin the exact sglang version that matches your VERL image (e.g. the image may expect `sglang==0.5.2` without extras):  
     `pip uninstall -y sglang && pip install sglang[all]==0.5.2 --force-reinstall`  
  2. If you’re on a different PyTorch than the image was built with, either reinstall PyTorch to match the image or try a different sglang version (e.g. `pip install sglang[all]==0.5.2.post1` if available, or a newer 0.5.x if the fix is there).  
  3. Check [sglang releases](https://github.com/sgl-project/sglang/releases) and [VERL issues](https://github.com/volcengine/verl/issues) for “patch_torch” or “update_weights” tuple/index errors.
- **Workaround:** If you’re blocked, use **vLLM for rollout** instead of SGLang: in `train/docker_run_verl.sh` switch to the vLLM image, then run with `ROLLOUT_BACKEND=vllm bash train/scripts/openspiel-ppo-trainer.sh`. You may need a vLLM version that supports your model (e.g. Qwen3) or use a Qwen2 base model.

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

