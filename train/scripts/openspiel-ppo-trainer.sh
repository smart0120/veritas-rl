# OpenSpiel PPO (GRPO) trainer: fixed tasks per game via OpenSpiel.
#
# HOW TO USE
# ----------
# 1. Prerequisites
#    - HF_TOKEN set (e.g. in .env) for model download.
#    - Model: use HuggingFace repo_id only (e.g. rhuanmatias/trained-top-sn120). Optional: set AGENT_MODEL_REPO_ID in .env. Do not use a local path.
#    - GPU: if you see "Free memory ... less than desired GPU memory utilization", set GPU_MEMORY_UTILIZATION=0.85 (default) or lower (e.g. 0.8).
#    - OpenSpiel is installed by this script if missing (pip install open_spiel).
#    - Run inside the VERL Docker container: from repo root run "cd train && ./docker_run_verl.sh", then run this script.
#
# 2. Run (from repo root, or from inside the container at /workspace/veritas-rl)
#    bash train/scripts/openspiel-ppo-trainer.sh
#
# 3. Optional: restrict to specific games via GAME_TYPES (comma-separated). Empty = all games.
#    GAME_TYPES="hex,go,chess" bash train/scripts/openspiel-ppo-trainer.sh
#
# 4. Optional: adapter overrides (use + prefix for Hydra struct)
#    bash train/scripts/openspiel-ppo-trainer.sh \
#      +data.custom_cls.config.adapter_config.num_tasks_per_case=200 \
#      +data.custom_cls.config.adapter_config.max_random_steps=5
#
# 5. LoRA is default (VERL: fsdp/fsdp2 + vllm rollout; PEFT). Only adapter params trained; base frozen.
#    Saves only trainable LoRA adapters; use merge-ppo.sh then merge adapter into base if you want a full model.
#    To full fine-tune (train all params): ENABLE_LORA=0 bash train/scripts/openspiel-ppo-trainer.sh
#    Resume from adapter: LORA_ADAPTER_PATH=/path/to/adapter bash train/scripts/openspiel-ppo-trainer.sh
#    Large model / low GPU: LAYERED_SUMMON=1 (sync LoRA to vLLM per-layer to reduce peak memory).
#
# 6. Rollout and weight updates default is vLLM. Use the vLLM VERL image: train/docker_run_verl.sh
#    (default IMAGE=verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2).
#    To use SGLang instead: set the SGLang image in train/docker_run_verl.sh and run with ROLLOUT_BACKEND=sglang.
#    If you see "no module or parameter named 'block' in Qwen3ForCausalLM", use a Qwen2 base (AGENT_MODEL_REPO_ID=Qwen/Qwen2.5-4B-Instruct) or see docs/training.md.
#
# See docs/training.md and train/tools/openspiel_adapter.py.

set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/train/verl:${PYTHONPATH}"
if [ -f "${PROJECT_ROOT}/.env" ]; then
    echo "📋 Loading .env file from ${PROJECT_ROOT}/.env"
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1
# vLLM attention backend. If you see "flash_attn undefined symbol" (e.g. SetDevice), reinstall flash-attn in the container to match PyTorch/CUDA; see docs/training.md.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set"
    exit 1
fi
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0

DATASET_PATH="${PWD}/train/tools/dataset-manager.py"
REWARD_FN_PATH="${PWD}/train/tools/reward-manager.py"
# Must be a HuggingFace repo_id (namespace/name). Do not use a local path or VERL/vLLM will fail with "Repo id must be in the form 'repo_name' or 'namespace/repo_name'".
# Default: Sota26/Affine-38-... (Qwen3-based). If you see "no module or parameter named 'block' in Qwen3ForCausalLM", upgrade vLLM in the VERL container to >=0.8.5 or use a Qwen2 base (e.g. AGENT_MODEL_REPO_ID=Qwen/Qwen2.5-4B-Instruct).
pure_agent_model_name="${AGENT_MODEL_REPO_ID:-Sota26/Affine-38-5HpqTamztoLsVqrHKv1aY4auSQKerdLBKHHTfvgebqGynTeq}"
if [ "${pure_agent_model_name#/}" != "$pure_agent_model_name" ] || [ "${pure_agent_model_name#train/}" != "$pure_agent_model_name" ]; then
    echo "ERROR: actor_rollout_ref.model.path must be a HuggingFace repo_id (e.g. org/model-name), not a path. Got: $pure_agent_model_name"
    echo "Set AGENT_MODEL_REPO_ID=org/model-name in .env or the environment and run again."
    exit 1
fi

# Ensure OpenSpiel is available (required for openspiel_adapter). Install inside container if missing.
if ! python3 -c "import pyspiel" 2>/dev/null; then
    echo "Installing open_spiel (required for OpenSpiel PPO)..."
    pip install -q open_spiel
fi

DATE=$(date +%Y%m%d)
TIME=$(date +%Y%m%d_%H%M%S)
if [ -z "$WANDB_API_KEY" ] && [ "$WANDB_MODE" != "offline" ]; then
    echo "WARNING: WANDB_API_KEY not set. Use WANDB_MODE=offline for local logging."
fi
export WANDB_DIR=train/artifacts/wandb/openspiel-${DATE}
export WANDB_PROJECT=openspiel-grpo
export WANDB_EXP=openspiel-${TIME}-grpo

model_save_dir="train/artifacts/RL/checkpoints"
mkdir -p ${model_save_dir}
model_save_path=${model_save_dir}

# --- Tuned for: 4B model, pre-trained on 2 envs, training 1 game, 2x H200, ~30 steps (ping-pong) per prompt ---
kl_coef=0.001
policy_learning_rate=6e-6
rollout_sample_num=12
# LoRA default: only train adapter params (base frozen); new env training does not affect pre-trained old env.
ENABLE_LORA="${ENABLE_LORA:-1}"
lora_rank=32
lora_alpha=32
# When LoRA enabled, use higher LR (VERL recommends ~1 order of magnitude)
policy_learning_rate_lora=6e-5
train_batch_size=32
ppo_mini_batch_size=16
ppo_micro_batch_size_per_gpu=8
ppo_inner_epochs=1
# 2x H200 (141GB each): use higher utilization; override with GPU_MEMORY_UTILIZATION if needed
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.85}"
# ~30 user/assistant exchanges per task -> need room for long prompt + response
max_prompt_length=12000
max_response_length=4096
# Single-game training: more steps to converge
total_steps=100
tensor_model_parallel_size=2
gpu_count=2

GAME_TYPES="${GAME_TYPES:-}"
export GAME_TYPES
if [ -n "$GAME_TYPES" ]; then
  echo "🎮 Restricting to game_types: ${GAME_TYPES}"
fi

# Rollout and weight updates backend: vllm (default) or sglang. Use matching VERL image in train/docker_run_verl.sh (vLLM image for vllm).
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"

# Ensure SGLang is installed when using sglang rollout (VERL needs sglang[all]==0.5.2). Install in container if image lacks it.
if [ "$ROLLOUT_BACKEND" = "sglang" ]; then
  if ! python3 -c "import sglang.srt.entrypoints.engine" 2>/dev/null; then
    echo "Installing sglang (required for SGLang rollout; VERL uses sglang[all]==0.5.2)..."
    pip install "sglang[all]==0.5.2" || { echo "ERROR: Failed to install sglang. Use the SGLang VERL image (train/docker_run_verl.sh) or set ROLLOUT_BACKEND=vllm."; exit 1; }
  fi
fi

# LoRA: required for VERL LoRA = strategy fsdp/fsdp2 + rollout.name=vllm/sglang + peft (lora_rank, lora_alpha, load_format=safetensors, target_modules).
# use_shm omitted: only safe for local model paths; this script uses HuggingFace repo_id (VERL would try copy_to_local and fail with FileNotFoundError).
# Optional: layered_summon (large model / low GPU), lora_adapter_path (resume adapter).
LORA_OVERRIDES=""
FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp}"
if [ "$ENABLE_LORA" = "1" ]; then
  echo "🔧 LoRA enabled (default): only training adapter params (rank=${lora_rank}, alpha=${lora_alpha}); base frozen — new env won't affect old"
  LORA_OVERRIDES="+trainer.strategy=${FSDP_STRATEGY} \
    actor_rollout_ref.model.lora_rank=${lora_rank} \
    actor_rollout_ref.model.lora_alpha=${lora_alpha} \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.rollout.load_format=safetensors"
  if [ -n "${LORA_ADAPTER_PATH:-}" ]; then
    echo "📂 Loading pretrained LoRA adapter from: $LORA_ADAPTER_PATH"
    LORA_OVERRIDES="${LORA_OVERRIDES} actor_rollout_ref.model.lora_adapter_path=$LORA_ADAPTER_PATH"
  fi
  if [ "${LAYERED_SUMMON:-0}" = "1" ]; then
    echo "📦 Layered summon enabled (recommended for 70B+ or GPU < 48GB)"
    LORA_OVERRIDES="${LORA_OVERRIDES} actor_rollout_ref.rollout.layered_summon=True"
  fi
else
  echo "⚠️ Full fine-tune (ENABLE_LORA=0): training all params; new env training will affect base model"
fi
# Use LoRA LR when LoRA enabled, else full fine-tune LR
CURRENT_LR="${ENABLE_LORA:+$policy_learning_rate_lora}"
CURRENT_LR="${CURRENT_LR:-$policy_learning_rate}"

HYDRA_FULL_ERROR=1 WANDB_MODE=offline python3 -m verl.trainer.main_ppo \
    hydra.run.dir=train/outputs \
    algorithm.adv_estimator=grpo \
    data.filter_overlong_prompts=True \
    data.custom_cls.path="$DATASET_PATH" \
    data.custom_cls.name="OpenSpielDataset" \
    data.train_files="[]" \
    data.val_files="[]" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.dataloader_num_workers=8 \
    actor_rollout_ref.model.path=${pure_agent_model_name} \
    $LORA_OVERRIDES \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.reward_fn_key="data_source" \
    data.shuffle=True \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name="compute_score" \
    actor_rollout_ref.actor.optim.lr=${CURRENT_LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.rollout.max_model_len=65536 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.actor.kl_loss_coef=${kl_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=${ROLLOUT_BACKEND} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=${rollout_sample_num} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.default_local_dir=${model_save_path} \
    trainer.experiment_name=${WANDB_EXP} \
    trainer.n_gpus_per_node=${gpu_count} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_training_steps=${total_steps} \
    trainer.val_before_train=False "$@"
