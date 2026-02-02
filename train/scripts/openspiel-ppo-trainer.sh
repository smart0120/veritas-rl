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

kl_coef=0.001
policy_learning_rate=6e-6
rollout_sample_num=8
train_batch_size=16
ppo_mini_batch_size=8
ppo_micro_batch_size_per_gpu=4
ppo_inner_epochs=1
# 0.85 fits ~69 GiB free on 79 GiB GPUs; increase to 0.9 only if more memory is free. Override with GPU_MEMORY_UTILIZATION=0.9
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.5}"
total_steps=30
tensor_model_parallel_size=2
gpu_count=2

GAME_TYPES="${GAME_TYPES:-}"
export GAME_TYPES
if [ -n "$GAME_TYPES" ]; then
  echo "🎮 Restricting to game_types: ${GAME_TYPES}"
fi

HYDRA_FULL_ERROR=1 WANDB_MODE=offline python3 -m verl.trainer.main_ppo \
    hydra.run.dir=train/outputs \
    algorithm.adv_estimator=grpo \
    data.filter_overlong_prompts=True \
    data.custom_cls.path="$DATASET_PATH" \
    data.custom_cls.name="OpenSpielDataset" \
    data.train_files="[]" \
    data.val_files="[]" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=9182 \
    data.max_response_length=2048 \
    data.dataloader_num_workers=8 \
    actor_rollout_ref.model.path=${pure_agent_model_name} \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.reward_fn_key="data_source" \
    data.shuffle=True \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name="compute_score" \
    actor_rollout_ref.actor.optim.lr=${policy_learning_rate} \
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
    actor_rollout_ref.rollout.name=vllm \
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
