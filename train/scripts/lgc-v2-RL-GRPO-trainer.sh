set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Set PYTHONPATH to include verl module
export PYTHONPATH="${PROJECT_ROOT}/train/verl:${PYTHONPATH}"
# Load .env file if it exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f "${PROJECT_ROOT}/.env" ]; then
    echo "📋 Loading .env file from ${PROJECT_ROOT}/.env"
    set -a  # automatically export all variables
    source "${PROJECT_ROOT}/.env"
    set +a
fi

export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# HF_TOKEN must be set as environment variable
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set"
    echo "Please set it with: export HF_TOKEN=your_token_here"
    echo "Or add it to ${PROJECT_ROOT}/.env file"
    exit 1
fi
# Performance optimizations
export OMP_NUM_THREADS=1  # Reduce CPU thread contention
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings
export CUDA_LAUNCH_BLOCKING=0  # Async CUDA operations

# ================================================
# Setting Model Path and Download Model and Dataset
# ================================================
DATASET_PATH="${PWD}/train/tools/dataset-manager.py"
REWARD_FN_PATH="${PWD}/train/tools/reward-manager.py"
pure_agent_model_name="rhuanmatias/trained-top-sn120"
agent_model_path="${PWD}/train/models/${pure_agent_model_name}"

# ================================================
# Setting WANDB
# ================================================
DATE=$(date +%Y%m%d)
TIME=$(date +%Y%m%d_%H%M%S)
# WANDB_API_KEY must be set as environment variable (optional for offline mode)
if [ -z "$WANDB_API_KEY" ] && [ "$WANDB_MODE" != "offline" ]; then
    echo "WARNING: WANDB_API_KEY not set. WandB logging may fail."
    echo "Set it with: export WANDB_API_KEY=your_key_here"
    echo "Or run with: WANDB_MODE=offline"
fi
export WANDB_DIR=train/artifacts/wandb/rhuanmatias/trained-top-sn120-${DATE}
export WANDB_PROJECT=rhuanmatias/trained-top-sn120
export WANDB_EXP=${pure_agent_model_name}-${TIME}-grpo

# ================================================
# Setting Model Save Path
# ================================================
model_save_dir="train/artifacts/RL/checkpoints"
mkdir -p ${model_save_dir}

model_save_path=${model_save_dir}

mkdir -p ${model_save_path}


kl_coef=0.001
policy_learning_rate=6e-6
rollout_sample_num=8  # Reduced from 6 to speed up (fewer generations per prompt)
train_batch_size=32  # Increased from 4 (process more samples per step)
ppo_mini_batch_size=16  # Increased from 2 (larger mini-batches)
ppo_micro_batch_size_per_gpu=4  # Increased from 2 (better GPU utilization)
ppo_inner_epochs=1
gpu_memory_utilization=0.9  # Increased from 0.8 (use more GPU memory)
total_steps=10000
tensor_model_parallel_size=2
gpu_count=4

HYDRA_FULL_ERROR=1 WANDB_MODE=online python3 -m verl.trainer.main_ppo \
    hydra.run.dir=train/outputs \
    algorithm.adv_estimator=grpo \
    data.filter_overlong_prompts=True \
    data.custom_cls.path="$DATASET_PATH" \
    data.custom_cls.name="LGCV2DynamicDataset" \
    data.train_files="[]" \
    data.val_files="[]" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=9182 \
    data.max_response_length=2048 \
    data.dataloader_num_workers=8 \
    actor_rollout_ref.model.path=${agent_model_path} \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.reward_fn_key="data_source" \
    data.shuffle=True \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name="compute_lgc_v2_score" \
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
    trainer.project_name='LGC_V2' \
    trainer.experiment_name="${pure_agent_model_name}-${TIME}-grpo" \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.default_local_dir=${model_save_path} \
    trainer.experiment_name=${WANDB_EXP} \
    trainer.n_gpus_per_node=${gpu_count} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_training_steps=${total_steps} \
    trainer.val_before_train=False $@ 
