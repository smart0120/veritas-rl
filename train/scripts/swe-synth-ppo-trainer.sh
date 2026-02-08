# SWE-bench / mini-swe-agent PPO (GRPO) trainer: fixer model on SWE-bench tasks.
#
# Uses [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) and SWE-bench:
# - By default loads from HuggingFace princeton-nlp/SWE-bench_Lite (split test).
# - Optional: SWE_SYNTH_DATASET_NAME (e.g. princeton-nlp/SWE-bench_Verified), SWE_SYNTH_CACHE_DIR.
# - Evaluation: mini-extra swebench + sb-cli, or in-process when mini-swe-agent/swe-rex exposes API.
#
# HOW TO USE
# ----------
# 1. Install: pip install datasets mini-swe-agent (optional: mini-swe-agent[full] for swe-rex).
# 2. Optional: TASK_ID_MIN, TASK_ID_MAX (default 0..99), TASK_IDS_FILE, SWE_SYNTH_DATASET_NAME.
# 3. Run: bash train/scripts/swe-synth-ppo-trainer.sh
# 4. For full metrics: mini-extra swebench, then sb-cli submit (see docs).
#
# See docs/training.md and train/tools/swe_synth_adapter.py.

set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/train/verl:${PROJECT_ROOT}/train/tools:${PYTHONPATH}"
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

DATASET_PATH="${PROJECT_ROOT}/train/tools/dataset-manager.py"
REWARD_FN_PATH="${PROJECT_ROOT}/train/tools/reward-manager.py"
pure_agent_model_name="${AGENT_MODEL_REPO_ID:-Qwen/Qwen2.5-4B-Instruct}"
if [ "${pure_agent_model_name#/}" != "$pure_agent_model_name" ] || [ "${pure_agent_model_name#train/}" != "$pure_agent_model_name" ]; then
    echo "ERROR: actor_rollout_ref.model.path must be a HuggingFace repo_id. Got: $pure_agent_model_name"
    exit 1
fi

# Task set: task_id_range [min, max] inclusive, or task_ids from file
TASK_ID_MIN="${TASK_ID_MIN:-0}"
TASK_ID_MAX="${TASK_ID_MAX:-99}"
# Adapter: task range (default 0..99) and optional cache / dataset
SWE_SYNTH_ADAPTER_OVERRIDES="+data.custom_cls.config.adapter_config.task_id_range=[${TASK_ID_MIN},${TASK_ID_MAX}]"
if [ -n "${SWE_SYNTH_CACHE_DIR:-}" ]; then
    SWE_SYNTH_ADAPTER_OVERRIDES="${SWE_SYNTH_ADAPTER_OVERRIDES} +data.custom_cls.config.adapter_config.cache_dir=${SWE_SYNTH_CACHE_DIR}"
fi
if [ -n "${SWE_SYNTH_DATASET_NAME:-}" ]; then
    SWE_SYNTH_ADAPTER_OVERRIDES="${SWE_SYNTH_ADAPTER_OVERRIDES} +data.custom_cls.config.adapter_config.dataset_name=${SWE_SYNTH_DATASET_NAME}"
fi
if [ -n "${TASK_IDS_FILE:-}" ] && [ -f "$TASK_IDS_FILE" ]; then
    TASK_IDS_STR=$(head -1 "$TASK_IDS_FILE" | tr -d ' \n')
    if [ -n "$TASK_IDS_STR" ]; then
        SWE_SYNTH_ADAPTER_OVERRIDES="+data.custom_cls.config.adapter_config.task_ids=[${TASK_IDS_STR}]"
        [ -n "${SWE_SYNTH_CACHE_DIR:-}" ] && SWE_SYNTH_ADAPTER_OVERRIDES="${SWE_SYNTH_ADAPTER_OVERRIDES} +data.custom_cls.config.adapter_config.cache_dir=${SWE_SYNTH_CACHE_DIR}"
    fi
fi

DATE=$(date +%Y%m%d)
TIME=$(date +%Y%m%d_%H%M%S)
# W&B: set WANDB_API_KEY in .env for online sync; otherwise logs go to train/artifacts/wandb/ (offline).
if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_MODE="${WANDB_MODE:-run}"
    echo "W&B: online (WANDB_API_KEY set). Project: swe-synth-grpo"
else
    export WANDB_MODE="${WANDB_MODE:-offline}"
    echo "W&B: offline (WANDB_API_KEY not set). Logs under train/artifacts/wandb/"
fi
export WANDB_DIR="${WANDB_DIR:-train/artifacts/wandb/swe-synth-${DATE}}"
export WANDB_PROJECT="${WANDB_PROJECT:-swe-synth-grpo}"
export WANDB_EXP="${WANDB_EXP:-swe-synth-${TIME}-grpo}"

model_save_dir="train/artifacts/RL/checkpoints"
mkdir -p "${model_save_dir}"
model_save_path="${model_save_dir}"

kl_coef=0.001
policy_learning_rate=6e-6
policy_learning_rate_lora=6e-5
rollout_sample_num=8
ENABLE_LORA="${ENABLE_LORA:-1}"
lora_rank=32
lora_alpha=32
train_batch_size=16
ppo_mini_batch_size=8
ppo_micro_batch_size_per_gpu=4
ppo_inner_epochs=1
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.85}"
max_prompt_length=8192
max_response_length=4096
total_steps=100
tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
gpu_count="${GPU_COUNT:-1}"

ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"

# Exclude lm_head and embed_tokens so merged model stays same size as base (e.g. 4B); keeps pretrained envs compatible.
LORA_EXCLUDE_MODULES="${LORA_EXCLUDE_MODULES:-[lm_head,embed_tokens]}"
LORA_OVERRIDES=""
if [ "$ENABLE_LORA" = "1" ]; then
    LORA_OVERRIDES="+trainer.strategy=fsdp \
    actor_rollout_ref.model.lora_rank=${lora_rank} \
    actor_rollout_ref.model.lora_alpha=${lora_alpha} \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.exclude_modules=${LORA_EXCLUDE_MODULES} \
    actor_rollout_ref.rollout.load_format=safetensors"
    if [ -n "${LORA_ADAPTER_PATH:-}" ]; then
        LORA_OVERRIDES="${LORA_OVERRIDES} actor_rollout_ref.model.lora_adapter_path=$LORA_ADAPTER_PATH"
    fi
fi
CURRENT_LR="${ENABLE_LORA:+$policy_learning_rate_lora}"
CURRENT_LR="${CURRENT_LR:-$policy_learning_rate}"

# data.custom_cls.name=DynamicEnvDataset with env=swe-synth (adapter_config sets task_ids or task_id_range)
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    hydra.run.dir=train/outputs \
    algorithm.adv_estimator=grpo \
    data.filter_overlong_prompts=True \
    data.custom_cls.path="$DATASET_PATH" \
    data.custom_cls.name=DynamicEnvDataset \
    data.custom_cls.config.env=swe-synth \
    $SWE_SYNTH_ADAPTER_OVERRIDES \
    data.train_files="[]" \
    data.val_files="[]" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.dataloader_num_workers=4 \
    actor_rollout_ref.model.path=${pure_agent_model_name} \
    $LORA_OVERRIDES \
    data.truncation=left \
    data.return_raw_chat=True \
    data.reward_fn_key=data_source \
    data.shuffle=True \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name=compute_score \
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
