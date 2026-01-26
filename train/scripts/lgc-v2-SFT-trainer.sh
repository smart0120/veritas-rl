set -x
# git clone https://github.com/volcengine/verl.git src/verl

# Load .env file if it exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Set PYTHONPATH to include verl module
export PYTHONPATH="${PROJECT_ROOT}/train/verl:${PYTHONPATH}"
if [ -f "${PROJECT_ROOT}/.env" ]; then
    echo "📋 Loading .env file from ${PROJECT_ROOT}/.env"
    set -a  # automatically export all variables
    source "${PROJECT_ROOT}/.env"
    set +a
fi

export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# HF_TOKEN must be set as environment variable
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set"
    echo "Please set it with: export HF_TOKEN=your_token_here"
    echo "Or add it to ${PROJECT_ROOT}/.env file"
    exit 1
fi


python -m pip install -U flashinfer-python==0.4.1 flashinfer-cubin==0.4.1


# NOTE:
# - This SFT recipe expects parquet rows containing BOTH:
#   - prompt (string) and response (string)
# - If your source data is prompt-only (like LGC-V2 JSONL prompts), use the RL/GRPO script instead.

# ================================================
# Setting Model Path and Download Model and Dataset
# ================================================
pure_agent_model_name="Qwen/Qwen3Guard-Gen-8B"
agent_model_path="${PWD}/train/models/${pure_agent_model_name}"
dataset_path="train/parquet"

hf download $pure_agent_model_name --local-dir $agent_model_path
# ================================================
# Setting WANDB
# ================================================
task_name="lgc-v2"
train_batch_size=8
max_length=2048
total_epoches=1
gpu_count=1

DATE=$(date +%Y%m%d)
TIME=$(date +%Y%m%d_%H%M%S)
# WANDB_API_KEY must be set as environment variable (optional for offline mode)
if [ -z "$WANDB_API_KEY" ] && [ "$WANDB_MODE" != "offline" ]; then
    echo "WARNING: WANDB_API_KEY not set. WandB logging may fail."
    echo "Set it with: export WANDB_API_KEY=your_key_here"
    echo "Or run with: WANDB_MODE=offline"
fi
export WANDB_DIR=train/artifacts/wandb/Qwen3-4B-Instruct-${task_name}-${DATE}
export WANDB_PROJECT=Qwen3-4B-Instruct
export WANDB_EXP=${pure_agent_model_name}-${TIME}-"batch_size-"${train_batch_size}_"max_length-"${max_length}
# ================================================
# Setting Model Save Path
# ================================================
model_save_dir="train/artifacts/checkpoints"
mkdir -p ${model_save_dir}
model_save_path=${model_save_dir}/${WANDB_EXP}
mkdir -p ${model_save_path}


# ================================================
# Start Training
# ================================================
WANDB_MODE=online torchrun --nnodes=1 --nproc_per_node=${gpu_count} \
     -m verl.trainer.fsdp_sft_trainer \
    hydra.run.dir=train/outputs \
    data.train_files=${dataset_path}/train.parquet \
    data.val_files=${dataset_path}/val.parquet \
    data.train_batch_size=${train_batch_size} \
    data.prompt_key=prompt \
    data.response_key=response \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=${max_length} \
    model.fsdp_config.model_dtype=bf16 \
    model.partial_pretrain=${agent_model_path} \
    optim.lr=2e-6 \
    optim.betas="[0.9, 0.98]" \
    optim.weight_decay=0.05 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=wsd \
    trainer.default_local_dir=${model_save_path} \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXP} \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=${total_epoches} \
    trainer.save_freq=5 \
    trainer.test_freq=5