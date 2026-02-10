# LGC-V2 SFT trainer: fine-tune on parquet (prompt + response).
#
# HOW TO USE
# ----------
# 1. Prerequisites
#    - HF_TOKEN set (e.g. in .env) for model download.
#    - Parquet files with "prompt" and "response" columns (see data_processing/convert_to_parquet.py).
#
# 2. .env (optional) — all params can be overridden here
#    AGENT_MODEL_REPO_ID=Qwen/Qwen2.5-4B-Instruct
#    SFT_TRAIN_FILES=train/parquet/train.parquet,train/parquet/extra.parquet
#    SFT_VAL_FILES=train/parquet/val.parquet
#    SFT_PARQUET_DIR=train/parquet
#    WANDB_PROJECT=my-sft
#    WANDB_EXP=run1
#    SFT_TRAIN_BATCH_SIZE=8
#    SFT_MAX_LENGTH=8192
#    SFT_TOTAL_EPOCHS=10
#    GPU_COUNT=2
#
# 3. Run (from repo root or inside container at /workspace/veritas-rl)
#    bash train/scripts/lgc-v2-SFT-trainer.sh
#
# 4. Multiple train parquets: set SFT_TRAIN_FILES to comma-separated paths (no spaces).
#    SFT_TRAIN_FILES=train/parquet/train1.parquet,train/parquet/train2.parquet bash train/scripts/lgc-v2-SFT-trainer.sh
#
# 5. SWE-synth style (instance_id, messages, model_patch, run_name): point prompt/response to your columns.
#    The trainer needs one string column as prompt and one as response. instance_id and run_name are kept as metadata (ignored for loss).
#    - If you have problem_statement + model_patch:
#      SFT_PROMPT_KEY=problem_statement SFT_RESPONSE_KEY=model_patch
#    - If you have a single "prompt" string column (e.g. built from messages) and model_patch:
#      SFT_PROMPT_KEY=prompt SFT_RESPONSE_KEY=model_patch
#    Parquet can include instance_id, messages, run_name; only the prompt_key and response_key columns are used for training.
#
# 6. Use HuggingFace SWE-Synth SFT dataset directly (converts to parquet on the fly):
#    SFT_HF_DATASET=swesynth/SWE-Synth_Moatless-SFT-Trajectories SFT_VAL_RATIO=0.1 bash train/scripts/lgc-v2-SFT-trainer.sh
#    This runs data_processing/convert_swe_synth_sft.py and sets SFT_TRAIN_FILES/SFT_VAL_FILES to the output.
#    Optional: SFT_VAL_RATIO=0.1 (default), SFT_PARQUET_DIR=train/parquet.
#
# 7. LoRA mode (fewer trainable params, less overfitting / forgetting):
#    SFT_LORA_RANK=16 SFT_LORA_ALPHA=32 bash train/scripts/lgc-v2-SFT-trainer.sh
#    Optional: SFT_LORA_TARGET_MODULES=all-linear (default) or a list, e.g. [q_proj,v_proj,k_proj,o_proj].
#
# 8. Checkpoint limit: keep only the last N checkpoints to save disk (e.g. SFT_MAX_CKPT_TO_KEEP=3).
#    Unset or null = keep all checkpoints.
#
# See docs/training.md and data_processing/convert_to_parquet.py, data_processing/convert_swe_synth_sft.py.

set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/train/verl:${PYTHONPATH}"
if [ -f "${PROJECT_ROOT}/.env" ]; then
    echo "📋 Loading .env from ${PROJECT_ROOT}/.env"
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Set it in .env or export HF_TOKEN=..."
    exit 1
fi

# ---------------------------------------------------------------------------
# Model: HuggingFace repo_id (script downloads to local for SFT). Set AGENT_MODEL_REPO_ID in .env.
# ---------------------------------------------------------------------------
pure_agent_model_name="${AGENT_MODEL_REPO_ID:-Qwen/Qwen2.5-4B-Instruct}"
agent_model_path="${SFT_MODEL_PATH:-${PROJECT_ROOT}/train/models/${pure_agent_model_name//\//_}}"

# Download base model if not using a local path
if [ -z "$SFT_MODEL_PATH" ]; then
    echo "Downloading model: $pure_agent_model_name -> $agent_model_path"
    hf download "$pure_agent_model_name" --local-dir "$agent_model_path" || true
fi

# ---------------------------------------------------------------------------
# Data: multiple train parquets + val (comma-separated, no spaces)
# Optional: SFT_HF_DATASET → convert HF dataset to parquet and use it
# ---------------------------------------------------------------------------
SFT_PARQUET_DIR="${SFT_PARQUET_DIR:-train/parquet}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.1}"

if [ -n "$SFT_HF_DATASET" ]; then
    echo "Converting HuggingFace dataset: $SFT_HF_DATASET -> ${SFT_PARQUET_DIR}"
    mkdir -p "${PROJECT_ROOT}/${SFT_PARQUET_DIR}"
    # Truncate by char length so tokenized length stays under model max (avoids tokenizer warning in verl)
    _max_len="${SFT_MAX_LENGTH:-8192}"
    _max_chars="$((_max_len * 4))"
    python "${PROJECT_ROOT}/data_processing/convert_swe_synth_sft.py" \
        --dataset "$SFT_HF_DATASET" \
        --split train \
        --output-dir "${PROJECT_ROOT}/${SFT_PARQUET_DIR}" \
        --val-ratio "$SFT_VAL_RATIO" \
        --max-chars-prompt "$_max_chars" \
        --max-chars-response "$_max_chars"
    SFT_TRAIN_FILES="${SFT_PARQUET_DIR}/swe_synth_sft_train.parquet"
    if [ -f "${PROJECT_ROOT}/${SFT_PARQUET_DIR}/swe_synth_sft_val.parquet" ]; then
        SFT_VAL_FILES="${SFT_PARQUET_DIR}/swe_synth_sft_val.parquet"
    else
        SFT_VAL_FILES="${SFT_TRAIN_FILES}"
    fi
    [ -z "$SFT_PROMPT_KEY" ] && export SFT_PROMPT_KEY=prompt
    [ -z "$SFT_RESPONSE_KEY" ] && export SFT_RESPONSE_KEY=response
fi

# Default single train/val if not set
if [ -z "$SFT_TRAIN_FILES" ]; then
    SFT_TRAIN_FILES="${SFT_PARQUET_DIR}/train.parquet"
fi
if [ -z "$SFT_VAL_FILES" ]; then
    SFT_VAL_FILES="${SFT_PARQUET_DIR}/val.parquet"
fi
# Hydra list format: [path1,path2,...]
TRAIN_FILES_ARG="[${SFT_TRAIN_FILES}]"
VAL_FILES_ARG="[${SFT_VAL_FILES}]"

# ---------------------------------------------------------------------------
# W&B: online when WANDB_API_KEY set, else offline
# ---------------------------------------------------------------------------
DATE=$(date +%Y%m%d)
TIME=$(date +%Y%m%d_%H%M%S)
if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_MODE="${WANDB_MODE:-run}"
    echo "W&B: online. Project: ${WANDB_PROJECT:-lgc-v2-sft}"
else
    export WANDB_MODE="${WANDB_MODE:-offline}"
    echo "W&B: offline. Logs under train/artifacts/wandb/"
fi
export WANDB_DIR="${WANDB_DIR:-train/artifacts/wandb/sft-${DATE}}"
export WANDB_PROJECT="${WANDB_PROJECT:-lgc-v2-sft}"
export WANDB_EXP="${WANDB_EXP:-sft-${TIME}}"

# ---------------------------------------------------------------------------
# Training params (override via .env)
# ---------------------------------------------------------------------------
train_batch_size="${SFT_TRAIN_BATCH_SIZE:-8}"
max_length="${SFT_MAX_LENGTH:-8192}"
micro_batch_size_per_gpu="${SFT_MICRO_BATCH_SIZE_PER_GPU:-4}"
truncation_mode="${SFT_TRUNCATION:-right}"
total_epoches="${SFT_TOTAL_EPOCHS:-10}"
gpu_count="${GPU_COUNT:-2}"
prompt_key="${SFT_PROMPT_KEY:-prompt}"
response_key="${SFT_RESPONSE_KEY:-response}"
model_save_dir="${SFT_SAVE_DIR:-train/artifacts/checkpoints}"
optim_lr="${SFT_LR:-2e-6}"
save_freq="${SFT_SAVE_FREQ:-2000}"
test_freq="${SFT_TEST_FREQ:-10000}"
# Keep only the last N checkpoints (saves disk). Unset or null = keep all.
max_ckpt_to_keep="${SFT_MAX_CKPT_TO_KEEP:-null}"

# LoRA: set SFT_LORA_RANK > 0 to enable (e.g. 8, 16, 32). Reduces overfitting and preserves base model.
lora_rank="${SFT_LORA_RANK:-8}"
lora_alpha="${SFT_LORA_ALPHA:-16}"
lora_target_modules="${SFT_LORA_TARGET_MODULES:-all-linear}"

model_save_path="${model_save_dir}/${WANDB_EXP}"
mkdir -p "${model_save_path}"

# ---------------------------------------------------------------------------
# Run SFT
# ---------------------------------------------------------------------------
torchrun --nnodes=1 --nproc_per_node="${gpu_count}" \
    -m verl.trainer.fsdp_sft_trainer \
    hydra.run.dir=train/outputs \
    data.train_files="${TRAIN_FILES_ARG}" \
    data.val_files="${VAL_FILES_ARG}" \
    data.train_batch_size="${train_batch_size}" \
    data.micro_batch_size_per_gpu="${micro_batch_size_per_gpu}" \
    data.prompt_key="${prompt_key}" \
    data.response_key="${response_key}" \
    data.max_length="${max_length}" \
    data.truncation="${truncation_mode}" \
    model.fsdp_config.model_dtype=bf16 \
    model.partial_pretrain="${agent_model_path}" \
    model.lora_rank="${lora_rank}" \
    model.lora_alpha="${lora_alpha}" \
    model.target_modules="${lora_target_modules}" \
    optim.lr="${optim_lr}" \
    optim.betas="[0.9, 0.98]" \
    optim.weight_decay=0.05 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=wsd \
    trainer.default_local_dir="${model_save_path}" \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${WANDB_EXP}" \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs="${total_epoches}" \
    trainer.save_freq="${save_freq}" \
    trainer.test_freq="${test_freq}" \
    trainer.max_ckpt_to_keep="${max_ckpt_to_keep}" \
    "$@"
