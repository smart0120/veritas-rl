# Merge PPO (GRPO) actor checkpoint to a single HuggingFace model.
#
# Same behavior as merge-ppo.sh but with defaults suitable for quick runs.
# The merger expects the *actor* checkpoint dir (global_step_N/actor).
#
# HOW TO USE
# ----------
# 1. Run from repo root (or from VERL container at /workspace/veritas-rl):
#    bash train/scripts/merge.sh
#
# 2. Optional: use same env vars as merge-ppo.sh (CHECKPOINT_BASE, GLOBAL_STEP, TARGET_DIR).
#    CHECKPOINT_BASE=train/artifacts/RL/checkpoints GLOBAL_STEP=30 TARGET_DIR=train/merged_model bash train/scripts/merge.sh
#
# 3. With several envs / experiment subdir:
#    CHECKPOINT_BASE=train/artifacts/RL/checkpoints/openspiel-20250202-grpo bash train/scripts/merge.sh
#
# Requires: run from repo root or container; PYTHONPATH will point to train/verl.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

export PYTHONPATH="${PROJECT_ROOT}/train/verl:${PYTHONPATH}"

CHECKPOINT_BASE="${CHECKPOINT_BASE:-train/artifacts/RL/checkpoints}"
TARGET_DIR="${TARGET_DIR:-train/merged_model}"

# Resolve GLOBAL_STEP: use env, else latest from latest_checkpointed_iteration.txt, else 30.
if [ -n "$GLOBAL_STEP" ]; then
    REQUESTED_STEP="$GLOBAL_STEP"
else
    REQUESTED_STEP=""
fi
LATEST_FILE="${CHECKPOINT_BASE}/latest_checkpointed_iteration.txt"
if [ -f "$LATEST_FILE" ]; then
    LATEST_STEP=$(cat "$LATEST_FILE" | tr -d '[:space:]')
else
    LATEST_STEP="30"
fi
if [ -n "$REQUESTED_STEP" ]; then
    GLOBAL_STEP="$REQUESTED_STEP"
else
    GLOBAL_STEP="$LATEST_STEP"
fi

# Merger expects the *actor* checkpoint dir (contains huggingface/ and shards).
LOCAL_DIR="${CHECKPOINT_BASE}/global_step_${GLOBAL_STEP}/actor"

if [ ! -d "$LOCAL_DIR" ]; then
    echo "ERROR: Actor checkpoint dir not found: $LOCAL_DIR"
    if [ -f "$LATEST_FILE" ]; then
        echo "Latest checkpoint step is: $LATEST_STEP. Use GLOBAL_STEP=$LATEST_STEP or unset GLOBAL_STEP to use latest."
    fi
    echo "Set CHECKPOINT_BASE and/or GLOBAL_STEP (e.g. CHECKPOINT_BASE=train/artifacts/RL/checkpoints GLOBAL_STEP=30)"
    exit 1
fi

mkdir -p "$TARGET_DIR"
echo "Merging PPO actor checkpoint (step $GLOBAL_STEP): $LOCAL_DIR -> $TARGET_DIR"
# Verify actor config (param count / size comes from here; run show_model_config to spot 4.4B vs 4B)
if [ -f "${LOCAL_DIR}/huggingface/config.json" ]; then
    echo "Actor config (param count comes from here):"
    python3 "${PROJECT_ROOT}/train/tools/show_model_config.py" "${LOCAL_DIR}/huggingface" 2>/dev/null || true
fi

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$LOCAL_DIR" \
    --target_dir "$TARGET_DIR"

echo "Done. Merged model saved to: $TARGET_DIR"
