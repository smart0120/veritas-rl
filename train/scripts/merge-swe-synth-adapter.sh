# Merge SWE-Synth adapter into base 14B model.
#
# Adapter: swesynth/Qwen2.5-Coder-14B-Instruct-SFT-3k-Moatless-Traj-SWE-Synth (SFT on SWE-Synth)
# Base:    Qwen/Qwen2.5-Coder-14B-Instruct
#
# HOW TO USE
# ----------
# 1. From repo root (or anywhere with Python + transformers):
#    bash train/scripts/merge-swe-synth-adapter.sh
#
# 2. Optional: custom paths or output dir
#    BASE_MODEL=Qwen/Qwen2.5-Coder-14B-Instruct \
#    ADAPTER_MODEL=swesynth/Qwen2.5-Coder-14B-Instruct-SFT-3k-Moatless-Traj-SWE-Synth \
#    OUTPUT_DIR=train/merged_models/my-swe-synth-14b \
#    bash train/scripts/merge-swe-synth-adapter.sh
#
# 3. If adapter is a full SFT checkpoint (not LoRA), script still works: it saves the
#    adapter as the merged model to OUTPUT_DIR.
#
# Requires: pip install transformers (and pip install peft if adapter is LoRA)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct}"
ADAPTER_MODEL="${ADAPTER_MODEL:-swesynth/Qwen2.5-Coder-14B-Instruct-SFT-3k-Moatless-Traj-SWE-Synth}"
OUTPUT_DIR="${OUTPUT_DIR:-train/merged_models/qwen2.5-coder-14b-swe-synth}"

echo "Base:   $BASE_MODEL"
echo "Adapter: $ADAPTER_MODEL"
echo "Output: $OUTPUT_DIR"

python3 "${PROJECT_ROOT}/train/tools/merge_adapter_into_base.py" \
    --base_model "$BASE_MODEL" \
    --adapter_model "$ADAPTER_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --trust_remote_code \
    "$@"

echo "Merged model saved to: $OUTPUT_DIR"
