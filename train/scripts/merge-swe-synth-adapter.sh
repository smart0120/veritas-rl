# Merge an adapter (LoRA or full SFT) into a base model.
#
#
# HOW TO USE
# ----------
# 1. From repo root (or anywhere with Python + transformers):
#    bash train/scripts/merge-swe-synth-adapter.sh
#
# 2. Use a different base model (adapter must have been trained on that base’s architecture):
#    BASE_MODEL=kiwikiw/Affine-0001-5GxTqXLzESa6FThGdcfHANa1b8XmafCshj4yw7PVKwDZuUE2 \
#    ADAPTER_MODEL=<your-adapter-repo> \
#    OUTPUT_DIR=train/merged_models/affine-15b-merged \
#    bash train/scripts/merge-swe-synth-adapter.sh
#
# 3. Optional: custom paths or output dir
#    BASE_MODEL=Qwen/Qwen2.5-Coder-14B-Instruct \
#    OUTPUT_DIR=train/merged_models/my-swe-synth-14b \
#    bash train/scripts/merge-swe-synth-adapter.sh
#
# 4. If adapter is a full SFT checkpoint (not LoRA), script still works: it saves the
#    adapter as the merged model to OUTPUT_DIR.
#
# 5. If you see "size mismatch" with 13824 vs 17408: this adapter was trained on a 7B
#    base (intermediate 13824). Use the 7B base (default) or an adapter trained on your base.
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

# Optional: use Affine 15B as base (set BASE_MODEL and an adapter trained on that architecture)
# AFFINE_15B_BASE="kiwikiw/Affine-0001-5GxTqXLzESa6FThGdcfHANa1b8XmafCshj4yw7PVKwDZuUE2"
# Adapter has LoRA shapes with intermediate 13824 (7B); use 7B base to avoid size mismatch.
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
ADAPTER_MODEL="${ADAPTER_MODEL:-swesynth/Qwen2.5-Coder-14B-Instruct-SFT-3k-Moatless-Traj-SWE-Synth}"
OUTPUT_DIR="${OUTPUT_DIR:-train/merged_models/qwen2.5-coder-7b-swe-synth}"

echo "Base:   $BASE_MODEL"
echo "Adapter: $ADAPTER_MODEL"
echo "Output: $OUTPUT_DIR"

# Use --use_base_model so we use our BASE_MODEL (7B) instead of adapter_config's base (avoids 13824 vs 17408 mismatch)
python3 "${PROJECT_ROOT}/train/tools/merge_adapter_into_base.py" \
    --base_model "$BASE_MODEL" \
    --adapter_model "$ADAPTER_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --trust_remote_code \
    --use_base_model \
    "$@"

echo "Merged model saved to: $OUTPUT_DIR"
