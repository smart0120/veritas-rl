# Merge two full models (same architecture) by averaging weights (model soup).
#
# Both models must have the same config (e.g. both 4B or both 15B, same hidden_size, num_layers).
# Output: merged = alpha * model_a + (1-alpha) * model_b
#
# HOW TO USE
# ----------
#   MODEL_A=path/or/repo_a MODEL_B=path/or/repo_b bash train/scripts/merge-two-models.sh
#
#   # 50/50 blend (default)
#   MODEL_A=Qwen/Qwen2.5-Coder-7B-Instruct MODEL_B=org/my-7b-checkpoint \
#     OUTPUT_DIR=train/merged_models/7b-blend bash train/scripts/merge-two-models.sh
#
#   # 70% A, 30% B
#   MODEL_A=... MODEL_B=... ALPHA=0.7 bash train/scripts/merge-two-models.sh
#
# Requires: pip install transformers torch

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -z "$MODEL_A" ] || [ -z "$MODEL_B" ]; then
    echo "Usage: MODEL_A=<repo_or_path> MODEL_B=<repo_or_path> [ALPHA=0.5] [OUTPUT_DIR=...] bash train/scripts/merge-two-models.sh"
    echo "  ALPHA=0.5 means 50%% model A, 50%% model B. merged = alpha*A + (1-alpha)*B"
    exit 1
fi

ALPHA="${ALPHA:-0.5}"
OUTPUT_DIR="${OUTPUT_DIR:-train/merged_models/merged_two_models}"

python3 "${PROJECT_ROOT}/train/tools/merge_two_models.py" \
    --model_a "$MODEL_A" \
    --model_b "$MODEL_B" \
    --output_dir "$OUTPUT_DIR" \
    --alpha "$ALPHA" \
    --trust_remote_code \
    "$@"

echo "Merged model saved to: $OUTPUT_DIR"
