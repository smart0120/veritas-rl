# ================================================
# SELECT THE IMAGE
# ================================================
# IMAGE="verlai/verl:ngc-th2.4.0-cu124-vllm0.6.3-te1.7-v0.0.4"
# IMAGE="verlai/verl:app-verl0.6-transformers4.56.1-sglang0.5.2-mcore0.13.0-te2.2"
# IMAGE="verlai/verl:base-verl0.6-cu128-cudnn9.8-torch2.8.0-fa2.7.4"
IMAGE="verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2"

# ================================================
# CHECK FOR .env FILE AND LOAD IT
# ================================================
# Get the veritas-rl directory (parent of train/ directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

if [ -f "$ENV_FILE" ]; then
    echo "📋 Loading environment variables from .env file..."
    ENV_FILE_ARG="--env-file $ENV_FILE"
else
    echo "ℹ️  No .env file found. Using environment variables from shell."
    echo "   Create .env file from env.example if you want to use it."
    ENV_FILE_ARG=""
fi

# ================================================
# RUN THE CONTAINER
# ================================================
# Mount veritas-rl directory to /workspace/veritas-rl
# and set working directory to veritas-rl
# Note: --ulimit nofile/memlock removed to avoid "rlimit type 8: operation not permitted"
# on restricted hosts (e.g. Kubernetes). Add back if your host allows it.
docker run -it \
  $ENV_FILE_ARG \
  --name verl \
  --hostname verl-container \
  --runtime=nvidia \
  --gpus all \
  --shm-size=16g \
  --ipc=host \
  --privileged \
  --network=host \
  -v "${PWD}/..":/workspace/veritas-rl \
  -w /workspace/veritas-rl \
  "$IMAGE" bash -lc 'DATE=$(date +%Y%m%d) \
    && TIME=$(date +%Y%m%d_%H%M%S) \
    && echo "📅 DATE: ${DATE}" \
    && echo "🕒 TIME: ${TIME}" \
    && read -p "📝 Please enter the project name: " PROJECT_NAME \
    && echo "ℹ️  Environment variables loaded from .env file (if present) or shell." \
    && export WANDB_DIR="${PROJECT_NAME}-${DATE}" \
    && export WANDB_PROJECT="$WANDB_DIR" \
    && export WANDB_EXP="${PROJECT_NAME}-${TIME}" \
    && echo "🎉 WANDB Initialized" \
    && echo "📁 WANDB_DIR: ${WANDB_DIR}" \
    && echo "🚀 Ready to run!" \
    && exec bash -l
    '