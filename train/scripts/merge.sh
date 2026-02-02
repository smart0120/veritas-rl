export PYTHONPATH=/workspace/veritas-rl/veritas-rl/train/verl:
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir train/artifacts/RL/checkpoints/global_step_30/ \
    --target_dir train/merged_model