export PYTHONPATH=/workspace/veritas-rl/train/verl:
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir train/artifacts/checkpoints/Qwen/Qwen3Guard-Gen-8B-20260120_210611-batch_size-8_max_length-2048/global_step_5/ \
    --target_dir train/merged_model