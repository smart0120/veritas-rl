#!/usr/bin/env python3
"""
Print a model's architecture (config) from Hugging Face repo or local path.

Use to check if two models have the same architecture before merging.
Key fields for compatibility: model_type, hidden_size, num_hidden_layers,
intermediate_size, num_attention_heads, vocab_size.

Example:
  python train/tools/show_model_config.py Qwen/Qwen2.5-Coder-7B-Instruct
  python train/tools/show_model_config.py train/merged_models/my-model
"""

from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Print model config (architecture) from HF repo or local path.")
    parser.add_argument("model", type=str, help="Model: HuggingFace repo_id or local path.")
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Print only this config key (e.g. hidden_size). Default: print all.",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoConfig
    except ImportError:
        raise SystemExit("pip install transformers")

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    d = config.to_dict()

    if args.key:
        if args.key not in d:
            print(f"Key '{args.key}' not in config. Keys: {list(d.keys())}")
            return
        print(d[args.key])
        return

    # Sort so important merge-related keys appear first
    priority = ("model_type", "hidden_size", "num_hidden_layers", "intermediate_size", "num_attention_heads", "vocab_size", "max_position_embeddings")
    rest = [k for k in d if k not in priority]
    for k in priority:
        if k in d:
            print(f"  {k}: {d[k]}")
    for k in sorted(rest):
        print(f"  {k}: {d[k]}")


if __name__ == "__main__":
    main()
