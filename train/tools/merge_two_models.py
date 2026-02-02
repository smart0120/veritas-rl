#!/usr/bin/env python3
"""
Merge two full models with the same architecture by averaging their weights (model soup).

Use when both models are the same size (e.g. both 4B or both 15B) and same config
(hidden_size, num_layers, etc.). Output: one model with weights
  merged = alpha * model_a + (1 - alpha) * model_b

Example (50/50 blend):
  python train/tools/merge_two_models.py \
    --model_a Qwen/Qwen2.5-Coder-7B-Instruct \
    --model_b org/other-7B-checkpoint \
    --output_dir train/merged_models/7b-blend \
    --alpha 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Merge two same-architecture models by weight averaging (alpha * A + (1-alpha) * B)."
    )
    parser.add_argument("--model_a", type=str, required=True, help="First model: HF repo_id or local path.")
    parser.add_argument("--model_b", type=str, required=True, help="Second model: HF repo_id or local path.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="train/merged_models/merged_two_models",
        help="Directory to save the merged model.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for model_a. merged = alpha * A + (1-alpha) * B. Default 0.5 (50/50).",
    )
    parser.add_argument("--trust_remote_code", action="store_true", help="Allow trust_remote_code when loading.")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        raise SystemExit("pip install transformers torch")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading model A: {args.model_a}")
    model_a = AutoModelForCausalLM.from_pretrained(
        args.model_a,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Loading model B: {args.model_b}")
    model_b = AutoModelForCausalLM.from_pretrained(
        args.model_b,
        trust_remote_code=args.trust_remote_code,
    )

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    if set(state_a.keys()) != set(state_b.keys()):
        raise RuntimeError(
            "Models have different parameter names; they are not the same architecture. "
            "Merge requires identical keys."
        )

    alpha = args.alpha
    merged_state = {}
    for k in state_a:
        a, b = state_a[k], state_b[k]
        if a.shape != b.shape:
            raise RuntimeError(
                f"Shape mismatch for {k}: A {a.shape} vs B {b.shape}. "
                "Both models must have the same architecture (same shapes)."
            )
        merged_state[k] = (alpha * a + (1 - alpha) * b).to(a.dtype)

    # Load structure from A and set merged state
    model_a.load_state_dict(merged_state, strict=True)
    print(f"Saving merged model (alpha={alpha}) to {out}")
    model_a.save_pretrained(out)

    # Tokenizer from model_a (or could average tokenizer embeddings if needed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_a,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.save_pretrained(out)
    print("Done.")


if __name__ == "__main__":
    main()
