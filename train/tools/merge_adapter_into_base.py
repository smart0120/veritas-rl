#!/usr/bin/env python3
"""
Merge an adapter model (LoRA or full SFT) into a base model and save the result.

Use case: merge swesynth/Qwen2.5-Coder-14B-Instruct-SFT-3k-Moatless-Traj-SWE-Synth
into Qwen/Qwen2.5-Coder-14B-Instruct and save a single deployable model.

Adapter repo structure (PEFT/LoRA):
  - adapter_config.json   (PEFT config, e.g. base_model, r, lora_alpha)
  - adapter_model.safetensors  (adapter weights)
  → Script loads base, loads adapter with PEFT, merge_and_unload(), saves.

Full SFT checkpoint (no adapter_config.json):
  → Script loads adapter as full model and saves to output_dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Merge adapter model into base model (or save full adapter as merged)."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-Coder-14B-Instruct",
        help="Base model: HuggingFace repo_id or local path. Used only when adapter is PEFT/LoRA.",
    )
    parser.add_argument(
        "--adapter_model",
        type=str,
        default="swesynth/Qwen2.5-Coder-14B-Instruct-SFT-3k-Moatless-Traj-SWE-Synth",
        help="Adapter model: HuggingFace repo_id or local path (LoRA or full SFT).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="train/merged_models/qwen2.5-coder-14b-swe-synth",
        help="Directory to save the merged model.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow trust_remote_code when loading models.",
    )
    parser.add_argument(
        "--no_peft_merge",
        action="store_true",
        help="Skip PEFT merge: treat adapter as full model and just save it to output_dir.",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise SystemExit("pip install transformers")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_peft_merge:
        # Full model: load adapter and save
        print(f"Loading adapter as full model: {args.adapter_model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.adapter_model,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.adapter_model,
            trust_remote_code=args.trust_remote_code,
        )
        print(f"Saving to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        # Try PEFT merge first; if adapter is not LoRA, load as full model
        try:
            from peft import PeftModel
            print(f"Loading base model: {args.base_model}")
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                trust_remote_code=args.trust_remote_code,
            )
            print(f"Loading adapter: {args.adapter_model}")
            model = PeftModel.from_pretrained(
                model,
                args.adapter_model,
                is_trainable=False,
            )
            print("Merging adapter into base...")
            model = model.merge_and_unload()
            print(f"Loading tokenizer from base: {args.base_model}")
            tokenizer = AutoTokenizer.from_pretrained(
                args.base_model,
                trust_remote_code=args.trust_remote_code,
            )
            print(f"Saving merged model to {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        except Exception as e:
            if "adapter_config" in str(e).lower() or "peft" in str(e).lower():
                raise
            print(f"Adapter is not PEFT/LoRA ({e}), loading as full model...")
            model = AutoModelForCausalLM.from_pretrained(
                args.adapter_model,
                trust_remote_code=args.trust_remote_code,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                args.adapter_model,
                trust_remote_code=args.trust_remote_code,
            )
            print(f"Saving to {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
