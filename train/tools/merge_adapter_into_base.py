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
import json
from pathlib import Path


def get_base_model_from_adapter_config(adapter_model: str) -> str | None:
    """Read adapter_config.json and return base_model_name_or_path if present."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(adapter_model, "adapter_config.json")
        with open(path) as f:
            config = json.load(f)
        return config.get("base_model_name_or_path") or config.get("base_model")
    except Exception:
        pass
    # Local path
    if Path(adapter_model).is_dir():
        config_path = Path(adapter_model) / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            return config.get("base_model_name_or_path") or config.get("base_model")
    return None


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
    parser.add_argument(
        "--use_base_model",
        action="store_true",
        help="Use --base_model instead of base from adapter_config.json (avoids mismatch when adapter was trained on different size, e.g. 7B).",
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
            # Use base model from adapter config unless --use_base_model (e.g. adapter is 7B but config says 14B)
            if args.use_base_model:
                base_model = args.base_model
                print(f"Using base (--base_model): {base_model}")
            else:
                base_from_config = get_base_model_from_adapter_config(args.adapter_model)
                base_model = base_from_config or args.base_model
                if base_from_config:
                    print(f"Using base from adapter_config.json: {base_model}")
                else:
                    print(f"Using base (--base_model): {base_model}")
            print("Loading base model...")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
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
            print(f"Loading tokenizer from base: {base_model}")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=args.trust_remote_code,
            )
            print(f"Saving merged model to {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        except RuntimeError as e:
            err_str = str(e)
            if "size mismatch" in err_str and "13824" in err_str and "17408" in err_str:
                print(
                    "\nThis adapter was trained on a 7B-size base (intermediate 13824), "
                    "but the base model has 14B-size (intermediate 17408).\n"
                    "Use the 7B base instead, e.g.:\n"
                    "  --base_model Qwen/Qwen2.5-Coder-7B-Instruct\n"
                    "Or: BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct bash train/scripts/merge-swe-synth-adapter.sh\n"
                )
            raise
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
