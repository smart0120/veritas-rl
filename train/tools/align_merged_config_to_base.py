#!/usr/bin/env python3
"""
Overwrite a merged model's config (and generation_config) with the base model's config.

Use after merging a PPO actor checkpoint when the checkpoint's saved config reports
a different parameter count than the base (e.g. 4.4B vs 4B). The merger uses the
checkpoint's huggingface/config.json; if that was wrong, the merged model inherits it.
This script replaces the merged output's config with the base model's so the
reported architecture (and param count) matches the base.

The merged weights are unchanged; only config.json (and generation_config.json if
present) are overwritten. Only run this when the checkpoint was trained from the
given base model (same architecture).

Running example
--------------
Merge PPO checkpoint (align runs automatically; param count will match base):
  bash train/scripts/merge-ppo.sh

Standalone: merge first, then fix config:
  bash train/scripts/merge-ppo.sh
  # If you need to re-apply base config only (e.g. different base):
  python train/tools/align_merged_config_to_base.py \
    --base_model Sota26/Affine-38-5HpqTamztoLsVqrHKv1aY4auSQKerdLBKHHTfvgebqGynTeq \
    --target_dir train/merged_model_ppo \
    --trust_remote_code
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Overwrite merged model config with base model config (fix param count)."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model: HuggingFace repo_id or local path (same as used for training).",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Merged model directory (config.json will be overwritten here).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow trust_remote_code when loading config.",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoConfig
    except ImportError:
        raise SystemExit("pip install transformers")

    target_dir = Path(args.target_dir)
    if not target_dir.is_dir():
        raise SystemExit(f"Target dir not found: {target_dir}")

    print(f"Loading config from base: {args.base_model}")
    config = AutoConfig.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )
    config_path = target_dir / "config.json"
    config.to_json_file(config_path)
    print(f"Wrote {config_path} (param count now matches base).")

    # Optionally copy generation_config.json if base has it (local path only to avoid HF dependency)
    base_path = Path(args.base_model)
    if base_path.is_dir():
        gen_src = base_path / "generation_config.json"
        if gen_src.exists():
            out_gen = target_dir / "generation_config.json"
            with open(gen_src) as f:
                gen_cfg = json.load(f)
            with open(out_gen, "w") as f:
                json.dump(gen_cfg, f, indent=2)
            print(f"Wrote {out_gen}")

    print("Done. Merged model config now matches base (param count fixed).")


if __name__ == "__main__":
    main()
