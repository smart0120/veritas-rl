"""
Convert SWE-Synth Moatless SFT Trajectories (HuggingFace) to parquet for SFT training.

Dataset: https://huggingface.co/datasets/swesynth/SWE-Synth_Moatless-SFT-Trajectories
Schema: instance_id, messages (list of {role, content, tool_calls?}), model_patch, run_name.

Output parquet has columns: prompt (string, built from messages), response (from model_patch),
and optionally instance_id, run_name. Use with SFT_PROMPT_KEY=prompt, SFT_RESPONSE_KEY=response.

Usage:
  python data_processing/convert_swe_synth_sft.py \\
    --dataset swesynth/SWE-Synth_Moatless-SFT-Trajectories \\
    --split train \\
    --output train/parquet/swe_synth_sft.parquet

  # Train/val split (e.g. 90% train, 10% val)
  python data_processing/convert_swe_synth_sft.py \\
    --dataset swesynth/SWE-Synth_Moatless-SFT-Trajectories \\
    --output-dir train/parquet \\
    --val-ratio 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from typing import Any


def _normalize_message(m: Any) -> tuple[str, str]:
    """Extract (role, content) from a message that may be a dict or a list."""
    if isinstance(m, dict):
        role = m.get("role", "unknown")
        content = m.get("content")
    elif isinstance(m, (list, tuple)) and len(m) >= 2:
        role = str(m[0]) if m[0] is not None else "unknown"
        content = m[1]
    elif isinstance(m, (list, tuple)) and len(m) == 1:
        role = "unknown"
        content = m[0]
    else:
        role = "unknown"
        content = m if m is not None else ""
    if content is None:
        content = ""
    if isinstance(content, list):
        content = " ".join(
            (str(block.get("text", "")) for block in content if isinstance(block, dict))
        )
    return role, str(content)


def messages_to_prompt(messages: list[Any]) -> str:
    """Turn a list of chat messages (dicts or lists) into a single prompt string (for single-turn SFT)."""
    parts = []
    for m in messages:
        role, content = _normalize_message(m)
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert SWE-Synth SFT Trajectories (HF) to prompt/response parquet."
    )
    parser.add_argument(
        "--dataset",
        default="swesynth/SWE-Synth_Moatless-SFT-Trajectories",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet path (single file). Ignored if --output-dir is set.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory; write train.parquet and val.parquet when --val-ratio > 0",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Fraction of data for validation (0 = no val file). Used only with --output-dir.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to convert (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("pandas is required: pip install pandas", file=sys.stderr)
        return 1

    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets is required: pip install datasets", file=sys.stderr)
        return 1

    print(f"Loading {args.dataset} split={args.split}...")
    ds = load_dataset(args.dataset, split=args.split, trust_remote_code=True)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    n = len(ds)
    print(f"Converting {n} rows...")

    prompts = []
    responses = []
    instance_ids = []
    run_names = []
    for i in range(n):
        row = ds[i]
        msgs = row.get("messages")
        if not msgs:
            msgs = []
        prompts.append(messages_to_prompt(msgs))
        responses.append(row.get("model_patch") or "")
        instance_ids.append(row.get("instance_id", ""))
        run_names.append(row.get("run_name", ""))

    df = pd.DataFrame({
        "prompt": prompts,
        "response": responses,
        "instance_id": instance_ids,
        "run_name": run_names,
    })

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.val_ratio > 0 and args.val_ratio < 1:
            df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
            n_val = max(1, int(len(df) * args.val_ratio))
            n_train = len(df) - n_val
            train_df = df.iloc[:n_train]
            val_df = df.iloc[n_train:]
            train_path = out_dir / "swe_synth_sft_train.parquet"
            val_path = out_dir / "swe_synth_sft_val.parquet"
            train_df.to_parquet(train_path, index=False)
            val_df.to_parquet(val_path, index=False)
            print(f"Wrote {len(train_df)} train -> {train_path}")
            print(f"Wrote {len(val_df)} val   -> {val_path}")
        else:
            path = out_dir / "swe_synth_sft_train.parquet"
            df.to_parquet(path, index=False)
            print(f"Wrote {len(df)} rows -> {path}")
        return 0

    out_path = args.output or "train/parquet/swe_synth_sft.parquet"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
