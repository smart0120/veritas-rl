"""
Convert JSON files from results folder into parquet files.

This script targets VERL's RLHF/RL dataset format (see `verl/verl/utils/dataset/README.md`).

Supports multiple dataset types:
  1. Normal datasets from generate_dataset.py (with "answer" field)
  2. Custom JSONL files (with various field names)

Field Detection Priority:
  - For SFT format response field: response > answer > completion > output > assistant > label
  - For RL format ground_truth: ground_truth > answer
  - All original fields are preserved in extra_info for reference

Minimum output columns expected by common VERL pipelines:
  - data_source: str (used to pick reward function when reward_fn_key=data_source)
  - prompt: list[{"role": ..., "content": ...}]  (chat messages for RL) or str (for SFT)
  - reward_model: {"ground_truth": ...}          (required by NaiveRewardManager)
  - extra_info: dict                            (optional metadata; used by custom reward)
  - response: str                                (for SFT format only)

Example 1: Normal Dataset (from generate_dataset.py)
  Input: {"task_id": 500, "task_type": "dyck_language", "prompt": "...", "answer": "...", "metadata": {...}}
  
  RL format output:
    {
      "data_source": "dyck_language",
      "prompt": [{"role":"user","content":"..."}],
      "reward_model": {"ground_truth": "..."},  # from answer field
      "extra_info": {"env":"lgc-v2", "task_id":500, "task_type":"dyck_language", "answer": "...", ...metadata...}
    }
  
  SFT format output:
    {
      "data_source": "dyck_language",
      "prompt": "...",
      "response": "...",  # from answer field
      "extra_info": {"env":"lgc-v2", "task_id":500, "task_type":"dyck_language", "answer": "...", ...}
    }

"""

from __future__ import annotations

# Fix for multiprocess ResourceTracker AttributeError in Python 3.12
# This must be done before any multiprocessing imports
try:
    import multiprocess.resource_tracker as rt
    _orig_stop_locked = rt.ResourceTracker._stop_locked
    
    def _safe_stop_locked(self, *args, **kwargs):
        """Wrapper that suppresses the harmless _recursion_count AttributeError."""
        try:
            return _orig_stop_locked(self, *args, **kwargs)
        except AttributeError as e:
            if "_recursion_count" in str(e):
                # Harmless cleanup bug in Python 3.12 + multiprocess; ignore it
                return
            raise
    
    rt.ResourceTracker._stop_locked = _safe_stop_locked
except (ImportError, AttributeError):
    # If multiprocess isn't available or structure changed, skip patching
    pass

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import datasets
from validation import (
    validate_file_path,
    validate_directory_path,
    validate_non_negative_int,
    validate_ratio,
)

# Constants
DEFAULT_RESPONSE_FIELDS = [
    "response",
    "answer",
    "completion",
    "output",
    "assistant",
    "label",
]

COMMON_METADATA_FIELDS = ("task_id", "task_type", "task_type_id", "seed")
GENERATED_DATASET_FIELDS = ("raw_answer", "transformed_code", "inputs", "dataset_index")
TOP_LEVEL_FIELDS = ("task_id", "task_type", "seed")


def _infer_first_present_field(
    raw: Dict[str, Any],
    candidates: List[str]
) -> Optional[str]:
    """Find first present field from candidates."""
    for k in candidates:
        if k in raw and raw.get(k) is not None:
            return k
    return None


def _iter_input_files(input_path: Path) -> List[Path]:
    """Iterate over input files (single file or directory)."""
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    files: List[Path] = []
    for pattern in ("*.jsonl", "*.json"):
        files.extend(sorted(input_path.glob(pattern)))
    if not files:
        raise FileNotFoundError(f"No .jsonl/.json files found under: {input_path}")
    return files


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Read JSONL file line by line."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_no}: {e}"
                ) from e
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object (dict) at {path}:{line_no}, got {type(obj)}"
                )
            yield obj


def _read_json(path: Path) -> Iterable[Dict[str, Any]]:
    """Read JSON file (single object or array)."""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e
    
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Expected list[dict] in {path}, item {i} is {type(item)}"
                )
            yield item
    elif isinstance(obj, dict):
        yield obj
    else:
        raise ValueError(f"Expected dict or list in {path}, got {type(obj)}")


def _extract_prompt_text(
    prompt_val: Any,
    prompt_field: str,
    raw: Dict[str, Any],
    output_format: str
) -> str:
    """Extract prompt text from various formats."""
    if isinstance(prompt_val, list):
        # If prompt already comes as messages, extract last user message for SFT
        if output_format == "sft":
            user_msgs = [
                m for m in prompt_val
                if isinstance(m, dict) and m.get("role") == "user"
            ]
            return str((user_msgs[-1].get("content") if user_msgs else "") or "")
        return ""
    
    # Try multiple field names
    prompt_text = (
        raw.get(prompt_field) or
        raw.get("question") or
        raw.get("instruction") or
        ""
    )
    return str(prompt_text)


def _extract_ground_truth(raw: Dict[str, Any]) -> str:
    """Extract ground truth from various fields."""
    return (
        raw.get("ground_truth") or
        raw.get("answer") or  # Generated datasets use "answer" field
        ""
    )


def _normalize_value_for_parquet(value: Any) -> Any:
    """
    Normalize a value to be PyArrow/Parquet compatible.
    
    Converts non-serializable types to strings or JSON strings.
    """
    if value is None:
        return None
    elif isinstance(value, (str, int, float)):
        return value
    elif isinstance(value, bool):
        # Keep booleans as-is (PyArrow supports them)
        return value
    elif isinstance(value, (list, tuple)):
        # Recursively normalize list items
        return [_normalize_value_for_parquet(item) for item in value]
    elif isinstance(value, dict):
        # Recursively normalize dict values
        return {k: _normalize_value_for_parquet(v) for k, v in value.items()}
    else:
        # Convert other types to string (e.g., custom objects)
        return str(value)


def _build_extra_info(
    raw: Dict[str, Any],
    env: str
) -> Dict[str, Any]:
    """Build extra_info dict from raw data."""
    extra_info = raw.get("extra_info")
    if not isinstance(extra_info, dict):
        extra_info = {}
    else:
        extra_info = dict(extra_info)
    
    extra_info.setdefault("env", env)
    
    # Copy common fields to extra_info
    for k in COMMON_METADATA_FIELDS:
        if k in raw and k not in extra_info:
            extra_info[k] = _normalize_value_for_parquet(raw[k])
    
    # For generated datasets: preserve metadata and other fields
    if "metadata" in raw and isinstance(raw["metadata"], dict):
        # Merge metadata into extra_info (metadata takes precedence)
        for k, v in raw["metadata"].items():
            if k not in extra_info:
                extra_info[k] = _normalize_value_for_parquet(v)
    
    # Preserve other generated dataset fields (including answer for reference)
    for k in GENERATED_DATASET_FIELDS:
        if k in raw and k not in extra_info:
            extra_info[k] = _normalize_value_for_parquet(raw[k])
    
    # Also preserve answer field if present (for normal datasets, even if used as response)
    if "answer" in raw and "answer" not in extra_info:
        extra_info["answer"] = _normalize_value_for_parquet(raw["answer"])
    
    # Normalize all values in extra_info to ensure Parquet compatibility
    return {k: _normalize_value_for_parquet(v) for k, v in extra_info.items()}


def _normalize_row(
    raw: Dict[str, Any],
    *,
    env: str,
    default_data_source: str = "unknown",
    prompt_field: str = "prompt",
    data_source_field: Optional[str] = None,
    output_format: str = "rl",
    response_field: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Normalize a raw row into VERL-compatible format.
    
    Args:
        raw: Raw input row
        env: Environment name
        default_data_source: Default data_source if not found
        prompt_field: Field name containing prompt
        data_source_field: Optional explicit data_source field
        output_format: "rl" or "sft"
        response_field: Optional explicit response field (SFT only)
    
    Returns:
        Normalized row dict
    """
    # Extract data_source
    if data_source_field:
        data_source = raw.get(data_source_field, default_data_source)
    else:
        data_source = (
            raw.get("data_source") or
            raw.get("task_type") or
            default_data_source
        )
    data_source = str(data_source)

    # Extract prompt
    prompt_val = raw.get(prompt_field)
    prompt_text = _extract_prompt_text(prompt_val, prompt_field, raw, output_format)

    # Extract reward_model ground_truth
    reward_model = raw.get("reward_model")
    if not isinstance(reward_model, dict):
        ground_truth = _extract_ground_truth(raw)
        reward_model = {
            "ground_truth": "" if ground_truth is None else str(ground_truth)
        }
    else:
        reward_model = dict(reward_model)
        reward_model.setdefault("ground_truth", "")

    # Build extra_info
    extra_info = _build_extra_info(raw, env)

    # Build output based on format
    if output_format == "sft":
        if not response_field:
            # Auto-detect common response fields
            response_field = _infer_first_present_field(raw, DEFAULT_RESPONSE_FIELDS)
            if not response_field:
                raise ValueError(
                    "SFT format requires a response field. Either pass --response-field, "
                    "or include one of: " + ", ".join(DEFAULT_RESPONSE_FIELDS) + "."
                )

        response_val = raw.get(response_field)
        if response_val is None:
            raise ValueError(
                f"SFT format: missing response field '{response_field}' in row."
            )
        response_text = str(response_val)

        # Convert extra_info to JSON string for Parquet compatibility
        # PyArrow has issues with nested dicts containing mixed types (bool, str, int, etc.)
        extra_info_json = json.dumps(extra_info, ensure_ascii=False)

        out: Dict[str, Any] = {
            "data_source": data_source,
            "prompt": prompt_text,
            "response": response_text,
            "extra_info": extra_info_json,
        }
    else:
        # RL format expects prompt as chat messages
        messages = (
            prompt_val if isinstance(prompt_val, list)
            else [{"role": "user", "content": prompt_text}]
        )
        # Convert extra_info to JSON string for Parquet compatibility
        # PyArrow has issues with nested dicts containing mixed types (bool, str, int, etc.)
        extra_info_json = json.dumps(extra_info, ensure_ascii=False)
        
        out = {
            "data_source": data_source,
            "prompt": messages,
            "reward_model": reward_model,
            "extra_info": extra_info_json,
        }

    # Preserve some common fields at top-level (helpful for debugging)
    for k in TOP_LEVEL_FIELDS:
        if k in raw and k not in out:
            out[k] = raw[k]

    return out


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point for parquet conversion."""
    p = argparse.ArgumentParser(description="Convert JSON files from results folder to parquet")
    p.add_argument(
        "--out-dir",
        type=str,
        default="parquet",
        help="Output directory for parquet files (default: same as results folder).",
    )
    p.add_argument(
        "--env",
        type=str,
        default="lgc-v2",
        help="Environment name for normalization (e.g. lgc-v2, trace).",
    )
    p.add_argument(
        "--format",
        type=str,
        default="rl",
        choices=["rl", "sft"],
        help="Output format: rl (prompt as chat messages) or sft (prompt/response strings).",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="If >0, split into train/val with this ratio (default: 0.2).",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle before splitting.",
    )

    args = p.parse_args(argv)
    
    # Input validation
    if args.val_ratio > 0:
        validate_ratio(args.val_ratio, "--val-ratio")

    # Setup paths
    data_processing_dir = Path(__file__).parent
    results_dir = data_processing_dir / "results"
    
    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}. "
            f"Make sure to run evaluate_dataset.py first."
        )
    
    if not results_dir.is_dir():
        raise ValueError(f"Path exists but is not a directory: {results_dir}")

    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = data_processing_dir / out_dir
    else:
        # Default: save parquet files in results folder
        out_dir = results_dir
    
    # Create output directory if it doesn't exist
    validate_directory_path(out_dir, must_exist=False, create=True)

    # Find all JSON files
    json_files = sorted(results_dir.glob("*.json"))
    
    if not json_files:
        raise ValueError(
            f"No JSON files found in {results_dir}. "
            f"Make sure to run evaluate_dataset.py first."
        )
    
    print(f"Found {len(json_files)} JSON file(s) to convert in {results_dir}")
    
    # Process each file
    total_processed = 0
    total_errors = 0
    
    for json_file in json_files:
        # Output parquet file with same name
        parquet_file = out_dir / f"{json_file.stem}.parquet"
        
        # Skip if output already exists
        if parquet_file.exists():
            print(f"Skipping {json_file.name} (parquet file already exists: {parquet_file.name})")
            continue
        
        print(f"\n{'='*60}")
        print(f"Converting: {json_file.name} -> {parquet_file.name}")
        print(f"{'='*60}")
        
        try:
            # Read JSON file
            json_data = json.loads(json_file.read_text(encoding="utf-8"))
            
            # Convert to list if it's a single dict
            if isinstance(json_data, dict):
                # For evaluation results, wrap in a list to create a single-row dataset
                rows = [json_data]
            elif isinstance(json_data, list):
                rows = json_data
            else:
                raise ValueError(f"Expected dict or list in {json_file}, got {type(json_data)}")
            
            # Create dataset
            ds = datasets.Dataset.from_list(rows)
            
            # Shuffle if requested
            if args.shuffle:
                ds = ds.shuffle(seed=42)
            
            # Split and save
            if args.val_ratio and args.val_ratio > 0 and len(ds) >= 2:
                split = ds.train_test_split(test_size=args.val_ratio, seed=42)
                train_ds, val_ds = split["train"], split["test"]
                train_path = out_dir / f"{json_file.stem}_train.parquet"
                val_path = out_dir / f"{json_file.stem}_val.parquet"
                
                train_ds.to_parquet(str(train_path))
                val_ds.to_parquet(str(val_path))
                
                print(f"✓ Converted {len(train_ds)} train + {len(val_ds)} val rows to:")
                print(f"    - {train_path}")
                print(f"    - {val_path}")
                
                # Clean up
                del train_ds, val_ds, split
            else:
                # Save single file
                ds.to_parquet(str(parquet_file))
                print(f"✓ Converted {len(rows)} row(s) to {parquet_file}")
            
            total_processed += 1
            
            # Clean up
            del ds
            
        except Exception as e:
            total_errors += 1
            print(f"Error converting {json_file.name}: {e}", file=sys.stderr)
            # Continue with next file
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Files processed: {total_processed}")
    if total_errors > 0:
        print(f"  Files with errors: {total_errors}", file=sys.stderr)
    print(f"{'='*60}")
    
    # Force cleanup before exit
    import gc
    gc.collect()


if __name__ == "__main__":
    import sys
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        raise
