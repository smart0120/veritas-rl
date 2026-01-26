"""
Evaluate answers in generated dataset JSONL files using environment adapters.

This script reads all JSONL files from data_processing/{env}/{from}/ folder and evaluates
each answer using the appropriate environment adapter to compute accuracy scores.

Results are automatically saved to:
    data_processing/results/{filename}.json

Usage:
    # Evaluate reasoned datasets (default)
    python data_processing/evaluate_dataset.py --env trace

    # Evaluate generated datasets
    python data_processing/evaluate_dataset.py --env trace --from generated

    # Show detailed per-sample results
    python data_processing/evaluate_dataset.py --env lgc-v2 --verbose
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add train/tools directory to path for imports
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "train" / "tools"))

from env_adapter import get_env_adapter, SimpleChallenge
from validation import (
    validate_file_path,
    validate_directory_path,
)

# Constants
PROGRESS_INTERVAL = 10


def extract_final_answer(answer: str) -> str:
    """
    Extract the final answer from reasoning-enhanced answer field.
    
    If the answer contains "Final Answer:" (English) or "最终答案：" (Chinese) marker, 
    extract only the part after the LAST occurrence of it. Otherwise, return the answer as-is (for backward compatibility).
    
    Args:
        answer: Answer string that may contain reasoning + final answer
    
    Returns:
        str: Just the final answer part
    """
    if not answer:
        return answer
    
    # Look for "Final Answer:" marker (English, case-insensitive)
    # Use rfind to get the LAST occurrence (most recent final answer)
    english_markers = ["Final Answer:", "Final answer:", "final answer:", "FINAL ANSWER:"]
    for marker in english_markers:
        idx = answer.rfind(marker)  # Find last occurrence
        if idx != -1:
            # Extract everything after the last marker
            final = answer[idx + len(marker):].strip()
            # Remove any trailing "Final Answer:" duplicates or extra text
            # Also remove trailing periods or other punctuation that might be added
            final = final.rstrip('.\n')
            # Remove any remaining "Final Answer:" markers that might appear after
            while any(final.startswith(m) for m in english_markers):
                final = final.split(":", 1)[1].strip() if ":" in final else final
            return final
    
    # Look for "最终答案：" marker (Chinese)
    # Use rfind to get the LAST occurrence
    chinese_markers = ["最终答案：", "最终答案:", "最终答案"]
    for marker in chinese_markers:
        idx = answer.rfind(marker)  # Find last occurrence
        if idx != -1:
            # Extract everything after the last marker
            final = answer[idx + len(marker):].strip()
            # Remove trailing punctuation
            final = final.rstrip('。\n')
            # Remove any remaining Chinese markers
            while any(final.startswith(m) for m in chinese_markers):
                final = final.split("：", 1)[1].strip() if "：" in final else final.split(":", 1)[1].strip() if ":" in final else final
            return final
    
    # No marker found, return as-is (original format)
    return answer


@dataclass
class EvaluationResult:
    """Result of evaluating a single sample."""
    task_id: int
    task_type: Optional[str] = None
    score: float = 0.0
    correct: bool = False
    error: Optional[str] = None


@dataclass
class EvaluationStats:
    """Aggregated evaluation statistics."""
    environment: str
    input_file: str
    total_samples: int
    correct: int
    incorrect: int
    accuracy: float
    error_count: int
    task_type_accuracy: Optional[Dict[str, Dict[str, Any]]] = None
    results: Optional[List[Dict[str, Any]]] = None


async def evaluate_lgc_v2_sample(
    adapter: Any,
    sample: Dict[str, Any],
    verbose: bool = False
) -> EvaluationResult:
    """
    Evaluate a single LGC-V2 sample.
    
    Args:
        adapter: LGC-V2 adapter instance
        sample: Sample dict with prompt, answer, task_id, etc.
        verbose: If True, print detailed information
    
    Returns:
        EvaluationResult: Evaluation result with score and metadata
    """
    task_id = sample.get("task_id")
    task_type = sample.get("task_type", "unknown")
    answer = sample.get("answer", "")
    
    # Extract final answer if reasoning is present
    final_answer = extract_final_answer(answer)
    
    try:
        # Reconstruct challenge from stored data
        logic_task = adapter._task
        ch = await logic_task.generate(task_id=task_id)
        
        # Create SimpleChallenge for evaluation
        challenge = SimpleChallenge(
            env=adapter.env_name,
            prompt=ch.prompt,
            extra=dict(ch.extra)
        )
        
        # Evaluate the final answer (not the reasoning)
        score = await logic_task.evaluate(final_answer, ch)
        score = float(score)
        
        if verbose:
            status = "✓" if score > 0 else "✗"
            print(f"Task {task_id} ({task_type}): {status} (score={score})")
        
        return EvaluationResult(
            task_id=task_id,
            task_type=task_type,
            score=score,
            correct=score > 0,
            error=None
        )
    except Exception as e:
        error_msg = str(e)
        if verbose:
            print(f"Task {task_id} ({task_type}): ERROR - {error_msg}")
        return EvaluationResult(
            task_id=task_id,
            task_type=task_type,
            score=0.0,
            correct=False,
            error=error_msg
        )


async def evaluate_trace_sample(
    adapter: Any,
    sample: Dict[str, Any],
    verbose: bool = False
) -> EvaluationResult:
    """
    Evaluate a single Trace sample.
    
    Args:
        adapter: Trace adapter instance
        sample: Sample dict with prompt, answer, task_id, etc.
        verbose: If True, print detailed information
    
    Returns:
        EvaluationResult: Evaluation result with score and metadata
    """
    task_id = sample.get("task_id")
    prompt = sample.get("prompt", "")
    answer = sample.get("answer", "")
    
    # Extract final answer if reasoning is present
    final_answer = extract_final_answer(answer)
    
    # Get ground_truth for challenge_extra
    ground_truth = sample.get("ground_truth", final_answer)
    
    challenge_extra = {
        "ground_truth": ground_truth,
        "transformed_code": sample.get("transformed_code", ""),
        "inputs": sample.get("inputs", ""),
        "seed": sample.get("seed"),
        "dataset_index": sample.get("dataset_index"),
        "task_id": task_id
    }
    
    try:
        # Create challenge from stored data
        challenge = SimpleChallenge(
            env=adapter.env_name,
            prompt=prompt,
            extra=challenge_extra
        )
        
        # Evaluate using adapter
        trace_task = adapter._task
        from models import Challenge as TraceChallenge  # type: ignore
        
        ch = TraceChallenge(env="trace", prompt=prompt, extra=challenge_extra)
        score, _test_result = await trace_task.evaluate(final_answer, ch)
        score = float(score)
        
        if verbose:
            status = "✓" if score > 0 else "✗"
            print(f"Task {task_id}: {status} (score={score})")
        
        return EvaluationResult(
            task_id=task_id,
            score=score,
            correct=score > 0,
            error=None
        )
    except Exception as e:
        error_msg = str(e)
        if verbose:
            print(f"Task {task_id}: ERROR - {error_msg}")
        return EvaluationResult(
            task_id=task_id,
            score=0.0,
            correct=False,
            error=error_msg
        )


def _load_samples(input_file: Path) -> List[Dict[str, Any]]:
    """Load samples from JSONL file."""
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(
                    f"Warning: Skipping invalid JSON at line {line_no}: {e}",
                    file=sys.stderr
                )
                continue
    
    if not samples:
        raise ValueError(f"No valid samples found in {input_file}")
    
    return samples


def _compute_task_type_stats(
    results: List[EvaluationResult],
    env: str
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Compute per-task-type statistics (LGC-V2 only)."""
    if env != "lgc-v2":
        return None
    
    task_type_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for result in results:
        if result.task_type:
            task_type_stats[result.task_type]["total"] += 1
            if result.correct:
                task_type_stats[result.task_type]["correct"] += 1
    
    return {
        task_type: {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": (
                stats["correct"] / stats["total"] 
                if stats["total"] > 0 else 0.0
            )
        }
        for task_type, stats in task_type_stats.items()
    }


async def evaluate_dataset(
    input_file: Path,
    env: str,
    adapter_config: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> EvaluationStats:
    """
    Evaluate all samples in a dataset file.
    
    Args:
        input_file: Path to input JSONL file
        env: Environment name ("lgc-v2" or "trace")
        adapter_config: Optional adapter configuration
        verbose: If True, print detailed progress
    
    Returns:
        EvaluationStats: Evaluation results with statistics
    """
    adapter = get_env_adapter(env, config=adapter_config or {})
    
    # Load samples
    samples = _load_samples(input_file)
    print(f"Evaluating {len(samples)} samples from {input_file}...")
    
    # Evaluate each sample
    results: List[EvaluationResult] = []
    for i, sample in enumerate(samples):
        if verbose and (i + 1) % PROGRESS_INTERVAL == 0:
            print(f"Progress: {i + 1}/{len(samples)}", file=sys.stderr)
        
        if env == "lgc-v2":
            result = await evaluate_lgc_v2_sample(adapter, sample, verbose)
        elif env == "trace":
            result = await evaluate_trace_sample(adapter, sample, verbose)
        else:
            raise ValueError(f"Unknown environment: {env}")
        
        results.append(result)
    
    # Compute statistics
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / total if total > 0 else 0.0
    error_count = sum(1 for r in results if r.error is not None)
    
    # Per-task-type statistics
    task_type_accuracy = _compute_task_type_stats(results, env)
    
    # Convert results to dict for JSON serialization
    results_dict = [
        {
            "task_id": r.task_id,
            "task_type": r.task_type,
            "score": r.score,
            "correct": r.correct,
            "error": r.error
        }
        for r in results
    ] if verbose else None
    
    return EvaluationStats(
        environment=env,
        input_file=str(input_file),
        total_samples=total,
        correct=correct,
        incorrect=total - correct,
        accuracy=accuracy,
        error_count=error_count,
        task_type_accuracy=task_type_accuracy,
        results=results_dict
    )


def _print_summary(stats: EvaluationStats) -> None:
    """Print evaluation summary."""
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Environment: {stats.environment}")
    print(f"Total Samples: {stats.total_samples}")
    print(f"Correct: {stats.correct}")
    print(f"Incorrect: {stats.incorrect}")
    print(f"Accuracy: {stats.accuracy:.2%}")
    print(f"Errors: {stats.error_count}")
    
    if stats.task_type_accuracy:
        print("\nPer-Task-Type Accuracy:")
        for task_type, task_stats in sorted(stats.task_type_accuracy.items()):
            print(
                f"  {task_type}: {task_stats['correct']}/{task_stats['total']} "
                f"({task_stats['accuracy']:.2%})"
            )


def _stats_to_dict(stats: EvaluationStats) -> Dict[str, Any]:
    """Convert EvaluationStats to dictionary for JSON serialization."""
    return {
        "environment": stats.environment,
        "input_file": stats.input_file,
        "total_samples": stats.total_samples,
        "correct": stats.correct,
        "incorrect": stats.incorrect,
        "accuracy": stats.accuracy,
        "error_count": stats.error_count,
        "task_type_accuracy": stats.task_type_accuracy,
        "results": stats.results
    }


async def main() -> None:
    """Main entry point for dataset evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate answers in dataset JSONL files"
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=["lgc-v2", "trace"],
        help="Environment type"
    )
    parser.add_argument(
        "--from",
        type=str,
        default="reasoned",
        choices=["generated", "reasoned"],
        dest="from_dir",
        help="Source directory: 'generated' or 'reasoned' (default: reasoned)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-sample results"
    )
    parser.add_argument(
        "--adapter-config",
        type=str,
        default=None,
        help="JSON string for adapter configuration"
    )
    
    args = parser.parse_args()
    
    # Parse adapter config
    adapter_config: Optional[Dict[str, Any]] = None
    if args.adapter_config:
        try:
            adapter_config = json.loads(args.adapter_config)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --adapter-config: {e}") from e
    
    # Setup paths
    data_processing_dir = Path(__file__).parent
    input_dir = data_processing_dir / args.env / args.from_dir
    results_dir = data_processing_dir / "results"
    
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}. "
            f"Make sure to generate datasets first using generate_dataset.py"
        )
    
    if not input_dir.is_dir():
        raise ValueError(f"Path exists but is not a directory: {input_dir}")
    
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all input files
    input_files = sorted(input_dir.glob("*.jsonl"))
    
    if not input_files:
        raise ValueError(
            f"No JSONL files found in {input_dir}. "
            f"Make sure to generate datasets first."
        )
    
    print(f"Found {len(input_files)} file(s) to evaluate in {input_dir}")
    
    # Process each file
    total_processed = 0
    total_errors = 0
    
    for input_path in input_files:
        # Output to results directory with same filename but .json extension
        output_path = results_dir / f"{input_path.stem}.json"
        
        # Skip if output already exists
        if output_path.exists():
            print(f"Skipping {input_path.name} (results already exist: {output_path.name})")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {input_path.name}")
        print(f"{'='*60}")
        
        try:
            # Evaluate dataset
            stats = await evaluate_dataset(
                input_path,
                args.env,
                adapter_config,
                args.verbose
            )
            
            # Print summary
            _print_summary(stats)
            
            # Save results
            results_dict = _stats_to_dict(stats)
            output_json = json.dumps(results_dict, indent=2, ensure_ascii=False)
            
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(output_json)
                print(f"\n✓ Results saved to: {output_path}")
                total_processed += 1
            except (OSError, IOError) as e:
                raise RuntimeError(f"Failed to write output file {output_path}: {e}") from e
                
        except Exception as e:
            total_errors += 1
            print(f"Error evaluating {input_path.name}: {e}", file=sys.stderr)
            # Continue with next file
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Files processed: {total_processed}")
    if total_errors > 0:
        print(f"  Files with errors: {total_errors}", file=sys.stderr)
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
