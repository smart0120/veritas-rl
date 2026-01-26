"""
Generate dataset (problems + answers) from environments.

This script can generate datasets from both lgc-v2 and trace environments,
including both the problem (prompt) and the answer/ground truth.

Files are automatically saved in chunks (default: 1000 records) to:
    data_processing/{env_name}/generated/{start_num}-{end_num}.jsonl

Usage:
    # Generate LGC-V2 dataset
    python data_processing/generate_dataset.py --env lgc-v2 --num-samples 1000

    # Generate with higher concurrency for faster generation
    python data_processing/generate_dataset.py --env lgc-v2 --num-samples 1000 --concurrency 8

    # Generate with custom chunk size
    python data_processing/generate_dataset.py --env trace --num-samples 5000 --chunk-size 2000

    # Generate Trace dataset
    python data_processing/generate_dataset.py --env trace --num-samples 1000

    # Generate specific task types from LGC-V2
    python data_processing/generate_dataset.py --env lgc-v2 --task-types dyck_language game_of_24 --num-samples 500
"""

import argparse
import asyncio
import itertools
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

# Add train/tools directory to path for imports
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "train" / "tools"))

from env_adapter import get_env_adapter, SimpleChallenge
from validation import (
    validate_file_path,
    validate_directory_path,
    validate_positive_int,
    validate_non_negative_int,
)

# Constants
MAX_RETRIES = 5
PROGRESS_INTERVAL = 100
DEFAULT_CONCURRENCY = 4  # Default number of concurrent generations
CHUNK_SIZE = 1000  # Number of samples per file


@dataclass
class GenerationConfig:
    """Configuration for dataset generation."""
    env: str
    num_samples: int
    seed: int
    task_types: Optional[List[str]] = None
    start_task_id: int = 0
    random_selection: bool = False
    adapter_config: Optional[Dict[str, Any]] = None
    concurrency: int = DEFAULT_CONCURRENCY  # Number of concurrent generations
    chunk_size: int = CHUNK_SIZE  # Number of samples per file


def format_answer_for_task_type(answer: str, task_type: str, prompt: str = "") -> str:
    """
    Format answer according to task type requirements.
    
    Different task types require different answer formats:
    - boolean_expressions: \boxed{answer}
    - operation: \boxed{answer}
    - sudoku: ```\nanswer\n``` (triple backticks)
    - dyck_language2: `answer` (single backticks)
    - dyck_language: answer (as is - full sequence)
    - dyck_language_reasoning_errors: answer (as is - comma-separated error indices)
    - game_of_24: ```python\nanswer\n``` (python code block)
    - cryptarithm: "The answer is ..." format
    
    Args:
        answer: Raw answer from game_data
        task_type: Task type name
        prompt: Optional prompt text to check format requirements
    
    Returns:
        str: Formatted answer matching prompt requirements
    """
    task_type_lower = task_type.lower()
    
    # Special handling for game_of_24: empty answer should be formatted as None
    # This must be checked before the early return since game_of_24 always has empty answer by design
    if task_type_lower == "game_of_24":
        if not answer or answer.lower() in ("none", "null", ""):
            return "```python\nNone\n```"
        else:
            return f"```python\n{answer}\n```"
    
    # Early return for empty answers (other task types shouldn't have empty answers)
    if not answer:
        return answer
    
    # Format mapping for cleaner code
    format_map = {
        "boolean_expressions": lambda a: f"\\boxed{{{a}}}",
        "operation": lambda a: f"\\boxed{{{a}}}",
        "sudoku": lambda a: f"```\n{a}\n```",
        "dyck_language2": lambda a: f"`{a}`",
    }
    
    if task_type_lower in format_map:
        return format_map[task_type_lower](answer)
    
    # Special handling for cryptarithm (language detection)
    if task_type_lower == "cryptarithm":
        is_chinese = any(ord(char) > 127 for char in prompt[:100]) if prompt else False
        return f"答案是 {answer}。" if is_chinese else f"The answer is {answer}."
    
    # dyck_language and dyck_language_reasoning_errors: no special formatting needed
    # - dyck_language: full sequence as is
    # - dyck_language_reasoning_errors: comma-separated error indices as is
    return answer


def _parse_game_data(game_data_str: Any) -> Dict[str, Any]:
    """Parse game_data from various formats."""
    if isinstance(game_data_str, str):
        try:
            return json.loads(game_data_str)
        except json.JSONDecodeError:
            return {}
    return game_data_str if isinstance(game_data_str, dict) else {}


def _extract_answer_from_challenge(
    challenge: SimpleChallenge, 
    task_type: str
) -> Tuple[str, str]:
    """
    Extract raw and formatted answer from challenge.
    
    Returns:
        Tuple of (raw_answer, formatted_answer)
    """
    game_data_str = challenge.extra.get("game_data", "{}")
    game_data = _parse_game_data(game_data_str)
    metadata = challenge.extra.get("metadata", {})
    
    # For dyck_language and dyck_language2, use full_sequence from metadata
    if task_type in ("dyck_language", "dyck_language2"):
        raw_answer = metadata.get("full_sequence", game_data.get("answer", ""))
    else:
        raw_answer = game_data.get("answer", "")
    
    formatted_answer = format_answer_for_task_type(
        raw_answer, 
        task_type, 
        challenge.prompt
    )
    
    return raw_answer, formatted_answer


def _build_left_to_right_expression(nums: tuple, ops: tuple) -> str:
    """
    Build an expression with parentheses to enforce left-to-right evaluation.
    
    Example: (1 - 2) + (5 * 12) for nums=(1,2,5,12), ops=('-','+','*')
    But we want: ((1 - 2) + 5) * 12 for left-to-right
    """
    if len(nums) < 2:
        return str(nums[0]) if nums else ""
    
    expr = str(nums[0])
    for i, op in enumerate(ops):
        expr = f"({expr}) {op} {nums[i + 1]}"
    
    return expr


def _generate_game_of_24_solution(numbers: List[int], operators: List[str], target: int) -> Optional[str]:
    """
    Generate a valid solution expression for game_of_24.
    
    Returns a Python expression string that evaluates to the target, or None if no solution found.
    The expression must:
    - Use all numbers exactly once
    - Evaluate correctly using Python's eval() with standard operator precedence
    - Use parentheses to control evaluation order when needed
    """
    for nums in itertools.permutations(numbers):
        for ops in itertools.product(operators, repeat=len(nums) - 1):
            # Try expression with parentheses (left-to-right evaluation)
            expr_with_parens = _build_left_to_right_expression(nums, ops)
            try:
                evaluated = eval(expr_with_parens)
                if abs(float(evaluated) - float(target)) < 1e-10:
                    return expr_with_parens
            except (ZeroDivisionError, ValueError, IndexError):
                pass
            
            # Try expression without parentheses (using operator precedence)
            # Only if it might work (e.g., all same precedence operators)
            try:
                expr_no_parens = "".join([str(nums[0])] + [f" {op} {num}" for op, num in zip(ops, nums[1:])])
                evaluated = eval(expr_no_parens)
                if abs(float(evaluated) - float(target)) < 1e-10:
                    return expr_no_parens
            except (ZeroDivisionError, ValueError, IndexError):
                pass
    
    return None


async def _generate_single_lgc_v2_sample(
    adapter: Any,
    sample_idx: int,
    task_type: str,
    base_id: int,
    max_id: int,
    seed: int,
    semaphore: asyncio.Semaphore
) -> Optional[Dict[str, Any]]:
    """
    Generate a single LGC-V2 sample with retry logic.
    
    Returns:
        Sample dict if successful, None if failed after all retries
    """
    async with semaphore:
        range_width = max_id - base_id + 1
        rng = random.Random(seed)
        logic_task = adapter._task
        
        for retry_count in range(MAX_RETRIES):
            # Use deterministic but varied seeds
            rng.seed(seed + sample_idx + retry_count * 1000)
            task_seed = rng.randint(0, range_width - 1)
            task_id = base_id + task_seed
            
            try:
                # Generate challenge
                ch = await logic_task.generate(task_id=task_id)
                
                # Convert to SimpleChallenge format
                challenge = SimpleChallenge(
                    env=adapter.env_name,
                    prompt=ch.prompt,
                    extra=dict(ch.extra)
                )
                
                # For game_of_24: generate actual solution if solvable
                if task_type == "game_of_24":
                    metadata = challenge.extra.get("metadata", {})
                    is_solvable = metadata.get("is_solvable", False)
                    if not is_solvable:
                        continue  # Skip unsolvable puzzles, try another task_id
                    
                    # Generate a valid solution expression
                    numbers = metadata.get("numbers", [])
                    operators = metadata.get("operators", ["+", "-", "*", "/"])
                    target = metadata.get("target", 24)
                    
                    solution_expr = _generate_game_of_24_solution(numbers, operators, target)
                    if solution_expr is None:
                        continue  # Failed to generate solution, try another task_id
                    
                    # Format as Python code block
                    raw_answer = solution_expr
                    formatted_answer = f"```python\n{solution_expr}\n```"
                else:
                    # Extract answers for other task types
                    raw_answer, formatted_answer = _extract_answer_from_challenge(
                        challenge, task_type
                    )
                    
                    # Skip if answer is missing (shouldn't happen for non-game_of_24)
                    if not raw_answer:
                        continue
                
                return {
                    "task_id": task_id,
                    "task_type": task_type,
                    "prompt": challenge.prompt,
                    "answer": formatted_answer,
                    "raw_answer": raw_answer,
                    "seed": challenge.extra.get("seed"),
                    "metadata": challenge.extra.get("metadata", {}),
                }
                
            except Exception as e:
                if retry_count == MAX_RETRIES - 1:
                    print(
                        f"Error generating task_id {task_id} (task_type={task_type}) "
                        f"after {MAX_RETRIES} retries: {e}",
                        file=sys.stderr
                    )
                continue
        
        return None


async def generate_lgc_v2_samples(
    adapter: Any,
    num_samples: int,
    task_types: Optional[List[str]] = None,
    seed: int = 0,
    concurrency: int = DEFAULT_CONCURRENCY
) -> AsyncIterator[Dict[str, Any]]:
    """
    Generate LGC-V2 samples with problems and answers.
    
    Randomly selects task types for each sample (if multiple task types available).
    
    Args:
        adapter: LGC-V2 adapter instance
        num_samples: Number of samples to generate
        task_types: Optional list of task types to generate.
                    If None, uses all available task types and randomly selects for each sample.
        seed: Random seed for task type selection and task_id generation
    
    Yields:
        dict: Sample with prompt and answer
    """
    from env_adapter import _add_env_to_syspath
    _add_env_to_syspath("lgc-v2")
    from logic_task_v2 import LogicTaskV2
    
    # Determine task types
    if task_types is None:
        task_types = list(LogicTaskV2.SUPPORTED_TASKS.keys())
    
    # Generate task ID ranges for all task types
    task_id_ranges: Dict[str, Tuple[int, int]] = {}
    for task_type in task_types:
        if task_type not in LogicTaskV2.SUPPORTED_TASKS:
            print(f"Warning: Unknown task type '{task_type}', skipping", file=sys.stderr)
            continue
        
        task_info = LogicTaskV2.SUPPORTED_TASKS[task_type]
        task_type_id = task_info["task_type_id"]
        if isinstance(task_type_id, list):
            task_type_id = min(task_type_id)
        
        base_task_id = int(task_type_id) * int(LogicTaskV2.TASK_ID_RANGE)
        task_id_ranges[task_type] = (
            base_task_id, 
            base_task_id + LogicTaskV2.TASK_ID_RANGE - 1
        )
    
    if not task_id_ranges:
        raise ValueError("No valid task types found")
    
    # Initialize random number generator for task type selection
    rng = random.Random(seed)
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)
    
    # Prepare all sample generation tasks and start them concurrently
    tasks = []
    for sample_idx in range(num_samples):
        task_type = rng.choice(list(task_id_ranges.keys()))
        base_id, max_id = task_id_ranges[task_type]
        
        coro = _generate_single_lgc_v2_sample(
            adapter=adapter,
            sample_idx=sample_idx,
            task_type=task_type,
            base_id=base_id,
            max_id=max_id,
            seed=seed,
            semaphore=semaphore
        )
        # Create task to start execution immediately
        task = asyncio.create_task(coro)
        tasks.append((sample_idx, task))
    
    # Wait for tasks to complete and yield results in order
    for sample_idx, task in tasks:
        result = await task
        if result is not None:
            yield result
        else:
            print(
                f"Warning: Failed to generate sample {sample_idx + 1} after {MAX_RETRIES} retries",
                file=sys.stderr
            )


async def _generate_single_trace_sample(
    adapter: Any,
    task_id: int,
    semaphore: asyncio.Semaphore
) -> Optional[Dict[str, Any]]:
    """Generate a single Trace sample."""
    async with semaphore:
        try:
            trace_task = adapter._task
            # Generate challenge
            ch = await trace_task.generate(task_id=task_id)
            
            # Convert to SimpleChallenge format
            challenge = SimpleChallenge(
                env=adapter.env_name,
                prompt=ch.prompt,
                extra=dict(ch.extra)
            )
            
            # Extract ground truth (the expected stdout)
            ground_truth = challenge.extra.get("ground_truth", "")
            
            return {
                "task_id": task_id,
                "prompt": challenge.prompt,
                "answer": ground_truth,
                "transformed_code": challenge.extra.get("transformed_code", ""),
                "inputs": challenge.extra.get("inputs", ""),
                "seed": challenge.extra.get("seed"),
                "dataset_index": challenge.extra.get("dataset_index"),
            }
        except Exception as e:
            print(f"Error generating task_id {task_id}: {e}", file=sys.stderr)
            return None


async def generate_trace_samples(
    adapter: Any,
    num_samples: int,
    start_task_id: int = 0,
    random_selection: bool = False,
    seed: Optional[int] = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Generate Trace samples with problems and answers.
    
    The underlying affinetes TraceTask wraps a HuggingFace dataset. Instead of
    relying on a hard-coded TRACE_DATASET_SIZE constant, we derive the dataset
    length directly from the adapter task. This keeps the generator correct even
    if the dataset, split, or configuration changes.
    
    Args:
        adapter: Trace adapter instance
        num_samples: Number of samples to generate
        start_task_id: Starting task_id (for deterministic generation)
        random_selection: If True, randomly select task_ids within the valid range
        seed: Random seed for random selection (only used if random_selection=True)
        concurrency: Maximum number of concurrent generations
    
    Yields:
        dict: Sample with prompt and ground truth (answer)
    """
    # Get dataset size from the underlying TraceTask
    dataset = adapter._task.dataset  # type: ignore[attr-defined]
    dataset_size = len(dataset)
    max_task_id = dataset_size - 1
    
    # Warn if generating more samples than available unique items
    if num_samples > dataset_size and not random_selection:
        print(
            f"Warning: Generating {num_samples} samples but dataset only has "
            f"{dataset_size} unique items.",
            file=sys.stderr,
        )
        print(
            f"  Samples will wrap around (task_id % {dataset_size}).",
            file=sys.stderr,
        )
        print(
            "  Consider using --random-selection to randomly sample from all available items.",
            file=sys.stderr,
        )
    
    # Initialize random number generator if needed
    rng = None
    if random_selection:
        rng = random.Random(seed if seed is not None else start_task_id)
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)
    
    # Prepare all sample generation tasks and start them concurrently
    tasks = []
    for i in range(num_samples):
        if random_selection:
            assert rng is not None  # for type checkers
            task_id = rng.randint(0, max_task_id)
        else:
            # Sequential generation (wraps around if exceeds dataset size)
            task_id = (start_task_id + i) % dataset_size
        
        coro = _generate_single_trace_sample(adapter, task_id, semaphore)
        # Create task to start execution immediately
        task = asyncio.create_task(coro)
        tasks.append(task)
    
    # Wait for tasks to complete and yield results in order
    for task in tasks:
        result = await task
        if result is not None:
            yield result


async def _generate_samples(config: GenerationConfig) -> AsyncIterator[Dict[str, Any]]:
    """Generate samples based on configuration."""
    adapter = get_env_adapter(config.env, config=config.adapter_config or {})
    
    if config.env == "lgc-v2":
        async for sample in generate_lgc_v2_samples(
            adapter, config.num_samples, config.task_types, config.seed, config.concurrency
        ):
            yield sample
    elif config.env == "trace":
        async for sample in generate_trace_samples(
            adapter,
            config.num_samples,
            config.start_task_id,
            config.random_selection,
            config.seed,
            config.concurrency,
        ):
            yield sample
    else:
        raise ValueError(f"Unsupported environment: {config.env}")


async def main() -> None:
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate dataset from environments")
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=["lgc-v2", "trace"],
        help="Environment to generate from"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=None,
        help="Task types to generate (LGC-V2 only). "
             "If not specified, generates all types with random selection."
    )
    parser.add_argument(
        "--start-task-id",
        type=int,
        default=0,
        help="Starting task_id/seed for deterministic generation "
             "(used as random seed if --seed not provided)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for task type selection. If not provided, uses --start-task-id"
    )
    parser.add_argument(
        "--random-selection",
        action="store_true",
        help=(
            "For Trace env: randomly select task_ids from the full logical ID range "
            "[0, 1,000,000,000) instead of sequential; underlying dataset index is "
            "task_id % len(dataset) (matches PRINT dataset_range in affine config)."
        ),
    )
    parser.add_argument(
        "--adapter-config",
        type=str,
        default=None,
        help="JSON string for adapter configuration "
             "(e.g., '{\"dataset_name\": \"satpalsr/rl-python\"}')"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Number of concurrent generations (default: {DEFAULT_CONCURRENCY}). "
             "Higher values speed up generation but use more resources."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Number of samples per file (default: {CHUNK_SIZE}). "
             "Files are saved in chunks of this size."
    )
    
    args = parser.parse_args()
    
    # Input validation
    validate_positive_int(args.num_samples, "--num-samples")
    validate_non_negative_int(args.start_task_id, "--start-task-id")
    validate_positive_int(args.concurrency, "--concurrency")
    validate_positive_int(args.chunk_size, "--chunk-size")
    
    # Parse adapter config
    adapter_config: Optional[Dict[str, Any]] = None
    if args.adapter_config:
        try:
            adapter_config = json.loads(args.adapter_config)
            if not isinstance(adapter_config, dict):
                raise ValueError("--adapter-config must be a JSON object")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --adapter-config: {e}") from e
    
    # Create configuration
    seed = args.seed if args.seed is not None else args.start_task_id
    config = GenerationConfig(
        env=args.env,
        num_samples=args.num_samples,
        seed=seed,
        task_types=args.task_types,
        start_task_id=args.start_task_id,
        random_selection=args.random_selection,
        adapter_config=adapter_config,
        concurrency=args.concurrency,
        chunk_size=args.chunk_size
    )
    
    # Print generation info
    print(f"Generating {config.num_samples} samples from {config.env} environment...")
    print(f"  Concurrency: {config.concurrency} parallel generations")
    if config.env == "lgc-v2" and config.task_types is None:
        print(f"  Using all task types with random selection (seed={config.seed})")
    elif config.env == "trace":
        if config.random_selection:
            print(
                "  Using random selection from dataset (range determined by adapter dataset size, "
                f"seed={config.seed})"
            )
        else:
            print(
                f"  Using sequential task_ids starting from {config.start_task_id} "
                "(wraps around at dataset_size derived from adapter)"
            )
    
    # Generate and write samples in chunks
    samples_generated = 0
    errors_count = 0
    max_errors = config.num_samples // 10  # Allow up to 10% errors
    
    # Create output directory: data_processing/{env_name}/generated/
    output_dir = Path(__file__).parent / config.env / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    current_file = None
    current_file_samples = 0
    chunk_start_task_id = None  # Track first task_id in current chunk
    chunk_end_task_id = None    # Track last task_id in current chunk
    generated_files = []
    
    def close_current_file():
        """Close current file and rename with correct task_id range."""
        nonlocal current_file, chunk_start_task_id, chunk_end_task_id, generated_files
        if current_file is not None:
            current_file.close()
            if chunk_start_task_id is not None and chunk_end_task_id is not None:
                # Create correct filename with actual task_id range
                correct_path = output_dir / f"{chunk_start_task_id}-{chunk_end_task_id}.jsonl"
                # Find the temporary file we created
                if generated_files:
                    temp_path, _ = generated_files[-1]
                    if temp_path != correct_path and temp_path.exists():
                        temp_path.rename(correct_path)
                        generated_files[-1] = (correct_path, chunk_start_task_id)
            current_file = None
    
    try:
        async for sample in _generate_samples(config):
            try:
                # Get task_id from sample
                task_id = sample.get("task_id", samples_generated)
                
                # Check if we need to start a new file (every chunk_size samples)
                if current_file is None or current_file_samples >= config.chunk_size:
                    # Close previous file if open
                    close_current_file()
                    
                    # Start new chunk - use current task_id as start
                    chunk_start_task_id = task_id
                    chunk_end_task_id = task_id
                    current_file_samples = 0
                    
                    # Create temporary file path (will be renamed with correct end_id when closed)
                    temp_path = output_dir / f"{chunk_start_task_id}-temp.jsonl"
                    current_file = open(temp_path, "w", encoding="utf-8")
                    generated_files.append((temp_path, chunk_start_task_id))
                    print(f"Starting new chunk (task_id {chunk_start_task_id})...", file=sys.stderr)
                
                # Write sample to current file
                current_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                current_file.flush()
                samples_generated += 1
                current_file_samples += 1
                chunk_end_task_id = task_id  # Update end task_id
                
                if samples_generated % PROGRESS_INTERVAL == 0:
                    print(
                        f"Generated {samples_generated}/{config.num_samples} samples...",
                        file=sys.stderr
                    )
                    
            except (OSError, IOError) as e:
                errors_count += 1
                print(
                    f"Error writing sample {samples_generated + 1}: {e}",
                    file=sys.stderr
                )
                if errors_count > max_errors:
                    raise RuntimeError(
                        f"Too many write errors ({errors_count}). "
                        f"Stopping generation. Check disk space and permissions."
                    ) from e
        
        # Close the last file and rename it with correct end task_id
        close_current_file()
            
    except KeyboardInterrupt:
        print(f"\nInterrupted. Generated {samples_generated} samples.", file=sys.stderr)
        close_current_file()
        if samples_generated > 0:
            print(f"Partial dataset saved to {len(generated_files)} file(s) in {output_dir}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        close_current_file()
        print(f"Error during generation: {e}", file=sys.stderr)
        if samples_generated > 0:
            print(f"Partial dataset ({samples_generated} samples) saved to {len(generated_files)} file(s) in {output_dir}", file=sys.stderr)
        raise
    
    print(f"✓ Generated {samples_generated} samples to {len(generated_files)} file(s) in {output_dir}")
    for file_path, _ in generated_files:
        print(f"  - {file_path}")


if __name__ == "__main__":
    asyncio.run(main())
