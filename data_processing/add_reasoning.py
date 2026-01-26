"""
Add reasoning to dataset using DeepSeek API with deepseek-reasoner model.

This script reads a dataset JSONL file and generates reasoning for each sample
using DeepSeek API with deepseek-reasoner model. The reasoning is combined with 
the original answer and placed in the "answer" field. The answer field contains: 
reasoning + final answer.

Rate Limit:
DeepSeek API does NOT constrain user's rate limit. We will try our best to serve 
every request.

However, when servers are under high traffic pressure, requests may take time to 
receive a response. During this period:
- HTTP request remains connected
- Non-streaming requests: Continuously return empty lines
- Streaming requests: Continuously return SSE keep-alive comments (: keep-alive)

The OpenAI SDK automatically handles these empty lines and keep-alive comments 
correctly. They do not affect JSON body parsing.

If the request has not started inference after 10 minutes, the server will close 
the connection. No retry logic is implemented - requests wait up to the timeout 
for response.

Usage:
    python data_processing/add_reasoning.py \
      --input data_processing/dataset/swe-synth.jsonl \
      --output data_processing/dataset/swe-synth-with-reasoning.jsonl \
      --timeout 600
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    # dotenv is optional, but recommended
    load_dotenv = None
    import warnings
    warnings.warn(
        "python-dotenv not installed. .env file will not be loaded. "
        "Install with: pip install python-dotenv or activate virtual environment.",
        UserWarning
    )

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: openai package not installed. Install with: pip install openai", file=sys.stderr)
    sys.exit(1)

# Load .env file if it exists (doesn't override existing env vars)
if load_dotenv is not None:
    # Try loading from project root
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        # Fallback to default location (current directory)
        load_dotenv()

from validation import validate_file_path, validate_directory_path

# Constants
MODEL = "deepseek-reasoner"  # Always use deepseek-reasoner model
# Default timeout: 10 minutes (600 seconds)
# DeepSeek will close connection if inference hasn't started after 10 minutes
DEFAULT_TIMEOUT = 600
PROGRESS_INTERVAL = 10
DEEPSEEK_API_BASE = "https://api.deepseek.com"


def detect_language(prompt: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Detect the language of the prompt.
    
    Args:
        prompt: The problem/prompt text
        metadata: Optional metadata dict that may contain language info
    
    Returns:
        str: "chinese" or "english"
    """
    # First check metadata for language field
    if metadata and isinstance(metadata, dict):
        lang = metadata.get("language", "").lower()
        if lang in ("chinese", "zh", "zh-cn", "zh-tw"):
            return "chinese"
        if lang in ("english", "en"):
            return "english"
    
    # Detect by checking for Chinese characters
    # Chinese characters are typically in the range 0x4E00-0x9FFF
    chinese_char_count = sum(1 for char in prompt[:500] if '\u4e00' <= char <= '\u9fff')
    total_chars = len([c for c in prompt[:500] if c.isalnum() or '\u4e00' <= c <= '\u9fff'])
    
    # If more than 10% of characters are Chinese, consider it Chinese
    if total_chars > 0 and chinese_char_count / total_chars > 0.1:
        return "chinese"
    
    # Default to English
    return "english"


def get_deepseek_client() -> AsyncOpenAI:
    """Get DeepSeek API client from environment variable."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        # Check if .env file exists and provide helpful message
        project_root = Path(__file__).parent.parent
        env_file = project_root / ".env"
        env_hint = ""
        if env_file.exists():
            env_hint = (
                f"\nNote: .env file exists at {env_file}, but DEEPSEEK_API_KEY was not loaded. "
                "Make sure to:\n"
                "  1. Activate the virtual environment: source .venv/bin/activate\n"
                "  2. Or install python-dotenv: pip install python-dotenv\n"
                "  3. Or set the environment variable: export DEEPSEEK_API_KEY=your_key_here"
            )
        raise ValueError(
            "DEEPSEEK_API_KEY environment variable is not set. "
            "Set it with: export DEEPSEEK_API_KEY=your_key_here"
            + env_hint
        )
    # DeepSeek uses OpenAI-compatible API with custom base URL
    return AsyncOpenAI(api_key=api_key, base_url=DEEPSEEK_API_BASE)


async def generate_reasoning(
    client: AsyncOpenAI,
    prompt: str,
    original_answer: str,
    timeout: int = DEFAULT_TIMEOUT,
    language: str = "english",
) -> str:
    """
    Generate reasoning using DeepSeek API with deepseek-reasoner model.
    The reasoning will be generated in the same language as the problem.
    
    Rate Limit & Keep-Alive:
        DeepSeek API does NOT enforce rate limits. During high traffic, the HTTP
        connection remains open and may receive:
        - Non-streaming: Empty lines continuously
        - Streaming: SSE keep-alive comments (: keep-alive)
        
        The OpenAI SDK handles these automatically - they don't affect JSON parsing.
        No retry logic is needed as DeepSeek serves all requests.
        
        If inference hasn't started after 10 minutes, the server closes the connection.
    
    Args:
        client: DeepSeek API async client (OpenAI-compatible)
        prompt: The problem/prompt text
        original_answer: The original answer (final solution)
        timeout: Request timeout in seconds (default: 600s = 10 minutes)
        language: Language of the prompt ("chinese" or "english")
    
    Returns:
        str: Generated reasoning text in the same language as the problem
    """
    # Generate prompts in the appropriate language with JSON format instructions
    if language == "chinese":
        system_prompt = """你是一个有用的助手，能够为解决问题提供逐步的推理过程。
你的任务是提供清晰、逻辑严密的推理，引导出给定的答案。
推理应该：
1. 将问题分解为步骤
2. 清楚地解释每个步骤
3. 展示步骤如何导向最终答案
4. 简洁但完整

请以纯文本格式提供推理，最后清楚地标记最终答案。"""
        
        user_prompt = f"""问题：
{prompt}

最终答案：
{original_answer}

请提供逐步推理，解释如何得出这个答案。
推理应该清晰且逻辑严密，逐步展示思考过程。
在推理的最后清楚地说明最终答案。"""
    else:
        # English (default)
        system_prompt = """You are a helpful assistant that explains step-by-step reasoning for solving problems.
Your task is to provide clear, logical reasoning that leads to the given answer.
The reasoning should:
1. Break down the problem into steps
2. Explain each step clearly
3. Show how the steps lead to the final answer
4. Be concise but complete

Format your response as plain text reasoning, ending with the final answer clearly marked."""
        
        user_prompt = f"""Problem:
{prompt}

Final Answer:
{original_answer}

Please provide step-by-step reasoning that explains how to arrive at this answer. 
The reasoning should be clear and logical, showing the thought process step by step.
End your reasoning with a clear statement of the final answer.

Please output your response in JSON format as follows:
{{
    "answer": "Your complete reasoning process and final answer"
}}"""

    # Make single request - no retry logic needed
    # DeepSeek API behavior during high traffic:
    # - Non-streaming: Continuously returns empty lines (handled by OpenAI SDK)
    # - Streaming: Continuously returns SSE keep-alive comments (: keep-alive)
    # - Connection stays open until inference starts or 10 minutes timeout
    # - OpenAI SDK automatically handles empty lines/keep-alive, doesn't affect JSON parsing
    try:
        request_params = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }
        
        response = await asyncio.wait_for(
            client.chat.completions.create(**request_params),
            timeout=timeout,
        )
        
        content = response.choices[0].message.content
        
        if not content:
            raise ValueError("Empty response content from API")
        
        try:
            response_json = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}. Content: {content[:200]}")
        
        # Extract answer field from JSON response
        answer_content = response_json.get("answer", "").strip()
        
        if not answer_content:
            raise ValueError(
                f"Empty 'answer' field in JSON response. "
                f"Available keys: {list(response_json.keys())}"
            )
        
        return answer_content
        
    except asyncio.TimeoutError:
        # DeepSeek closes connection if inference hasn't started after 10 minutes
        raise RuntimeError(
            f"Request timed out after {timeout}s. "
            f"DeepSeek closes connection if inference hasn't started after 10 minutes."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to generate reasoning: {e}") from e


def combine_reasoning_and_answer(reasoning: str, original_answer: str, language: str = "english") -> str:
    """
    Combine reasoning and answer into a single string.
    The reasoning and answer are combined into one cohesive answer field.
    
    Args:
        reasoning: The generated reasoning text
        original_answer: The original answer
        language: Language of the reasoning ("chinese" or "english")
    
    Returns:
        str: Combined text with reasoning followed by answer (all in answer field)
    """
    # Ensure reasoning ends properly
    reasoning = reasoning.rstrip()
    if language == "chinese":
        if not reasoning.endswith(('。', '！', '？', '.', '!', '?')):
            reasoning += "。"
        # Use Chinese separator
        separator = "\n\n最终答案：\n"
    else:
        # English
        if not reasoning.endswith(('.', '!', '?')):
            reasoning += "."
        separator = "\n\nFinal Answer:\n"
    
    # Combine reasoning and answer - both are part of the answer field
    # Format: reasoning text, then clear separator, then final answer
    combined = f"{reasoning}{separator}{original_answer}"
    return combined


async def process_sample(
    client: AsyncOpenAI,
    sample: Dict[str, Any],
    timeout: int,
) -> Dict[str, Any]:
    """
    Process a single sample to add reasoning.
    
    Args:
        client: DeepSeek API async client (OpenAI-compatible)
        sample: Sample dictionary with prompt and answer
        timeout: Request timeout
    
    Returns:
        dict: Sample with updated answer field containing reasoning + answer
    """
    prompt = sample.get("prompt", "")
    original_answer = sample.get("answer", "")
    metadata = sample.get("metadata", {})
    
    if not prompt:
        raise ValueError("Sample missing 'prompt' field")
    if not original_answer:
        raise ValueError("Sample missing 'answer' field")
    
    # Detect language of the prompt
    language = detect_language(prompt, metadata)
    
    # Generate reasoning in the same language as the prompt
    reasoning = await generate_reasoning(
        client, prompt, original_answer, timeout, language
    )
    
    # Combine reasoning and answer - this becomes the new answer field
    enhanced_answer = combine_reasoning_and_answer(reasoning, original_answer, language)
    
    # Create new sample with reasoning+answer in the answer field only
    new_sample = dict(sample)
    new_sample["answer"] = enhanced_answer  # Reasoning + final answer combined
    
    return new_sample


async def process_dataset(
    input_path: Path,
    output_path: Path,
    timeout: int,
    max_concurrent: int = 5,
) -> None:
    """
    Process entire dataset to add reasoning.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        timeout: Request timeout
        max_concurrent: Maximum concurrent requests
    """
    client = get_deepseek_client()
    
    # Load samples
    samples = []
    print(f"Loading samples from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
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
        raise ValueError(f"No valid samples found in {input_path}")
    
    print(f"Processing {len(samples)} samples with model {MODEL}...")
    
    # Process samples with concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    processed = 0
    errors = 0
    
    async def process_with_semaphore(sample: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        nonlocal processed, errors
        async with semaphore:
            try:
                result = await process_sample(client, sample, timeout)
                processed += 1
                if processed % PROGRESS_INTERVAL == 0:
                    print(
                        f"Processed {processed}/{len(samples)} samples...",
                        file=sys.stderr
                    )
                return result
            except Exception as e:
                errors += 1
                print(
                    f"Error processing sample {index} (task_id={sample.get('task_id', 'unknown')}): {e}",
                    file=sys.stderr
                )
                return None
    
    # Process all samples
    tasks = [
        process_with_semaphore(sample, i)
        for i, sample in enumerate(samples)
    ]
    results = await asyncio.gather(*tasks)
    
    # Write results
    print(f"Writing {len([r for r in results if r is not None])} samples to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            if result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"✓ Successfully processed {processed} samples")
    if errors > 0:
        print(f"⚠ {errors} samples failed to process", file=sys.stderr)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Add reasoning to dataset using DeepSeek API with thinking mode"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT}s = 10 minutes). "
             f"DeepSeek closes connection if inference hasn't started after 10 minutes. "
             f"Keep-alive (empty lines/SSE comments) during high traffic is handled automatically by SDK."
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent requests (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    input_path = Path(args.input)
    validate_file_path(input_path, must_exist=True, extensions=(".jsonl", ".json"))
    
    output_path = Path(args.output)
    validate_directory_path(output_path.parent, must_exist=False, create=True)
    
    if args.timeout < 1:
        raise ValueError("--timeout must be at least 1")
    if args.max_concurrent < 1:
        raise ValueError("--max-concurrent must be at least 1")
    
    # Process dataset
    await process_dataset(
        input_path,
        output_path,
        args.timeout,
        args.max_concurrent,
    )


if __name__ == "__main__":
    asyncio.run(main())

