"""
Dynamic env dataset for VERL.

Supports single-turn verifiable envs in `tools/env_adapter.py`:
- lgc-v2, trace: use DynamicEnvDataset with config.env=lgc-v2 or trace.
- openspiel: use OpenSpielDataset (or DynamicEnvDataset with config.env=openspiel).

OpenSpielDataset: dedicated dataset for OpenSpiel; forces env=openspiel and uses
adapter_config (cases, num_tasks_per_case, max_random_steps, reward_mode, seed).
"""

import asyncio
import random
import threading
from collections import deque
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from env_adapter import get_env_adapter


class DynamicEnvDataset(Dataset):
    """
    Dynamic dataset that generates env tasks on-the-fly.
    
    This dataset doesn't require pre-generated parquet files. It generates
    challenges from a local env adapter during training.
    
    Compatible with VERL's RLHFDataset interface.
    
    Args:
        data_files: Ignored (for compatibility with VERL)
        tokenizer: Tokenizer for tokenization
        processor: Processor (ignored, for compatibility)
        config: Dataset configuration
        max_samples: Number of samples (for __len__). If -1, infinite dataset
    """
    
    def __init__(
        self,
        data_files=None,  # Ignored - for compatibility with VERL
        tokenizer: PreTrainedTokenizer = None,
        processor=None,  # Ignored - for compatibility
        config: DictConfig = None,
        max_samples: int = -1,
    ):
        self.tokenizer = tokenizer
        self.config = config or DictConfig({})
        self.max_samples = max_samples
        
        # Which env to generate from (default lgc-v2).
        self.env = self.config.get("env", "lgc-v2")
        self.adapter_config = self.config.get("adapter_config", {}) or {}
        self.adapter = get_env_adapter(self.env, config=self.adapter_config)
        # OpenSpiel: fixed number of tasks per game (total = num_games * num_tasks_per_game)
        self._openspiel_total_tasks = None
        if str(self.env).lower() == "openspiel" and hasattr(self.adapter, "get_total_fixed_tasks"):
            self._openspiel_total_tasks = self.adapter.get_total_fixed_tasks()

        # Configuration
        self.prompt_key = self.config.get("prompt_key", "prompt")
        self.max_prompt_length = self.config.get("max_prompt_length", 2048)
        self.return_raw_chat = self.config.get("return_raw_chat", False)
        self.shuffle = self.config.get("shuffle", True)
        self.seed = self.config.get("seed", None)
        
        # Task id generation strategy:
        # - For lgc-v2 we support selecting specific task types via task_types + deterministic ranges.
        # - For other envs, task_id is just `index` (or shuffled by seed) unless overridden.
        self.task_types = self.config.get("task_types", None)
        self.task_id_ranges = None
        if str(self.env).lower() in {"lgc-v2", "lgc_v2", "logic-v2", "logic_v2"}:
            # Late import: only when needed
            from env_adapter import _add_env_to_syspath  # type: ignore

            _add_env_to_syspath("lgc-v2")
            from logic_task_v2 import LogicTaskV2  # type: ignore

            if self.task_types is None:
                self.task_types = list(LogicTaskV2.SUPPORTED_TASKS.keys())
            else:
                self.task_types = self.task_types if isinstance(self.task_types, list) else [self.task_types]

        self.task_id_ranges = {}
        if self.task_types is not None:
            for task_type in self.task_types:
                task_info = LogicTaskV2.SUPPORTED_TASKS[task_type]
                task_type_id = task_info["task_type_id"]
                if isinstance(task_type_id, list):
                    # multi-range task types exist (e.g., dyck_language2). Use the first range.
                    task_type_id = min(task_type_id)
                base_task_id = int(task_type_id) * int(LogicTaskV2.TASK_ID_RANGE)
                self.task_id_ranges[task_type] = (
                    base_task_id,
                    base_task_id + int(LogicTaskV2.TASK_ID_RANGE) - 1,
                )
        
        # Random state for reproducibility
        if self.seed is not None:
            self.rng = random.Random(self.seed)
        else:
            self.rng = random.Random()
        
        # Performance optimizations
        self.cache_size = config.get("cache_size", 1000)  # LRU cache for generated tasks
        self._task_cache = {}
        self._cache_lock = threading.Lock()
        
        # Progress tracking
        self._generation_count = 0
        self._generation_lock = threading.Lock()
        
        # Per-worker event loop (for async generation)
        self._loop = None
        self._loop_lock = threading.Lock()
        
        # Pre-generation pool (optional)
        self.prefetch_size = config.get("prefetch_size", 0)  # 0 = disabled
        self._prefetch_queue = deque(maxlen=self.prefetch_size * 2) if self.prefetch_size > 0 else None
        self._prefetch_thread = None
        if self.prefetch_size > 0:
            self._start_prefetch_thread()
    
    def __len__(self):
        """Return dataset length"""
        if self._openspiel_total_tasks is not None:
            # OpenSpiel: fixed set of tasks (len(cases) * num_tasks_per_case)
            if self.max_samples >= 0:
                return min(self.max_samples, self._openspiel_total_tasks)
            return self._openspiel_total_tasks
        if self.max_samples == -1:
            # Infinite dataset - return a large number
            return 10_000_000  # Large number for "infinite"
        return self.max_samples
    
    def _generate_task_id(self, index: int) -> tuple[int, str]:
        """
        Generate a task_id for the given index.
        
        Args:
            index: Dataset index
            
        Returns:
            tuple: (task_id, task_type)
        """
        if self.task_id_ranges is not None and self.task_types is not None:
            # LGC-V2 style: choose task type and sample seed within that range.
            if self.shuffle:
                task_type = self.rng.choice(self.task_types)
            else:
                task_type = self.task_types[index % len(self.task_types)]
            
            base_id, _max_id = self.task_id_ranges[task_type]

            if self.seed is not None:
                self.rng.seed(self.seed + index)

            # TASK_ID_RANGE is 100M for lgc-v2; keep it in sync by using range width.
            range_width = (self.task_id_ranges[task_type][1] - self.task_id_ranges[task_type][0]) + 1
            task_id = base_id + int(self.rng.randint(0, range_width - 1))
            return int(task_id), str(task_type)

        # OpenSpiel: task_id in [0, total_fixed_tasks); shuffle or cycle by index
        if self._openspiel_total_tasks is not None and self._openspiel_total_tasks > 0:
            total = self._openspiel_total_tasks
            if self.shuffle and self.seed is not None:
                self.rng.seed(self.seed + index)
                task_id = self.rng.randint(0, total - 1)
            else:
                task_id = index % total
            return int(task_id), str(self.env)

        # Generic env: task_id is based on index (optionally shuffled)
        if self.seed is not None:
            self.rng.seed(self.seed + index)
            task_id = self.rng.randint(0, 2**31 - 1)
        else:
            task_id = index
        return int(task_id), str(self.env)
    
    def _get_event_loop(self):
        """Get or create a per-worker event loop for async operations"""
        if self._loop is None or self._loop.is_closed():
            with self._loop_lock:
                if self._loop is None or self._loop.is_closed():
                    self._loop = asyncio.new_event_loop()
                    # Start loop in background thread
                    def run_loop():
                        asyncio.set_event_loop(self._loop)
                        self._loop.run_forever()
                    thread = threading.Thread(target=run_loop, daemon=True)
                    thread.start()
        return self._loop
    
    def _generate_challenge_sync(self, task_id: int):
        """Generate challenge synchronously (for use in __getitem__).

        Note:
            The underlying adapters already handle any async logic (they use
            `asyncio.run(...)` internally), so we keep this method purely
            synchronous to avoid nested event-loop issues in DataLoader workers.
        """
        # Check cache first
        with self._cache_lock:
            if task_id in self._task_cache:
                return self._task_cache[task_id]
        
        # Try to get from prefetch queue (use any available task as fallback)
        if self._prefetch_queue is not None and len(self._prefetch_queue) > 0:
            try:
                # Use any prefetched task (they're random anyway)
                cached_id, challenge = self._prefetch_queue.popleft()
                # Cache it with the requested task_id
                with self._cache_lock:
                    if len(self._task_cache) >= self.cache_size:
                        # Remove oldest (simple FIFO)
                        self._task_cache.pop(next(iter(self._task_cache)))
                    self._task_cache[task_id] = challenge
                return challenge
            except (IndexError, ValueError):
                pass  # Queue empty or error, continue to generate
        
        # Generate synchronously – adapters wrap their own async with asyncio.run
        challenge = self.adapter.generate(int(task_id))

        if challenge is None:
            raise RuntimeError(f"Failed to generate challenge (task_id={task_id})")
        
        # Cache the result
        with self._cache_lock:
            if len(self._task_cache) >= self.cache_size:
                # Remove oldest entry (simple FIFO eviction)
                self._task_cache.pop(next(iter(self._task_cache)))
            self._task_cache[task_id] = challenge
        
        return challenge
    
    def _start_prefetch_thread(self):
        """Start background thread to pre-generate tasks"""
        def prefetch_worker():
            rng = random.Random(self.seed) if self.seed else random.Random()
            
            while True:
                try:
                    # Generate a random task_id (env-dependent)
                    if self.task_id_ranges is not None and self.task_types is not None:
                        # LGC-V2 style: sample within the selected task type's range.
                        task_type = rng.choice(self.task_types)
                        base_id, max_id = self.task_id_ranges[task_type]
                        range_width = (max_id - base_id) + 1
                        task_id = base_id + rng.randint(0, range_width - 1)
                    else:
                        # Generic env: just sample a random integer id
                        task_id = rng.randint(0, 2**31 - 1)
                    
                    # Generate challenge (adapter is sync)
                    challenge = self.adapter.generate(int(task_id))
                    
                    # Add to queue
                    if len(self._prefetch_queue) < self._prefetch_queue.maxlen:
                        self._prefetch_queue.append((task_id, challenge))
                    
                    # Small sleep to avoid CPU spinning
                    import time
                    time.sleep(0.001)
                except Exception as e:
                    # Continue on error
                    import time
                    time.sleep(0.1)
        
        self._prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self._prefetch_thread.start()
    
    def __getitem__(self, index: int) -> dict:
        """
        Generate a task on-the-fly and return in VERL format.
        
        Args:
            index: Dataset index
            
        Returns:
            dict: Sample in VERL format
        """
        # Generate task_id and task_type
        task_id, task_type = self._generate_task_id(index)
        
        # Generate challenge from environment with retry on failure
        max_retries = self.config.get("max_generation_retries", 3)
        last_error = None
        challenge = None
        
        for retry in range(max_retries):
            try:
                challenge = self._generate_challenge_sync(task_id)
                break
            except (ValueError, RuntimeError, TimeoutError) as e:
                last_error = e
                error_msg = str(e).lower()
                # Retry on specific errors that might be transient
                if any(keyword in error_msg for keyword in [
                    "failed to generate", "timeout", "rate limit", 
                    "temporary", "retry"
                ]) and retry < max_retries - 1:
                    # Retry with a different task_id
                    task_id, task_type = self._generate_task_id(index + retry + 1)
                    continue
                else:
                    # Re-raise if it's the last retry or a non-retryable error
                    if retry == max_retries - 1:
                        raise RuntimeError(
                            f"Failed to generate challenge after {max_retries} retries. "
                            f"Last error: {e}"
                        ) from e
                    raise
            except Exception as e:
                # Unexpected errors - don't retry, just raise
                raise RuntimeError(
                    f"Unexpected error generating challenge (task_id={task_id}): {e}"
                ) from e
        
        if challenge is None:
            raise RuntimeError(
                f"Failed to generate challenge after {max_retries} retries. "
                f"Last error: {last_error}"
            )
        
        # Track generation count
        with self._generation_lock:
            self._generation_count += 1
            if self._generation_count % 100 == 0:
                print(f"[Dataset] Generated {self._generation_count} samples so far (cache hit rate: {len(self._task_cache)}/{self.cache_size})")
        
        # Build messages in VERL format (always list of messages)
        messages = [
            {
                "role": "user",
                "content": challenge.prompt,
            }
        ]
        
        # Tokenize the prompt (same as RLHFDataset)
        if self.tokenizer is not None:
            # Apply chat template
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            # Tokenize
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")[0]  # Remove batch dimension
            attention_mask = model_inputs.pop("attention_mask")[0]
            
            # Postprocess (padding, truncation) - same as RLHFDataset
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids.unsqueeze(0),  # Add batch dim for postprocess
                attention_mask=attention_mask.unsqueeze(0),
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.get("truncation", "error"),
            )
            input_ids = input_ids[0]  # Remove batch dim
            attention_mask = attention_mask[0]
            
            # Compute position_ids (required by vLLM rollout)
            position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]  # Remove batch dim

            # Also compute raw_prompt_ids for rollout backends (match RLHFDataset)
            raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
            if len(raw_prompt_ids) > self.max_prompt_length:
                truncation = self.config.get("truncation", "error")
                if truncation == "left":
                    raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
                elif truncation == "right":
                    raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
                elif truncation == "middle":
                    left_half = self.max_prompt_length // 2
                    right_half = self.max_prompt_length - left_half
                    raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
                elif truncation == "error":
                    raise RuntimeError(
                        f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                    )
        else:
            # Fallback if tokenizer not available (shouldn't happen)
            input_ids = torch.tensor([], dtype=torch.long)
            attention_mask = torch.tensor([], dtype=torch.long)
            position_ids = torch.tensor([], dtype=torch.long)
            raw_prompt_ids = []
        
        # Build sample dict (same format as RLHFDataset)
        sample = {
            "data_source": task_type,
            "prompt": messages,
            "raw_prompt": messages,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "reward_model": {
                # Optional: for some envs (trace) the "ground truth" is included here.
                "ground_truth": challenge.extra.get("ground_truth")
                or challenge.extra.get("game_data", {}).get("answer", ""),
            },
            "extra_info": {
                "env": str(self.env),
                "adapter_config": dict(self.adapter_config),
                "task_id": task_id,
                "task_type": task_type,
                # Serialize challenge so reward-manager can evaluate without re-generating.
                "prompt": challenge.prompt,
                "challenge_extra": dict(challenge.extra),
                "seed": challenge.extra.get("seed"),
                "index": index,
            },
            # non-tensor field used by rollout backends
            "raw_prompt_ids": raw_prompt_ids,
        }
        
        return sample


class OpenSpielDataset(DynamicEnvDataset):
    """
    Dataset for OpenSpiel env only: fixed tasks per case (e.g. per board size).

    Forces env=openspiel and builds adapter from config.adapter_config.
    Compatible with VERL's RLHFDataset interface. Use in scripts with:
      data.custom_cls.name=OpenSpielDataset
      +data.custom_cls.config.adapter_config.cases=...  # optional (use + so Hydra adds to struct)
      +data.custom_cls.config.adapter_config.num_tasks_per_case=100
    """

    def __init__(
        self,
        data_files=None,
        tokenizer: PreTrainedTokenizer = None,
        processor=None,
        config: DictConfig = None,
        max_samples: int = -1,
    ):
        config = config or DictConfig({})
        # Build config with env=openspiel and adapter_config for OpenSpiel only
        adapter_config = dict(config.get("adapter_config", {}) or {})
        openspiel_config = OmegaConf.create({
            "env": "openspiel",
            "adapter_config": adapter_config,
            "prompt_key": config.get("prompt_key", "prompt"),
            "max_prompt_length": config.get("max_prompt_length", 2048),
            "return_raw_chat": config.get("return_raw_chat", False),
            "shuffle": config.get("shuffle", True),
            "seed": config.get("seed", None),
            "cache_size": config.get("cache_size", 1000),
            "prefetch_size": config.get("prefetch_size", 0),
            "truncation": config.get("truncation", "error"),
            "max_generation_retries": config.get("max_generation_retries", 3),
        })
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            processor=processor,
            config=openspiel_config,
            max_samples=max_samples,
        )


# Backwards-compatible alias (older configs/scripts may refer to this name)
LGCV2DynamicDataset = DynamicEnvDataset



