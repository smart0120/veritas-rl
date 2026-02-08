"""
SWE-bench / mini-swe-agent environment adapter for VERL.

Uses [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) and SWE-bench
datasets: load tasks (problem statements), model produces a patch, reward from
test results (via mini-swe-agent evaluation flow or sb-cli).

Data: HuggingFace SWE-bench datasets (e.g. princeton-nlp/SWE-bench_Lite,
princeton-nlp/SWE-bench_Verified) or local cache with same schema.

Evaluation: If mini-swe-agent (and optional swe-rex) is installed, the adapter
can use it for in-process verification when available; otherwise use
mini-extra swebench / sb-cli for evaluation (see docs).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from env_adapter import SimpleChallenge

# Lazy-loaded HuggingFace dataset
_dataset_cache: Optional[Any] = None
_dataset_name_cache: Optional[str] = None
_dataset_split_cache: Optional[str] = None


def _load_hf_dataset(dataset_name: str, split: str):
    """Load SWE-bench–style dataset from HuggingFace (lazy, cached)."""
    global _dataset_cache, _dataset_name_cache, _dataset_split_cache
    if (
        _dataset_cache is not None
        and _dataset_name_cache == dataset_name
        and _dataset_split_cache == split
    ):
        return _dataset_cache
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install datasets: pip install datasets (required for SWE-bench dataset). "
            "For mini-swe-agent: pip install mini-swe-agent"
        )
    _dataset_cache = load_dataset(dataset_name, split=split, trust_remote_code=True)
    _dataset_name_cache = dataset_name
    _dataset_split_cache = split
    return _dataset_cache


def _row_to_instance(row: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    """Convert HF row (or cache dict) to instance shape for prompt/extra (SWE-bench schema)."""
    if hasattr(row, "keys") and callable(getattr(row, "get", None)):
        d = dict(row) if not isinstance(row, dict) else row
    else:
        d = row
    problem = (d.get("problem_statement") or "").strip()
    if not problem and d.get("hints_text"):
        problem = (d.get("hints_text") or "").strip()
    if not problem:
        problem = "Fix the bug described in the issue."
    return {
        "instance_id": d.get("instance_id"),
        "repo": d.get("repo"),
        "base_commit": d.get("base_commit"),
        "version": d.get("version"),
        "problem_statement": problem,
        "FAIL_TO_PASS": d.get("FAIL_TO_PASS"),
        "PASS_TO_PASS": d.get("PASS_TO_PASS"),
        "environment_setup_commit": d.get("environment_setup_commit"),
        "patch": d.get("patch"),  # gold patch if present
    }


class SweSynthAdapter:
    """
    Adapter for SWE-bench + mini-swe-agent: load tasks from SWE-bench (HF) or
    local cache; present problem_statement as prompt; evaluate patch via
    mini-swe-agent / swe-rex / sb-cli when available.
    """

    env_name = "swe-synth"

    def __init__(
        self,
        task_ids: Optional[List[int]] = None,
        task_id_range: Optional[tuple[int, int]] = None,
        dataset_name: str = "princeton-nlp/SWE-bench_Lite",
        split: str = "test",
        cache_dir: Optional[str] = None,
        use_cache_only: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            task_ids: List of task indices. If set, total tasks = len(task_ids).
            task_id_range: (min_id, max_id) inclusive when task_ids is None.
            dataset_name: HuggingFace dataset (e.g. princeton-nlp/SWE-bench_Lite,
                princeton-nlp/SWE-bench_Verified). Ignored if use_cache_only and cache_dir set.
            split: Dataset split (e.g. test, dev).
            cache_dir: Optional local dir with task_0.json, task_1.json, ... (SWE-bench schema).
            use_cache_only: If True and cache_dir set, load only from cache (no HF).
            seed: Optional seed for sampling.
        """
        self._config = {
            "dataset_name": dataset_name,
            "split": split,
            "cache_dir": cache_dir or "",
            "use_cache_only": use_cache_only,
            "seed": seed,
        }
        self._task_indices: List[int] = []
        if task_ids is not None and len(task_ids) > 0:
            self._task_indices = list(task_ids)
        elif task_id_range is not None:
            lo, hi = int(task_id_range[0]), int(task_id_range[1])
            self._task_indices = list(range(lo, hi + 1))
        else:
            self._task_indices = []
        self._dataset = None
        self._cache_dir_path: Optional[Path] = None
        if cache_dir:
            self._cache_dir_path = Path(cache_dir).resolve()
        self._evaluator = None  # Lazy: mini-swe-agent / swe-rex if available

    def _get_raw_task_count(self) -> int:
        if self._config["use_cache_only"] and self._config["cache_dir"]:
            return self._get_task_count_from_cache()
        ds = self._get_dataset()
        return len(ds) if ds is not None else self._get_task_count_from_cache()

    def _resolve_task_indices(self) -> List[int]:
        if self._task_indices:
            return self._task_indices
        n = self._get_raw_task_count()
        self._task_indices = list(range(min(n, 100)))
        return self._task_indices

    def _get_dataset(self):
        if self._config["use_cache_only"] and self._config["cache_dir"]:
            return None
        if self._dataset is None:
            self._dataset = _load_hf_dataset(
                self._config["dataset_name"],
                self._config["split"],
            )
        return self._dataset

    def _get_task_count_from_cache(self) -> int:
        if not self._cache_dir_path or not self._cache_dir_path.is_dir():
            return 0
        return sum(1 for _ in self._cache_dir_path.glob("task_*.json"))

    def get_total_fixed_tasks(self) -> int:
        return len(self._resolve_task_indices())

    def get_task_id_for_index(self, index: int) -> int:
        indices = self._resolve_task_indices()
        if not indices:
            return index
        return indices[index % len(indices)]

    def _load_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        if self._cache_dir_path:
            p = self._cache_dir_path / f"task_{task_id}.json"
            if p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                return _row_to_instance(raw)
        ds = self._get_dataset()
        if ds is not None and 0 <= task_id < len(ds):
            return _row_to_instance(ds[task_id])
        return None

    def generate(self, task_id: int) -> SimpleChallenge:
        """Load task (SWE-bench instance) and return prompt + challenge_extra."""
        instance = self._load_task(task_id)
        if instance is None:
            raise ValueError(
                f"SWE-bench task_id {task_id} not found. "
                "Use HuggingFace princeton-nlp/SWE-bench_Lite (or _Verified) or populate cache_dir with task_*.json."
            )
        problem = (instance.get("problem_statement") or "").strip()
        if not problem:
            problem = "Fix the bug described in the issue."
        prompt = (
            "You are a fixer agent. Apply a minimal patch to fix the bug.\n\n"
            "Problem statement:\n" + problem + "\n\n"
            "Reply with a single unified diff patch (no explanation, only the patch)."
        )
        extra: Dict[str, Any] = {
            "task_id": task_id,
            "bug_instance": instance,
        }
        return SimpleChallenge(env=self.env_name, prompt=prompt, extra=extra)

    def _get_evaluator(self):
        """Lazy-init: try mini-swe-agent / swe-rex for patch evaluation if available."""
        if self._evaluator is not None:
            return self._evaluator
        # Optional: swe-rex (mini-swe-agent[full]) can run SWE-bench evaluation
        try:
            import minisweagent  # noqa: F401
            # Check for swe-rex or minisweagent evaluation API
            try:
                from minisweagent.benchmarks import swebench
                self._evaluator = getattr(swebench, "evaluate_instance", None) or getattr(swebench, "run_instance", None)
            except ImportError:
                pass
            if self._evaluator is None:
                try:
                    import swe_rex
                    self._evaluator = getattr(swe_rex, "evaluate_patch", None) or getattr(swe_rex, "run_evaluation", None)
                except ImportError:
                    pass
        except ImportError:
            pass
        if self._evaluator is None:
            self._evaluator = False
        return self._evaluator

    def evaluate(self, solution_str: str, challenge: SimpleChallenge) -> float:
        """
        Verify the model's patch. Returns 1.0 if required tests pass, else 0.0.
        Uses mini-swe-agent / swe-rex when installed; otherwise use
        mini-extra swebench and sb-cli for evaluation (see docs).
        """
        extra = challenge.extra
        fix_patch = (solution_str or "").strip()
        instance = extra.get("bug_instance")
        if not instance or not fix_patch:
            return 0.0
        evaluator = self._get_evaluator()
        if callable(evaluator):
            try:
                result = evaluator(instance, fix_patch)
                return float(result) if result is not None else 0.0
            except Exception:
                if os.environ.get("VERBOSE_REWARD"):
                    import traceback
                    traceback.print_exc()
                return 0.0
        # No in-process evaluator: use mini-extra swebench + sb-cli for full metrics
        return 0.0
