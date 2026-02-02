"""
Environment adapters to make VERL custom datasets / reward functions work across envs.

VERL’s custom reward function API is (roughly):
  score = compute_score(data_source, solution_str, ground_truth, extra_info)

And dataset samples can carry arbitrary `extra_info` (json-serializable) through the
DataProto so reward functions can evaluate without re-generating tasks.

This file provides a thin adapter layer so different *single-turn* verifiable envs
(currently: lgc-v2, trace) can be integrated consistently.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol


def _project_root() -> Path:
    # veritas-rl/train/tools/ -> veritas-rl/
    return Path(__file__).resolve().parent.parent.parent


def _add_env_to_syspath(env_dirname: str) -> Path:
    """
    Add the environment implementation directory to sys.path for local imports.

    Historically, environments lived under:
        /repo/environments/<env_dirname>

    After integrating `affinetes`, they now live under:
        /repo/affinetes/environments/primeintellect/lgc-v2
        /repo/affinetes/environments/trace

    This helper transparently supports both layouts so callers can keep using
    `_add_env_to_syspath("lgc-v2")` or `"trace"` without caring about the
    underlying directory structure.
    """
    project_root = _project_root()

    # 1) Original layout: /environments/<env_dirname>
    legacy_path = project_root / "environments" / env_dirname
    if legacy_path.exists():
        env_path = legacy_path
    else:
        # 2) New Affinetes layout
        affinetes_root = project_root / "affinetes" / "environments"

        # Special-case mappings for known envs
        if env_dirname in {"lgc-v2", "lgc_v2", "logic-v2", "logic_v2"}:
            env_path = affinetes_root / "primeintellect" / "lgc-v2"
        elif env_dirname == "trace":
            env_path = affinetes_root / "trace"
        else:
            # Fallback: direct match under affinetes/environments/<env_dirname>
            env_path = affinetes_root / env_dirname

    if not env_path.exists():
        raise FileNotFoundError(f"Environment path not found: {env_path}")

    if str(env_path) not in sys.path:
        sys.path.insert(0, str(env_path))

    return env_path


@dataclass(frozen=True)
class SimpleChallenge:
    """Minimal challenge container (prompt + extra metadata) that is JSON-serializable."""
    env: str
    prompt: str
    extra: Dict[str, Any]


class EnvAdapter(Protocol):
    """
    Adapter surface that works for single-turn verifiable tasks:
    - generate(): produces a prompt + metadata (challenge)
    - evaluate(): scores a model response against that challenge
    """

    env_name: str

    def generate(self, task_id: int) -> SimpleChallenge: ...
    def evaluate(self, solution_str: str, challenge: SimpleChallenge) -> float: ...


class LGCV2Adapter:
    env_name = "lgc-v2"

    def __init__(self, task_configs: Optional[Dict[str, Any]] = None, max_cache_size: int = 1000):
        _add_env_to_syspath("lgc-v2")
        from logic_task_v2 import LogicTaskV2  # type: ignore

        self._LogicTaskV2 = LogicTaskV2
        self._task = LogicTaskV2(task_configs=task_configs, max_cache_size=max_cache_size)

    def generate(self, task_id: int) -> SimpleChallenge:
        # LogicTaskV2.generate is async; run it in a local loop.
        import asyncio

        ch = asyncio.run(self._task.generate(task_id=task_id))
        return SimpleChallenge(env=self.env_name, prompt=ch.prompt, extra=dict(ch.extra))

    def evaluate(self, solution_str: str, challenge: SimpleChallenge) -> float:
        import asyncio

        _add_env_to_syspath("lgc-v2")
        from models import Challenge  # type: ignore

        # Rehydrate a Challenge object expected by LogicTaskV2.evaluate
        ch = Challenge(env="logic-v2", prompt=challenge.prompt, extra=challenge.extra)
        task = self._LogicTaskV2()
        return float(asyncio.run(task.evaluate(solution_str, ch)))


class TraceAdapter:
    env_name = "trace"

    def __init__(
        self,
        dataset_name: str = "satpalsr/rl-python",
        dataset_split: str = "train",
        dataset_shuffle: bool = False,
    ):
        _add_env_to_syspath("trace")
        from trace_task import TraceTask  # type: ignore

        self._task = TraceTask(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            dataset_shuffle=dataset_shuffle,
        )

    def generate(self, task_id: int) -> SimpleChallenge:
        import asyncio

        ch = asyncio.run(self._task.generate(task_id=task_id))
        return SimpleChallenge(env=self.env_name, prompt=ch.prompt, extra=dict(ch.extra))

    def evaluate(self, solution_str: str, challenge: SimpleChallenge) -> float:
        import asyncio

        _add_env_to_syspath("trace")
        from models import Challenge  # type: ignore

        ch = Challenge(env="trace", prompt=challenge.prompt, extra=challenge.extra)
        score, _test_result = asyncio.run(self._task.evaluate(solution_str, ch))
        return float(score)


def get_env_adapter(env: str, config: Optional[Dict[str, Any]] = None) -> EnvAdapter:
    """
    Factory for adapters.

    `config` is adapter-specific (e.g. Trace dataset settings, LGC task_configs).
    """
    config = config or {}
    env = env.strip().lower()

    if env in {"lgc-v2", "lgc_v2", "logic-v2", "logic_v2"}:
        return LGCV2Adapter(
            task_configs=config.get("task_configs"),
            max_cache_size=int(config.get("max_cache_size", 1000)),
        )

    if env == "trace":
        return TraceAdapter(
            dataset_name=str(config.get("dataset_name", "satpalsr/rl-python")),
            dataset_split=str(config.get("dataset_split", "train")),
            dataset_shuffle=bool(config.get("dataset_shuffle", False)),
        )

    if env == "openspiel":
        import os
        from openspiel_adapter import OpenSpielAdapter
        gt = config.get("game_types")
        if gt is not None and not isinstance(gt, list):
            gt = [gt] if gt else []
        if not gt and os.environ.get("GAME_TYPES"):
            # Avoid Hydra struct: read game_types from env instead of data.custom_cls.config
            gt = [s.strip() for s in os.environ["GAME_TYPES"].split(",") if s.strip()]
        return OpenSpielAdapter(
            cases=config.get("cases"),
            game_list=config.get("game_list"),
            game_configs=config.get("game_configs"),
            game_types=gt,
            num_tasks_per_case=int(config.get("num_tasks_per_case", config.get("num_tasks_per_game", 100))),
            max_random_steps=int(config.get("max_random_steps", 0)),
            reward_mode=str(config.get("reward_mode", "outcome")),
            seed=config.get("seed"),
        )

    raise ValueError(f"Unknown env adapter: {env}")