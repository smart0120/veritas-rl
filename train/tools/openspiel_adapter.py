"""
OpenSpiel environment adapter for VERL dynamic dataset + reward.

Uses OpenSpiel (https://github.com/google-deepmind/open_spiel) to generate
game states as prompts and evaluate model moves. Supports the 20-game suite
(Leduc Poker, Liar's Dice, Battleship, Goofspiel, Gin Rummy, Backgammon, Pig,
Blackjack, Phantom Tic-Tac-Toe, Breakthrough, Hex, Hearts, Cribbage, Euchre,
Othello, Go, Chess, Checkers, Dots and Boxes, Clobber, Quoridor) via OpenSpiel
short names and configs.

Fixed tasks per case:
  - A "case" is one (game + config), e.g. hex 5x5, hex 7x7, go 9x9, breakthrough 6x6.
  - You provide a list of cases (game load strings, e.g. ["hex(board_size=5)", "hex(board_size=7)"])
    or use game_list + game_configs to build one case per (game, config). You set num_tasks_per_case
    (e.g. 100). Total fixed tasks = len(cases) * num_tasks_per_case.
  - task_id in [0, total): case_index = task_id // num_tasks_per_case, task_index = task_id % num_tasks_per_case.
    Same task_id always yields the same state (seed = base_seed + task_index).
  - evaluate(solution_str, challenge): deserialize state, parse action, apply move; reward = outcome or legal.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from env_adapter import SimpleChallenge

# Optional import; caller must install open_spiel
try:
    import pyspiel
    _OPENSPIEL_AVAILABLE = True
except ImportError:
    pyspiel = None  # type: ignore
    _OPENSPIEL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Focus games (7): clobber, gin_rummy, goofspiel, hex, leduc_poker, liars_dice, othello.
# Available cases per game with config variants (OpenSpiel param names).
# Default adapter uses FOCUS_CASES when cases and game_configs are not set.
# If a case fails to load (unsupported param in your OpenSpiel build), remove it or set game_types.
# ---------------------------------------------------------------------------
FOCUS_CASES = [
    # Clobber: 3 configs (5x5, 6x6, 7x7)
    "clobber(rows=5,columns=5)",
    "clobber(rows=6,columns=6)",
    "clobber(rows=7,columns=7)",
    # Gin Rummy: default (OpenSpiel may support hand_size / knock_card variants)
    "gin_rummy",
    # Goofspiel: default + num_cards variants if supported (8–16 cards)
    "goofspiel",
    "goofspiel(num_cards=8)",
    "goofspiel(num_cards=10)",
    "goofspiel(num_cards=12)",
    "goofspiel(num_cards=14)",
    "goofspiel(num_cards=16)",
    # Hex: 4 configs (5x5, 7x7, 9x9, 11x11)
    "hex(board_size=5)",
    "hex(board_size=7)",
    "hex(board_size=9)",
    "hex(board_size=11)",
    # Leduc Poker: 1 config
    "leduc_poker",
    # Liar's Dice: default + num_dice variants (if supported)
    "liars_dice",
    # Othello: 2 configs (6x6, 8x8) — standard 8x8 is often just "othello"
    "othello",
]

# ---------------------------------------------------------------------------
# All available cases: Category A (high-randomness) + Category B (deterministic).
# Order: focus games first (FOCUS_CASES), then the rest with config variants.
# ---------------------------------------------------------------------------
AVAILABLE_CASES = list(FOCUS_CASES) + [
    # Category A (high-randomness) — remaining
    "battleship",
    "backgammon",
    "pig",
    "blackjack",
    "latent_tic_tac_toe",  # Phantom Tic-Tac-Toe
    "breakthrough(rows=6,columns=6)",
    "breakthrough(rows=8,columns=8)",
    # Category B (large deterministic)
    "hearts",
    "cribbage",
    "euchre",
    "go(go_board_size=9)",
    "go(go_board_size=13)",
    "chess",
    "checkers",
    "dots_and_boxes(num_rows=3,num_cols=3)",
    "dots_and_boxes(num_rows=4,num_cols=4)",
    "dots_and_boxes(num_rows=5,num_cols=5)",
]

# Legacy: board-size–style example list (subset). Prefer FOCUS_CASES or AVAILABLE_CASES.
DEFAULT_CASES_BOARD_SIZES = [
    "hex(board_size=5)",
    "hex(board_size=7)",
    "hex(board_size=9)",
    "go(go_board_size=9)",
    "go(go_board_size=13)",
    "breakthrough(rows=6,columns=6)",
    "breakthrough(rows=8,columns=8)",
    "dots_and_boxes(num_rows=3,num_cols=3)",
    "dots_and_boxes(num_rows=4,num_cols=4)",
    "dots_and_boxes(num_rows=5,num_cols=5)",
    "clobber(rows=5,columns=5)",
    "clobber(rows=6,columns=6)",
    "clobber(rows=7,columns=7)",
    "othello",
    "chess",
    "checkers",
]

# Default game short names for the full suite (OpenSpiel names).
# Config variants can be passed via adapter config (game_configs).
DEFAULT_GAME_LIST = [
    "leduc_poker",           # 0  Leduc Poker
    "liars_dice",            # 1  Liar's Dice
    "battleship",            # 2  Battleship
    "goofspiel",             # 3  Goofspiel
    "gin_rummy",             # 4  Gin Rummy
    "backgammon",            # 5  Backgammon
    "pig",                   # 6  Pig
    "blackjack",             # 7  Blackjack
    "latent_tic_tac_toe",    # 8  Phantom Tic-Tac-Toe (imperfect info)
    "breakthrough",          # 9  Breakthrough
    "hex",                   # 10 Hex
    "hearts",                # 11 Hearts
    "cribbage",              # 12 Cribbage
    "euchre",                # 13 Euchre
    "othello",               # 14 Othello
    "go",                    # 15 Go
    "chess",                 # 16 Chess
    "checkers",              # 17 Checkers
    "dots_and_boxes",       # 18 Dots and Boxes
    "clobber",               # 19 Clobber
    # Quoridor has known issues in OpenSpiel; omit or add when fixed
]


def _game_short_name_with_params(game_spec: str, params: Optional[Dict[str, Any]] = None) -> str:
    """Build OpenSpiel load_game string, e.g. 'hex(board_size=7)' or 'go'."""
    if not params:
        return game_spec
    parts = [f"{k}={v}" for k, v in sorted(params.items())]
    return f"{game_spec}({','.join(parts)})"


def _decode_task_id_by_cases(
    task_id: int,
    num_cases: int,
    num_tasks_per_case: int,
) -> Tuple[int, int]:
    """
    Decode task_id when using fixed tasks per case.
    task_id in [0, num_cases * num_tasks_per_case).
    Returns: (case_index, task_index). Seed for state = base_seed + task_index.
    """
    if num_cases <= 0 or num_tasks_per_case <= 0:
        raise ValueError("num_cases and num_tasks_per_case must be positive")
    total = num_cases * num_tasks_per_case
    tid = task_id % total if task_id >= 0 else (task_id % total + total) % total
    case_index = tid // num_tasks_per_case
    task_index = tid % num_tasks_per_case
    return case_index, task_index


def _build_cases_from_game_list(
    game_list: List[str],
    game_configs: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    """Build list of game load strings: one case per (game, config). E.g. hex 5x5, hex 7x7, go 9x9."""
    cases: List[str] = []
    for name in game_list:
        configs = game_configs.get(name)
        if not configs:
            cases.append(name)
            continue
        for params in configs:
            cases.append(_game_short_name_with_params(name, params))
    return cases


def _filter_cases_by_game_types(cases: List[str], game_types: List[str]) -> List[str]:
    """Keep only cases that match game_types. If game_types is empty, return cases unchanged.
    Each entry in game_types can be a game short name (e.g. 'hex', 'go') or a full case string (e.g. 'hex(board_size=5)').
    A case matches if it equals an entry or starts with entry + '(' (so 'hex' matches 'hex(board_size=5)').
    """
    if not game_types:
        return cases
    out: List[str] = []
    for c in cases:
        for gt in game_types:
            gt = (gt or "").strip()
            if not gt:
                continue
            if c == gt or c.startswith(gt + "("):
                out.append(c)
                break
    return out


def _get_observation_string(state) -> str:
    """Get a string description of the state for the current player (for prompt)."""
    try:
        if state.is_chance_node():
            return "Chance node (random event)."
        cur = state.current_player()
        try:
            g = state.get_game()
            if getattr(g.get_type(), "information", None) == getattr(
                pyspiel.GameType.Information, "IMPERFECT_INFORMATION", None
            ):
                return state.information_state_string(cur)
        except Exception:
            pass
        return state.observation_string(cur)
    except Exception:
        pass
    try:
        return state.to_string()
    except Exception:
        return str(state)


def _legal_actions_for_prompt(state, game) -> List[Tuple[int, str]]:
    """Return list of (action_id, action_string) for current player."""
    out: List[Tuple[int, str]] = []
    try:
        for a in state.legal_actions():
            try:
                s = game.action_to_string(state.current_player(), a)
            except Exception:
                s = str(a)
            out.append((a, s))
    except Exception:
        pass
    return out


def _parse_action_from_response(
    solution_str: str,
    legal_pairs: List[Tuple[int, str]],
) -> Optional[int]:
    """Parse model response to a legal action id. Prefer string match, then int."""
    text = (solution_str or "").strip().lower()
    # Try exact string match (case-insensitive)
    for a, s in legal_pairs:
        if s.lower().strip() == text:
            return a
    # Try substring
    for a, s in legal_pairs:
        if s.lower() in text or text in s.lower():
            return a
    # Try integer
    try:
        idx = int(text)
        for a, _ in legal_pairs:
            if a == idx:
                return a
        if 0 <= idx < len(legal_pairs):
            return legal_pairs[idx][0]
    except ValueError:
        pass
    return None


def _play_out_random(state, rng: random.Random) -> "pyspiel.State":
    """Play out from state to terminal using random legal actions."""
    while not state.is_terminal():
        if state.is_chance_node():
            # chance_outcomes() returns a list of (outcome, prob) pairs, not (outcomes, probs)
            chance_list = state.chance_outcomes()
            actions = [o[0] for o in chance_list]
            probs_list = [o[1] for o in chance_list]
            a = rng.choices(actions, weights=probs_list, k=1)[0]
        else:
            legal = state.legal_actions()
            a = rng.choice(legal)
        state.apply_action(a)
    return state


class OpenSpielAdapter:
    """
    Adapter that uses OpenSpiel to generate (prompt, extra) and evaluate (response -> reward).
    """

    env_name = "openspiel"

    def __init__(
        self,
        cases: Optional[List[str]] = None,
        game_list: Optional[List[str]] = None,
        game_configs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        game_types: Optional[List[str]] = None,
        num_tasks_per_case: int = 100,
        max_random_steps: int = 0,
        reward_mode: str = "outcome",
        seed: Optional[int] = None,
    ):
        """
        Args:
            cases: List of OpenSpiel game load strings, one per "case" (e.g. ["hex(board_size=5)", "hex(board_size=7)", "go(go_board_size=9)"]).
                   Each case gets num_tasks_per_case tasks. If None, built from game_list + game_configs (one case per (game, config)).
            game_list: Used when cases is None. List of game short names (default: DEFAULT_GAME_LIST).
            game_configs: Used when cases is None. Dict game_short_name -> list of param dicts (e.g. {"hex": [{"board_size": 5}, {"board_size": 7}]}).
            game_types: If set, only train on these games/cases. Each entry is a game short name (e.g. 'hex', 'go') or full case string (e.g. 'hex(board_size=5)').
                        If not set or empty, train on all cases.
            num_tasks_per_case: Fixed number of tasks per case. Total = len(cases) * num_tasks_per_case.
            max_random_steps: Max random steps from root before presenting state (0 = root only).
            reward_mode: "outcome" = play out and use normalized return; "legal" = 1 if legal else 0.
            seed: Base random seed; per-task seed = seed + task_index for reproducibility.
        """
        if not _OPENSPIEL_AVAILABLE:
            raise ImportError(
                "open_spiel is not installed. Install with: pip install open_spiel"
            )
        self.game_configs = game_configs or {}
        self.game_list = game_list or list(DEFAULT_GAME_LIST)
        if cases is not None and len(cases) > 0:
            self.cases = list(cases)
        elif game_list is not None or (game_configs and len(game_configs) > 0):
            self.cases = _build_cases_from_game_list(self.game_list, self.game_configs)
        else:
            # Default: focus games (clobber, gin_rummy, goofspiel, hex, leduc_poker, liars_dice, othello)
            self.cases = list(FOCUS_CASES)
        if not self.cases:
            self.cases = [ _game_short_name_with_params(g, None) for g in self.game_list ]
        # Filter by game_types if specified; if not set, train on all cases
        game_types_list = (game_types or []) if isinstance(game_types, list) else ([game_types] if game_types else [])
        if game_types_list:
            filtered = _filter_cases_by_game_types(self.cases, game_types_list)
            if not filtered:
                raise ValueError(
                    f"game_types {game_types_list} matched no cases. "
                    "Use game short names (e.g. 'hex', 'go') or full case strings (e.g. 'hex(board_size=5)')."
                )
            self.cases = filtered
        self.num_tasks_per_case = max(1, num_tasks_per_case)
        self.max_random_steps = max(0, max_random_steps)
        self.reward_mode = reward_mode if reward_mode in ("outcome", "legal") else "outcome"
        self._base_seed = seed if seed is not None else 0
        self.rng = random.Random(self._base_seed)

    def get_total_fixed_tasks(self) -> int:
        """Total number of fixed tasks (len(cases) * num_tasks_per_case). Used by the dataset for __len__ and task_id range."""
        return len(self.cases) * self.num_tasks_per_case

    def _load_game_by_case(self, case_index: int):
        """Load game instance by case_index (into self.cases)."""
        game_str = self.cases[case_index % len(self.cases)]
        return pyspiel.load_game(game_str)

    def generate(self, task_id: int) -> SimpleChallenge:
        """Generate a single-turn challenge from fixed task set: (case_index, task_index) from task_id; same task_id -> same state."""
        if not _OPENSPIEL_AVAILABLE:
            raise RuntimeError("open_spiel is not installed")
        case_index, task_index = _decode_task_id_by_cases(
            task_id, len(self.cases), self.num_tasks_per_case,
        )
        game = self._load_game_by_case(case_index)
        seed = self._base_seed + task_index
        self.rng.seed(seed)
        state = game.new_initial_state()

        # Optional random rollout to get a non-root state
        steps = 0
        while not state.is_terminal() and steps < self.max_random_steps:
            if state.is_chance_node():
                # chance_outcomes() returns a list of (outcome, prob) pairs, not (outcomes, probs)
                chance_list = state.chance_outcomes()
                actions = [o[0] for o in chance_list]
                probs_list = [o[1] for o in chance_list]
                a = self.rng.choices(actions, weights=probs_list, k=1)[0]
            else:
                legal = state.legal_actions()
                if not legal:
                    break
                a = self.rng.choice(legal)
            state.apply_action(a)
            steps += 1

        if state.is_terminal():
            # Restart from root if we ended up terminal
            state = game.new_initial_state()
            self.rng.seed(seed + 1)

        obs = _get_observation_string(state)
        legal_pairs = _legal_actions_for_prompt(state, game)
        legal_str = ", ".join(s for _, s in legal_pairs[:50])  # cap for prompt length
        if len(legal_pairs) > 50:
            legal_str += f", ... ({len(legal_pairs)} total)"

        game_name = getattr(game.get_type(), "short_name", None) or getattr(game, "short_name", None) or "unknown"
        prompt = (
            f"Game: {game_name}\n\n"
            f"Current state (your turn):\n{obs}\n\n"
            f"Legal actions: {legal_str}\n\n"
            "Reply with exactly one legal action (the exact string or its index)."
        )

        serialized = pyspiel.serialize_game_and_state(game, state)
        game_name = getattr(game.get_type(), "short_name", None) or getattr(game, "short_name", None) or "unknown"
        extra: Dict[str, Any] = {
            "serialized_game_and_state": serialized,
            "game_short_name": game_name,
            "legal_actions": [[a, s] for a, s in legal_pairs],
            "current_player": state.current_player(),
            "task_id": task_id,
            "num_players": game.num_players(),
        }
        return SimpleChallenge(env=self.env_name, prompt=prompt, extra=extra)

    def evaluate(self, solution_str: str, challenge: SimpleChallenge) -> float:
        """Parse action from model response, apply it, and return reward (outcome or legal indicator)."""
        if not _OPENSPIEL_AVAILABLE:
            return 0.0
        extra = challenge.extra
        serialized = extra.get("serialized_game_and_state")
        legal_actions = extra.get("legal_actions")
        current_player = extra.get("current_player", 0)
        if not serialized or not legal_actions:
            return 0.0
        try:
            game, state = pyspiel.deserialize_game_and_state(serialized)
        except Exception:
            return 0.0
        legal_pairs: List[Tuple[int, str]] = [
            (int(x[0]), str(x[1])) for x in legal_actions
        ]
        action = _parse_action_from_response(solution_str, legal_pairs)
        if action is None:
            return 0.0
        if action not in [a for a, _ in legal_pairs]:
            return 0.0
        try:
            state.apply_action(action)
        except Exception:
            return 0.0
        if self.reward_mode == "legal":
            return 1.0
        # Outcome: play out with random policy and return normalized return for current_player
        if state.is_terminal():
            returns = state.returns()
            r = returns[current_player] if current_player < len(returns) else 0.0
        else:
            state = _play_out_random(state, self.rng)
            returns = state.returns()
            r = returns[current_player] if current_player < len(returns) else 0.0
        # Normalize to [0, 1] (OpenSpiel often uses [-1, 1] for 2p zero-sum)
        return float((r + 1.0) / 2.0) if r is not None else 0.5
