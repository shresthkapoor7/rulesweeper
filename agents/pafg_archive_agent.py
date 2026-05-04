from __future__ import annotations

import argparse
import ast
from collections import deque
import inspect
import json
import os
import random
import re
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - environment-dependent
    OpenAI = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code_mutations import CodeValidationError, load_archive_file, register_all_from_archive
from config import GameConfig
from game import GameState, MinesweeperGame


LLM_TIMEOUT_SECONDS = 60.0
MODEL_NAME = "anthropic/claude-haiku-4.5"
DEFAULT_DELAY_SECONDS = 10.0
DEFAULT_OUTPUT_PATH = "pafg_agent_archive.json"
DEFAULT_MAX_TOKENS = 1800

_FORBIDDEN_NAMES = {
    "__import__", "eval", "exec", "compile", "open",
    "globals", "locals", "vars", "__builtins__",
}

_FORBIDDEN_ATTRS = {
    "__class__", "__subclasses__", "__bases__", "__mro__",
    "__globals__", "__dict__", "__builtins__", "__import__",
    "__getattribute__", "__setattr__", "__delattr__",
    "__class_getitem__", "__reduce__", "__reduce_ex__",
}

_GAME_API_CHEATSHEET = """\
MinesweeperGame:
  game.get_config() -> GameConfig
  game.get_cell(r, c) -> Cell
  game.get_state() -> GameState
  game.board -> Board
  game.player -> Player

Board:
  game.board.neighbors(r, c) -> list[(r, c)]
  game.board.grid[r][c] -> Cell

Player:
  game.player.is_alive() -> bool
  game.player.metrics() -> dict

Cell fields:
  cell.is_mine
  cell.is_revealed
  cell.is_flagged
  cell.adjacent_mines

Important:
  - Do not call methods that are not listed here.
  - There is no `game.get_health()` method.
  - If you need health-related logic, use `game.player.is_alive()` or
    `game.player.metrics()` only."""

_HOOK_SURFACE = """\
Subclass `ArchiveAwarePAFGAgent` and override only these hooks when needed:

  def opening_action(self, game, grid, rows, cols) -> tuple[str, int, int]
  def neighbor_positions(self, game, r, c, rows, cols) -> list[tuple[int, int]]
  def clue_value(self, game, grid, r, c) -> int | None
  def frontier_cell_score(self, game, grid, pos, prob_map, mine_set, neighbors_of) -> tuple
  def postprocess_prob_map(self, game, grid, prob_map, mine_set, neighbors_of) -> dict

Do not override large internal solver methods like `_stage_p`, `_stage_a`, or `_stage_g`
unless absolutely necessary. Prefer a class with 1-2 hook overrides."""


class ArchiveAwarePAFGAgent:
    """
    Self-contained PAFG-style baseline for archive-specific adaptation.

    Generated subclasses should extend this class, not import from
    `agents/pafg_agent.py`. The goal is to expose one stable file-local surface
    for LLM-written agent variants.
    """

    _ENUM_CAP = 20

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Hook surface intended for LLM adaptation
    # ------------------------------------------------------------------

    def neighbor_positions(
        self,
        game: MinesweeperGame,
        r: int,
        c: int,
        rows: int,
        cols: int,
    ) -> list[tuple[int, int]]:
        """
        Default topology hook.

        By default this asks the live board for neighbors, which makes the
        baseline automatically compatible with generated neighborhood mechanics.
        """
        return game.board.neighbors(r, c)

    def opening_action(
        self,
        game: MinesweeperGame,
        grid,
        rows: int,
        cols: int,
    ) -> tuple[str, int, int]:
        """Opening move hook."""
        return ("reveal", 0, 0)

    def clue_value(
        self,
        game: MinesweeperGame,
        grid,
        r: int,
        c: int,
    ) -> int | None:
        """
        Clue interpretation hook.

        Default behavior reads the true adjacent mine count from the board cell.
        Subclasses can override this to respect alternate clue semantics.
        """
        cell = grid[r][c]
        if not cell.is_revealed:
            return None
        return cell.adjacent_mines

    def frontier_cell_score(
        self,
        game: MinesweeperGame,
        grid,
        pos: tuple[int, int],
        prob_map: dict[tuple[int, int], float],
        mine_set: set[tuple[int, int]],
        neighbors_of: dict[tuple[int, int], list[tuple[int, int]]],
    ) -> tuple[float, float, float]:
        """
        Guess-stage hook.

        Lower score is better. Default prefers lower mine probability, then
        higher local information gain, then a random tie-break.
        """
        r, c = pos
        p = prob_map.get(pos, 0.5)
        nbr_count = sum(
            1
            for nr, nc in neighbors_of[(r, c)]
            if not grid[nr][nc].is_revealed and not grid[nr][nc].is_flagged
        )
        return (p, -nbr_count, self._rng.random())

    def postprocess_prob_map(
        self,
        game: MinesweeperGame,
        grid,
        prob_map: dict[tuple[int, int], float],
        mine_set: set[tuple[int, int]],
        neighbors_of: dict[tuple[int, int], list[tuple[int, int]]],
    ) -> dict[tuple[int, int], float]:
        """Optional hook for mechanic-aware probability adjustments."""
        return prob_map

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def choose_action(self, game: MinesweeperGame) -> tuple[str, int, int]:
        cfg = game.get_config()
        rows, cols = cfg.rows, cfg.cols
        grid = [[game.get_cell(r, c) for c in range(cols)] for r in range(rows)]
        neighbors_of = {
            (r, c): self.neighbor_positions(game, r, c, rows, cols)
            for r in range(rows)
            for c in range(cols)
        }

        flag_limit = cfg.flag_limit
        flags_placed = sum(
            1 for r in range(rows) for c in range(cols) if grid[r][c].is_flagged
        )
        can_flag = flag_limit is None or flags_placed < flag_limit

        revealed_any = any(grid[r][c].is_revealed for r in range(rows) for c in range(cols))
        if not revealed_any:
            return self.opening_action(game, grid, rows, cols)

        action, mine_set = self._stage_p(game, grid, rows, cols, neighbors_of, can_flag)
        if action:
            return action

        action, prob_map = self._stage_a(
            game,
            grid,
            rows,
            cols,
            neighbors_of,
            cfg.mine_count,
            mine_set,
            can_flag,
        )
        if action:
            return action

        prob_map = self.postprocess_prob_map(
            game, grid, prob_map, mine_set, neighbors_of
        )
        return self._stage_g(game, grid, rows, cols, neighbors_of, prob_map, mine_set)

    # ------------------------------------------------------------------
    # Stage P
    # ------------------------------------------------------------------

    def _stage_p(
        self,
        game: MinesweeperGame,
        grid,
        rows: int,
        cols: int,
        neighbors_of: dict[tuple[int, int], list[tuple[int, int]]],
        can_flag: bool,
    ) -> tuple[tuple[str, int, int] | None, set[tuple[int, int]]]:
        changed = True
        safe_set: set[tuple[int, int]] = set()
        mine_set: set[tuple[int, int]] = set()

        while changed:
            changed = False
            for r in range(rows):
                for c in range(cols):
                    clue = self.clue_value(game, grid, r, c)
                    if clue is None or clue == 0:
                        continue
                    flagged, unrevealed = self._classify_neighbors(
                        grid, neighbors_of, r, c, virtual_mines=mine_set
                    )
                    remaining = clue - len(flagged)
                    if remaining < 0:
                        continue
                    if remaining == 0:
                        for pos in unrevealed:
                            if pos not in safe_set and pos not in mine_set:
                                safe_set.add(pos)
                                changed = True
                    elif remaining == len(unrevealed):
                        for pos in unrevealed:
                            if pos not in mine_set:
                                mine_set.add(pos)
                                changed = True

        if mine_set and can_flag:
            r, c = next(iter(mine_set))
            return ("flag", r, c), mine_set
        if safe_set:
            r, c = next(iter(safe_set))
            return ("reveal", r, c), mine_set
        return None, mine_set

    # ------------------------------------------------------------------
    # Stage A
    # ------------------------------------------------------------------

    def _stage_a(
        self,
        game: MinesweeperGame,
        grid,
        rows: int,
        cols: int,
        neighbors_of: dict[tuple[int, int], list[tuple[int, int]]],
        mine_count: int,
        mine_set: set[tuple[int, int]],
        can_flag: bool,
    ) -> tuple[tuple[str, int, int] | None, dict[tuple[int, int], float]]:
        unrevealed = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if not grid[r][c].is_revealed and not grid[r][c].is_flagged
            and (r, c) not in mine_set
        ]
        if not unrevealed:
            return None, {}

        frontier_set: set[tuple[int, int]] = set()
        for r, c in unrevealed:
            for nr, nc in neighbors_of[(r, c)]:
                clue = self.clue_value(game, grid, nr, nc)
                if clue is not None and clue > 0:
                    frontier_set.add((r, c))
                    break

        frontier = sorted(frontier_set)
        var_index = {pos: i for i, pos in enumerate(frontier)}
        non_frontier = [pos for pos in unrevealed if pos not in frontier_set]

        equations: list[tuple[list[int], int]] = []
        seen_constraints: set[frozenset] = set()
        total_flagged = sum(
            1 for r in range(rows) for c in range(cols) if grid[r][c].is_flagged
        )

        for r in range(rows):
            for c in range(cols):
                clue = self.clue_value(game, grid, r, c)
                if clue is None or clue == 0:
                    continue
                flagged_nbrs, unrevealed_nbrs = self._classify_neighbors(
                    grid, neighbors_of, r, c
                )
                frontier_nbrs = [pos for pos in unrevealed_nbrs if pos in frontier_set]
                if not frontier_nbrs:
                    continue
                rhs = clue - len(flagged_nbrs)
                key = frozenset(frontier_nbrs)
                if key in seen_constraints:
                    continue
                seen_constraints.add(key)
                equations.append(([var_index[pos] for pos in frontier_nbrs], rhs))

        n = len(frontier)
        matrix = [[Fraction(0)] * (n + 1) for _ in equations]
        for i, (vars_in_eq, rhs) in enumerate(equations):
            for vi in vars_in_eq:
                matrix[i][vi] = Fraction(1)
            matrix[i][n] = Fraction(rhs)

        self._gauss_eliminate(matrix, n)

        forced: dict[int, int] = {}
        for row in matrix:
            pivots = [j for j in range(n) if row[j] != 0]
            if len(pivots) == 1:
                j = pivots[0]
                val = row[n] / row[j]
                if val == 0:
                    forced[j] = 0
                elif val == 1:
                    forced[j] = 1

        for idx, val in forced.items():
            r, c = frontier[idx]
            if val == 0:
                return ("reveal", r, c), {}
            if val == 1 and can_flag:
                return ("flag", r, c), {}

        components = self._connected_components(frontier, equations, n)
        prob_map: dict[tuple[int, int], float] = {}

        for comp_vars in components:
            comp_eqs = [
                (vars_in_eq, rhs) for vars_in_eq, rhs in equations
                if any(v in comp_vars for v in vars_in_eq)
            ]
            if len(comp_vars) <= self._ENUM_CAP:
                comp_probs = self._enumerate_component(comp_vars, comp_eqs, forced)
            else:
                comp_probs = {
                    v: float(forced[v]) if v in forced else 0.5
                    for v in comp_vars
                }
            for vi, p in comp_probs.items():
                prob_map[frontier[vi]] = p

        mines_in_frontier_expected = sum(prob_map.get(pos, 0.5) for pos in frontier)
        remaining_mines = mine_count - total_flagged - mines_in_frontier_expected
        p_non_frontier = (
            max(0.0, min(1.0, remaining_mines / len(non_frontier)))
            if non_frontier else 0.5
        )
        for pos in non_frontier:
            prob_map[pos] = p_non_frontier

        if can_flag:
            for pos, p in prob_map.items():
                if p >= 1.0:
                    r, c = pos
                    return ("flag", r, c), prob_map

        return None, prob_map

    # ------------------------------------------------------------------
    # Stage G
    # ------------------------------------------------------------------

    def _stage_g(
        self,
        game: MinesweeperGame,
        grid,
        rows: int,
        cols: int,
        neighbors_of: dict[tuple[int, int], list[tuple[int, int]]],
        prob_map: dict[tuple[int, int], float],
        mine_set: set[tuple[int, int]],
    ) -> tuple[str, int, int]:
        unrevealed = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if not grid[r][c].is_revealed and not grid[r][c].is_flagged
            and (r, c) not in mine_set
        ]
        if not unrevealed:
            r, c = self._rng.choice([(r, c) for r in range(rows) for c in range(cols)])
            return ("reveal", r, c)

        r, c = min(
            unrevealed,
            key=lambda pos: self.frontier_cell_score(
                game, grid, pos, prob_map, mine_set, neighbors_of
            ),
        )
        return ("reveal", r, c)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _classify_neighbors(
        self,
        grid,
        neighbors_of: dict[tuple[int, int], list[tuple[int, int]]],
        r: int,
        c: int,
        virtual_mines: set[tuple[int, int]] | None = None,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        flagged, unrevealed = [], []
        for nr, nc in neighbors_of[(r, c)]:
            nbr = grid[nr][nc]
            if nbr.is_flagged or (virtual_mines and (nr, nc) in virtual_mines):
                flagged.append((nr, nc))
            elif not nbr.is_revealed:
                unrevealed.append((nr, nc))
        return flagged, unrevealed

    def _gauss_eliminate(self, matrix: list[list[Fraction]], n: int) -> None:
        m = len(matrix)
        pivot_row = 0
        for col in range(n):
            found = -1
            for row in range(pivot_row, m):
                if matrix[row][col] != 0:
                    found = row
                    break
            if found == -1:
                continue
            matrix[pivot_row], matrix[found] = matrix[found], matrix[pivot_row]
            pivot = matrix[pivot_row][col]
            matrix[pivot_row] = [x / pivot for x in matrix[pivot_row]]
            for row in range(m):
                if row == pivot_row or matrix[row][col] == 0:
                    continue
                factor = matrix[row][col]
                matrix[row] = [
                    matrix[row][j] - factor * matrix[pivot_row][j]
                    for j in range(n + 1)
                ]
            pivot_row += 1
            if pivot_row == m:
                break

    def _connected_components(
        self,
        frontier: list[tuple[int, int]],
        equations: list[tuple[list[int], int]],
        n: int,
    ) -> list[set[int]]:
        graph = {i: set() for i in range(n)}
        for vars_in_eq, _rhs in equations:
            for i in vars_in_eq:
                for j in vars_in_eq:
                    if i != j:
                        graph[i].add(j)
        seen: set[int] = set()
        comps: list[set[int]] = []
        for start in range(n):
            if start in seen:
                continue
            stack = [start]
            comp = set()
            seen.add(start)
            while stack:
                cur = stack.pop()
                comp.add(cur)
                for nxt in graph[cur]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            comps.append(comp)
        return comps

    def _enumerate_component(
        self,
        comp_vars: set[int],
        comp_eqs: list[tuple[list[int], int]],
        forced: dict[int, int],
    ) -> dict[int, float]:
        vars_list = sorted(comp_vars)
        assignable = [v for v in vars_list if v not in forced]
        if not assignable:
            return {v: float(forced[v]) for v in vars_list}

        valid = 0
        mine_sum = {v: 0 for v in vars_list}

        def backtrack(i: int, assignment: dict[int, int]) -> None:
            nonlocal valid
            if i == len(assignable):
                full = {**forced, **assignment}
                for eq_vars, rhs in comp_eqs:
                    if sum(full[v] for v in eq_vars) != rhs:
                        return
                valid += 1
                for v in vars_list:
                    mine_sum[v] += full[v]
                return

            v = assignable[i]
            for bit in (0, 1):
                assignment[v] = bit
                if self._partial_consistent(assignment, comp_eqs, forced):
                    backtrack(i + 1, assignment)
                assignment.pop(v, None)

        backtrack(0, {})
        if valid == 0:
            return {v: float(forced[v]) if v in forced else 0.5 for v in vars_list}
        return {v: mine_sum[v] / valid for v in vars_list}

    def _partial_consistent(
        self,
        assignment: dict[int, int],
        comp_eqs: list[tuple[list[int], int]],
        forced: dict[int, int],
    ) -> bool:
        for eq_vars, rhs in comp_eqs:
            known = 0
            unknown = 0
            for v in eq_vars:
                if v in assignment:
                    known += assignment[v]
                elif v in forced:
                    known += forced[v]
                else:
                    unknown += 1
            if known > rhs:
                return False
            if known + unknown < rhs:
                return False
        return True


def _entry_config(entry: dict[str, Any]) -> GameConfig:
    return GameConfig(**entry["config_snapshot"])


def _format_stats(label: str, stats: dict[str, Any]) -> str:
    return (
        f"{label}: "
        f"win={stats['win_rate']*100:.1f}% "
        f"progress={stats['avg_progress_fraction']*100:.1f}% "
        f"revealed={stats['avg_cells_revealed']:.1f} "
        f"mines_hit={stats['avg_mines_hit']:.2f} "
        f"turns={stats['avg_turns']:.1f}"
    )


def _save_results(path: str, results: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _load_results(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _get_client() -> OpenAI:
    if OpenAI is None:
        raise EnvironmentError(
            "Missing dependency: install the OpenAI client with `pip install openai`"
        )
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def _archive_summary_block(entry: dict[str, Any]) -> str:
    config_snapshot = dict(entry["config_snapshot"])
    parent_snapshot = entry.get("parent_snapshot")
    changed_fields = []
    if parent_snapshot:
        changed_fields = [
            key
            for key, value in config_snapshot.items()
            if parent_snapshot.get(key) != value
        ]
    lines = [
        f"description: {entry.get('description') or '(none)'}",
        f"generation: {entry.get('generation')}",
        f"changed_fields: {changed_fields}",
        "config_snapshot:",
        json.dumps(config_snapshot, indent=2),
    ]
    if parent_snapshot:
        lines.extend(["parent_snapshot:", json.dumps(parent_snapshot, indent=2)])
    if entry.get("code_kind"):
        lines.append(f"code_kind: {entry['code_kind']}")
    if entry.get("code_key"):
        lines.append(f"code_key: {entry['code_key']}")
    if entry.get("code_source"):
        lines.extend(["generated_mechanic_source:", entry["code_source"]])
    return "\n".join(lines)


def build_agent_mutation_prompt(entry: dict[str, Any]) -> str:
    archive_block = _archive_summary_block(entry)

    return f"""You are improving a self-contained PAFG-style Minesweeper agent for ONE generated game mechanic from archive.json.

Goal:
- Adapt the current archive-aware PAFG agent so it plays this specific mechanic better.
- Keep the solver recognizable as PAFG-style logic, but change reasoning where needed.
- Prefer pragmatic extensions over total rewrites.

Scope constraints:
- Target ONLY this single archive entry.
- Do not generate a new game mechanic. Generate a better agent.
- You MUST subclass `ArchiveAwarePAFGAgent`.
- Do not depend on any code from `agents/pafg_agent.py`.
- Keep the subclass SHORT. Prefer 1 hook override, at most 2 unless strictly necessary.

Archive entry:
{archive_block}

Available game API:
{_GAME_API_CHEATSHEET}

Allowed adaptation surface:
{_HOOK_SURFACE}

Return JSON only:
{{
  "name": "DescriptiveAgentClassName",
  "description": "one sentence about how the agent was adapted",
  "code": "<full Python source for exactly one subclass of ArchiveAwarePAFGAgent>",
  "assumptions": ["short note", "short note"]
}}

Hard constraints:
- Output exactly one Python class.
- No markdown outside the JSON.
- No import statements inside the generated code.
- The class constructor must accept `seed: int | None = None`.
- Prefer overriding hook methods like `neighbor_positions`, `opening_action`,
  `clue_value`, `frontier_cell_score`, or `postprocess_prob_map`.
- Override large internal solver methods only if absolutely necessary.
- The runtime already provides: ArchiveAwarePAFGAgent, GameConfig, MinesweeperGame, Fraction, random, deque.
- Do not import `Fraction`, `random`, or `deque`; use the provided names directly.
- Do not include any comments in the generated Python code.
- Your response must be complete valid JSON. Do not truncate the code.
- Keep the generated class under 120 lines whenever possible."""


def parse_agent_mutation_response(response_text: str) -> dict[str, Any] | None:
    text = re.sub(r"```[a-z]*\n?", "", response_text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    code = data.get("code")
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(code, str) or not code.strip():
        return None
    return data


def generate_agent_candidate(
    entry: dict[str, Any],
    *,
    model: str = MODEL_NAME,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[dict[str, Any] | None, str]:
    client = _get_client()
    prompt = build_agent_mutation_prompt(entry)
    token_budget = max_tokens
    last_error = None

    for _attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=token_budget,
                temperature=0.7,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            text = response.choices[0].message.content
            parsed = parse_agent_mutation_response(text)
            if parsed is not None:
                return parsed, text

            # Retry likely truncation/format failures with a stricter, shorter ask.
            if _attempt < 2:
                prompt = (
                    build_agent_mutation_prompt(entry)
                    + "\n\nYour previous answer was invalid or incomplete."
                      " Retry with a SHORTER class, at most 2 overridden hooks,"
                      " and return ONLY one complete JSON object."
                )
                token_budget = max(512, int(token_budget * 0.8))
                continue
            return None, text
        except Exception as e:
            msg = str(e)
            last_error = e
            if "402" in msg and "fewer max_tokens" in msg.lower():
                token_budget = max(512, int(token_budget * 0.75))
                continue
            raise

    raise last_error


def validate_agent_candidate(source: str, expected_name: str) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise CodeValidationError(f"syntax error: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise CodeValidationError("generated agent code may not contain import statements")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise CodeValidationError(f"forbidden name {node.id!r}")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRS:
            raise CodeValidationError(f"forbidden attribute {node.attr!r}")

    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if len(classes) != 1:
        raise CodeValidationError(
            f"generated code must define exactly one class, found {len(classes)}"
        )
    if classes[0].name != expected_name:
        raise CodeValidationError(
            f"generated class name {classes[0].name!r} does not match expected {expected_name!r}"
        )


def _sanitize_agent_source(source: str) -> str:
    """
    Remove a small safe set of redundant imports that the runtime already
    provides. Anything outside this allowlist remains subject to validation.
    """
    sanitized_lines: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped in {
            "from fractions import Fraction",
            "from collections import deque",
            "import random",
        }:
            continue
        sanitized_lines.append(line)
    return "\n".join(sanitized_lines)


def compile_agent_candidate(source: str, expected_name: str) -> type[ArchiveAwarePAFGAgent]:
    source = _sanitize_agent_source(source)
    validate_agent_candidate(source, expected_name)

    globals_dict = {
        "__builtins__": __builtins__,
        "__name__": "<generated_archive_pafg_agent>",
        "ArchiveAwarePAFGAgent": ArchiveAwarePAFGAgent,
        "GameConfig": GameConfig,
        "MinesweeperGame": MinesweeperGame,
        "Fraction": Fraction,
        "random": random,
        "deque": deque,
    }
    local_ns: dict[str, Any] = {}
    try:
        exec(source, globals_dict, local_ns)
    except Exception as e:
        raise CodeValidationError(
            f"generated agent exec failed: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        ) from e

    cls = local_ns.get(expected_name) or globals_dict.get(expected_name)
    if not isinstance(cls, type):
        raise CodeValidationError(
            f"generated code did not define expected class {expected_name!r}"
        )
    if not issubclass(cls, ArchiveAwarePAFGAgent):
        raise CodeValidationError(
            f"generated class {expected_name!r} is not a subclass of ArchiveAwarePAFGAgent"
        )
    return cls


def run_game(
    agent,
    config: GameConfig,
    *,
    seed: int | None = None,
    max_turns: int = 1000,
) -> dict[str, Any]:
    game = MinesweeperGame(config, seed=seed)
    turns = 0
    while game.get_state() in (GameState.PENDING, GameState.ACTIVE) and turns < max_turns:
        action, r, c = agent.choose_action(game)
        if action == "reveal":
            game.reveal(r, c)
        else:
            game.flag(r, c)
        turns += 1
    return {**game.get_player_metrics(), "state": game.get_state().value, "turns": turns}


def evaluate_config(
    agent_class,
    config: GameConfig,
    *,
    n_games: int,
    base_seed: int = 0,
) -> dict[str, Any]:
    safe_cells = config.rows * config.cols - config.mine_count
    results = []
    for i in range(n_games):
        seed = base_seed + i
        agent = agent_class(seed=seed)
        results.append(run_game(agent, config, seed=seed))

    wins = sum(1 for r in results if r["state"] == "won")
    avg_revealed = sum(r["cells_revealed"] for r in results) / n_games
    return {
        "win_rate": wins / n_games,
        "avg_cells_revealed": avg_revealed,
        "avg_progress_fraction": avg_revealed / max(safe_cells, 1),
        "avg_mines_hit": sum(r["mines_hit"] for r in results) / n_games,
        "avg_turns": sum(r["turns"] for r in results) / n_games,
        "n_games": n_games,
    }


def try_evaluate_agent_candidate(
    entry: dict[str, Any],
    candidate: dict[str, Any],
    *,
    n_games: int,
    base_seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    config = _entry_config(entry)
    baseline_stats = evaluate_config(
        ArchiveAwarePAFGAgent,
        config,
        n_games=n_games,
        base_seed=base_seed,
    )
    try:
        candidate_cls = compile_agent_candidate(candidate["code"], candidate["name"])
        candidate_stats = evaluate_config(
            candidate_cls,
            config,
            n_games=n_games,
            base_seed=base_seed,
        )
    except Exception as e:
        return baseline_stats, None, f"{type(e).__name__}: {e}"
    return baseline_stats, candidate_stats, None


def _process_entry(
    entry: dict[str, Any],
    *,
    archive_index: int,
    model: str,
    max_tokens: int,
    n_games: int,
    base_seed: int,
    print_only: bool,
) -> dict[str, Any]:
    print(f"[{archive_index}] Archive entry:")
    print(_archive_summary_block(entry))

    result: dict[str, Any] = {
        "archive_index": archive_index,
        "archive_description": entry.get("description", ""),
        "config_snapshot": entry.get("config_snapshot"),
        "code_kind": entry.get("code_kind"),
        "code_key": entry.get("code_key"),
        "candidate": None,
        "raw_llm_response": None,
        "baseline_stats": None,
        "candidate_stats": None,
        "candidate_error": None,
    }

    candidate, raw_response = generate_agent_candidate(
        entry,
        model=model,
        max_tokens=max_tokens,
    )
    result["raw_llm_response"] = raw_response
    if candidate is None:
        result["candidate_error"] = "LLM response could not be parsed"
        print()
        print(f"[{archive_index}] Candidate parse failed.")
        return result

    result["candidate"] = candidate
    print()
    print(f"[{archive_index}] Generated candidate:")
    print(json.dumps(candidate, indent=2))

    if print_only:
        return result

    print()
    baseline_stats, candidate_stats, candidate_error = try_evaluate_agent_candidate(
        entry,
        candidate,
        n_games=n_games,
        base_seed=base_seed,
    )
    result["baseline_stats"] = baseline_stats
    result["candidate_stats"] = candidate_stats
    result["candidate_error"] = candidate_error

    print(f"[{archive_index}] Performance:")
    print(_format_stats("archive-aware-pafg", baseline_stats))
    if candidate_error is not None:
        print(f"{candidate['name']}: crashed during evaluation ({candidate_error})")
    else:
        print(_format_stats(candidate["name"], candidate_stats))
    return result


def _replay_entry(
    entry: dict[str, Any],
    saved_result: dict[str, Any],
    *,
    n_games: int,
    base_seed: int,
) -> dict[str, Any]:
    archive_index = saved_result.get("archive_index")
    print(f"[{archive_index}] Replaying saved candidate:")
    print(_archive_summary_block(entry))

    candidate = saved_result.get("candidate")
    updated = dict(saved_result)
    updated["baseline_stats"] = None
    updated["candidate_stats"] = None
    updated["candidate_error"] = None

    if not candidate:
        updated["candidate_error"] = saved_result.get("candidate_error") or "No saved candidate"
        print(f"[{archive_index}] No saved candidate to evaluate.")
        return updated

    print()
    print(f"[{archive_index}] Saved candidate:")
    print(json.dumps(candidate, indent=2))
    print()

    baseline_stats, candidate_stats, candidate_error = try_evaluate_agent_candidate(
        entry,
        candidate,
        n_games=n_games,
        base_seed=base_seed,
    )
    updated["baseline_stats"] = baseline_stats
    updated["candidate_stats"] = candidate_stats
    updated["candidate_error"] = candidate_error

    print(f"[{archive_index}] Replay performance:")
    print(_format_stats("archive-aware-pafg", baseline_stats))
    if candidate_error is not None:
        print(f"{candidate['name']}: crashed during evaluation ({candidate_error})")
    else:
        print(_format_stats(candidate["name"], candidate_stats))
    return updated


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate an archive-specific self-contained PAFG agent candidate with OpenRouter."
    )
    p.add_argument("--archive", default="archive.json", help="Archive path")
    p.add_argument("--index", type=int, default=0, help="0-based archive entry index")
    p.add_argument("--all", action="store_true", help="Process every archive entry")
    p.add_argument("--model", default=MODEL_NAME, help="OpenRouter model name")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max completion tokens for agent-generation calls (default: {DEFAULT_MAX_TOKENS})",
    )
    p.add_argument("--games", type=int, default=5, help="Games per agent evaluation")
    p.add_argument("--seed", type=int, default=0, help="Base seed for fair comparison")
    p.add_argument("--out", default=DEFAULT_OUTPUT_PATH, help="Results JSON path")
    p.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help=f"Seconds to wait between LLM calls in --all mode (default: {DEFAULT_DELAY_SECONDS})",
    )
    p.add_argument("--print-only", action="store_true", help="Skip evaluation")
    p.add_argument("--resume", action="store_true", help="Resume from existing output")
    p.add_argument(
        "--replay-from",
        default=None,
        help="Re-evaluate saved candidates from an existing results JSON without any LLM calls",
    )
    args = p.parse_args()

    entries = load_archive_file(args.archive)
    if not entries:
        raise SystemExit(f"No archive entries found in {args.archive!r}")
    if not args.all and not (0 <= args.index < len(entries)):
        raise SystemExit(f"Index {args.index} out of range for {len(entries)} entries")
    register_all_from_archive({i: e for i, e in enumerate(entries)})

    if args.replay_from:
        saved_results = _load_results(args.replay_from)
        if not saved_results:
            raise SystemExit(f"No saved results found in {args.replay_from!r}")

        filtered = saved_results
        if not args.all:
            filtered = [r for r in saved_results if r.get("archive_index") == args.index]
            if not filtered:
                raise SystemExit(f"No saved candidate found for archive index {args.index}")

        replayed_results: list[dict[str, Any]] = []
        for saved in filtered:
            archive_index = saved.get("archive_index")
            if archive_index is None or not (0 <= archive_index < len(entries)):
                updated = dict(saved)
                updated["candidate_error"] = f"archive_index {archive_index!r} missing from archive"
                replayed_results.append(updated)
                continue
            replayed_results.append(
                _replay_entry(
                    entries[archive_index],
                    saved,
                    n_games=args.games,
                    base_seed=args.seed,
                )
            )

        _save_results(args.out, replayed_results)
        print()
        print(f"Saved {len(replayed_results)} replayed result(s) to {args.out}")
        return

    indices = list(range(len(entries))) if args.all else [args.index]
    saved_results: list[dict[str, Any]] = _load_results(args.out) if args.resume else []
    done_indexes = {r.get("archive_index") for r in saved_results}
    if args.resume:
        indices = [i for i in indices if i not in done_indexes]

    for pos, archive_index in enumerate(indices):
        if pos > 0 and args.delay > 0:
            print(f"\nWaiting {args.delay:.1f}s before next LLM call...")
            time.sleep(args.delay)

        result = _process_entry(
            entries[archive_index],
            archive_index=archive_index,
            model=args.model,
            max_tokens=args.max_tokens,
            n_games=args.games,
            base_seed=args.seed,
            print_only=args.print_only,
        )
        saved_results.append(result)
        _save_results(args.out, saved_results)

    print()
    print(f"Saved {len(saved_results)} result(s) to {args.out}")


if __name__ == "__main__":
    main()
