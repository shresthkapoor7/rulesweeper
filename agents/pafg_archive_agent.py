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

from agents import Agent, evaluate_config, run_game
from code_mutations import CodeValidationError, load_archive_file, register_all_from_archive
from config import GameConfig
from game import MinesweeperGame


LLM_TIMEOUT_SECONDS = 60.0
MODEL_NAME = "anthropic/claude-sonnet-4.6"
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
  game.behavior_summary() -> dict[str, dict]
      keys: "mine_behavior", "info_strategy", "win_condition",
            "neighborhood", "reveal_strategy"
      each value: {"name": <registry-key str>, "params": {...}}
      Use this to read knobs like drift_prob, lie_prob, target_turns,
      neighborhood offsets, etc. The default `clue_value` hook still
      reads `cell.adjacent_mines` (the TRUE count) — these params are
      for tuning your hooks, not decoding the displayed clue.
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
  def action_priority(self, game, grid, rows, cols) -> str
  def frontier_cell_score(self, game, grid, pos, prob_map, mine_set, neighbors_of) -> tuple
  def postprocess_prob_map(self, game, grid, prob_map, mine_set, neighbors_of) -> dict

`action_priority` returns one of:
  "reveal-first" (default) — when both safe cells and known mines are deduced,
                              reveal the safe cell first. Optimal under the
                              standard reveal-all-safes objective.
  "flag-first"             — flag known mines first. Use when the win
                              condition rewards flagging (e.g. flag-all-mines).
  "survival"               — if Stage G's best guess has high mine probability,
                              flag a deduced mine instead of risking a reveal.
                              Use under survival-style objectives.

Available instance state on `self` (set by ArchiveAwarePAFGAgent.__init__):
  self._rng — random.Random, seeded from the constructor's `seed` arg.
              Use this for any randomness; do NOT introduce `self.rng`
              (no underscore) or call the bare `random` module for stochastic
              tie-breaks — that defeats reproducibility.

Do not override large internal solver methods like `_stage_p`, `_stage_a`, or `_stage_g`
unless absolutely necessary. Prefer a class with 1-2 hook overrides."""


class ArchiveAwarePAFGAgent(Agent):
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
        """
        Opening move hook.

        Default opens at the corner under cascade reveal (most flood-fill is
        likely to spill across the board from a corner zero). Under any other
        reveal strategy, the corner is a poor first click (no cascade follows
        and edge cells carry less information), so default to board center.
        """
        if game.get_config().reveal_strategy == "cascade":
            return ("reveal", 0, 0)
        return ("reveal", rows // 2, cols // 2)

    def action_priority(
        self,
        game: MinesweeperGame,
        grid,
        rows: int,
        cols: int,
    ) -> str:
        """
        Action-ordering hook.

        Returns one of "reveal-first" (default), "flag-first", or "survival".
        See `_HOOK_SURFACE` for semantics.
        """
        return "reveal-first"

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
        return self._stage_g(
            game, grid, rows, cols, neighbors_of, prob_map, mine_set, can_flag
        )

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

        prefer_flag = self.action_priority(game, grid, rows, cols) == "flag-first"
        if prefer_flag:
            if mine_set and can_flag:
                r, c = next(iter(mine_set))
                return ("flag", r, c), mine_set
            if safe_set:
                r, c = next(iter(safe_set))
                return ("reveal", r, c), mine_set
        else:
            if safe_set:
                r, c = next(iter(safe_set))
                return ("reveal", r, c), mine_set
            if mine_set and can_flag:
                r, c = next(iter(mine_set))
                return ("flag", r, c), mine_set
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

        prefer_flag = self.action_priority(game, grid, rows, cols) == "flag-first"
        if prefer_flag:
            for idx, val in forced.items():
                if val == 1 and can_flag:
                    r, c = frontier[idx]
                    return ("flag", r, c), {}
            for idx, val in forced.items():
                if val == 0:
                    r, c = frontier[idx]
                    return ("reveal", r, c), {}
        else:
            for idx, val in forced.items():
                if val == 0:
                    r, c = frontier[idx]
                    return ("reveal", r, c), {}
            for idx, val in forced.items():
                if val == 1 and can_flag:
                    r, c = frontier[idx]
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
        can_flag: bool = True,
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

        if self.action_priority(game, grid, rows, cols) == "survival" and can_flag:
            best_p = prob_map.get((r, c), 0.5)
            if best_p > 0.4:
                unflagged_mines = [
                    pos for pos in mine_set if not grid[pos[0]][pos[1]].is_flagged
                ]
                if unflagged_mines:
                    mr, mc = unflagged_mines[0]
                    return ("flag", mr, mc)
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


# ---------------------------------------------------------------------------
# Exemplar subclasses — embedded in the LLM prompt to anchor expectations.
#
# These are deliberately tiny: each one overrides 1-2 hooks with a focused
# rationale tied to a specific named mechanic. They are NOT used at runtime;
# their only purpose is to give the prompt concrete reference code so the LLM
# converges on the "small, surgical hook override" style rather than rewriting
# the solver.
# ---------------------------------------------------------------------------


class _ExemplarFlagFirstAgent(ArchiveAwarePAFGAgent):
    """Adapts to win_condition='flag-all-mines' by flagging deduced mines first."""

    def action_priority(self, game, grid, rows, cols):
        return "flag-first"


class _ExemplarChainReactionAgent(ArchiveAwarePAFGAgent):
    """Adapts to mine_behavior='chain-reaction' by penalizing high-density frontier guesses."""

    def frontier_cell_score(self, game, grid, pos, prob_map, mine_set, neighbors_of):
        r, c = pos
        p = prob_map.get(pos, 0.5)
        suspected = sum(
            1
            for nr, nc in neighbors_of[(r, c)]
            if (nr, nc) in mine_set or prob_map.get((nr, nc), 0.0) >= 0.5
        )
        return (p + 0.05 * suspected, -suspected, self._rng.random())


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


_NAMED_MECHANIC_PRIMER: dict[tuple[str, str], str] = {
    ("info_strategy", "count-flags"):
        "Cell shows count of FLAGGED neighbors, not mines. cell.adjacent_mines is "
        "still the true mine count, so the default clue_value remains correct for "
        "deduction. Players see flag counts in the rendered board.",
    ("info_strategy", "parity"):
        "Display shows 'E' / 'O' (even/odd) instead of an integer. cell.adjacent_mines "
        "is still the true integer count, so the default clue_value is correct. Do not "
        "decode parity into clue_value — that loses information.",
    ("info_strategy", "distance"):
        "Display shows Chebyshev distance to nearest mine on the whole board. "
        "cell.adjacent_mines is still the true LOCAL adjacent count for the agent.",
    ("info_strategy", "direction"):
        "Display shows a compass arrow toward the nearest mine. cell.adjacent_mines "
        "is still the true local adjacent count for the agent.",
    ("info_strategy", "noisy-count"):
        "Displayed clue is true count ±1 with 20% probability per cell (deterministic "
        "per-cell). cell.adjacent_mines is the true count; the default clue_value "
        "reads it. Decoding the displayed (noisy) value yields lossy constraints.",
    ("neighborhood", "von-neumann"):
        "Adjacency is the 4 orthogonal neighbors only. Each clue carries less "
        "information than Moore. neighbor_positions already delegates to "
        "board.neighbors() — do not hardcode offsets.",
    ("neighborhood", "diagonal"):
        "Adjacency is the 4 diagonal neighbors only. Topology already delegated.",
    ("neighborhood", "knight"):
        "Adjacency is the 8 chess-knight L-moves; cells are non-locally adjacent. "
        "Topology already delegated; cascade jumps non-locally.",
    ("neighborhood", "radius-2-moore"):
        "Adjacency is all 24 cells in a 5x5 box. Clues cover wide areas and "
        "constraints overlap densely. Topology already delegated.",
    ("win_condition", "reveal-quota"):
        "Win when at least half the safe cells are revealed (no need to clear the "
        "board). Reveals progress the win directly; flagging does not. Default "
        "action_priority='reveal-first' is correct.",
    ("win_condition", "flag-all-mines"):
        "Win requires EVERY mine flagged AND no false flags. Pure reveal cannot win. "
        "Override action_priority to return 'flag-first' so deduced mines are flagged "
        "before safe reveals. Never speculatively flag (false flags lose the game).",
    ("win_condition", "survival"):
        "Win by surviving N actions. Override action_priority to return 'survival' so "
        "the agent prefers flagging a deduced mine over a high-probability guess in "
        "Stage G.",
    ("mine_behavior", "drifting"):
        "Each turn, every unflagged mine has a 30% chance to walk into an adjacent "
        "unrevealed cell. Adjacency numbers update in place; past inferences decay. "
        "Flagging pins a mine. Prefer flagging deduced mines early to stabilize.",
    ("mine_behavior", "chain-reaction"):
        "Hitting one mine triggers BFS damage through every connected unflagged mine "
        "cluster. The cost of hitting a mine scales with cluster size. In "
        "frontier_cell_score, weigh mine_probability against suspected cluster size, "
        "not just probability.",
    ("reveal_strategy", "single"):
        "Each reveal uncovers EXACTLY one cell — no cascade. Information per turn "
        "drops sharply; the default opening_action at (0,0) yields almost nothing. "
        "Consider opening with a board-center cell or any cell most likely to be "
        "non-mine, even though no flood-fill follows.",
}


_DEFAULT_FIELD_VALUES: dict[str, Any] = {
    "info_strategy":   "count-mines",
    "reveal_strategy": "cascade",
    "mine_behavior":   "static",
    "neighborhood":    "moore",
    "win_condition":   "standard",
    "safe_first_click": True,
}


def _joint_mechanic_caution(config_snapshot: dict[str, Any]) -> str:
    """
    When 2+ mechanics are simultaneously off-default, warn the agent generator
    against trying to neutralize every mechanic at once. Returns an empty
    string when only one (or zero) non-default mechanic is active.
    """
    nondefault = [
        f"{field}={config_snapshot.get(field)!r}"
        for field, default in _DEFAULT_FIELD_VALUES.items()
        if config_snapshot.get(field) != default
    ]
    if len(nondefault) < 2:
        return ""
    return (
        f"Multiple non-default mechanics are active simultaneously: "
        f"{', '.join(nondefault)}. Pick the mechanic that most distorts "
        "deduction (usually info_strategy or win_condition) and adapt for "
        "that ONE with one or two hook overrides. Do not try to neutralize "
        "every mechanic at once — the resulting subclass tends to be brittle. "
        "Side effects from the other mechanics will still be handled by "
        "ArchiveAwarePAFGAgent's defaults."
    )


def _named_mechanic_primer(config_snapshot: dict[str, Any]) -> str:
    notes: list[str] = []
    for field in (
        "info_strategy",
        "neighborhood",
        "win_condition",
        "mine_behavior",
        "reveal_strategy",
    ):
        value = config_snapshot.get(field)
        if not isinstance(value, str) or value.startswith("gen-"):
            continue
        note = _NAMED_MECHANIC_PRIMER.get((field, value))
        if note:
            notes.append(f"- {field}={value}: {note}")
    return "\n".join(notes)


_GEN_FIELDS: tuple[str, ...] = (
    "mine_behavior", "reveal_strategy", "info_strategy",
    "neighborhood", "win_condition",
)


def _entry_key(entry: dict[str, Any]) -> str:
    """Stable key for an entry's config snapshot. Mirrors mortar._config_key."""
    return json.dumps(entry["config_snapshot"], sort_keys=True)


def _active_generated_mechanics(
    entry: dict[str, Any],
    archive: dict[str, dict] | None,
) -> list[dict[str, Any]]:
    """
    Walk the lineage to surface every ``gen-*`` mechanic currently active on
    this entry's config. Returns a list of dicts with keys ``field``, ``key``,
    ``description``, ``code_source``.

    When ``archive`` is None or the introducing ancestor is unreachable, falls
    back to whatever data lives directly on ``entry`` (the legacy single-source
    behavior). For multi-stack lineages this is the surface that lets the
    LLM-authored agent reason about complementarity at the code level.
    """
    config_snapshot = entry.get("config_snapshot", {})
    by_key: dict[str, dict] = {}
    if archive:
        by_key = {_entry_key(e): e for e in archive.values()}

    # Build the lineage chain (parent included) by walking parent_snapshot.
    chain: list[dict[str, Any]] = [entry]
    cur = entry
    seen_keys: set[str] = {_entry_key(entry)}
    while True:
        parent_snapshot = cur.get("parent_snapshot")
        if not parent_snapshot:
            break
        parent_key = json.dumps(parent_snapshot, sort_keys=True)
        if parent_key in seen_keys:
            break
        parent_entry = by_key.get(parent_key)
        if parent_entry is None:
            break
        chain.append(parent_entry)
        seen_keys.add(parent_key)
        cur = parent_entry
        if len(chain) > 32:
            break

    results: list[dict[str, Any]] = []
    for field in _GEN_FIELDS:
        value = config_snapshot.get(field)
        if not (isinstance(value, str) and value.startswith("gen-")):
            continue
        introducer = next(
            (
                e for e in chain
                if e.get("code_kind") == field and e.get("code_key") == value
            ),
            None,
        )
        if introducer is None:
            # Fallback: if the entry itself happens to carry the source for
            # this field, use it; otherwise emit a stub.
            if entry.get("code_kind") == field and entry.get("code_key") == value:
                introducer = entry
        if introducer is None:
            results.append({
                "field": field,
                "key": value,
                "description": "(introducing ancestor not in archive — source unavailable)",
                "code_source": None,
            })
            continue
        results.append({
            "field": field,
            "key": value,
            "description": introducer.get("description") or "(no description)",
            "code_source": introducer.get("code_source"),
        })
    return results


def _format_active_generated_block(
    mechanics: list[dict[str, Any]],
) -> str:
    if not mechanics:
        return ""
    blocks: list[str] = []
    for m in mechanics:
        header = f"# Active mechanic: {m['field']}={m['key']} — {m['description']}"
        source = m.get("code_source") or "# (source unavailable)"
        blocks.append(f"{header}\n{source}")
    return "\n\n".join(blocks)


def _archive_summary_block(
    entry: dict[str, Any],
    archive: dict[str, dict] | None = None,
) -> str:
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
    primer = _named_mechanic_primer(config_snapshot)
    if primer:
        lines.extend(["named_mechanic_primer:", primer])
    caution = _joint_mechanic_caution(config_snapshot)
    if caution:
        lines.extend(["joint_mechanic_caution:", caution])
    if parent_snapshot:
        lines.extend(["parent_snapshot:", json.dumps(parent_snapshot, indent=2)])

    # Surface every active gen-* mechanic on this config (not just the most
    # recently introduced one). Falls back to the entry's own code_source if
    # the lineage walk can't reach the introducing ancestor.
    active = _active_generated_mechanics(entry, archive)
    if active:
        lines.append("active_generated_mechanics:")
        lines.append(_format_active_generated_block(active))
    elif entry.get("code_source"):
        # No gen-* fields on this config but the entry itself carries source —
        # legacy fallback for synthesized one-off entries.
        if entry.get("code_kind"):
            lines.append(f"code_kind: {entry['code_kind']}")
        if entry.get("code_key"):
            lines.append(f"code_key: {entry['code_key']}")
        lines.extend(["generated_mechanic_source:", entry["code_source"]])
    return "\n".join(lines)


_EXEMPLAR_AGENTS: tuple[type, ...] = (
    _ExemplarFlagFirstAgent,
    _ExemplarChainReactionAgent,
)


def _exemplar_agents_block() -> str:
    """Render the exemplar subclasses' source so the prompt can embed them."""
    blocks: list[str] = []
    for cls in _EXEMPLAR_AGENTS:
        try:
            blocks.append(inspect.getsource(cls).rstrip())
        except OSError:
            continue
    return "\n\n".join(blocks)


def build_agent_mutation_prompt(
    entry: dict[str, Any],
    archive: dict[str, dict] | None = None,
) -> str:
    archive_block = _archive_summary_block(entry, archive)
    exemplars_block = _exemplar_agents_block()

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

Exemplar subclasses (the SHAPE you should aim for — short, hook-only):
{exemplars_block}

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
    archive: dict[str, dict] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    client = _get_client()
    prompt = build_agent_mutation_prompt(entry, archive)
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
                    build_agent_mutation_prompt(entry, archive)
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


def _build_repair_prompt(
    entry: dict[str, Any],
    candidate: dict[str, Any],
    error: str,
    archive: dict[str, dict] | None = None,
) -> str:
    return f"""Your previous PAFG agent candidate failed during evaluation.

Error:
{error}

Previous candidate code (class `{candidate.get("name")}`):
{candidate.get("code")}

Original task context:
{_archive_summary_block(entry, archive)}

Available game API:
{_GAME_API_CHEATSHEET}

Allowed adaptation surface:
{_HOOK_SURFACE}

Repair the candidate. Return JSON ONLY in this schema:
{{
  "name": "{candidate.get('name')}",
  "description": "one sentence about what was fixed",
  "code": "<full corrected Python source for exactly one subclass of ArchiveAwarePAFGAgent>",
  "assumptions": []
}}

Hard constraints:
- Keep the class name `{candidate.get('name')}` unchanged.
- Output exactly one Python class.
- No markdown outside the JSON.
- No import statements.
- The class constructor must accept `seed: int | None = None`.
- The runtime already provides: ArchiveAwarePAFGAgent, GameConfig, MinesweeperGame, Fraction, random, deque.
- Do not include any comments in the generated code."""


def repair_agent_candidate(
    entry: dict[str, Any],
    candidate: dict[str, Any],
    error: str,
    *,
    model: str = MODEL_NAME,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    archive: dict[str, dict] | None = None,
) -> dict[str, Any] | None:
    client = _get_client()
    prompt = _build_repair_prompt(entry, candidate, error, archive)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.5,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        text = response.choices[0].message.content
        return parse_agent_mutation_response(text)
    except Exception:
        return None


def try_evaluate_agent_candidate(
    entry: dict[str, Any],
    candidate: dict[str, Any],
    *,
    n_games: int,
    base_seed: int = 0,
    repair_on_error: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    config = _entry_config(entry)
    baseline_stats = evaluate_config(
        ArchiveAwarePAFGAgent,
        config,
        n_games=n_games,
        base_seed=base_seed,
    )

    def _run(cand: dict[str, Any]) -> dict[str, Any]:
        candidate_cls = compile_agent_candidate(cand["code"], cand["name"])
        return evaluate_config(
            candidate_cls,
            config,
            n_games=n_games,
            base_seed=base_seed,
        )

    try:
        return baseline_stats, _run(candidate), None
    except Exception as e:
        first_error = f"{type(e).__name__}: {e}"

    if not repair_on_error:
        return baseline_stats, None, first_error

    print(f"  [repair] candidate failed ({first_error}); attempting one repair...")
    repaired = repair_agent_candidate(entry, candidate, first_error)
    if repaired is None:
        return baseline_stats, None, f"{first_error} | repair LLM call failed"

    candidate.update(repaired)
    try:
        return baseline_stats, _run(candidate), None
    except Exception as e:
        return (
            baseline_stats,
            None,
            f"{first_error} | repair retry failed: {type(e).__name__}: {e}",
        )


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
        repair_on_error=False,
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
    p.add_argument("--games", type=int, default=20, help="Games per agent evaluation")
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
