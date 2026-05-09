from __future__ import annotations
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from board import Board


class InfoStrategy(ABC):
    """
    Decides what symbol/text a revealed safe cell shows the player.
    Purely a display concern — the cell's true adjacent_mines integer is still
    maintained by Board.recompute_adjacency, so cascade and reveal mechanics
    are unaffected. The renderer asks encode(board, r, c) when drawing.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed

    @abstractmethod
    def encode(self, board: "Board", r: int, c: int) -> str:
        """Return display text for cell (r, c). Empty string renders as blank."""
        ...

    def summary(self) -> dict:
        """Key parameters for this strategy. Override on knob-bearing subclasses."""
        return {}


class CountMinesInfo(InfoStrategy):
    """Canonical minesweeper: count of adjacent mines, blank for zero."""

    def encode(self, board, r, c):
        n = board.grid[r][c].adjacent_mines
        return str(n) if n > 0 else ""


class CountFlagsInfo(InfoStrategy):
    """
    Show how many neighbors are *flagged*, not how many are mines. Player
    must mentally subtract — turns flagging into part of the deduction loop.
    """

    def encode(self, board, r, c):
        n = sum(
            1 for nr, nc in board.neighbors(r, c)
            if board.grid[nr][nc].is_flagged
        )
        return str(n) if n > 0 else ""


class ParityInfo(InfoStrategy):
    """Show only even/odd of the mine count. Strictly less info — much harder."""

    def encode(self, board, r, c):
        n = board.grid[r][c].adjacent_mines
        if n == 0:
            return ""
        return "E" if n % 2 == 0 else "O"


class DistanceInfo(InfoStrategy):
    """Chebyshev distance to the nearest mine on the entire board (capped at 9)."""

    def encode(self, board, r, c):
        best = None
        for rr in range(board.config.rows):
            for cc in range(board.config.cols):
                if board.grid[rr][cc].is_mine:
                    d = max(abs(rr - r), abs(cc - c))
                    if best is None or d < best:
                        best = d
        if best is None or best == 0:
            return ""
        return str(min(best, 9))


class DirectionInfo(InfoStrategy):
    """Arrow toward the nearest mine (Chebyshev). 8 compass directions."""

    _ARROWS = {
        (-1, -1): "↖",  # ↖
        (-1,  0): "↑",  # ↑
        (-1,  1): "↗",  # ↗
        ( 0, -1): "←",  # ←
        ( 0,  1): "→",  # →
        ( 1, -1): "↙",  # ↙
        ( 1,  0): "↓",  # ↓
        ( 1,  1): "↘",  # ↘
    }

    def encode(self, board, r, c):
        best_d = None
        best_target = None
        for rr in range(board.config.rows):
            for cc in range(board.config.cols):
                if board.grid[rr][cc].is_mine:
                    d = max(abs(rr - r), abs(cc - c))
                    if best_d is None or d < best_d:
                        best_d = d
                        best_target = (rr, cc)
        if best_target is None:
            return ""
        dr = best_target[0] - r
        dc = best_target[1] - c
        # Reduce to a unit step in the dominant direction
        sr = (dr > 0) - (dr < 0)
        sc = (dc > 0) - (dc < 0)
        return self._ARROWS.get((sr, sc), "")


class NoisyCountInfo(InfoStrategy):
    """
    True adjacent-mine count, but with a fixed lie probability. Lies are
    deterministic per (board-seed, r, c) so the displayed value is stable
    across re-renders within a game.
    """

    DEFAULT_LIE_PROB = 0.2

    def __init__(
        self,
        seed: int | None = None,
        lie_prob: float | None = None,
    ) -> None:
        super().__init__(seed)
        self.lie_prob = lie_prob if lie_prob is not None else self.DEFAULT_LIE_PROB

    def summary(self) -> dict:
        return {"lie_prob": self.lie_prob}

    def encode(self, board, r, c):
        true_n = board.grid[r][c].adjacent_mines
        # Per-cell deterministic noise: hash board seed + coordinates so each
        # render of the same cell shows the same lie. random.Random accepts
        # int/str/bytes, so combine via hash().
        rng = random.Random(hash((self._seed, r, c)))
        if rng.random() < self.lie_prob:
            offset = rng.choice((-1, 1))
            displayed = max(0, true_n + offset)
        else:
            displayed = true_n
        return str(displayed) if displayed > 0 else ""


INFO_STRATEGIES: dict[str, type[InfoStrategy]] = {
    "count-mines": CountMinesInfo,
    "count-flags": CountFlagsInfo,
    "parity":      ParityInfo,
    "distance":    DistanceInfo,
    "direction":   DirectionInfo,
    "noisy-count": NoisyCountInfo,
}
