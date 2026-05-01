from __future__ import annotations
import random
from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from board import Board
    from game import MinesweeperGame


class MineBehavior(ABC):
    """
    Pluggable per-action behavior for mines. The MinesweeperGame invokes
    on_post_action after every reveal/flag while state == ACTIVE. A behavior
    may mutate the board (move mines, recompute adjacency), reveal additional
    cells, and/or apply player damage. The game re-checks win/loss after the
    hook returns, so behaviors must not call back into game.reveal/game.flag.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    @abstractmethod
    def on_post_action(
        self,
        board: Board,
        game: MinesweeperGame,
        action: dict,
    ) -> None:
        ...


class StaticMines(MineBehavior):
    """Default — preserves canonical minesweeper. Mines never move or react."""

    def on_post_action(self, board, game, action):
        return


class DriftingMines(MineBehavior):
    """
    Each turn, every unflagged mine has drift_prob chance of walking into an
    adjacent cell that is unrevealed, unflagged, and not already a mine.
    Flagged mines stay pinned, which keeps flagging meaningful as a deduction
    tool in this mode. After all drift moves resolve, adjacency is recomputed
    once so already-revealed numbers update in place.
    """

    DEFAULT_DRIFT_PROB = 0.3

    def __init__(
        self,
        seed: int | None = None,
        drift_prob: float | None = None,
    ) -> None:
        super().__init__(seed)
        self.drift_prob = drift_prob if drift_prob is not None else self.DEFAULT_DRIFT_PROB

    def on_post_action(self, board, game, action):
        moved = False
        # Snapshot mine positions up front so a mine we just moved doesn't
        # get processed again at its new location this same tick.
        mines = [
            (r, c)
            for r in range(board.config.rows)
            for c in range(board.config.cols)
            if board.grid[r][c].is_mine and not board.grid[r][c].is_flagged
        ]
        for src in mines:
            if self._rng.random() >= self.drift_prob:
                continue
            r, c = src
            candidates = [
                (nr, nc) for nr, nc in board.neighbors(r, c)
                if not board.grid[nr][nc].is_revealed
                and not board.grid[nr][nc].is_flagged
                and not board.grid[nr][nc].is_mine
            ]
            if not candidates:
                continue
            dst = self._rng.choice(candidates)
            if board.move_mine(src, dst):
                moved = True
        if moved:
            board.recompute_adjacency()


class ChainReactionMines(MineBehavior):
    """
    If the player hit a mine this turn, every adjacent unrevealed mine also
    detonates: it is revealed and the player takes mine_damage for each.
    Detonations cascade — each newly-detonated mine triggers its neighbors
    too — until the chain runs out of fresh mines or the player dies.
    """

    def on_post_action(self, board, game, action):
        if not action.get("hit_mine"):
            return
        origin = action["coords"]
        queue: deque[tuple[int, int]] = deque([origin])
        seen: set[tuple[int, int]] = {origin}
        while queue:
            cr, cc = queue.popleft()
            for nr, nc in board.neighbors(cr, cc):
                if (nr, nc) in seen:
                    continue
                cell = board.grid[nr][nc]
                if not (cell.is_mine and not cell.is_revealed):
                    continue
                seen.add((nr, nc))
                board.reveal_mine(nr, nc)
                action["newly_revealed"].append((nr, nc))
                game.player.take_damage()
                if not game.player.is_alive():
                    return
                queue.append((nr, nc))


MINE_BEHAVIORS: dict[str, type[MineBehavior]] = {
    "static":         StaticMines,
    "drifting":       DriftingMines,
    "chain-reaction": ChainReactionMines,
}
