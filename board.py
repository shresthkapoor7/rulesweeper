from __future__ import annotations
import random

from config import GameConfig
from cell import Cell
from reveal_strategies import REVEAL_STRATEGIES, RevealStrategy


class Board:
    """
    This class owns the grid of Cells and all topological operations.
    Mine placement is deferred until first reveal to support safe_first_click.
    """

    def __init__(self, config: GameConfig, seed: int | None = None) -> None:
        self.config = config
        self.mines_placed = False
        self.grid: list[list[Cell]] = [
            [Cell() for _ in range(config.cols)]
            for _ in range(config.rows)
        ]
        self._flags_placed = 0
        self._strategy: RevealStrategy = REVEAL_STRATEGIES[config.reveal_strategy]()
        # Isolated RNG — does not share state with the agent or any other component
        self._rng = random.Random(seed)

    ##############################################################
    # SETUP
    ##############################################################

    def place_mines(self, exclude: tuple[int, int]) -> None:
        """
        Randomly place config.mine_count mines.
        If safe_first_click, exclude the clicked cell and its 8 neighbors.
        """
        if self.config.safe_first_click:
            exclude_set = {exclude} | set(self.neighbors(*exclude))
        else:
            exclude_set = {exclude}

        candidates = [
            (r, c)
            for r in range(self.config.rows)
            for c in range(self.config.cols)
            if (r, c) not in exclude_set
        ]

        # Guard: if mine_count exceeds available candidates, don't place more
        count = min(self.config.mine_count, len(candidates))
        for r, c in self._rng.sample(candidates, count):
            self.grid[r][c].is_mine = True

        self.recompute_adjacency()
        self.mines_placed = True

    def recompute_adjacency(self) -> None:
        """
        Fill adjacent_mines for every non-mine cell. O(rows × cols).
        Public so MineBehavior implementations (e.g. DriftingMines) can call it
        after relocating mines mid-game.
        """
        for r in range(self.config.rows):
            for c in range(self.config.cols):
                if not self.grid[r][c].is_mine:
                    self.grid[r][c].adjacent_mines = sum(
                        1 for nr, nc in self.neighbors(r, c)
                        if self.grid[nr][nc].is_mine
                    )

    ##############################################################
    # QUERIES
    ##############################################################

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.config.rows and 0 <= c < self.config.cols

    def neighbors(self, r: int, c: int) -> list[tuple[int, int]]:
        """Return valid (r, c) pairs for all 8 Moore-neighborhood cells."""
        return [
            (r + dr, c + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0) and self.in_bounds(r + dr, c + dc)
        ]

    def is_solved(self) -> bool:
        """True when every non-mine cell is revealed."""
        return all(
            cell.is_revealed
            for row in self.grid
            for cell in row
            if not cell.is_mine
        )

    def flags_placed(self) -> int:
        return self._flags_placed

    ##############################################################
    # MUTATIONS
    ##############################################################

    def reveal(self, r: int, c: int) -> list[tuple[int, int]]:
        """
        Reveal a cell via the configured RevealStrategy.
        Mine cells are handled by the game layer; this method is only called for safe cells.
        Returns all newly revealed (r, c) pairs.
        """
        cell = self.grid[r][c]
        if cell.is_revealed or cell.is_flagged or cell.is_mine:
            return []
        return self._strategy.reveal(self, r, c)

    def reveal_mine(self, r: int, c: int) -> None:
        """Mark a mine cell as revealed (called by MinesweeperGame on hit)."""
        self.grid[r][c].is_revealed = True

    def move_mine(self, src: tuple[int, int], dst: tuple[int, int]) -> bool:
        """
        Move a mine from src to dst. Returns True iff the move was applied.
        Refuses if src is not a mine, or dst is out-of-bounds, revealed, flagged,
        or already a mine. Does not recompute adjacency — callers should batch
        moves and call recompute_adjacency() once when done.
        """
        sr, sc = src
        dr, dc = dst
        if not self.in_bounds(dr, dc):
            return False
        src_cell = self.grid[sr][sc]
        dst_cell = self.grid[dr][dc]
        if not src_cell.is_mine:
            return False
        if dst_cell.is_revealed or dst_cell.is_flagged or dst_cell.is_mine:
            return False
        src_cell.is_mine = False
        dst_cell.is_mine = True
        return True

    def toggle_flag(self, r: int, c: int) -> bool:
        """
        Toggle flag on an unrevealed cell. Respects flag_limit.
        Returns True if the flag state changed.
        """
        cell = self.grid[r][c]
        if cell.is_revealed:
            return False

        if not cell.is_flagged:
            limit = self.config.flag_limit
            if limit is not None and self._flags_placed >= limit:
                return False
            cell.is_flagged = True
            self._flags_placed += 1
        else:
            cell.is_flagged = False
            self._flags_placed -= 1

        return True
