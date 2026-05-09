from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from board import Board


class RevealStrategy(ABC):
    """
    Base class for all reveal behaviors. This is used by the MinesweeperGame class to reveal cells.
    Implement reveal() to define how cells are uncovered.
    
    """

    @abstractmethod
    def reveal(self, board: Board, r: int, c: int) -> list[tuple[int, int]]:
        """Reveal cell(s) starting from (r, c). Returns list of newly revealed (r, c) pairs."""
        ...

    def summary(self) -> dict:
        """Key parameters for this strategy. Override on knob-bearing subclasses."""
        return {}


class CascadeReveal(RevealStrategy):
    """
    Standard minesweeper cascade: BFS flood-fill from zero-adjacent cells.
    Numbered border cells are revealed but not used as BFS seeds.
    """

    def reveal(self, board: Board, r: int, c: int) -> list[tuple[int, int]]:
        cell = board.grid[r][c]
        if cell.adjacent_mines > 0:
            cell.is_revealed = True
            return [(r, c)]

        queue: deque[tuple[int, int]] = deque([(r, c)])
        visited: set[tuple[int, int]] = {(r, c)}
        revealed: list[tuple[int, int]] = []

        while queue:
            cr, cc = queue.popleft()
            curr = board.grid[cr][cc]
            if curr.is_revealed or curr.is_flagged or curr.is_mine:
                continue
            curr.is_revealed = True
            revealed.append((cr, cc))
            if curr.adjacent_mines == 0:
                for nr, nc in board.neighbors(cr, cc):
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        return revealed


class SingleReveal(RevealStrategy):
    """Reveals exactly one cell with no cascade. Increases difficulty and number of clicks"""

    def reveal(self, board: Board, r: int, c: int) -> list[tuple[int, int]]:
        board.grid[r][c].is_revealed = True
        return [(r, c)]


# Add new strategies here.
REVEAL_STRATEGIES: dict[str, type[RevealStrategy]] = {
    "cascade": CascadeReveal,
    "single":  SingleReveal,
}
