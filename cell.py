from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Cell:
    """
    Pure data object representing a single grid cell.
    Only the Board mutates cell state; no logic lives here.
    """
    is_mine: bool = False
    is_revealed: bool = False
    is_flagged: bool = False
    adjacent_mines: int = 0
