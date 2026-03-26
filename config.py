from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GameConfig:
    """
    Central parameter object for all configurable game mechanics.
    MORTAR mechanic generation can mutate fields of this object to explore mechanic variants.
    """
    # Standard board size
    rows: int = 16
    cols: int = 16
    mine_count: int = 40

    # Player has health, which is reduced by mine_damage every time a mine is hit.
    # Standard gameplay is just a single life.
    starting_health: int = 1
    mine_damage: int = 1

    # Safe first click guarantees the first reveal is mine-free.
    safe_first_click: bool = True

    # Reveal strategy controls how cells are uncovered. See reveal_strategies.py for options.
    # MORTAR can swap this to explore different reveal mechanics.
    reveal_strategy: str = "cascade"

    # Flag limit is the maximum number of flags that can be placed on the board.
    flag_limit: int | None = None
