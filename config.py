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

    # Mine behavior controls how mines react between turns. See mine_behaviors.py for options.
    # MORTAR can swap this to explore mechanics like drifting or chain-reacting mines.
    mine_behavior: str = "static"

    # Info strategy controls what symbol/text a revealed safe cell shows the player.
    # See info_strategies.py for options. Default reproduces canonical mine-count display.
    info_strategy: str = "count-mines"

    # Neighborhood defines which cells are 'adjacent' for adjacency counts, cascades,
    # safe-first-click zones, and any MineBehavior that walks neighbors.
    # See neighborhoods.py for options. Default is the canonical 8-Moore neighborhood.
    neighborhood: str = "moore"
