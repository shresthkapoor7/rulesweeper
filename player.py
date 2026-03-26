from __future__ import annotations
from config import GameConfig


class Player:
    """
    Tracks player health and action statistics.
    This class intentionally knows nothing about the Board, instead the MinesweeperGame class will mediate.
    metrics() is the primary MORTAR fitness signal.
    """

    def __init__(self, config: GameConfig) -> None:
        self.config = config
        self.health: int = config.starting_health
        self.moves: int = 0
        self.mines_hit: int = 0
        self.cells_revealed: int = 0

    def take_damage(self) -> None:
        """Reduce health by config.mine_damage, clamped to 0."""
        self.health = max(0, self.health - self.config.mine_damage)
        self.mines_hit += 1

    def is_alive(self) -> bool:
        return self.health > 0

    def record_reveal(self, count: int) -> None:
        self.cells_revealed += count
        self.moves += 1

    def record_flag(self) -> None:
        self.moves += 1

    # Contains performance metrics for the player
    # Eventually will be used to calculate the fitness of the player.
    def metrics(self) -> dict:
        """
        Flat, serializable performance snapshot.
        MORTAR reads this dict as the fitness signal for a given config.
        """
        return {
            "moves": self.moves,
            "cells_revealed": self.cells_revealed,
            "mines_hit": self.mines_hit,
            "health_remaining": self.health,
            "efficiency": self.cells_revealed / max(self.moves, 1),
        }
