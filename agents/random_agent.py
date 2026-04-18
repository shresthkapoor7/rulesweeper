from __future__ import annotations
import random

from agents import Agent
from game import MinesweeperGame


class RandomAgent(Agent):
    """
    Reveals a random unrevealed, unflagged cell each turn. Never flags.
    Useful as a baseline for fitness evaluation and sanity-checking configs.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose_action(self, game: MinesweeperGame) -> tuple[str, int, int]:
        cfg = game.get_config()
        candidates = [
            (r, c)
            for r in range(cfg.rows)
            for c in range(cfg.cols)
            if not game.get_cell(r, c).is_revealed
            and not game.get_cell(r, c).is_flagged
        ]
        r, c = self._rng.choice(candidates)
        return ("reveal", r, c)
