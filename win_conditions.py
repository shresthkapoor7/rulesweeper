from __future__ import annotations
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from board import Board
    from game import MinesweeperGame


class WinCondition(ABC):
    """
    Decides what counts as winning (and optionally losing) the game. Consulted
    after every player action while the game is ACTIVE, then again after the
    configured MineBehavior runs. Health-based loss is checked BEFORE this
    hook fires, so the player is always alive when evaluate() is called.

    evaluate() must return one of three values:
      - the literal string "won"  — game is won
      - the literal string "lost" — game is lost (custom loss; standard
                                    health-based loss still triggers separately)
      - None                      — game continues
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    @abstractmethod
    def evaluate(
        self,
        board: "Board",
        game: "MinesweeperGame",
        action: dict,
    ) -> str | None:
        """
        action: same dict passed to MineBehavior.on_post_action
                (action, coords, hit_mine, newly_revealed).
        """
        ...

    def summary(self) -> dict:
        """Key parameters for this condition. Override on knob-bearing subclasses."""
        return {}


class StandardWin(WinCondition):
    """Canonical: win when every non-mine cell has been revealed."""

    def evaluate(self, board, game, action):
        return "won" if board.is_solved() else None


class RevealQuotaWin(WinCondition):
    """
    Win once at least `quota` fraction of safe cells has been revealed.
    Cheaper-than-perfect win — useful on dense boards where total clearing
    is impractical.
    """

    DEFAULT_QUOTA = 0.5

    def __init__(
        self,
        seed: int | None = None,
        quota: float | None = None,
    ) -> None:
        super().__init__(seed)
        q = quota if quota is not None else self.DEFAULT_QUOTA
        self.quota = max(0.0, min(1.0, q))

    def summary(self) -> dict:
        return {"quota": self.quota}

    def evaluate(self, board, game, action):
        total_safe = board.config.rows * board.config.cols - board.config.mine_count
        if total_safe <= 0:
            return "won"
        revealed_safe = sum(
            1
            for row in board.grid
            for cell in row
            if cell.is_revealed and not cell.is_mine
        )
        if revealed_safe / total_safe >= self.quota:
            return "won"
        return None


class FlagAllMinesWin(WinCondition):
    """
    Win when every mine cell is flagged AND no non-mine cell is flagged.
    Forces flagging into the deduction loop — pure-reveal play cannot win.
    """

    def evaluate(self, board, game, action):
        for row in board.grid:
            for cell in row:
                if cell.is_mine and not cell.is_flagged:
                    return None
                if cell.is_flagged and not cell.is_mine:
                    return None
        return "won"


class SurvivalWin(WinCondition):
    """
    Win after acting `target_turns` times without dying. Mine hits still cost
    HP via the standard health rule — this just changes what counts as success.
    """

    DEFAULT_TARGET = 20

    def __init__(
        self,
        seed: int | None = None,
        target_turns: int | None = None,
    ) -> None:
        super().__init__(seed)
        self.target_turns = (
            target_turns if target_turns is not None else self.DEFAULT_TARGET
        )

    def summary(self) -> dict:
        return {"target_turns": self.target_turns}

    def evaluate(self, board, game, action):
        if game.player.metrics()["moves"] >= self.target_turns:
            return "won"
        return None


WIN_CONDITIONS: dict[str, type[WinCondition]] = {
    "standard":       StandardWin,
    "reveal-quota":   RevealQuotaWin,
    "flag-all-mines": FlagAllMinesWin,
    "survival":       SurvivalWin,
}
