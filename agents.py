from __future__ import annotations
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from config import GameConfig
from game import MinesweeperGame, GameState

if TYPE_CHECKING:
    from renderer import TerminalRenderer


class Agent(ABC):
    """
    Base class for all game-playing agents.
    Implement choose_action() to define how the agent selects its next move.
    """

    @abstractmethod
    def choose_action(self, game: MinesweeperGame) -> tuple[str, int, int]:
        """
        Return (action, row, col) where action is "reveal" or "flag".
        Called once per turn. Must not mutate game state.
        """
        ...


class RandomAgent(Agent):
    """
    Reveals a random unrevealed, unflagged cell each turn. Never flags.
    Useful as a baseline for fitness evaluation and sanity-checking configs.
    """

    def __init__(self, seed: int | None = None) -> None:
        # Isolated RNG instance — seeding the agent does not affect Board.place_mines()
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


# Registry — add new agents here; CLI selects them by name via --agent
AGENTS: dict[str, type[Agent]] = {
    "random": RandomAgent,
}


def run_game(
    agent: Agent,
    config: GameConfig | None = None,
    renderer: TerminalRenderer | None = None,
    seed: int | None = None,
) -> dict:
    """
    Run a complete game with the given agent and config.

    - Omit renderer for headless/batch use (MORTAR fitness evaluation).
    - Pass a TerminalRenderer to watch the agent play interactively.
    - Pass seed to make both board mine placement and agent choices reproducible.

    Returns the final player_metrics dict augmented with 'state' and 'turns'.
    This dict is the primary result consumed by MORTAR's fitness evaluator.
    """
    if seed is not None and isinstance(agent, RandomAgent):
        agent._rng = random.Random(seed)
    game = MinesweeperGame(config, seed=seed)
    turns = 0

    while game.get_state() in (GameState.PENDING, GameState.ACTIVE):
        if renderer:
            renderer.render(game)
            renderer.render_status(game)

        action, r, c = agent.choose_action(game)
        if action == "reveal":
            game.reveal(r, c)
        else:
            game.flag(r, c)
        turns += 1

    if renderer:
        renderer.render(game)
        renderer.render_result(game)

    return {**game.get_player_metrics(), "state": game.get_state().value, "turns": turns}
