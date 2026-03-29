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


def evaluate_config(
    agent_class: type[Agent],
    config: GameConfig,
    n_games: int = 50,
    base_seed: int | None = None,
) -> dict:
    """
    Run N games and return aggregated fitness metrics for the given config.

    Uses a fresh agent instance per game to avoid RNG state accumulation.
    Pass base_seed for reproducible results: game i uses seed base_seed+i.

    Returns:
        win_rate              — fraction of games won
        avg_cells_revealed    — mean safe cells uncovered
        avg_progress_fraction — avg_cells_revealed / safe_cells (board-size-normalized)
        avg_mines_hit         — mean mines struck per game
        avg_turns             — mean total actions per game
        n_games               — number of games played
    """
    safe_cells = config.rows * config.cols - config.mine_count
    results = []
    for i in range(n_games):
        seed = (base_seed + i) if base_seed is not None else None
        agent = agent_class(seed=seed)
        results.append(run_game(agent, config, seed=seed))

    wins = sum(1 for r in results if r["state"] == "won")
    avg_revealed = sum(r["cells_revealed"] for r in results) / n_games
    return {
        "win_rate":              wins / n_games,
        "avg_cells_revealed":    avg_revealed,
        "avg_progress_fraction": avg_revealed / max(safe_cells, 1),
        "avg_mines_hit":         sum(r["mines_hit"] for r in results) / n_games,
        "avg_turns":             sum(r["turns"]    for r in results) / n_games,
        "n_games":               n_games,
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
