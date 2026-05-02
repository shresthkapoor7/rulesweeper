from __future__ import annotations
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


# Registry — add new agents here; CLI selects them by name via --agent
from agents.random_agent import RandomAgent
from agents.pafg_agent import PAFGAgent

AGENTS: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "pafg":   PAFGAgent,
}

try:
    from agents.neural_agent import NeuralAgent
    AGENTS["neural"] = NeuralAgent
except ImportError:
    pass  # torch not installed — neural agent unavailable


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


def evaluate_config_multi(
    agent_classes: dict[str, type[Agent]],
    config: GameConfig,
    n_games: int = 20,
    base_seed: int | None = None,
) -> dict[str, dict]:
    """
    Evaluate `config` with each agent in `agent_classes` for n_games.
    Returns {agent_name: fitness_dict} where fitness_dict matches evaluate_config().

    Same base_seed across agents means each agent plays the same boards — fair comparison.
    """
    return {
        name: evaluate_config(cls, config, n_games=n_games, base_seed=base_seed)
        for name, cls in agent_classes.items()
    }


def run_game(
    agent: Agent,
    config: GameConfig | None = None,
    renderer: TerminalRenderer | None = None,
    seed: int | None = None,
    max_turns: int = 1000,
) -> dict:
    """
    Run a complete game with the given agent and config.

    - Omit renderer for headless/batch use (MORTAR fitness evaluation).
    - Pass a TerminalRenderer to watch the agent play interactively.
    - Pass seed to make both board mine placement and agent choices reproducible.
    - max_turns caps a single game to prevent pathological mechanics (e.g. an
      LLM-authored MineBehavior that spawns mines faster than they're cleared)
      from hanging the eval loop. The cap is intentionally generous; legitimate
      play on a 30x30 board with single-reveal stays well under it.

    Returns the final player_metrics dict augmented with 'state' and 'turns'.
    If the game hits max_turns, 'state' will still be 'active' — callers can
    detect this case via `result['turns'] >= max_turns and result['state'] == 'active'`.
    This dict is the primary result consumed by MORTAR's fitness evaluator.
    """
    game = MinesweeperGame(config, seed=seed)
    turns = 0

    while game.get_state() in (GameState.PENDING, GameState.ACTIVE) and turns < max_turns:
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
