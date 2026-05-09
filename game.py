from __future__ import annotations
from enum import Enum

from config import GameConfig
from board import Board
from player import Player
from cell import Cell
from mine_behaviors import MINE_BEHAVIORS
from info_strategies import INFO_STRATEGIES
from win_conditions import WIN_CONDITIONS

# Represents the current state of the game.
# "Pending" is the state before the first reveal.
class GameState(Enum):
    PENDING = "pending"
    ACTIVE  = "active"
    WON     = "won"
    LOST    = "lost"


class MinesweeperGame:
    """
    This is the Orchestrator for the game, and will enforce rules from the GameConfig
    - 1 Board
    - 1 Player
  
    This is the primary interface for the TerminalRenderer, which is how humans can play the game,
    as well as the eventual MORTAR agents.

    MORTAR programmatic usage:
        config = dataclasses.replace(base_config, mine_damage=2, starting_health=3)
        game = MinesweeperGame(config)
        while game.get_state() == GameState.ACTIVE:
            result = game.reveal(r, c)
        metrics = game.get_player_metrics()
    """

    def __init__(self, config: GameConfig | None = None, seed: int | None = None) -> None:
        self.config = config or GameConfig()
        self._seed = seed
        self._init_components()

    def _init_components(self) -> None:
        self.board = Board(self.config, seed=self._seed)
        self.player = Player(self.config)
        self._mine_behavior = MINE_BEHAVIORS[self.config.mine_behavior](seed=self._seed)
        self._info_strategy = INFO_STRATEGIES[self.config.info_strategy](seed=self._seed)
        self._win_condition = WIN_CONDITIONS[self.config.win_condition](seed=self._seed)
        self.state = GameState.PENDING

    ##############################################################
    # CORE ACTIONS
    ##############################################################

    def reveal(self, r: int, c: int) -> dict:
        """
        Reveal a cell. 
        - On the very first call, transitions game state from PENDING → ACTIVE
        - If safe_first_click is enabled, mines will be placed after the first reveal
        - Returns a structured action-result dict.
        """

        # If the game is won or lost, return an empty result.
        if self.state in (GameState.WON, GameState.LOST):
            return self._action_result("reveal", (r, c), False, [])

        cell = self.board.grid[r][c]

        # If the cell is already revealed or flagged, return an empty result.
        if cell.is_revealed or cell.is_flagged:
            return self._action_result("reveal", (r, c), False, [])

        # First reveal: place mines then transition to ACTIVE
        if self.state == GameState.PENDING:
            self.board.place_mines(exclude=(r, c))
            self.state = GameState.ACTIVE
            cell = self.board.grid[r][c]

        hit_mine = cell.is_mine
        newly_revealed: list[tuple[int, int]] = []

        # If the cell is a mine, reveal the mine and damage the player.
        if hit_mine:
            self.board.reveal_mine(r, c)
            newly_revealed = [(r, c)]
            self.player.take_damage()
        else:
            newly_revealed = self.board.reveal(r, c)
            # Defensive count: a generated RevealStrategy may not dedup or may
            # include mine cells. Count unique safe-cell reveals only so
            # progress_fraction stays bounded by safe_cells.
            unique_safe = {
                (rr, cc) for (rr, cc) in newly_revealed
                if not self.board.grid[rr][cc].is_mine
            }
            self.player.record_reveal(len(unique_safe))

        self._check_endgame("reveal", (r, c), hit_mine, newly_revealed)
        self._run_mine_behavior("reveal", (r, c), hit_mine, newly_revealed)

        return self._action_result("reveal", (r, c), hit_mine, newly_revealed)

    def flag(self, r: int, c: int) -> dict:
        """
        Toggle a flag on an unrevealed cell.
        - If the game is won or lost, return an empty result.
        - Returns a structured action-result dict.
        """
        if self.state in (GameState.WON, GameState.LOST):
            return self._action_result("flag", (r, c), False, [])

        changed = self.board.toggle_flag(r, c)
        if changed:
            self.player.record_flag()

        self._check_endgame("flag", (r, c), False, [])
        self._run_mine_behavior("flag", (r, c), False, [])

        return self._action_result("flag", (r, c), False, [])

    def reset(self, config: GameConfig | None = None) -> None:
        """
        Re-initialize board and player. Accepts an optional new config for MORTAR sweeps.
        - If a new config is provided, update the current config.
        - Reset the board and player.
        """
        if config is not None:
            self.config = config
        self._init_components()

    ##############################################################
    # ACCESSORS
    ##############################################################

    def get_cell(self, r: int, c: int) -> Cell:
        return self.board.grid[r][c]

    def info_at(self, r: int, c: int) -> str:
        """
        Display text for a revealed safe cell, per the configured InfoStrategy.
        The renderer calls this instead of reading cell.adjacent_mines directly.
        """
        return self._info_strategy.encode(self.board, r, c)

    def get_state(self) -> GameState:
        return self.state

    def get_config(self) -> GameConfig:
        return self.config

    def get_player_metrics(self) -> dict:
        return self.player.metrics()

    def behavior_summary(self) -> dict:
        """
        Snapshot of mechanic identities and key params, intended for agents
        (notably pafg-llm) that want to reason about the live mechanic without
        poking at private attributes.

        Returns a dict shaped:
            {
              "mine_behavior":   {"name": "drifting", "params": {"drift_prob": 0.3}},
              "info_strategy":   {"name": "noisy-count", "params": {"lie_prob": 0.2}},
              "win_condition":   {"name": "survival",   "params": {"target_turns": 20}},
              "neighborhood":    {"name": "moore",      "params": {"offsets": [...]}},
              "reveal_strategy": {"name": "cascade",    "params": {}},
            }

        ``name`` is the registry key from GameConfig (which is the canonical
        handle the agent already knows). ``params`` comes from each strategy's
        ``summary()`` method — empty dict for strategies that have no knobs.
        """
        return {
            "mine_behavior": {
                "name":   self.config.mine_behavior,
                "params": self._mine_behavior.summary(),
            },
            "info_strategy": {
                "name":   self.config.info_strategy,
                "params": self._info_strategy.summary(),
            },
            "win_condition": {
                "name":   self.config.win_condition,
                "params": self._win_condition.summary(),
            },
            "neighborhood": {
                "name":   self.config.neighborhood,
                "params": self.board._neighborhood.summary(),
            },
            "reveal_strategy": {
                "name":   self.config.reveal_strategy,
                "params": self.board._strategy.summary(),
            },
        }

    ##############################################################
    # ENDGAME + MINE BEHAVIOR HOOKS
    ##############################################################

    def _check_endgame(
        self,
        action: str,
        coords: tuple[int, int],
        hit_mine: bool,
        newly_revealed: list[tuple[int, int]],
    ) -> None:
        """
        Resolve game state after an action. Health-based loss is checked first
        and is non-overridable; then the configured WinCondition can declare
        WON or a custom LOST.
        """
        if self.state != GameState.ACTIVE:
            return
        if not self.player.is_alive():
            self.state = GameState.LOST
            return
        snapshot = {
            "action":         action,
            "coords":         coords,
            "hit_mine":       hit_mine,
            "newly_revealed": newly_revealed,
        }
        result = self._win_condition.evaluate(self.board, self, snapshot)
        if result == GameState.WON.value:
            self.state = GameState.WON
        elif result == GameState.LOST.value:
            self.state = GameState.LOST

    def _run_mine_behavior(
        self,
        action: str,
        coords: tuple[int, int],
        hit_mine: bool,
        newly_revealed: list[tuple[int, int]],
    ) -> None:
        """
        Invoke the configured MineBehavior. Only runs while the game is ACTIVE
        (mines exist and game is not over). Behavior may mutate the board, the
        player, and append to newly_revealed; we re-check endgame afterward.
        """
        if self.state != GameState.ACTIVE:
            return
        snapshot = {
            "action":         action,
            "coords":         coords,
            "hit_mine":       hit_mine,
            "newly_revealed": newly_revealed,
        }
        self._mine_behavior.on_post_action(self.board, self, snapshot)
        self._check_endgame(action, coords, hit_mine, newly_revealed)

    ##############################################################
    # ACTION RESULT
    ##############################################################

    def _action_result(
        self,
        action: str,
        coords: tuple[int, int],
        hit_mine: bool,
        newly_revealed: list[tuple[int, int]],
    ) -> dict:
        """
        Returns a structured result dict for the action.
        This is used by the TerminalRenderer and MORTAR agents.
        """
        return {
            "action": action,
            "coords": coords,
            "hit_mine": hit_mine,
            "newly_revealed": newly_revealed,
            "state": self.state.value,
            "player_metrics": self.player.metrics(),
        }
