from __future__ import annotations
import dataclasses
from config import GameConfig


# This file contains the named mechanic presets.
# MORTAR adds entries here when it produces an interesting evolved config.

def standard() -> GameConfig:
    """Canonical 16×16 standard minesweeper. The MORTAR seed config."""
    return GameConfig()


def extra_life() -> GameConfig:
    """Player survives one mine hit before game over."""
    return dataclasses.replace(
        GameConfig(),
        starting_health=2,
        mine_damage=1,
    )


def drifting_mines() -> GameConfig:
    """Mines wander into adjacent unrevealed cells each turn; adjacency numbers update in place."""
    return dataclasses.replace(GameConfig(), mine_behavior="drifting")


def chain_reaction() -> GameConfig:
    """Hitting a mine cascades to every adjacent mine. Player gets extra health to make it survivable."""
    return dataclasses.replace(
        GameConfig(),
        mine_behavior="chain-reaction",
        starting_health=3,
    )


def parity_vision() -> GameConfig:
    """Numbers show only even/odd of adjacent mine counts. Extra HP to compensate for harder deduction."""
    return dataclasses.replace(
        GameConfig(),
        info_strategy="parity",
        starting_health=2,
    )


def knight_moves() -> GameConfig:
    """Adjacency follows chess-knight moves. Cascade jumps non-locally; surreal but solvable."""
    return dataclasses.replace(
        GameConfig(),
        neighborhood="knight",
    )


def flag_hunter() -> GameConfig:
    """Win by flagging every mine and only mines — pure-reveal play cannot win."""
    return dataclasses.replace(
        GameConfig(),
        win_condition="flag-all-mines",
    )


def quota_rush() -> GameConfig:
    """Win after revealing half the safe cells; mine count bumped to keep the partial-clear meaningful."""
    return dataclasses.replace(
        GameConfig(),
        win_condition="reveal-quota",
        mine_count=60,
    )


# Archive of named mechanic presets.
# MORTAR adds entries here when it produces an interesting evolved config.
MECHANICS: dict[str, callable] = {
    "standard":       standard,
    "extra-life":     extra_life,
    "drifting-mines": drifting_mines,
    "chain-reaction": chain_reaction,
    "parity-vision":  parity_vision,
    "knight-moves":   knight_moves,
    "flag-hunter":    flag_hunter,
    "quota-rush":     quota_rush,
}
