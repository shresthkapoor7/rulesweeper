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


# Archive of named mechanic presets.
# MORTAR adds entries here when it produces an interesting evolved config.
MECHANICS: dict[str, callable] = {
    "standard":       standard,
    "extra-life":     extra_life,
    "drifting-mines": drifting_mines,
    "chain-reaction": chain_reaction,
}
