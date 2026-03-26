# CLAUDE.md — MORTAR Minesweeper

## Project Purpose

This is a base Minesweeper implementation built for the [MORTAR](https://arxiv.org/pdf/2601.00105) mechanic generation research project. MORTAR uses evolutionary algorithms to mutate game mechanics and discover novel, playable configurations. The game is designed so that every rule is isolated behind a configurable parameter, making it straightforward for MORTAR to swap or mutate mechanics without touching core game logic.

## Dev Rules

- **Python 3.10+** required (`int | None` union syntax is used throughout)
- **Run scripts from inside `mortar-minesweeper/`** — imports are flat/sibling-relative, not package-based
- **No external dependencies** — stdlib only (`dataclasses`, `abc`, `collections`, `enum`, `argparse`)
- No `__init__.py`; the directory is not a package

## Architecture

All files are flat in `mortar-minesweeper/`. Each has a single responsibility.

| File | Responsibility |
|---|---|
| `config.py` | `GameConfig` dataclass — the sole MORTAR mutation surface; every configurable mechanic lives here |
| `cell.py` | `Cell` dataclass — pure data (mine, revealed, flagged, adjacent count); only `Board` mutates it |
| `board.py` | `Board` — owns the grid, mine placement (deferred to first click), adjacency computation, flagging, and delegates reveals to a `RevealStrategy` |
| `reveal_strategies.py` | `RevealStrategy` ABC + concrete implementations (`CascadeReveal`, `SingleReveal`) + `REVEAL_STRATEGIES` registry |
| `mechanics_archive.py` | Named `GameConfig` presets (`extra_life`, …) + `MECHANICS` registry — catalog of evolved mechanic variants |
| `agents.py` | `Agent` ABC + `RandomAgent` + `run_game()` helper — headless game execution for MORTAR fitness evaluation |
| `player.py` | `Player` — tracks health, moves, and statistics; exposes `metrics()` as the MORTAR fitness signal |
| `game.py` | `MinesweeperGame` + `GameState` enum — orchestrates `Board` and `Player`; primary interface for both human play and MORTAR agents |
| `renderer.py` | `TerminalRenderer` — stateless display; all output lives here; ANSI color when stdout is a tty |
| `main.py` | Entry point — argparse CLI, input parsing, game loop |

## MORTAR Integration

### Mutating mechanics

`GameConfig` is a dataclass; MORTAR mutates it via `dataclasses.replace()`:

```python
import dataclasses
config = dataclasses.replace(GameConfig(), mine_damage=1, starting_health=3, reveal_strategy="single")
game = MinesweeperGame(config)
```

Every `GameConfig` field is a candidate gene. The table below maps each field to the code location it controls:

| Field | Default | Controls |
|---|---|---|
| `rows`, `cols` | `16`, `16` | Grid dimensions (`Board.__init__`) |
| `mine_count` | `40` | Mine placement (`Board.place_mines`) |
| `starting_health` | `1` | Player starting HP (`Player.__init__`) |
| `mine_damage` | `1` | Damage per mine hit (`Player.take_damage`) |
| `safe_first_click` | `True` | First-click mine exclusion (`Board.place_mines`) |
| `reveal_strategy` | `"cascade"` | Which `RevealStrategy` to use (`Board.__init__`) |
| `flag_limit` | `None` | Max flags on board (`Board.toggle_flag`) |

### Driving a game programmatically

```python
from game import MinesweeperGame, GameState

game = MinesweeperGame(config)
while game.get_state() == GameState.ACTIVE:
    result = game.reveal(r, c)   # agent picks (r, c)

metrics = game.get_player_metrics()  # fitness signal
```

Every call to `game.reveal()` and `game.flag()` returns a structured dict:

```python
{
    "action":          "reveal" | "flag",
    "coords":          (r, c),
    "hit_mine":        bool,
    "newly_revealed":  [(r, c), ...],
    "state":           "pending" | "active" | "won" | "lost",
    "player_metrics":  { ... },  # see below
}
```

### Fitness signal — `Player.metrics()`

```python
{
    "moves":            int,    # total reveal + flag actions
    "cells_revealed":   int,    # cumulative safe cells uncovered
    "mines_hit":        int,    # > 0 only when mine_damage < starting_health
    "health_remaining": int,
    "efficiency":       float,  # cells_revealed / moves
}
```

## Named Mechanics

`mechanics_archive.py` is the catalog of named mechanic presets — the kind of configs MORTAR would produce after evolving interesting variants. Each preset is a function that returns a `GameConfig`.

| Name | Description |
|---|---|
| `extra-life` | Player starts with 2 HP; survives one mine hit before game over |

Use a preset from the CLI — additional flags compose on top:
```
python main.py --mechanic extra-life
python main.py --mechanic extra-life --mines 60   # override mine count on top of preset
```

To add a new preset: write a function returning a `GameConfig`, register it in `MECHANICS`.

## How to Extend

### Adding a new reveal strategy

1. Subclass `RevealStrategy` in `reveal_strategies.py`
2. Implement `reveal(self, board, r, c) -> list[tuple[int, int]]`
3. Register it in `REVEAL_STRATEGIES` with a string key
4. Reference it via `GameConfig(reveal_strategy="your_key")`

```python
class IslandReveal(RevealStrategy):
    """Reveals all connected non-mine cells regardless of adjacency counts."""
    def reveal(self, board, r, c):
        ...

REVEAL_STRATEGIES["island"] = IslandReveal
```

### Adding a new configurable mechanic

1. Add a field to `GameConfig` with a sensible default that preserves standard behavior
2. Read the field in exactly one place in the codebase (the class responsible for that behavior)
3. Document it in the table above

## CLI Reference

```
python main.py [options]

  --rows INT              Grid height (default: 16)
  --cols INT              Grid width  (default: 16)
  --mines INT             Mine count  (default: 40)
  --health INT            Starting health (default: 1)
  --damage INT            HP lost per mine hit (default: 1)
  --flag-limit INT        Max flags allowed (default: unlimited)
  --reveal-strategy STR   cascade | single (default: cascade)
  --no-safe-click         Disable safe first click guarantee
  --mechanic NAME         Start from a named preset (extra-life)
```

In-game commands:
```
  r <row> <col>   reveal a cell  (e.g. r 7 D)
  f <row> <col>   toggle flag    (e.g. f 2 J)
  h               show help
  q               quit
```

Columns are labeled A–P; rows are 0-indexed integers.
