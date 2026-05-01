# CLAUDE.md — MORTAR Minesweeper

## Project Purpose

This is a base Minesweeper implementation built for the [MORTAR](https://arxiv.org/pdf/2601.00105) mechanic generation research project. MORTAR uses evolutionary algorithms to mutate game mechanics and discover novel, playable configurations. The game is designed so that every rule is isolated behind a configurable parameter, making it straightforward for MORTAR to swap or mutate mechanics without touching core game logic.

## Dev Rules

- **Python 3.10+** required (`int | None` union syntax is used throughout)
- **Run scripts from inside `mortar-minesweeper/`** — imports assume this as the working directory
- **No external dependencies in the core game** — stdlib only (`dataclasses`, `abc`, `collections`, `enum`, `argparse`)
- **`mortar.py` requires `openai`** (`pip install openai`) and an `OPENROUTER_API_KEY` in `.env` — copy `.env.example`
- **Neural agent requires `torch`** (`pip install torch`) — only imported when the neural agent is used

## Architecture

Top-level files each have a single responsibility. Agents live in the `agents/` package.

| File | Responsibility |
|---|---|
| `config.py` | `GameConfig` dataclass — the sole MORTAR mutation surface; every configurable mechanic lives here |
| `cell.py` | `Cell` dataclass — pure data (mine, revealed, flagged, adjacent count); only `Board` mutates it |
| `board.py` | `Board` — owns the grid, mine placement (deferred to first click), adjacency computation, flagging, and delegates reveals to a `RevealStrategy` |
| `reveal_strategies.py` | `RevealStrategy` ABC + concrete implementations (`CascadeReveal`, `SingleReveal`) + `REVEAL_STRATEGIES` registry |
| `mine_behaviors.py` | `MineBehavior` ABC + concrete implementations (`StaticMines`, `DriftingMines`, `ChainReactionMines`) + `MINE_BEHAVIORS` registry |
| `mechanics_archive.py` | Named `GameConfig` presets + `MECHANICS` registry — catalog of evolved mechanic variants; `standard` is the MORTAR seed |
| `mortar.py` | MORTAR mutation engine — LLM-guided `GameConfig` evolution via OpenRouter, flat JSON archive, CLI entry point |
| `code_mutations.py` | Compile / AST-validate / smoke-test / register LLM-authored `MineBehavior` and `RevealStrategy` subclasses (used by `mortar.py --mode code`) |
| `player.py` | `Player` — tracks health, moves, and statistics; exposes `metrics()` as the MORTAR fitness signal |
| `game.py` | `MinesweeperGame` + `GameState` enum — orchestrates `Board` and `Player`; primary interface for both human play and MORTAR agents |
| `renderer.py` | `TerminalRenderer` — stateless display; all output lives here; ANSI color when stdout is a tty |
| `main.py` | Entry point — argparse CLI, input parsing, game loop |

### `agents/` package

| File | Responsibility |
|---|---|
| `__init__.py` | `Agent` ABC, `AGENTS` registry, `run_game()`, `evaluate_config()` |
| `random_agent.py` | `RandomAgent` — reveals random unrevealed cells; baseline agent |
| `pafg_agent.py` | `PAFGAgent` — four-stage constraint solver (First, Primary, Advanced, Guess) |
| `neural_agent.py` | `NeuralAgent` — CNN-based DQN; includes model (`MinesweeperDQN`), state encoding, replay buffer, target network, and `Trainer` class |
| `train_neural.py` | CLI training script for the neural agent (`python -m agents.train_neural`) |

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
| `mine_behavior` | `"static"` | Which `MineBehavior` runs after each action (`MinesweeperGame._run_mine_behavior`) |

### Code-mutation mode

`mortar.py --mode code` (or 50% of `mixed` iterations) asks the LLM to author
a brand-new `MineBehavior` or `RevealStrategy` subclass instead of tuning a
`GameConfig` field. Pipeline (in `code_mutations.py`):

1. `ast.parse` — syntax check.
2. AST denylist — no `import`, no `__class__/__bases__/__subclasses__/__globals__/__dict__/__import__/eval/exec/open/compile`.
3. `exec` in curated globals (`random`, `deque`, the target ABC, restricted `__builtins__`).
4. Locate exactly one subclass of the ABC.
5. Smoke test — two short games on 8×8 with `RandomAgent`, 30-turn cap, 5s `SIGALRM` guard.
6. Register under `gen-<sha1[:8]>` in `MINE_BEHAVIORS` / `REVEAL_STRATEGIES`. Idempotent.
7. Standard panel evaluation + admission criteria.

Accepted entries persist with `code_kind`, `code_source`, `code_key` fields in
`archive.json`. On startup, `register_all_from_archive` recompiles every code
entry; entries that fail to recompile are dropped with a warning. `main.py`
loads the archive too, so `python main.py --mine-behavior gen-abc12345` works.

**Trust caveat.** Generated code is `exec()`'d in-process. The AST denylist
catches obvious foot-guns but is not a security boundary — `archive.json` with
non-null `code_source` is executable code, not data. Run only against trusted
models. Subprocess isolation is a follow-up.

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
    "efficiency":       float,  # cells_revealed / moves (note: mortar.py uses progress_fraction instead)
}
```

## Named Mechanics

`mechanics_archive.py` is the catalog of named mechanic presets — the kind of configs MORTAR would produce after evolving interesting variants. Each preset is a function that returns a `GameConfig`.

| Name | Description |
|---|---|
| `standard` | Canonical 16×16 minesweeper — the MORTAR seed config |
| `extra-life` | Player starts with 2 HP; survives one mine hit before game over |
| `drifting-mines` | Unflagged mines wander into adjacent unrevealed/unflagged cells each turn; adjacency numbers update in place. Flagging pins a mine in place |
| `chain-reaction` | Hitting a mine cascades to every adjacent mine; player starts with 3 HP to make it survivable |

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

### Adding a new mine behavior

`MineBehavior` runs after every reveal/flag while the game is `ACTIVE`. It can mutate the board (move mines, recompute adjacency), reveal additional cells, and apply player damage. The game re-checks win/loss after the hook returns.

1. Subclass `MineBehavior` in `mine_behaviors.py`
2. Implement `on_post_action(self, board, game, action) -> None`
3. Register it in `MINE_BEHAVIORS` with a string key
4. Reference it via `GameConfig(mine_behavior="your_key")`

The `action` dict has `action`, `coords`, `hit_mine`, and `newly_revealed` (a list — append to it if the behavior reveals more cells). Use `board.move_mine(src, dst)` for relocation (refuses dst that is revealed/flagged/already a mine) and `board.recompute_adjacency()` afterward. Use `game.player.take_damage()` for damage.

```python
class TeleportingMines(MineBehavior):
    """Each turn, every mine teleports to a random unrevealed, non-mine cell."""
    def on_post_action(self, board, game, action):
        ...

MINE_BEHAVIORS["teleporting"] = TeleportingMines
```

### Adding a new agent

1. Create a new file in `agents/` (e.g. `agents/my_agent.py`)
2. Subclass `Agent` and implement `choose_action(self, game) -> tuple[str, int, int]`
3. Constructor must accept `seed: int | None = None` for compatibility with `evaluate_config()`
4. Import and register it in `agents/__init__.py` in the `AGENTS` dict

```python
from agents import Agent
from game import MinesweeperGame

class MyAgent(Agent):
    def __init__(self, seed: int | None = None) -> None:
        ...
    def choose_action(self, game: MinesweeperGame) -> tuple[str, int, int]:
        ...  # return ("reveal", row, col) or ("flag", row, col)
```

### Adding a new configurable mechanic

1. Add a field to `GameConfig` with a sensible default that preserves standard behavior
2. Read the field in exactly one place in the codebase (the class responsible for that behavior)
3. Document it in the table above

## CLI Reference

```
python main.py [options]

  --agent NAME            Run a named agent: random, pafg, neural
  --watch                 Render board while agent plays
  --games INT             Number of games (agent mode, default: 1)
  --seed INT              Random seed for reproducibility

  --rows INT              Grid height (default: 16)
  --cols INT              Grid width  (default: 16)
  --mines INT             Mine count  (default: 40)
  --health INT            Starting health (default: 1)
  --damage INT            HP lost per mine hit (default: 1)
  --flag-limit INT        Max flags allowed (default: unlimited)
  --reveal-strategy STR   cascade | single | gen-XXXXXXXX (default: cascade)
  --mine-behavior STR     static | drifting | chain-reaction | gen-XXXXXXXX (default: static)
  --no-safe-click         Disable safe first click guarantee
  --mechanic NAME         Start from a named preset (standard, extra-life, drifting-mines, chain-reaction)
```

In-game commands:
```
  r <row> <col>   reveal a cell  (e.g. r 7 D)
  f <row> <col>   toggle flag    (e.g. f 2 J)
  h               show help
  q               quit
```

Columns are labeled A–P; rows are 0-indexed integers.

### Training the neural agent

```
python -m agents.train_neural [options]

  --episodes INT          Total training episodes (default: 50000)
  --rows INT              Board rows for training (default: 9)
  --cols INT              Board cols for training (default: 9)
  --mines INT             Mine count for training (default: 10)
  --lr FLOAT              Learning rate (default: 1e-4)
  --batch-size INT        Batch size (default: 64)
  --checkpoint-dir STR    Checkpoint directory (default: checkpoints/)
  --eval-freq INT         Evaluate every N episodes (default: 500)
  --resume PATH           Resume from checkpoint
  --device STR            cpu or cuda (default: auto-detect)
```
