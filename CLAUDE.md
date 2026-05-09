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
| `info_strategies.py` | `InfoStrategy` ABC + concrete implementations (`CountMinesInfo`, `CountFlagsInfo`, `ParityInfo`, `DistanceInfo`, `DirectionInfo`, `NoisyCountInfo`) + `INFO_STRATEGIES` registry — controls what symbol a revealed cell shows |
| `neighborhoods.py` | `Neighborhood` ABC + concrete implementations (`MooreNeighborhood`, `VonNeumannNeighborhood`, `DiagonalNeighborhood`, `KnightNeighborhood`, `Radius2MooreNeighborhood`) + `NEIGHBORHOODS` registry — controls what counts as 'adjacent' |
| `win_conditions.py` | `WinCondition` ABC + concrete implementations (`StandardWin`, `RevealQuotaWin`, `FlagAllMinesWin`, `SurvivalWin`) + `WIN_CONDITIONS` registry — decides what counts as winning (and optionally losing) the game |
| `mechanics_archive.py` | Named `GameConfig` presets + `MECHANICS` registry — catalog of evolved mechanic variants; `standard` is the MORTAR seed |
| `mortar.py` | MORTAR mutation engine — LLM-guided `GameConfig` evolution via OpenRouter, flat JSON archive, CLI entry point |
| `code_mutations.py` | Compile / AST-validate / smoke-test / register LLM-authored `MineBehavior`, `RevealStrategy`, `InfoStrategy`, `Neighborhood`, and `WinCondition` subclasses (used by `mortar.py --mode code`) |
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
| `pafg_archive_agent.py` | `ArchiveAwarePAFGAgent` (PAFG with hook surface for LLM-tuned subclasses) + the LLM generation/compile/eval pipeline used by the `pafg-llm` panel slot and the standalone `python -m agents.pafg_archive_agent` post-hoc tool |
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
| `info_strategy` | `"count-mines"` | Which `InfoStrategy` decides the symbol shown for a revealed safe cell (`MinesweeperGame.info_at` → `TerminalRenderer._cell_str`) |
| `neighborhood` | `"moore"` | Which `Neighborhood` defines adjacency for cascade, adjacency counts, safe-first-click, and mine behaviors (`Board.neighbors`) |
| `win_condition` | `"standard"` | Which `WinCondition` is consulted after every action to decide WON / LOST / continue (`MinesweeperGame._check_endgame`) |

### Code-mutation mode

`mortar.py --mode code` (or 50% of `mixed` iterations) asks the LLM to author
a brand-new `MineBehavior`, `RevealStrategy`, `InfoStrategy`, `Neighborhood`,
or `WinCondition` subclass instead of tuning a `GameConfig` field. Pipeline
(in `code_mutations.py`):

1. `ast.parse` — syntax check.
2. AST denylist — no `import`, no `__class__/__bases__/__subclasses__/__globals__/__dict__/__import__/eval/exec/open/compile`.
3. `exec` in curated globals (`random`, `deque`, the target ABC, restricted `__builtins__`).
4. Locate exactly one subclass of the ABC.
5. Smoke test — two short games on 8×8 with `RandomAgent`, 30-turn cap, 5s `SIGALRM` guard.
6. Register under `gen-<sha1[:8]>` in the corresponding registry (`MINE_BEHAVIORS`, `REVEAL_STRATEGIES`, `INFO_STRATEGIES`, `NEIGHBORHOODS`, `WIN_CONDITIONS`). Idempotent.
7. Standard panel evaluation + admission criteria.

Accepted entries persist with `code_kind`, `code_source`, `code_key` fields in
`archive.json`. On startup, `register_all_from_archive` recompiles every code
entry; entries that fail to recompile are dropped with a warning. `main.py`
loads the archive too, so `python main.py --mine-behavior gen-abc12345` works.

**Trust caveat.** Generated code is `exec()`'d in-process. The AST denylist
catches obvious foot-guns but is not a security boundary — `archive.json` with
non-null `code_source` is executable code, not data. Run only against trusted
models. Subprocess isolation is a follow-up.

**Agent-fairness caveat for `info_strategy`.** `PAFGAgent` now consults
`MinesweeperGame.info_at()` and only forms constraints when
`info_strategy="count-mines"`; under any other named or generated info strategy
its `_read_clue` returns `None` and the cell silently drops out of the
equation system, so it can no longer "cheat" by reading `cell.adjacent_mines`
behind the obfuscation. `NeuralAgent` still reads `cell.adjacent_mines`
directly — fitness for `info_strategy` mutations is honest only on the
`random + pafg + pafg-llm` panel (the new default after Pass 1). Run with the
neural agent at your own risk on non-`count-mines` configs.

**Agent-fairness caveat for `neighborhood`.** `PAFGAgent` now reads adjacency
from `game.board.neighbors(r, c)` (the configured Neighborhood). It works
correctly under `knight`, `radius-2-moore`, `von-neumann`, etc. The previous
hardcoded 3×3 generator is gone.

**Agent-fairness caveat for `win_condition`.** Agents do not adapt their
strategy to non-standard objectives. `PAFGAgent` and `NeuralAgent` are tuned
around "reveal every safe cell"; under `win_condition="flag-all-mines"` they
still play to clear the board, so the win is largely incidental (`PAFG` does
flag mines as part of its constraint loop, `NeuralAgent` does not). Under
`"survival"` they keep clicking and may die before reaching the turn target.
Fitness signals for `win_condition` mutations therefore lean on the random
agent until the agents are refactored to consult the configured objective.

### `pafg-llm` — per-mechanic LLM-tuned PAFG

The MORTAR panel accepts a special name `pafg-llm` (in the default panel as of
the generation-pipeline overhaul) that, instead of resolving against the static
`AGENTS` registry, asks the LLM to write a `ArchiveAwarePAFGAgent` subclass
tailored to the specific mechanic being evaluated. Generation happens in `mortar.py:_materialize_panel`,
immediately before each `evaluate_config_multi` call, via the helpers in
`agents/pafg_archive_agent.py`:

1. Build a synthetic archive entry from the live `(snapshot, description, code_meta)`.
2. `generate_agent_candidate` — OpenRouter call (currently
   `anthropic/claude-sonnet-4.6`) returning JSON with `name`, `description`, `code`.
3. `compile_agent_candidate` — AST validation (no imports, no forbidden
   names/attrs), exec in a curated namespace, must be exactly one subclass of
   `ArchiveAwarePAFGAgent`.
4. The compiled class is dropped into the panel under the slot name
   `pafg-llm`; `evaluate_config_multi` runs it like any other agent.

On any failure (LLM error, parse fail, validation fail, exec fail) the slot
is silently dropped for that single evaluation — mortar continues with the
remaining panel agents, no crash.

**Caching.** Each archive entry gains an optional `pafg_llm_agent` field with
`{name, description, code}`. On subsequent visits to the same parent (or on
mortar restart), `_recompile_cached_pafg_llm` reuses the saved source instead
of paying for another LLM call. If the cached source ever fails to recompile
(e.g. ABC moved between branches), the cache is dropped and regenerated.

**Cost.** Each iteration that uses `pafg-llm` makes one extra LLM call per
config that needs evaluating (the mechanic-mutation call still happens
separately). The default panel is `["random", "pafg", "pafg-llm"]`. `neural`
was dropped from the default because it underperforms on the mutated-mechanic
configs MORTAR generates; opt in explicitly with `--agents random pafg neural
pafg-llm` if you want it.

**Hook surface for the generated subclass** (defined on
`ArchiveAwarePAFGAgent`):

- `opening_action(game, grid, rows, cols)` — first move (default `("reveal", 0, 0)`)
- `neighbor_positions(game, r, c, rows, cols)` — adjacency topology (default delegates to `board.neighbors`)
- `clue_value(game, grid, r, c)` — what the cell's number means (default `cell.adjacent_mines`)
- `frontier_cell_score(game, grid, pos, prob_map, mine_set, neighbors_of)` — guess-stage tiebreaker
- `postprocess_prob_map(game, grid, prob_map, mine_set, neighbors_of)` — last-step probability adjustments

Note that `ArchiveAwarePAFGAgent` already calls `game.board.neighbors()` by
default, so it is mechanic-aware out of the box for non-Moore neighborhoods.
The standard `PAFGAgent` uses a hardcoded Moore generator — under
`neighborhood="knight"` etc. its constraint propagation reads the wrong
adjacency. This is one reason `pafg-llm` is worth running on
neighborhood/win-condition mutations even before the LLM-tuned variant kicks in.

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
| `parity-vision` | Numbers show only `E`/`O` (even/odd of adjacent mine count); player starts with 2 HP |
| `knight-moves` | Adjacency follows chess-knight moves; cascade jumps non-locally |
| `flag-hunter` | Win by flagging every mine and only mines — pure-reveal play cannot win |
| `quota-rush` | Win after revealing half the safe cells; mine count bumped to 60 to keep the partial-clear meaningful |

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

`MineBehavior` runs after every reveal/flag while the game is `ACTIVE`. It can mutate the board (move mines, recompute adjacency), reveal additional cells, and apply player damage. The game re-runs the configured `WinCondition` (and the standard health-based loss check) after the hook returns.

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
  --info-strategy STR     count-mines | count-flags | parity | distance | direction | noisy-count | gen-XXXXXXXX (default: count-mines)
  --neighborhood STR      moore | von-neumann | diagonal | knight | radius-2-moore | gen-XXXXXXXX (default: moore)
  --win-condition STR     standard | reveal-quota | flag-all-mines | survival | gen-XXXXXXXX (default: standard)
  --no-safe-click         Disable safe first click guarantee
  --mechanic NAME         Start from a named preset (standard, extra-life, drifting-mines, chain-reaction, parity-vision, knight-moves, flag-hunter, quota-rush)
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
