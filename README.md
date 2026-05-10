# MORTAR Minesweeper

This repository adapts the [MORTAR](https://arxiv.org/pdf/2601.00105) quality-diversity and LLM-guided mechanic-generation system to Minesweeper. The engine evolves Minesweeper variants through LLM-guided mutation and an LLM-coded adaptive agent that re-tunes itself per mechanic — letting MORTAR discover game variants that fixed-agent benchmarks cannot evaluate.


## Playing the game

### Human

To play as a human, from the `mortar-minesweeper/` directory run

```
python main.py
```

To play with a particular mechanic, run

```
python main.py --mechanic extra-life
```

Available named mechanics: `standard`, `extra-life`, `drifting-mines`, `chain-reaction`, `parity-vision`, `knight-moves`, `flag-hunter`, `quota-rush`.

In-game commands:

| Action            | Command         | Example |
| ----------------- | --------------- | ------- |
| Reveal a cell     | `r <row> <col>` | r 7 D   |
| Flag a cell       | `f <row> <col>` | f 2 J   |
| Show all commands | `h`             |         |
| Quit              | `q`             |         |

Any individual mechanic can also be set directly via CLI flag — `--reveal-strategy`, `--mine-behavior`, `--info-strategy`, `--neighborhood`, `--win-condition`, `--health`, `--damage`, `--flag-limit`. These compose on top of `--mechanic` if both are given.

### Agent

Built-in agents: `random`, `pafg`, `pafg-llm`, `neural`.

| Command                                    | Outcome                                               |
| ------------------------------------------ | ----------------------------------------------------- |
| Single, headless game                      | `python main.py --agent random`                       |
| To see board outcome                       | `python main.py --agent random --watch`               |
| To batch the summary across a set of games | `python main.py --agent random --games 100`           |
| Reproducible run with a seed               | `python main.py --agent random --seed 42`             |
| With a mechanic enabled                    | `python main.py --agent random --mechanic extra-life` |

`pafg` is a deterministic four-stage constraint solver (First / Primary / Advanced / Guess) — the canonical skilled-but-non-adaptive baseline. `pafg-llm` is the *adaptive* agent: at evaluation time it generates a fresh `ArchiveAwarePAFGAgent` subclass tailored to the live mechanic. See the MORTAR section below.

#### Neural Agent

The `neural` agent uses a CNN-based DQN (Deep Q-Network) trained via reinforcement learning. It requires PyTorch.

```bash
# CPU
pip install torch

# GPU (CUDA 12.x recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Training:**

```bash
python -m agents.train_neural --device cuda --episodes 100000 --rows 9 --cols 9 --mines 10
```

| Flag                | Default      | Description                                        |
| ------------------- | ------------ | -------------------------------------------------- |
| `--episodes`        | 50000        | Total training episodes                            |
| `--rows/cols/mines` | 9 / 9 / 10   | Board config for training                          |
| `--lr`              | 1e-4         | Learning rate                                      |
| `--batch-size`      | 64           | Gradient update batch size                         |
| `--buffer-size`     | 100000       | Replay buffer capacity                             |
| `--eps-decay`       | 100000       | Steps over which epsilon decays from 1.0 → 0.05    |
| `--target-update`   | 1000         | Sync target network every N steps                  |
| `--eval-freq`       | 500          | Evaluate every N episodes                          |
| `--checkpoint-dir`  | checkpoints/ | Where to save model checkpoints                    |
| `--resume`          | —            | Resume from a checkpoint path                      |
| `--device`          | auto         | `cpu` or `cuda`                                    |

**Playing with a trained model:**

```bash
python main.py --agent neural
python main.py --agent neural --watch
```

The agent loads `checkpoints/best.pt` by default. A GPU is recommended for training.


## Running the MORTAR evolution loop

`mortar.py` drives LLM-guided mechanic mutation. It requires an OpenRouter API key — copy `.env.example` to `.env` and fill in the key.

```bash
pip install openai
python mortar.py --iterations 200 --games 20
```

| Flag                  | Default                | Description                                                                     |
| --------------------- | ---------------------- | ------------------------------------------------------------------------------- |
| `--iterations`        | 10                     | Number of mutation steps                                                        |
| `--games`             | 10                     | Games per agent per config evaluation                                           |
| `--delay`             | 10                     | Seconds between iterations (rate limiting)                                      |
| `--archive`           | archive.json           | Archive file path                                                               |
| `--agents`            | random pafg pafg-llm   | Agents in the evaluation panel                                                  |
| `--mode`              | mixed                  | `param` (tune fields), `code` (generate new mechanic classes), `mixed`          |
| `--admit-all`         | off                    | Skip the admission gate; admit every parseable mutation                         |
| `--parent-temperature`| 0.2                    | Softmax temperature for parent selection. Lower = more elitist                  |
| `--qd` / `--no-qd`    | on                     | Toggle 2-axis MAP-Elites admission and bin-uniform parent selection             |
| `--admission-metric`  | skill_spread           | `skill_spread` (best-skilled vs random) or `llm_lift` (pafg-llm vs pafg)        |

Results are saved to `archive.json` after each accepted config.

### MORTAR adaptations in this repo

This repo extends the original MORTAR proposal in three ways. Each addresses a methodological gap in evaluating mechanic-generation when the agent is itself LLM-coded.

**1. `pafg-llm` — adaptive per-mechanic agent.** Before each candidate config is evaluated, an LLM call writes a fresh subclass of `ArchiveAwarePAFGAgent` tailored to the live mechanic. The subclass overrides one or two of six hooks (`opening_action`, `neighbor_positions`, `clue_value`, `action_priority`, `frontier_cell_score`, `postprocess_prob_map`) — solver internals are inherited. The compiled subclass is cached on the archive entry so revisits cost zero LLM calls. This breaks MORTAR's original assumption of a fixed evaluator-agent suite and lets the engine reach mechanics whose obfuscation defeats vanilla PAFG.

**2. 2-axis Quality-Diversity admission.** Each entry is binned on `(mechanic_difficulty, agent_complexity)` — `random_progress` and pafg-llm patch line count, respectively. Admission requires either filling an empty bin or beating the bin's elite on the active metric; parent selection picks a bin uniformly then softmax-samples within. This addresses *co-evolution drift* — the tendency for skill_spread to silently become "can the LLM write code for this" rather than "is this a good mechanic." Toggle with `--no-qd`.

**3. Two admission metrics.** `--admission-metric skill_spread` (default) measures playability — best non-random agent versus random. `--admission-metric llm_lift` measures *adaptive-agent uplift* — pafg-llm versus vanilla pafg. The two surface different research questions: skill_spread admits any mechanic a skilled agent can play; llm_lift deliberately concentrates the search on mechanics that *require* per-mechanic LLM-coded adaptation. Recommended pairing for `llm_lift` is `--no-qd`, since QD's bin-uniform sampling defeats the lift bias.

### Recommended runs

| Goal                                         | Command                                                                              |
| -------------------------------------------- | ------------------------------------------------------------------------------------ |
| Diverse, playability-floor archive           | `python mortar.py --iterations 200 --games 20 --qd --admission-metric skill_spread`   |
| Legacy flat skill_spread (matches paper)     | `python mortar.py --iterations 200 --games 20 --no-qd --admission-metric skill_spread`|
| LLM-uplift-maximizing benchmark              | `python mortar.py --iterations 200 --games 20 --no-qd --admission-metric llm_lift`    |

### Code-mutation mode

In `--mode code` (or 50% of `mixed` iterations) the LLM authors a brand-new
`MineBehavior`, `RevealStrategy`, `InfoStrategy`, `Neighborhood`, or `WinCondition`
subclass. The source is AST-validated (no imports, no introspection escapes),
exec'd in a curated namespace, and smoke-tested before going through the same
panel evaluation as parameter mutations. Accepted classes are stored in
`archive.json` under a deterministic `gen-XXXXXXXX` key and can be played
directly:

```bash
python main.py --mine-behavior gen-abc12345
python main.py --info-strategy gen-def67890
python main.py --neighborhood gen-fedcba98
```

**Trust caveat.** Generated code is `exec()`'d in-process. The AST denylist
catches the obvious foot-guns but is not a security boundary — a determined
snippet can still escape via attribute introspection. Treat `archive.json`
with non-null `code_source` fields as executable code, not data, and only
run MORTAR against trusted models. Subprocess isolation is a follow-up.

### Archive schema

Every accepted entry persists with these fields:

- `config_snapshot` — the `GameConfig` as a dict.
- `description` — human-readable summary written by the LLM.
- `fitness` — `{ per_agent: {...}, skill_spread: float, llm_lift: float, n_games: int }`.
- `parent_snapshot`, `generation`, `n_children_attempted`, `n_children_admitted`.
- `code_kind`, `code_key`, `code_source` — present only on entries that introduced a `gen-*` class.
- `pafg_llm_agent` — the cached LLM-coded subclass `{ name, description, code, assumptions }`.
- `descriptors` — the QD bin coordinates `{ mechanic_random_progress, agent_complexity_lines, mechanic_bin, agent_bin, mech_bin_label, agent_bin_label }`. Backfilled at load time on legacy archives.

Loading an archive is idempotent: missing `descriptors` and missing `llm_lift` are computed from existing data on read.

## Setup

- Python 3.10+ required
- Core game (`main.py`, agents, etc.) — stdlib only, no install needed
- Neural agent — requires `pip install torch`
- MORTAR engine (`mortar.py`) and `pafg-llm` agent — require `pip install openai` and an OpenRouter API key in `.env`
