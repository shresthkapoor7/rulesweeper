# MORTAR Minesweeper

This repository adapts the [MORTAR](https://arxiv.org/pdf/2601.00105) quality-diversity and LLM-guided system to Minesweeper, in order generate novel and interesting game mechanics.



## Playing the game

### Human

To play as a human, from the mortar-minesweeper directory run

```
python main.py       
```

To play with a particular mechanic, run

```
python main.py --mechanic extra-life
```

Use the following commands to play the game


| Action            | Command         | Example |
| ----------------- | --------------- | ------- |
| Reveal a cell     | `r <row> <col>` | r 7 D   |
| Flag a cell       | `f <row> <col>` | f 2 J   |
| Show all commands | `h`             |         |
| Quit              | `q`             |         |


### Agent

Available agents: `random`, `pafg`, `neural`

| Command                                    | Outcome                                               |
| ------------------------------------------ | ----------------------------------------------------- |
| Single, headless game                      | `python main.py --agent random`                       |
| To see board outcome                       | `python main.py --agent random --watch`               |
| To batch the summary across a set of games | `python main.py --agent random --games 100`           |
| Reproducable run with a seed               | `python main.py --agent random --seed 42`             |
| With a mechanic enabled                    | `python main.py --agent random --mechanic extra-life` |

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
| `--eps-decay`       | 100000       | Steps over which epsilon decays from 1.0 → 0.05   |
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
python mortar.py --iterations 10 --games 50
```


| Flag           | Default      | Description                                |
| -------------- | ------------ | ------------------------------------------ |
| `--iterations` | 10           | Number of mutation steps                   |
| `--games`      | 50           | Games per config evaluation                |
| `--delay`      | 10           | Seconds between iterations (rate limiting) |
| `--archive`    | archive.json | Archive file path                          |


Results are saved to `archive.json` after each accepted config.

## Setup

- Python 3.10+ required
- Core game (`main.py`, agents, etc.) — stdlib only, no install needed
- Neural agent — requires `pip install torch`
- MORTAR engine (`mortar.py`) — requires `pip install openai` and an OpenRouter API key in `.env`

