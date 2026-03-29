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


| Command                                    | Outcome                                               |
| ------------------------------------------ | ----------------------------------------------------- |
| Single, headless game                      | `python main.py --agent random`                       |
| To see board outcome                       | `python main.py --agent random --watch`               |
| To batch the summary across a set of games | `python main.py --agent random --games 100`           |
| Reproducable run with a seed               | `python main.py --agent random --seed 42`             |
| With a mechanic enabled                    | `python main.py --agent random --mechanic extra-life` |


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
- MORTAR engine (`mortar.py`) — requires `pip install openai` and an OpenRouter API key in `.env`

