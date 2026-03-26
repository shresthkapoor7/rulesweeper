# MORTAR Minesweeper

This repository adapts the [MORTAR](https://arxiv.org/pdf/2601.00105) quality-diversity and LLM-guided system to Minesweeper, in order generate novel and interesting game mechanics.

![Minesweeper screenshot](assets/screenshot.png)

## Running the game
To play as a human, from the mortar-minesweeper directory run
```
python main.py       
```

To play with a particular mechanic, run
```
python main.py --mechanic extra_life
```

## Playing the game
To play as a human, use the following commands

| Action | Command | Example |
|---|---|---|
| Reveal a cell | `r <row> <col>` | r 7 D |
| Flag a cell | `f <row> <col>` | f 2 J |
| Show all commands | `h` | |
| Quit | `q` | |


## Setup
- Uses pure python stdlib, no dependencies to install
- Python 3.10+ required
