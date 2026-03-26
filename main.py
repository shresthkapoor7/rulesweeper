from __future__ import annotations
import argparse
import dataclasses
import sys

from config import GameConfig
from game import MinesweeperGame, GameState
from renderer import TerminalRenderer
from reveal_strategies import REVEAL_STRATEGIES
from mechanics_archive import MECHANICS


def parse_args() -> GameConfig:
    p = argparse.ArgumentParser(description="Minesweeper (MORTAR base game)")
    p.add_argument("--mechanic",        default=None, dest="mechanic",
                   choices=list(MECHANICS.keys()),
                   help="Start from a named mechanic preset")
    p.add_argument("--rows",         type=int,  default=None)
    p.add_argument("--cols",         type=int,  default=None)
    p.add_argument("--mines",        type=int,  default=None, dest="mine_count")
    p.add_argument("--health",       type=int,  default=None, dest="starting_health")
    p.add_argument("--damage",       type=int,  default=None, dest="mine_damage")
    p.add_argument("--flag-limit",      type=int,  default=None, dest="flag_limit")
    p.add_argument("--reveal-strategy", default=None, dest="reveal_strategy",
                   choices=list(REVEAL_STRATEGIES.keys()))
    p.add_argument("--no-safe-click",   action="store_false", dest="safe_first_click")
    args = p.parse_args()

    # Start from named preset or bare defaults; then apply any explicit CLI overrides
    base = MECHANICS[args.mechanic]() if args.mechanic else GameConfig()
    overrides = {k: v for k, v in vars(args).items()
                 if k != "mechanic" and v is not None}
    # safe_first_click is store_false so False is an explicit override; True is the default
    if args.safe_first_click is True and args.mechanic:
        overrides.pop("safe_first_click", None)
    return dataclasses.replace(base, **overrides)


def parse_command(raw: str) -> tuple[str, int, int] | str | None:
    """
    Parse player input. Returns:
      (action, row, col)  for reveal/flag
      "quit"              for q/quit
      "help"              for h/help
      None                on parse failure
    """
    parts = raw.lower().split()
    if not parts:
        return None
    if parts[0] in ("q", "quit"):
        return "quit"
    if parts[0] in ("h", "help"):
        return "help"
    if parts[0] in ("r", "reveal", "f", "flag") and len(parts) == 3:
        try:
            action = "reveal" if parts[0] in ("r", "reveal") else "flag"
            row = int(parts[1])
            col_raw = parts[2]
            # Accept either a letter (A–Z) or a numeric index
            col = ord(col_raw) - ord("a") if col_raw.isalpha() else int(col_raw)
            return (action, row, col)
        except ValueError:
            return None
    return None


def game_loop(game: MinesweeperGame, renderer: TerminalRenderer) -> None:
    cfg = game.get_config()
    while True:
        renderer.render(game)
        renderer.render_status(game)

        try:
            raw = renderer.prompt()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        parsed = parse_command(raw)

        if parsed is None:
            print("  Unknown command. Type 'h' for help.\n")
            continue
        if parsed == "quit":
            break
        if parsed == "help":
            renderer.render_help()
            continue

        action, r, c = parsed

        if not (0 <= r < cfg.rows and 0 <= c < cfg.cols):
            print(f"  Out of bounds. Row 0–{cfg.rows - 1}, col 0–{cfg.cols - 1}.\n")
            continue

        if action == "reveal":
            game.reveal(r, c)
        else:
            game.flag(r, c)

        state = game.get_state()
        if state in (GameState.WON, GameState.LOST):
            renderer.render(game)
            renderer.render_result(game)
            break


def main() -> None:
    config = parse_args()
    game = MinesweeperGame(config)
    renderer = TerminalRenderer()
    renderer.render_help()
    game_loop(game, renderer)


if __name__ == "__main__":
    main()
