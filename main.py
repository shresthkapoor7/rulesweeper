from __future__ import annotations
import argparse
import dataclasses
import sys

from config import GameConfig
from game import MinesweeperGame, GameState
from renderer import TerminalRenderer
from reveal_strategies import REVEAL_STRATEGIES
from mechanics_archive import MECHANICS
from agents import AGENTS, run_game


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minesweeper (MORTAR base game)")

    # --- Agent mode ---
    p.add_argument("--agent",  default=None, choices=list(AGENTS.keys()),
                   help="Run a named agent instead of human play")
    p.add_argument("--watch",  action="store_true",
                   help="Render the board while the agent plays")
    p.add_argument("--games",  type=int, default=1,
                   help="Number of games to run (agent mode only)")
    p.add_argument("--seed",   type=int, default=None,
                   help="Random seed for reproducible runs")

    # --- Game config ---
    p.add_argument("--mechanic",        default=None, dest="mechanic",
                   choices=list(MECHANICS.keys()),
                   help="Start from a named mechanic preset")
    p.add_argument("--rows",            type=int, default=None)
    p.add_argument("--cols",            type=int, default=None)
    p.add_argument("--mines",           type=int, default=None, dest="mine_count")
    p.add_argument("--health",          type=int, default=None, dest="starting_health")
    p.add_argument("--damage",          type=int, default=None, dest="mine_damage")
    p.add_argument("--flag-limit",      type=int, default=None, dest="flag_limit")
    p.add_argument("--reveal-strategy", default=None, dest="reveal_strategy",
                   choices=list(REVEAL_STRATEGIES.keys()))
    p.add_argument("--no-safe-click",   action="store_false", dest="safe_first_click")

    return p.parse_args()


def build_config(args: argparse.Namespace) -> GameConfig:
    """Build a GameConfig from parsed args, applying preset then CLI overrides."""
    base = MECHANICS[args.mechanic]() if args.mechanic else GameConfig()
    config_fields = {"rows", "cols", "mine_count", "starting_health",
                     "mine_damage", "flag_limit", "reveal_strategy"}
    overrides = {k: v for k, v in vars(args).items()
                 if k in config_fields and v is not None}
    if args.safe_first_click is True and args.mechanic:
        overrides.pop("safe_first_click", None)
    elif not args.safe_first_click:
        overrides["safe_first_click"] = False
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


def run_agent_cli(args: argparse.Namespace, config: GameConfig) -> None:
    agent = AGENTS[args.agent](seed=args.seed)
    renderer = TerminalRenderer() if args.watch else None
    # Only forward seed for single-game runs; batch runs use fresh randomness each game
    game_seed = args.seed if args.games == 1 else None
    results = [
        run_game(agent, config, renderer=renderer, seed=game_seed)
        for _ in range(args.games)
    ]
    if args.games == 1:
        print(results[0])
    else:
        wins = sum(1 for r in results if r["state"] == "won")
        avg_revealed  = sum(r["cells_revealed"] for r in results) / len(results)
        avg_efficiency = sum(r["efficiency"]     for r in results) / len(results)
        print(f"Games:          {args.games}")
        print(f"Wins:           {wins} ({100 * wins // args.games}%)")
        print(f"Avg revealed:   {avg_revealed:.1f}")
        print(f"Avg efficiency: {avg_efficiency:.2f}")


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
    args = parse_args()
    config = build_config(args)

    if args.agent:
        run_agent_cli(args, config)
    else:
        renderer = TerminalRenderer()
        renderer.render_help()
        game_loop(MinesweeperGame(config), renderer)


if __name__ == "__main__":
    main()
