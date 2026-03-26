from __future__ import annotations
import sys

from game import MinesweeperGame, GameState
from cell import Cell


# ANSI color codes — only applied when stdout is a tty
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_BLUE   = "\033[34m"

# Per-number colors matching classic minesweeper convention
_NUM_COLORS = {
    1: "\033[34m",   # blue
    2: "\033[32m",   # green
    3: "\033[31m",   # red
    4: "\033[34;1m", # dark blue
    5: "\033[31;1m", # dark red
    6: "\033[36m",   # cyan
    7: "\033[35m",   # magenta
    8: "\033[37m",   # white
}


class TerminalRenderer:
    """
    Stateless display component. Receives a MinesweeperGame at render time;
    never holds a reference to it, so it works cleanly across game.reset() calls.
    """

    def __init__(self) -> None:
        self._color = sys.stdout.isatty()

    ##############################################################
    # PUBLIC API
    ##############################################################

    @staticmethod
    def col_label(c: int) -> str:
        """Map column index to a letter label (0→A, 1→B, …, 25→Z)."""
        return chr(ord("A") + c)

    def render(self, game: MinesweeperGame) -> None:
        cfg = game.get_config()
        # Column header — one letter per cell, each padded to 2 chars to align with cells
        col_header = "     " + "".join(f" {self.col_label(c)}" for c in range(cfg.cols))
        print(col_header)
        print("    " + "--" * cfg.cols)
        for r in range(cfg.rows):
            row_str = f"{r:2} | "
            for c in range(cfg.cols):
                row_str += self._cell_str(game.get_cell(r, c))
            print(row_str)
        print()

    def render_status(self, game: MinesweeperGame) -> None:
        m = game.get_player_metrics()
        cfg = game.get_config()
        flags = game.board.flags_placed()
        total_safe = cfg.rows * cfg.cols - cfg.mine_count
        remaining = total_safe - m["cells_revealed"]
        print(
            f"  HP: {m['health_remaining']}/{cfg.starting_health}  "
            f"Moves: {m['moves']}  "
            f"Flags: {flags}  "
            f"Remaining: {remaining}"
        )

    def render_result(self, game: MinesweeperGame) -> None:
        state = game.get_state()
        m = game.get_player_metrics()
        if state == GameState.WON:
            msg = self._color_str(_GREEN + _BOLD, "You won!")
        else:
            msg = self._color_str(_RED + _BOLD, "Game over.")
        print(f"\n{msg}")
        print(
            f"  Cells revealed: {m['cells_revealed']}  "
            f"Mines hit: {m['mines_hit']}  "
            f"Moves: {m['moves']}  "
            f"Efficiency: {m['efficiency']:.2f}"
        )

    def render_help(self) -> None:
        print("Commands:")
        print("  r <row> <col>  — reveal a cell  (e.g. r 7 D)")
        print("  f <row> <col>  — toggle flag    (e.g. f 2 J)")
        print("  h              — show this help")
        print("  q              — quit")
        print()

    def prompt(self) -> str:
        return input("> ").strip()

    ##############################################################
    # INTERNALS
    ##############################################################

    def _cell_str(self, cell: Cell) -> str:
        if not cell.is_revealed:
            if cell.is_flagged:
                return self._color_str(_YELLOW, " F")
            return " ."
        if cell.is_mine:
            return self._color_str(_RED, " *")
        if cell.adjacent_mines == 0:
            return "  "
        n = cell.adjacent_mines
        color = _NUM_COLORS.get(n, "")
        return self._color_str(color, f" {n}")

    def _color_str(self, color: str, text: str) -> str:
        if self._color and color:
            return f"{color}{text}{_RESET}"
        return text
