// The mechanic source that each playable game runs — the generated Python
// subclass (or core engine class) that defines the rule. The browser engine in
// `minesweeper.ts` is an algorithm-faithful port of these.

export const MECHANIC_CODE: Record<string, string> = {
  "ranked-neighborhood": `class RankedNeighborInfo(InfoStrategy):
    """Each revealed cell shows the RANK of its adjacent-mine count relative to
    its revealed neighbors: 1 = fewest mines nearby. Falls back to the raw count
    when the cell has no revealed neighbors yet."""

    def encode(self, board, r, c):
        my_count = board.grid[r][c].adjacent_mines
        revealed_neighbors = [
            board.grid[nr][nc].adjacent_mines
            for nr, nc in board.neighbors(r, c)
            if board.grid[nr][nc].is_revealed and not board.grid[nr][nc].is_mine
        ]
        if not revealed_neighbors:
            return str(my_count) if my_count > 0 else ""
        ordered = sorted(set(revealed_neighbors + [my_count]))
        rank = ordered.index(my_count) + 1  # 1-based
        return str(rank)`,

  "radius-drift": `class Radius2MooreNeighborhood(Neighborhood):
    """24 cells in a 5x5 box — every clue counts mines over this whole region."""

    def offsets(self):
        return [(dr, dc)
                for dr in range(-2, 3)
                for dc in range(-2, 3)
                if (dr, dc) != (0, 0)]


class DriftingMines(MineBehavior):
    """Each turn, every unflagged mine has drift_prob chance of walking into an
    adjacent unrevealed, unflagged, non-mine cell. Flagged mines stay pinned."""

    DEFAULT_DRIFT_PROB = 0.3

    def on_post_action(self, board, game, action):
        moved = False
        mines = [(r, c)
                 for r in range(board.config.rows)
                 for c in range(board.config.cols)
                 if board.grid[r][c].is_mine and not board.grid[r][c].is_flagged]
        for (r, c) in mines:
            if self._rng.random() >= self.drift_prob:
                continue
            candidates = [(nr, nc) for nr, nc in board.neighbors(r, c)
                          if not board.grid[nr][nc].is_revealed
                          and not board.grid[nr][nc].is_flagged
                          and not board.grid[nr][nc].is_mine]
            if candidates:
                dst = self._rng.choice(candidates)
                if board.move_mine((r, c), dst):
                    moved = True
        if moved:
            board.recompute_adjacency()`,

  "telegraphed-mines": `class TelegraphedMines(MineBehavior):
    """Each turn a rotating ~20% of hidden, un-flagged mines auto-flag themselves
    as one-turn warnings. Last turn's warnings clear first, so a different subset
    is telegraphed on every move."""

    DEFAULT_TELEGRAPH_FRACTION = 0.20

    def on_post_action(self, board, game, action):
        # clear the mines we auto-flagged last turn
        for (r, c) in self._prev_auto_flagged:
            cell = board.grid[r][c]
            if not cell.is_revealed and cell.is_mine and cell.is_flagged:
                cell.is_flagged = False

        hidden_mines = [(r, c)
                        for r in range(board.config.rows)
                        for c in range(board.config.cols)
                        if board.grid[r][c].is_mine
                        and not board.grid[r][c].is_revealed
                        and not board.grid[r][c].is_flagged]
        if not hidden_mines:
            self._prev_auto_flagged = set()
            return

        count = max(1, int(len(hidden_mines) * self.telegraph_fraction))
        chosen = self._rng.sample(hidden_mines, min(count, len(hidden_mines)))
        for (r, c) in chosen:
            board.grid[r][c].is_flagged = True
        self._prev_auto_flagged = set(chosen)`,

  "checkerboard-reveal": `class CheckerboardReveal(RevealStrategy):
    """Cascade that only propagates through cells sharing the clicked cell's
    (r + c) parity. Opposite-parity neighbors are revealed as a border but never
    used as seeds, leaving an interleaved hidden lattice."""

    def reveal(self, board, r, c):
        cell = board.grid[r][c]
        parity = (r + c) % 2
        if cell.adjacent_mines > 0:
            cell.is_revealed = True
            return [(r, c)]

        queue = deque([(r, c)])
        visited = {(r, c)}
        revealed = []
        while queue:
            cr, cc = queue.popleft()
            curr = board.grid[cr][cc]
            if curr.is_revealed or curr.is_flagged or curr.is_mine:
                continue
            curr.is_revealed = True
            revealed.append((cr, cc))
            if curr.adjacent_mines == 0:
                for nr, nc in board.neighbors(cr, cc):
                    if (nr, nc) in visited:
                        continue
                    if (nr + nc) % 2 == parity:          # same parity: keep flooding
                        visited.add((nr, nc))
                        queue.append((nr, nc))
                    else:                                 # opposite parity: reveal, no cascade
                        nb = board.grid[nr][nc]
                        if not nb.is_revealed and not nb.is_flagged and not nb.is_mine:
                            visited.add((nr, nc))
                            nb.is_revealed = True
                            revealed.append((nr, nc))
        return revealed`,

  "ripple-reveal": `class RippleReveal(RevealStrategy):
    """Reveal expands in concentric BFS rings from the click, but halts the
    entire expansion the moment a ring contains any numbered cell."""

    def reveal(self, board, r, c):
        cell = board.grid[r][c]
        if cell.adjacent_mines > 0:
            cell.is_revealed = True
            return [(r, c)]

        revealed = []
        visited = {(r, c)}
        ring = [(r, c)]
        while ring:
            ring_revealed = []
            for cr, cc in ring:
                curr = board.grid[cr][cc]
                if curr.is_mine or curr.is_flagged:
                    continue
                if not curr.is_revealed:
                    curr.is_revealed = True
                    ring_revealed.append((cr, cc))
            revealed.extend(ring_revealed)

            # stop the whole ripple once any numbered cell shows up in this ring
            if any(board.grid[cr][cc].adjacent_mines > 0 for cr, cc in ring_revealed):
                break

            next_ring = []
            for cr, cc in ring_revealed:
                if board.grid[cr][cc].adjacent_mines != 0:
                    continue
                for nr, nc in board.neighbors(cr, cc):
                    if (nr, nc) not in visited:
                        nb = board.grid[nr][nc]
                        if not nb.is_mine and not nb.is_flagged and not nb.is_revealed:
                            visited.add((nr, nc))
                            next_ring.append((nr, nc))
            ring = next_ring
        return revealed`,
};
