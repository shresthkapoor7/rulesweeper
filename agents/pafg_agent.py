from __future__ import annotations
import random
from fractions import Fraction

from agents import Agent
from game import MinesweeperGame


def _neighbors(r: int, c: int, rows: int, cols: int):
    """Yield valid (r, c) pairs for all 8 Moore-neighborhood cells."""
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr, dc) != (0, 0):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    yield nr, nc


class PAFGAgent(Agent):
    """
    PAFG hierarchical constraint solver.

    Executes four stages in priority order each turn:
      F (First)  — corner heuristic on the opening move
      P (Primary)— simple constraint propagation (deterministic free moves)
      A (Advanced)— Gaussian elimination + enumeration for exact mine probabilities
      G (Guess)  — pick lowest mine-probability cell when all else fails

    Based on: minesweepergame.com/math/a-solver-of-single-agent-stochastic-puzzle.pdf
    """

    _ENUM_CAP = 20  # max frontier vars per connected component before skipping enumeration

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def choose_action(self, game: MinesweeperGame) -> tuple[str, int, int]:
        cfg = game.get_config()
        rows, cols = cfg.rows, cfg.cols

        # Build board snapshot once — avoid repeated get_cell calls
        grid = [[game.get_cell(r, c) for c in range(cols)] for r in range(rows)]

        flag_limit = cfg.flag_limit
        flags_placed = sum(
            1 for r in range(rows) for c in range(cols) if grid[r][c].is_flagged
        )
        can_flag = flag_limit is None or flags_placed < flag_limit

        # Stage F — first move
        revealed_any = any(grid[r][c].is_revealed for r in range(rows) for c in range(cols))
        if not revealed_any:
            return ("reveal", 0, 0)

        # Stage P — constraint propagation
        # Returns the action to take (if any) and the full mine_set so later
        # stages can treat deduced mines as virtual flags even when can_flag=False.
        action, mine_set = self._stage_p(grid, rows, cols, can_flag)
        if action:
            return action

        # Stage A — Gaussian elimination + enumeration
        action, prob_map = self._stage_a(grid, rows, cols, cfg.mine_count, mine_set, can_flag)
        if action:
            return action

        # Stage G — guess on lowest mine probability; skip known mines
        return self._stage_g(grid, rows, cols, prob_map, mine_set)

    # ------------------------------------------------------------------
    # Stage P — constraint propagation
    # ------------------------------------------------------------------

    def _stage_p(
        self, grid, rows, cols, can_flag
    ) -> tuple[tuple[str, int, int] | None, set[tuple[int, int]]]:
        """
        Iterate constraint propagation until fixpoint.
        Returns (action, mine_set). mine_set is always returned so later stages
        can treat deduced mines as virtual flags even when can_flag=False.
        """
        changed = True
        safe_set: set[tuple[int, int]] = set()
        mine_set: set[tuple[int, int]] = set()

        while changed:
            changed = False
            for r in range(rows):
                for c in range(cols):
                    cell = grid[r][c]
                    if not cell.is_revealed or cell.adjacent_mines == 0:
                        continue
                    # Pass mine_set so already-deduced mines count as virtual flags,
                    # enabling second-order deductions within the same fixpoint loop.
                    flagged, unrevealed = self._classify_neighbors(
                        grid, rows, cols, r, c, virtual_mines=mine_set
                    )
                    remaining = cell.adjacent_mines - len(flagged)
                    if remaining < 0:
                        continue
                    if remaining == 0:
                        for pos in unrevealed:
                            if pos not in safe_set and pos not in mine_set:
                                safe_set.add(pos)
                                changed = True
                    elif remaining == len(unrevealed):
                        for pos in unrevealed:
                            if pos not in mine_set:
                                mine_set.add(pos)
                                changed = True

        if mine_set and can_flag:
            r, c = next(iter(mine_set))
            return ("flag", r, c), mine_set
        if safe_set:
            r, c = next(iter(safe_set))
            return ("reveal", r, c), mine_set
        return None, mine_set

    # ------------------------------------------------------------------
    # Stage A — Gaussian elimination + enumeration
    # ------------------------------------------------------------------

    def _stage_a(
        self, grid, rows, cols, mine_count,
        mine_set: set[tuple[int, int]],
        can_flag: bool,
    ) -> tuple[tuple[str, int, int] | None, dict[tuple[int, int], float]]:
        """
        Build and solve the constraint system. Returns (action, prob_map).
        action is set only when a deterministic deduction is found.
        prob_map maps every unrevealed/unflagged cell to P(mine).
        mine_set: cells already deduced as mines by Stage P (treated as virtual flags).
        can_flag: when False, skip flag actions to avoid infinite loops at flag limit.
        """
        # Enumerate unrevealed/unflagged cells (exclude Stage P virtual mines)
        unrevealed = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if not grid[r][c].is_revealed and not grid[r][c].is_flagged
            and (r, c) not in mine_set
        ]
        if not unrevealed:
            return None, {}

        # Frontier: unrevealed cells adjacent to at least one revealed numbered cell
        frontier_set: set[tuple[int, int]] = set()
        for r, c in unrevealed:
            for nr, nc in _neighbors(r, c, rows, cols):
                if grid[nr][nc].is_revealed and grid[nr][nc].adjacent_mines > 0:
                    frontier_set.add((r, c))
                    break

        frontier = sorted(frontier_set)
        var_index = {pos: i for i, pos in enumerate(frontier)}
        non_frontier = [pos for pos in unrevealed if pos not in frontier_set]

        # Build equations: each revealed cell with unrevealed frontier neighbors
        equations: list[tuple[list[int], int]] = []  # (row of var indices, rhs)
        seen_constraints: set[frozenset] = set()
        total_flagged = sum(
            1 for r in range(rows) for c in range(cols) if grid[r][c].is_flagged
        )

        for r in range(rows):
            for c in range(cols):
                cell = grid[r][c]
                if not cell.is_revealed or cell.adjacent_mines == 0:
                    continue
                flagged_nbrs, unrevealed_nbrs = self._classify_neighbors(grid, rows, cols, r, c)
                frontier_nbrs = [pos for pos in unrevealed_nbrs if pos in frontier_set]
                if not frontier_nbrs:
                    continue
                rhs = cell.adjacent_mines - len(flagged_nbrs)
                key = frozenset(frontier_nbrs)
                if key in seen_constraints:
                    continue
                seen_constraints.add(key)
                equations.append(([var_index[pos] for pos in frontier_nbrs], rhs))

        n = len(frontier)

        # Gaussian elimination using Fraction for exact arithmetic
        # Matrix: n_eq x (n+1), last column is RHS
        matrix = [[Fraction(0)] * (n + 1) for _ in equations]
        for i, (vars_in_eq, rhs) in enumerate(equations):
            for vi in vars_in_eq:
                matrix[i][vi] = Fraction(1)
            matrix[i][n] = Fraction(rhs)

        self._gauss_eliminate(matrix, n)

        # Extract forced values (0 or 1) from reduced matrix
        forced: dict[int, int] = {}  # var_index -> 0 or 1
        for row in matrix:
            pivots = [j for j in range(n) if row[j] != 0]
            if len(pivots) == 1:
                j = pivots[0]
                val = row[n] / row[j]
                if val == 0:
                    forced[j] = 0
                elif val == 1:
                    forced[j] = 1

        # Deterministic action from forced values
        for idx, val in forced.items():
            r, c = frontier[idx]
            if val == 0:
                return ("reveal", r, c), {}
            if val == 1 and can_flag:
                return ("flag", r, c), {}

        # Enumerate per connected component for probability
        components = self._connected_components(frontier, equations, n)
        prob_map: dict[tuple[int, int], float] = {}

        for comp_vars in components:
            comp_eqs = [
                (vars_in_eq, rhs) for vars_in_eq, rhs in equations
                if any(v in comp_vars for v in vars_in_eq)
            ]
            if len(comp_vars) <= self._ENUM_CAP:
                comp_probs = self._enumerate_component(comp_vars, comp_eqs, forced)
            else:
                # Too large to enumerate — use forced values if available, else 0.5
                comp_probs = {
                    v: float(forced[v]) if v in forced else 0.5
                    for v in comp_vars
                }
            for vi, p in comp_probs.items():
                prob_map[frontier[vi]] = p

        # Non-frontier cells share uniform probability from remaining mine budget
        mines_in_frontier_expected = sum(prob_map.get(pos, 0.5) for pos in frontier)
        remaining_mines = mine_count - total_flagged - mines_in_frontier_expected
        if non_frontier:
            p_non_frontier = max(0.0, min(1.0, remaining_mines / len(non_frontier)))
        else:
            p_non_frontier = 0.5
        for pos in non_frontier:
            prob_map[pos] = p_non_frontier

        # Check for deterministic mines in prob_map (prob == 1.0)
        if can_flag:
            for pos, p in prob_map.items():
                if p >= 1.0:
                    r, c = pos
                    return ("flag", r, c), prob_map

        return None, prob_map

    # ------------------------------------------------------------------
    # Stage G — probabilistic guess
    # ------------------------------------------------------------------

    def _stage_g(
        self, grid, rows, cols, prob_map: dict[tuple[int, int], float],
        mine_set: set[tuple[int, int]],
    ) -> tuple[str, int, int]:
        unrevealed = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if not grid[r][c].is_revealed and not grid[r][c].is_flagged
            and (r, c) not in mine_set
        ]
        if not unrevealed:
            # Fallback: should never happen in a live game
            r, c = self._rng.choice([(r, c) for r in range(rows) for c in range(cols)])
            return ("reveal", r, c)

        # Sort: lowest P(mine), then most unrevealed neighbors (info gain), then random
        def score(pos):
            r, c = pos
            p = prob_map.get(pos, 0.5)
            nbr_count = sum(
                1 for nr, nc in _neighbors(r, c, rows, cols)
                if not grid[nr][nc].is_revealed and not grid[nr][nc].is_flagged
            )
            return (p, -nbr_count, self._rng.random())

        r, c = min(unrevealed, key=score)
        return ("reveal", r, c)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _classify_neighbors(
        self, grid, rows, cols, r, c,
        virtual_mines: set[tuple[int, int]] | None = None,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """
        Return (flagged_neighbors, unrevealed_unflagged_neighbors).
        virtual_mines: cells deduced as mines but not yet flagged on the grid;
        counted as flagged for constraint purposes.
        """
        flagged, unrevealed = [], []
        for nr, nc in _neighbors(r, c, rows, cols):
            nbr = grid[nr][nc]
            if nbr.is_flagged or (virtual_mines and (nr, nc) in virtual_mines):
                flagged.append((nr, nc))
            elif not nbr.is_revealed:
                unrevealed.append((nr, nc))
        return flagged, unrevealed

    def _gauss_eliminate(self, matrix: list[list[Fraction]], n: int) -> None:
        """In-place row echelon form (partial pivoting)."""
        m = len(matrix)
        pivot_row = 0
        for col in range(n):
            # Find pivot
            found = -1
            for row in range(pivot_row, m):
                if matrix[row][col] != 0:
                    found = row
                    break
            if found == -1:
                continue
            matrix[pivot_row], matrix[found] = matrix[found], matrix[pivot_row]
            # Eliminate column in all other rows
            for row in range(m):
                if row != pivot_row and matrix[row][col] != 0:
                    factor = matrix[row][col] / matrix[pivot_row][col]
                    for j in range(n + 1):
                        matrix[row][j] -= factor * matrix[pivot_row][j]
            pivot_row += 1

    def _connected_components(
        self,
        frontier: list[tuple[int, int]],
        equations: list[tuple[list[int], int]],
        n: int,
    ) -> list[list[int]]:
        """
        Partition frontier variable indices into connected components —
        groups that share at least one equation.
        """
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[a] = b

        for vars_in_eq, _ in equations:
            for i in range(1, len(vars_in_eq)):
                union(vars_in_eq[0], vars_in_eq[i])

        groups: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)
        return list(groups.values())

    def _enumerate_component(
        self,
        comp_vars: list[int],
        comp_eqs: list[tuple[list[int], int]],
        forced: dict[int, int],
    ) -> dict[int, float]:
        """
        Enumerate all 0/1 assignments for comp_vars that satisfy comp_eqs.
        Returns P(mine) for each variable index.
        """
        # Map global var indices to local indices for this component
        local = {v: i for i, v in enumerate(comp_vars)}
        k = len(comp_vars)

        # Pre-apply forced values; skip this component if a forced value
        # is contradicted (shouldn't happen with correct propagation)
        fixed: dict[int, int] = {}  # local index -> forced value
        free_locals: list[int] = []
        for li, vi in enumerate(comp_vars):
            if vi in forced:
                fixed[li] = forced[vi]
            else:
                free_locals.append(li)

        mine_counts = [0] * k
        total = 0

        for bits in range(1 << len(free_locals)):
            assignment = list(fixed.get(li, 0) for li in range(k))
            for bit_pos, li in enumerate(free_locals):
                assignment[li] = (bits >> bit_pos) & 1

            # Check all equations for this component
            valid = True
            for vars_in_eq, rhs in comp_eqs:
                s = sum(assignment[local[v]] for v in vars_in_eq if v in local)
                if s != rhs:
                    valid = False
                    break

            if valid:
                total += 1
                for li in range(k):
                    mine_counts[li] += assignment[li]

        if total == 0:
            return {vi: 0.5 for vi in comp_vars}

        return {vi: mine_counts[li] / total for li, vi in enumerate(comp_vars)}
