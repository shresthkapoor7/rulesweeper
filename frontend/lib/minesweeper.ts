// A faithful TypeScript port of the mechanics used by the two example
// mechanics from the RuleSweeper paper:
//   1. "ranked"       — RankedNeighborInfo clue encoding (moore, static mines)
//   2. "radius-drift" — radius-2 (5x5) neighborhood + drifting mines
//
// This mirrors the Python engine in the repo (board.py, reveal_strategies.py,
// mine_behaviors.py, info_strategies.py, neighborhoods.py) closely enough that
// the browser game plays the same as the research build.

export type Neighborhood = "moore" | "radius-2-moore";
export type MineBehavior = "static" | "drifting";
export type InfoStrategy = "count-mines" | "ranked";

export interface GameConfig {
  rows: number;
  cols: number;
  mineCount: number;
  startingHealth: number;
  neighborhood: Neighborhood;
  mineBehavior: MineBehavior;
  info: InfoStrategy;
  driftProb: number;
  safeFirstClick: boolean;
}

export interface Cell {
  isMine: boolean;
  isRevealed: boolean;
  isFlagged: boolean;
  adjacentMines: number;
}

export type Status = "pending" | "active" | "won" | "lost";

function offsets(n: Neighborhood): [number, number][] {
  if (n === "moore") {
    const out: [number, number][] = [];
    for (let dr = -1; dr <= 1; dr++)
      for (let dc = -1; dc <= 1; dc++)
        if (dr !== 0 || dc !== 0) out.push([dr, dc]);
    return out;
  }
  // radius-2-moore: 24 cells in a 5x5 box
  const out: [number, number][] = [];
  for (let dr = -2; dr <= 2; dr++)
    for (let dc = -2; dc <= 2; dc++)
      if (dr !== 0 || dc !== 0) out.push([dr, dc]);
  return out;
}

export class Minesweeper {
  readonly cfg: GameConfig;
  grid: Cell[][];
  health: number;
  status: Status;
  minesPlaced = false;
  moves = 0;
  private off: [number, number][];

  constructor(cfg: GameConfig) {
    this.cfg = cfg;
    this.health = cfg.startingHealth;
    this.status = "pending";
    this.off = offsets(cfg.neighborhood);
    this.grid = Array.from({ length: cfg.rows }, () =>
      Array.from({ length: cfg.cols }, () => ({
        isMine: false,
        isRevealed: false,
        isFlagged: false,
        adjacentMines: 0,
      }))
    );
  }

  private inBounds(r: number, c: number) {
    return r >= 0 && r < this.cfg.rows && c >= 0 && c < this.cfg.cols;
  }

  neighbors(r: number, c: number): [number, number][] {
    const out: [number, number][] = [];
    for (const [dr, dc] of this.off) {
      const nr = r + dr;
      const nc = c + dc;
      if (this.inBounds(nr, nc)) out.push([nr, nc]);
    }
    return out;
  }

  private placeMines(er: number, ec: number) {
    const exclude = new Set<string>();
    exclude.add(`${er},${ec}`);
    if (this.cfg.safeFirstClick) {
      for (const [nr, nc] of this.neighbors(er, ec)) exclude.add(`${nr},${nc}`);
    }
    const candidates: [number, number][] = [];
    for (let r = 0; r < this.cfg.rows; r++)
      for (let c = 0; c < this.cfg.cols; c++)
        if (!exclude.has(`${r},${c}`)) candidates.push([r, c]);

    const count = Math.min(this.cfg.mineCount, candidates.length);
    // Fisher-Yates partial shuffle
    for (let i = 0; i < count; i++) {
      const j = i + Math.floor(Math.random() * (candidates.length - i));
      [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
      const [r, c] = candidates[i];
      this.grid[r][c].isMine = true;
    }
    this.recomputeAdjacency();
    this.minesPlaced = true;
  }

  private recomputeAdjacency() {
    for (let r = 0; r < this.cfg.rows; r++)
      for (let c = 0; c < this.cfg.cols; c++) {
        if (this.grid[r][c].isMine) continue;
        let n = 0;
        for (const [nr, nc] of this.neighbors(r, c))
          if (this.grid[nr][nc].isMine) n++;
        this.grid[r][c].adjacentMines = n;
      }
  }

  private isSolved(): boolean {
    for (const row of this.grid)
      for (const cell of row)
        if (!cell.isMine && !cell.isRevealed) return false;
    return true;
  }

  // Standard cascade reveal (BFS flood-fill from zero-adjacent cells).
  private cascade(r: number, c: number): [number, number][] {
    const cell = this.grid[r][c];
    if (cell.adjacentMines > 0) {
      cell.isRevealed = true;
      return [[r, c]];
    }
    const queue: [number, number][] = [[r, c]];
    const visited = new Set<string>([`${r},${c}`]);
    const revealed: [number, number][] = [];
    while (queue.length) {
      const [cr, cc] = queue.shift()!;
      const cur = this.grid[cr][cc];
      if (cur.isRevealed || cur.isFlagged || cur.isMine) continue;
      cur.isRevealed = true;
      revealed.push([cr, cc]);
      if (cur.adjacentMines === 0) {
        for (const [nr, nc] of this.neighbors(cr, cc)) {
          const key = `${nr},${nc}`;
          if (!visited.has(key)) {
            visited.add(key);
            queue.push([nr, nc]);
          }
        }
      }
    }
    return revealed;
  }

  reveal(r: number, c: number) {
    if (this.status === "won" || this.status === "lost") return;
    if (!this.minesPlaced) this.placeMines(r, c);
    const cell = this.grid[r][c];
    if (cell.isRevealed || cell.isFlagged) return;
    this.moves++;

    if (cell.isMine) {
      cell.isRevealed = true;
      this.health -= 1;
      if (this.health <= 0) {
        this.status = "lost";
        return;
      }
    } else {
      this.cascade(r, c);
    }
    this.status = "active";
    this.runMineBehavior();
    this.checkEnd();
  }

  toggleFlag(r: number, c: number) {
    if (this.status === "won" || this.status === "lost") return;
    const cell = this.grid[r][c];
    if (cell.isRevealed) return;
    cell.isFlagged = !cell.isFlagged;
    this.moves++;
    if (this.minesPlaced) {
      this.status = "active";
      this.runMineBehavior();
      this.checkEnd();
    }
  }

  // DriftingMines: each unflagged mine has driftProb chance of walking into an
  // adjacent unrevealed, unflagged, non-mine cell. Flagged mines stay pinned.
  private runMineBehavior() {
    if (this.cfg.mineBehavior !== "drifting" || this.status !== "active") return;
    const mines: [number, number][] = [];
    for (let r = 0; r < this.cfg.rows; r++)
      for (let c = 0; c < this.cfg.cols; c++)
        if (this.grid[r][c].isMine && !this.grid[r][c].isFlagged)
          mines.push([r, c]);

    let moved = false;
    for (const [r, c] of mines) {
      if (Math.random() >= this.cfg.driftProb) continue;
      const candidates = this.neighbors(r, c).filter(([nr, nc]) => {
        const t = this.grid[nr][nc];
        return !t.isRevealed && !t.isFlagged && !t.isMine;
      });
      if (!candidates.length) continue;
      const [dr, dc] = candidates[Math.floor(Math.random() * candidates.length)];
      this.grid[r][c].isMine = false;
      this.grid[dr][dc].isMine = true;
      moved = true;
    }
    if (moved) this.recomputeAdjacency();
  }

  private checkEnd() {
    if (this.status === "lost") return;
    if (this.isSolved()) this.status = "won";
  }

  // What text a revealed safe cell shows the player.
  infoAt(r: number, c: number): string {
    const cell = this.grid[r][c];
    if (this.cfg.info === "count-mines") {
      return cell.adjacentMines > 0 ? String(cell.adjacentMines) : "";
    }
    // ranked: ordinal rank of this cell's mine count among its revealed,
    // non-mine neighbors (1 = lowest). Falls back to the raw count when the
    // cell has no revealed neighbors yet.
    const my = cell.adjacentMines;
    const neigh: number[] = [];
    for (const [nr, nc] of this.neighbors(r, c)) {
      const t = this.grid[nr][nc];
      if (t.isRevealed && !t.isMine) neigh.push(t.adjacentMines);
    }
    if (neigh.length === 0) return my > 0 ? String(my) : "";
    const uniq = Array.from(new Set([...neigh, my])).sort((a, b) => a - b);
    return String(uniq.indexOf(my) + 1);
  }

  minesRemaining(): number {
    let flags = 0;
    for (const row of this.grid) for (const cell of row) if (cell.isFlagged) flags++;
    return this.cfg.mineCount - flags;
  }
}
