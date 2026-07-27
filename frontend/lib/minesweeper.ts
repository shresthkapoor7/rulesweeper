// A faithful TypeScript port of the mechanics used by the two example
// mechanics from the RuleSweeper paper:
//   1. "ranked"       — RankedNeighborInfo clue encoding (moore, static mines)
//   2. "radius-drift" — radius-2 (5x5) neighborhood + drifting mines
//
// This mirrors the Python engine in the repo (board.py, reveal_strategies.py,
// mine_behaviors.py, info_strategies.py, neighborhoods.py) closely enough that
// the browser game plays the same as the research build.

export type Neighborhood = "moore" | "radius-2-moore";
export type MineBehavior = "static" | "drifting" | "telegraphed";
export type InfoStrategy = "count-mines" | "ranked";
export type RevealStrategy = "cascade" | "checkerboard" | "ripple";

export interface GameConfig {
  rows: number;
  cols: number;
  mineCount: number;
  startingHealth: number;
  neighborhood: Neighborhood;
  mineBehavior: MineBehavior;
  info: InfoStrategy;
  reveal: RevealStrategy;
  driftProb: number;
  telegraphFraction: number;
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
  // Cells that a mine behavior (Telegraphed Mines) has marked as a one-turn
  // warning. Tracked separately from player flags so it never touches the
  // player's own flagging state.
  autoFlagged = new Set<string>();
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

  private applyReveal(r: number, c: number): [number, number][] {
    switch (this.cfg.reveal) {
      case "checkerboard":
        return this.checkerboard(r, c);
      case "ripple":
        return this.ripple(r, c);
      default:
        return this.cascade(r, c);
    }
  }

  // Cascade restricted to same (r+c) parity: the flood only propagates through
  // matching-parity cells, and opposite-parity neighbors are revealed as a
  // border but never used as seeds — leaving an interleaved hidden lattice.
  private checkerboard(r: number, c: number): [number, number][] {
    const cell = this.grid[r][c];
    const parity = (r + c) % 2;
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
          if (visited.has(key)) continue;
          const nb = this.grid[nr][nc];
          if ((nr + nc) % 2 === parity) {
            visited.add(key);
            queue.push([nr, nc]);
          } else if (!nb.isRevealed && !nb.isFlagged && !nb.isMine) {
            // opposite-parity border: reveal but do not cascade
            visited.add(key);
            nb.isRevealed = true;
            revealed.push([nr, nc]);
          }
        }
      }
    }
    return revealed;
  }

  // Ripple reveal: expand in concentric BFS rings from the click and halt the
  // entire expansion the moment a ring contains any numbered cell.
  private ripple(r: number, c: number): [number, number][] {
    const cell = this.grid[r][c];
    if (cell.adjacentMines > 0) {
      cell.isRevealed = true;
      return [[r, c]];
    }
    const revealed: [number, number][] = [];
    const visited = new Set<string>([`${r},${c}`]);
    let ring: [number, number][] = [[r, c]];
    while (ring.length) {
      const ringRevealed: [number, number][] = [];
      for (const [cr, cc] of ring) {
        const cur = this.grid[cr][cc];
        if (cur.isMine || cur.isFlagged) continue;
        if (!cur.isRevealed) {
          cur.isRevealed = true;
          ringRevealed.push([cr, cc]);
        }
      }
      revealed.push(...ringRevealed);
      const ringHasNumbered = ringRevealed.some(
        ([cr, cc]) => this.grid[cr][cc].adjacentMines > 0
      );
      if (ringHasNumbered) break;
      const next: [number, number][] = [];
      for (const [cr, cc] of ringRevealed) {
        if (this.grid[cr][cc].adjacentMines !== 0) continue;
        for (const [nr, nc] of this.neighbors(cr, cc)) {
          const key = `${nr},${nc}`;
          if (visited.has(key)) continue;
          const nb = this.grid[nr][nc];
          if (!nb.isMine && !nb.isFlagged && !nb.isRevealed) {
            visited.add(key);
            next.push([nr, nc]);
          }
        }
      }
      ring = next;
    }
    return revealed;
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

  isAutoFlagged(r: number, c: number): boolean {
    return this.autoFlagged.has(`${r},${c}`);
  }

  reveal(r: number, c: number) {
    if (this.status === "won" || this.status === "lost") return;
    if (!this.minesPlaced) this.placeMines(r, c);
    const cell = this.grid[r][c];
    // A telegraphed cell is a known mine — treat it like a flag and ignore.
    if (cell.isRevealed || cell.isFlagged || this.isAutoFlagged(r, c)) return;
    this.moves++;

    if (cell.isMine) {
      cell.isRevealed = true;
      this.health -= 1;
      if (this.health <= 0) {
        this.status = "lost";
        return;
      }
    } else {
      this.applyReveal(r, c);
    }
    this.status = "active";
    this.runMineBehavior();
    this.checkEnd();
  }

  toggleFlag(r: number, c: number) {
    if (this.status === "won" || this.status === "lost") return;
    const cell = this.grid[r][c];
    if (cell.isRevealed || this.isAutoFlagged(r, c)) return;
    cell.isFlagged = !cell.isFlagged;
    this.moves++;
    if (this.minesPlaced) {
      this.status = "active";
      this.runMineBehavior();
      this.checkEnd();
    }
  }

  private runMineBehavior() {
    if (this.status !== "active") return;
    if (this.cfg.mineBehavior === "drifting") this.drift();
    else if (this.cfg.mineBehavior === "telegraphed") this.telegraph();
  }

  // TelegraphedMines: each turn a rotating fraction of hidden, un-flagged mines
  // reveal themselves as one-turn warnings. Last turn's warnings clear first,
  // so a different subset is telegraphed each move.
  private telegraph() {
    this.autoFlagged.clear();
    const mines: [number, number][] = [];
    for (let r = 0; r < this.cfg.rows; r++)
      for (let c = 0; c < this.cfg.cols; c++) {
        const cell = this.grid[r][c];
        if (cell.isMine && !cell.isRevealed && !cell.isFlagged)
          mines.push([r, c]);
      }
    if (mines.length === 0) return;
    const count = Math.min(
      mines.length,
      Math.max(1, Math.floor(mines.length * this.cfg.telegraphFraction))
    );
    // partial Fisher-Yates to pick `count` distinct mines
    for (let i = 0; i < count; i++) {
      const j = i + Math.floor(Math.random() * (mines.length - i));
      [mines[i], mines[j]] = [mines[j], mines[i]];
      const [r, c] = mines[i];
      this.autoFlagged.add(`${r},${c}`);
    }
  }

  // DriftingMines: each unflagged mine has driftProb chance of walking into an
  // adjacent unrevealed, unflagged, non-mine cell. Flagged mines stay pinned.
  private drift() {
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
