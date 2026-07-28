// Client-side ports of the three evaluation agents so you can watch them play
// each mechanic in the browser. Each agent exposes a `step` that returns the
// single next action (or null when it can make no move), so the UI can animate
// one move at a time.
//
// Faithfulness notes (mirrors the paper + CLAUDE.md fairness caveats):
//   * PAFG uses a HARDCODED 8-neighbour Moore adjacency for its constraints,
//     like the fixed symbolic solver. Correct on Moore games, wrong on the
//     5x5-radius game (it reads the wrong adjacency and flails).
//   * PAFG-LLM uses the game's CONFIGURED neighbourhood and reads the DISPLAYED
//     clue — so it handles radius-2 correctly and exploits the rank clue under
//     Ranked-Neighbourhood info, where PAFG can only guess.
//   * A cell the board is telegraphing (auto-flagged) is a known mine to both,
//     just like a flag.

import { Minesweeper } from "./minesweeper";

export type AgentKind = "random" | "pafg" | "pafg-llm";
export interface Action {
  type: "reveal" | "flag";
  r: number;
  c: number;
}

type NeighborFn = (g: Minesweeper, r: number, c: number) => [number, number][];

function inBounds(g: Minesweeper, r: number, c: number): boolean {
  return r >= 0 && r < g.cfg.rows && c >= 0 && c < g.cfg.cols;
}

// Fixed 8-neighbour Moore adjacency — what the paper's fixed PAFG assumes.
function mooreNeighbors(g: Minesweeper, r: number, c: number): [number, number][] {
  const out: [number, number][] = [];
  for (let dr = -1; dr <= 1; dr++)
    for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      if (inBounds(g, r + dr, c + dc)) out.push([r + dr, c + dc]);
    }
  return out;
}

// The game's actual configured neighbourhood — what PAFG-LLM uses.
function configuredNeighbors(g: Minesweeper, r: number, c: number): [number, number][] {
  return g.neighbors(r, c);
}

function hiddenCells(g: Minesweeper): [number, number][] {
  const out: [number, number][] = [];
  for (let r = 0; r < g.cfg.rows; r++)
    for (let c = 0; c < g.cfg.cols; c++) {
      const cell = g.grid[r][c];
      if (!cell.isRevealed && !cell.isFlagged && !g.isAutoFlagged(r, c))
        out.push([r, c]);
    }
  return out;
}

function anyRevealed(g: Minesweeper): boolean {
  for (const row of g.grid) for (const cell of row) if (cell.isRevealed) return true;
  return false;
}

function flaggedCount(g: Minesweeper): number {
  let n = 0;
  for (const row of g.grid) for (const cell of row) if (cell.isFlagged) n++;
  return n;
}

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
const key = (r: number, c: number) => `${r},${c}`;
const parse = (k: string): [number, number] => {
  const [r, c] = k.split(",").map(Number);
  return [r, c];
};

// ---- Random ----------------------------------------------------------------

function randomStep(g: Minesweeper): Action | null {
  const hidden = hiddenCells(g);
  if (!hidden.length) return null;
  const [r, c] = hidden[Math.floor(Math.random() * hidden.length)];
  return { type: "reveal", r, c };
}

// ---- Constraint solver (PAFG / PAFG-LLM shared core) ------------------------

interface Constraint {
  cells: string[]; // hidden unknown neighbours
  need: number; // remaining mines among them
}

function buildConstraints(g: Minesweeper, neigh: NeighborFn): Constraint[] {
  const cons: Constraint[] = [];
  for (let r = 0; r < g.cfg.rows; r++)
    for (let c = 0; c < g.cfg.cols; c++) {
      const cell = g.grid[r][c];
      if (!cell.isRevealed || cell.isMine) continue;
      const clue = cell.adjacentMines;
      let known = 0;
      const unknown: string[] = [];
      for (const [nr, nc] of neigh(g, r, c)) {
        const n = g.grid[nr][nc];
        if (n.isFlagged || g.isAutoFlagged(nr, nc)) known++;
        else if (!n.isRevealed) unknown.push(key(nr, nc));
      }
      if (unknown.length) cons.push({ cells: unknown, need: clue - known });
    }
  return cons;
}

function deduce(cons: Constraint[]): { safe: Set<string>; mines: Set<string> } {
  const safe = new Set<string>();
  const mines = new Set<string>();
  // Single-constraint rules.
  for (const con of cons) {
    if (con.need === 0) con.cells.forEach((k) => safe.add(k));
    else if (con.need === con.cells.length) con.cells.forEach((k) => mines.add(k));
  }
  // Subset rule: if A's cells ⊆ B's cells, the difference is constrained by
  // needB - needA.
  const sets = cons.map((c) => new Set(c.cells));
  for (let i = 0; i < cons.length; i++)
    for (let j = 0; j < cons.length; j++) {
      if (i === j) continue;
      const A = cons[i];
      const B = cons[j];
      if (A.cells.length >= B.cells.length) continue;
      if (!A.cells.every((k) => sets[j].has(k))) continue;
      const diff = B.cells.filter((k) => !sets[i].has(k));
      const nd = B.need - A.need;
      if (nd === 0) diff.forEach((k) => safe.add(k));
      else if (nd === diff.length) diff.forEach((k) => mines.add(k));
    }
  return { safe, mines };
}

// Lowest-probability guess when nothing is certain.
function guess(g: Minesweeper, cons: Constraint[]): Action | null {
  const hidden = hiddenCells(g);
  if (!hidden.length) return null;
  const prob = new Map<string, { sum: number; n: number }>();
  for (const con of cons) {
    const p = clamp(con.need / con.cells.length, 0, 1);
    for (const k of con.cells) {
      const e = prob.get(k) || { sum: 0, n: 0 };
      e.sum += p;
      e.n++;
      prob.set(k, e);
    }
  }
  const remaining = Math.max(0, g.cfg.mineCount - flaggedCount(g));
  const density = hidden.length ? remaining / hidden.length : 1;
  let best = hidden[0];
  let bestP = Infinity;
  for (const [r, c] of hidden) {
    const e = prob.get(key(r, c));
    const p = e ? e.sum / e.n : density;
    if (p < bestP) {
      bestP = p;
      best = [r, c];
    }
  }
  return { type: "reveal", r: best[0], c: best[1] };
}

// PAFG-LLM's clue-obfuscation hook: under Ranked info there are no numeric
// constraints, so it reads the DISPLAYED rank and reveals a hidden neighbour of
// the lowest-rank (relatively safest) revealed cell.
function rankHeuristicMove(g: Minesweeper): Action | null {
  let best: [number, number] | null = null;
  let bestRank = Infinity;
  for (let r = 0; r < g.cfg.rows; r++)
    for (let c = 0; c < g.cfg.cols; c++) {
      const cell = g.grid[r][c];
      if (!cell.isRevealed || cell.isMine) continue;
      const v = parseInt(g.infoAt(r, c), 10);
      if (Number.isNaN(v)) continue;
      const hids = g
        .neighbors(r, c)
        .filter(
          ([nr, nc]) =>
            !g.grid[nr][nc].isRevealed &&
            !g.grid[nr][nc].isFlagged &&
            !g.isAutoFlagged(nr, nc)
        );
      if (!hids.length) continue;
      if (v < bestRank) {
        bestRank = v;
        best = hids[0];
      }
    }
  return best ? { type: "reveal", r: best[0], c: best[1] } : null;
}

function solverStep(
  g: Minesweeper,
  opts: { neigh: NeighborFn; readClue: boolean; rankHeuristic: boolean }
): Action | null {
  if (hiddenCells(g).length === 0) return null;
  // Opening move: reveal the centre for a good first cascade.
  if (!anyRevealed(g))
    return {
      type: "reveal",
      r: Math.floor(g.cfg.rows / 2),
      c: Math.floor(g.cfg.cols / 2),
    };

  if (opts.readClue) {
    const cons = buildConstraints(g, opts.neigh);
    const { safe, mines } = deduce(cons);
    for (const k of safe) {
      const [r, c] = parse(k);
      const cell = g.grid[r][c];
      if (!cell.isRevealed && !cell.isFlagged && !g.isAutoFlagged(r, c))
        return { type: "reveal", r, c };
    }
    for (const k of mines) {
      const [r, c] = parse(k);
      if (!g.grid[r][c].isFlagged) return { type: "flag", r, c };
    }
    return guess(g, cons);
  }

  if (opts.rankHeuristic) {
    const move = rankHeuristicMove(g);
    if (move) return move;
  }
  return guess(g, []);
}

// ---- Public dispatch -------------------------------------------------------

export function agentStep(g: Minesweeper, kind: AgentKind): Action | null {
  if (kind === "random") return randomStep(g);
  const countBased = g.cfg.info === "count-mines";
  if (kind === "pafg")
    return solverStep(g, {
      neigh: mooreNeighbors,
      readClue: countBased,
      rankHeuristic: false,
    });
  // pafg-llm
  return solverStep(g, {
    neigh: configuredNeighbors,
    readClue: countBased,
    rankHeuristic: !countBased,
  });
}

export const AGENT_LABELS: Record<AgentKind, string> = {
  random: "Random",
  pafg: "PAFG",
  "pafg-llm": "PAFG-LLM",
};
