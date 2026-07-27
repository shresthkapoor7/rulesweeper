import type { GameConfig } from "./minesweeper";

export interface GameInfo {
  slug: string;
  title: string;
  lift: string;
  tagline: string;
  // Longer description of the mechanic + how to read the clues.
  description: string;
  howToRead: string[];
  config: GameConfig;
}

export const GAMES: GameInfo[] = [
  {
    slug: "ranked-neighborhood",
    title: "Ranked Neighborhood Info",
    lift: "+0.65",
    tagline: "Clues show comparative rank, not absolute mine counts.",
    description:
      "A code-level mutation of the clue encoding. Instead of printing the number of adjacent mines, each revealed cell shows the ordinal rank of its mine count relative to its already-revealed neighbors — 1 means this cell borders the fewest mines among the cells around it. This shifts play from absolute deduction to comparative reasoning: you can tell which direction is relatively safer, but never the exact count. Random play cannot exploit the relational ordering, which is what makes the mechanic separate skilled from unskilled solvers.",
    howToRead: [
      "A cell showing 1 borders the fewest mines among its revealed neighbors — relatively safe.",
      "Higher numbers mean relatively more mines nearby than the surrounding revealed cells.",
      "A freshly revealed cell with no revealed neighbors falls back to its true mine count.",
      "Everything else (8-cell Moore adjacency, cascade, single life) is canonical Minesweeper.",
    ],
    config: {
      rows: 16,
      cols: 16,
      mineCount: 40,
      startingHealth: 1,
      neighborhood: "moore",
      mineBehavior: "static",
      info: "ranked",
      driftProb: 0,
      safeFirstClick: true,
    },
  },
  {
    slug: "radius-drift",
    title: "5×5 Radius + Drifting Mines",
    lift: "+0.77",
    tagline: "Clues count mines over a 5×5 region while mines wander each turn.",
    description:
      "The highest-lift mechanic in the archive — it best illustrates how far a generated variant can drift from canonical Minesweeper while staying playable. Two mechanics compose: adjacency is widened to a 5×5 (radius-2) neighborhood so every clue counts mines over 24 surrounding cells, and mines drift — each turn every unflagged mine has a 30% chance of wandering into an adjacent empty cell. Flagging a mine pins it in place, which keeps flagging meaningful as a deduction tool. You start with 3 health because the moving hazards make a single life unforgiving.",
    howToRead: [
      "Numbers count mines within the whole 5×5 box around a cell — expect large values.",
      "After every move, unflagged mines may drift one step, so old clues update in place.",
      "Flag a mine to pin it; pinned mines stop drifting and stabilize nearby clues.",
      "You have 3 lives — hitting a mine costs one but the game continues.",
    ],
    config: {
      rows: 16,
      cols: 16,
      mineCount: 40,
      startingHealth: 3,
      neighborhood: "radius-2-moore",
      mineBehavior: "drifting",
      info: "count-mines",
      driftProb: 0.3,
      safeFirstClick: true,
    },
  },
];

export function getGame(slug: string): GameInfo | undefined {
  return GAMES.find((g) => g.slug === slug);
}
