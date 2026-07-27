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
    lift: "Skill-spread 0.56",
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
      reveal: "cascade",
      driftProb: 0,
      telegraphFraction: 0,
      safeFirstClick: true,
    },
  },
  {
    slug: "radius-drift",
    title: "5×5 Radius + Drifting Mines",
    lift: "Skill-spread 0.71",
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
      reveal: "cascade",
      driftProb: 0.3,
      telegraphFraction: 0,
      safeFirstClick: true,
    },
  },
  {
    slug: "telegraphed-mines",
    title: "Telegraphed Mines",
    lift: "Skill-spread 0.72",
    tagline: "A rotating subset of mines flashes a warning each turn.",
    description:
      "A generated mine behavior with one of the highest skill-spreads in the archive. After every move, roughly 20% of the still-hidden mines telegraph themselves — they flash a warning marker for a single turn. Next turn those clear and a different subset lights up, so the hints rotate around the board. A skilled player treats each telegraph as a confirmed mine and clears the guaranteed-safe cells around it; random play can't turn the flickering intel into progress. The warned cells can't be clicked (they're known mines), and everything else is canonical 16×16 Minesweeper.",
    howToRead: [
      "A ⚠ marker is a mine the board is telegraphing for this turn only.",
      "Use the warnings to deduce which neighboring cells must be safe, then reveal them.",
      "The telegraphed set rotates every move — don't expect the same mines flagged twice.",
      "You still place your own flags with right-click; telegraphs are separate and automatic.",
    ],
    config: {
      rows: 16,
      cols: 16,
      mineCount: 40,
      startingHealth: 1,
      neighborhood: "moore",
      mineBehavior: "telegraphed",
      info: "count-mines",
      reveal: "cascade",
      driftProb: 0,
      telegraphFraction: 0.2,
      safeFirstClick: true,
    },
  },
  {
    slug: "checkerboard-reveal",
    title: "Checkerboard Reveal",
    lift: "Skill-spread 0.72",
    tagline: "Cascades skip every other cell, leaving a hidden lattice.",
    description:
      "A generated reveal strategy. The flood-fill from an empty cell only propagates through cells that share the clicked cell's (row + column) parity, so a cascade opens a checkerboard: every other cell in the cleared region stays hidden. Opposite-parity cells at the border are revealed as numbered hints but never used to spread the flood. Skilled players read the exposed checkerboard layer to deduce the interleaved hidden cells; random play just wastes clicks. Standard 8-neighbor adjacency, single life.",
    howToRead: [
      "Opening an empty area reveals a checkerboard, not a solid blob.",
      "The revealed numbers still count all 8 neighbors — including the hidden lattice cells.",
      "Use two or more revealed cells to triangulate whether a hidden cell is a mine.",
      "You'll click far more often than in vanilla Minesweeper — that's the mechanic.",
    ],
    config: {
      rows: 16,
      cols: 16,
      mineCount: 40,
      startingHealth: 1,
      neighborhood: "moore",
      mineBehavior: "static",
      info: "count-mines",
      reveal: "checkerboard",
      driftProb: 0,
      telegraphFraction: 0,
      safeFirstClick: true,
    },
  },
  {
    slug: "ripple-reveal",
    title: "Ripple Reveal",
    lift: "Skill-spread 0.61",
    tagline: "Reveals expand in rings that halt at the first numbers.",
    description:
      "A generated reveal strategy that opens the board in concentric rings out from the click. Each ring is revealed all at once, but the moment a ring contains any numbered (mine-adjacent) cell, the whole ripple stops there. Instead of a single large flood you get a distance-layered clue structure — a clean expanding front bounded by the nearest numbers — that a skilled solver can read ring by ring, while random play gains little from the layering. Standard 8-neighbor adjacency, single life.",
    howToRead: [
      "A click on an empty cell ripples outward and stops as soon as numbers appear.",
      "The outermost revealed ring is your live constraint frontier — read it first.",
      "Because reveals are bounded, you'll open new fronts by clicking safe edge cells.",
      "Clue numbers are the ordinary count of the 8 surrounding mines.",
    ],
    config: {
      rows: 16,
      cols: 16,
      mineCount: 40,
      startingHealth: 1,
      neighborhood: "moore",
      mineBehavior: "static",
      info: "count-mines",
      reveal: "ripple",
      driftProb: 0,
      telegraphFraction: 0,
      safeFirstClick: true,
    },
  },
];

export function getGame(slug: string): GameInfo | undefined {
  return GAMES.find((g) => g.slug === slug);
}
