# RuleSweeper — research site

A minimal, static research website for the paper **"RuleSweeper: Procedurally
Generating Gameplay Mechanics in Minesweeper"** (Fleishman, Clark, Kapoor,
Borowski, Merino, Togelius — NYU).

It has two parts:

1. **The paper summary** — a black-and-white, centered write-up of what the
   MORTAR-adapted pipeline does, the moving-evaluation-target challenge, the
   agent panel (Random / PAFG / pafg-llm), and the results.
2. **Two playable mechanics** — two of the highest-lift evolved mechanics from
   the archive, ported faithfully to run entirely in the browser:
   - **Ranked Neighborhood Info** (LLM lift +0.65) — clues show comparative
     rank instead of absolute mine counts.
   - **5×5 Radius + Drifting Mines** (LLM lift +0.77) — clues count mines over
     a 5×5 region while unflagged mines wander each turn.

No backend. Next.js App Router, statically prerendered.

## Run

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000
```

```bash
npm run build && npm run start   # production
```

## Layout

| Path | Purpose |
|---|---|
| `app/page.tsx` | Home — paper summary + links to the games |
| `app/play/[slug]/page.tsx` | Per-game page (mechanic description + board) |
| `components/GameBoard.tsx` | Client-side interactive board |
| `lib/minesweeper.ts` | TS port of the game engine (neighborhoods, cascade, drifting mines, ranked/count clue encodings) |
| `lib/games.ts` | The two mechanic presets + copy |

The engine mirrors the Python research build (`board.py`,
`reveal_strategies.py`, `mine_behaviors.py`, `info_strategies.py`,
`neighborhoods.py`) so the browser game plays the same as the paper's engine.
