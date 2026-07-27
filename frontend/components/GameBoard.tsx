"use client";

import { useCallback, useMemo, useState } from "react";
import { Minesweeper, type GameConfig } from "@/lib/minesweeper";

export default function GameBoard({ config }: { config: GameConfig }) {
  const [game, setGame] = useState(() => new Minesweeper(config));
  // bump forces a re-render after mutating the game object in place.
  const [, setTick] = useState(0);
  const rerender = useCallback(() => setTick((t) => t + 1), []);

  const reset = useCallback(() => {
    setGame(new Minesweeper(config));
    rerender();
  }, [config, rerender]);

  const onReveal = useCallback(
    (r: number, c: number) => {
      game.reveal(r, c);
      rerender();
    },
    [game, rerender]
  );

  const onFlag = useCallback(
    (e: React.MouseEvent, r: number, c: number) => {
      e.preventDefault();
      game.toggleFlag(r, c);
      rerender();
    },
    [game, rerender]
  );

  const statusText = useMemo(() => {
    if (game.status === "won") return "You cleared it. ✓";
    if (game.status === "lost") return "Boom — out of lives. ✕";
    return "";
  }, [game.status]);

  return (
    <div className="board-wrap">
      <div className="hud">
        <span>
          Mines&nbsp;left <b>{game.minesRemaining()}</b>
        </span>
        {config.startingHealth > 1 && (
          <span>
            Lives <b>{Math.max(0, game.health)}</b>
          </span>
        )}
        <span>
          Moves <b>{game.moves}</b>
        </span>
        <button className="btn" onClick={reset} style={{ padding: "4px 14px" }}>
          New game
        </button>
      </div>

      <div className="status-line">{statusText}</div>

      <div
        className="board"
        style={{
          gridTemplateColumns: `repeat(${config.cols}, 26px)`,
          gridTemplateRows: `repeat(${config.rows}, 26px)`,
        }}
      >
        {game.grid.map((row, r) =>
          row.map((cell, c) => {
            const over = game.status === "won" || game.status === "lost";
            let content = "";
            let cls = "cell";
            if (cell.isRevealed) {
              cls += " revealed";
              if (cell.isMine) {
                cls += " mine";
                content = "✳";
              } else {
                content = game.infoAt(r, c);
              }
            } else if (cell.isFlagged) {
              content = "⚑";
            } else if (over && cell.isMine) {
              // reveal remaining mines when the game ends
              cls += " mine";
              content = "✳";
            }
            return (
              <button
                key={`${r}-${c}`}
                className={cls}
                onClick={() => !over && onReveal(r, c)}
                onContextMenu={(e) => (over ? e.preventDefault() : onFlag(e, r, c))}
                disabled={over || cell.isRevealed}
              >
                {content}
              </button>
            );
          })
        )}
      </div>

      <p className="hint">
        Left-click reveals · right-click flags (long-press on mobile). Flags pin
        a suspected mine.
      </p>
    </div>
  );
}
