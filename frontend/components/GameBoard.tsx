"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Minesweeper, type GameConfig } from "@/lib/minesweeper";
import {
  playReveal,
  playFlag,
  playHit,
  playWin,
  playLose,
  isSoundEnabled,
  setSoundEnabled,
  initSoundPref,
} from "@/lib/sound";

export default function GameBoard({ config }: { config: GameConfig }) {
  const [game, setGame] = useState(() => new Minesweeper(config));
  // bump forces a re-render after mutating the game object in place.
  const [, setTick] = useState(0);
  const rerender = useCallback(() => setTick((t) => t + 1), []);
  const [soundOn, setSoundOn] = useState(true);

  useEffect(() => {
    initSoundPref();
    setSoundOn(isSoundEnabled());
  }, []);

  const toggleSound = useCallback(() => {
    const next = !isSoundEnabled();
    setSoundEnabled(next);
    setSoundOn(next);
  }, []);

  // Pick the right sound from what changed between two game snapshots.
  const soundForTransition = useCallback(
    (prevStatus: string, prevHealth: number, action: "reveal" | "flag") => {
      if (game.status === "lost") return playLose();
      if (game.status === "won") return playWin();
      if (game.health < prevHealth) return playHit();
      if (action === "flag") return playFlag();
      return playReveal();
    },
    [game]
  );

  const reset = useCallback(() => {
    setGame(new Minesweeper(config));
    rerender();
  }, [config, rerender]);

  const onReveal = useCallback(
    (r: number, c: number) => {
      const cell = game.grid[r][c];
      if (cell.isRevealed || cell.isFlagged) return;
      const ps = game.status;
      const ph = game.health;
      game.reveal(r, c);
      soundForTransition(ps, ph, "reveal");
      rerender();
    },
    [game, rerender, soundForTransition]
  );

  const onFlag = useCallback(
    (e: React.MouseEvent, r: number, c: number) => {
      e.preventDefault();
      if (game.grid[r][c].isRevealed) return;
      const ps = game.status;
      const ph = game.health;
      game.toggleFlag(r, c);
      soundForTransition(ps, ph, "flag");
      rerender();
    },
    [game, rerender, soundForTransition]
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
        <button
          className="btn"
          onClick={toggleSound}
          style={{ padding: "4px 12px" }}
          aria-label={soundOn ? "Mute sound" : "Unmute sound"}
          title={soundOn ? "Mute sound" : "Unmute sound"}
        >
          {soundOn ? "♪ On" : "♪ Off"}
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
