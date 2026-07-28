"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Minesweeper, type GameConfig } from "@/lib/minesweeper";
import { agentStep, AGENT_LABELS, type AgentKind } from "@/lib/agents";
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

const AGENT_ORDER: AgentKind[] = ["random", "pafg", "pafg-llm"];

export default function GameBoard({ config }: { config: GameConfig }) {
  const [game, setGame] = useState(() => new Minesweeper(config));
  // bump forces a re-render after mutating the game object in place.
  const [, setTick] = useState(0);
  const rerender = useCallback(() => setTick((t) => t + 1), []);
  const [soundOn, setSoundOn] = useState(true);

  // Agent autoplay.
  const [agent, setAgent] = useState<AgentKind | null>(null);
  const runningRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [agentNote, setAgentNote] = useState("");

  useEffect(() => {
    initSoundPref();
    setSoundOn(isSoundEnabled());
  }, []);

  const stopAgent = useCallback(() => {
    runningRef.current = false;
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    setAgent(null);
  }, []);

  // Clean up the interval if the component unmounts (e.g. navigating away).
  useEffect(() => () => stopAgent(), [stopAgent]);

  const startAgent = useCallback(
    (kind: AgentKind) => {
      stopAgent();
      const g = new Minesweeper(config);
      setGame(g);
      setAgentNote("");
      setAgent(kind);
      runningRef.current = true;
      timerRef.current = setInterval(() => {
        if (!runningRef.current) return;
        const action = agentStep(g, kind);
        if (!action) {
          setAgentNote(
            `${AGENT_LABELS[kind]} stopped — no move it could prove.`
          );
          stopAgent();
          return;
        }
        if (action.type === "reveal") g.reveal(action.r, action.c);
        else g.toggleFlag(action.r, action.c);
        rerender();
        if (g.status === "won") {
          playWin();
          stopAgent();
        } else if (g.status === "lost") {
          playLose();
          stopAgent();
        }
      }, 170);
    },
    [config, rerender, stopAgent]
  );

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
    stopAgent();
    setAgentNote("");
    setGame(new Minesweeper(config));
    rerender();
  }, [config, rerender, stopAgent]);

  const onReveal = useCallback(
    (r: number, c: number) => {
      if (runningRef.current) return;
      const cell = game.grid[r][c];
      if (cell.isRevealed || cell.isFlagged || game.isAutoFlagged(r, c)) return;
      const ps = game.status;
      const ph = game.health;
      game.reveal(r, c);
      soundForTransition(ps, ph, "reveal");
      rerender();
    },
    [game, rerender, soundForTransition]
  );

  const flagCell = useCallback(
    (r: number, c: number) => {
      if (runningRef.current) return;
      if (game.grid[r][c].isRevealed || game.isAutoFlagged(r, c)) return;
      const ps = game.status;
      const ph = game.health;
      game.toggleFlag(r, c);
      soundForTransition(ps, ph, "flag");
      rerender();
    },
    [game, rerender, soundForTransition]
  );

  const onContextMenu = useCallback(
    (e: React.MouseEvent, r: number, c: number) => {
      e.preventDefault();
      flagCell(r, c);
    },
    [flagCell]
  );

  // Long-press to flag on touch devices (mobile has no right-click, and the
  // contextmenu event isn't reliably dispatched on long-press).
  const touch = useRef({ timer: 0 as unknown as ReturnType<typeof setTimeout>, fired: false, x: 0, y: 0 });

  const onTouchStart = useCallback(
    (e: React.TouchEvent, r: number, c: number) => {
      const t = e.touches[0];
      touch.current.fired = false;
      touch.current.x = t.clientX;
      touch.current.y = t.clientY;
      clearTimeout(touch.current.timer);
      touch.current.timer = setTimeout(() => {
        touch.current.fired = true;
        flagCell(r, c);
        if (typeof navigator !== "undefined" && navigator.vibrate) {
          navigator.vibrate(15);
        }
      }, 400);
    },
    [flagCell]
  );

  const onTouchMove = useCallback((e: React.TouchEvent) => {
    const t = e.touches[0];
    if (
      Math.abs(t.clientX - touch.current.x) > 10 ||
      Math.abs(t.clientY - touch.current.y) > 10
    ) {
      clearTimeout(touch.current.timer);
    }
  }, []);

  const onTouchEnd = useCallback((e: React.TouchEvent) => {
    clearTimeout(touch.current.timer);
    // If the long-press already flagged, swallow the tap so it doesn't reveal.
    if (touch.current.fired) e.preventDefault();
  }, []);

  const onCellClick = useCallback(
    (r: number, c: number) => {
      // A long-press that fired leaves this flag set; consume it and skip reveal.
      if (touch.current.fired) {
        touch.current.fired = false;
        return;
      }
      onReveal(r, c);
    },
    [onReveal]
  );

  const statusText = useMemo(() => {
    if (agent) return `${AGENT_LABELS[agent]} is playing…`;
    if (game.status === "won") return "Cleared. ✓";
    if (game.status === "lost") return "Boom — out of lives. ✕";
    if (agentNote) return agentNote;
    return "";
  }, [agent, agentNote, game.status]);

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

      <div className="hud agents">
        <span className="muted small">Watch an agent play:</span>
        {AGENT_ORDER.map((k) => (
          <button
            key={k}
            className={`btn${agent === k ? " active" : ""}`}
            onClick={() => startAgent(k)}
            style={{ padding: "4px 12px" }}
          >
            {AGENT_LABELS[k]}
          </button>
        ))}
        {agent && (
          <button
            className="btn"
            onClick={stopAgent}
            style={{ padding: "4px 12px" }}
          >
            Stop
          </button>
        )}
      </div>

      <div className="status-line">{statusText}</div>

      <div
        className={`board${agent ? " locked" : ""}`}
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
            } else if (game.isAutoFlagged(r, c)) {
              cls += " telegraph";
              content = "⚠";
            } else if (over && cell.isMine) {
              // reveal remaining mines when the game ends
              cls += " mine";
              content = "✳";
            }
            return (
              <button
                key={`${r}-${c}`}
                className={cls}
                onClick={() => !over && onCellClick(r, c)}
                onContextMenu={(e) =>
                  over ? e.preventDefault() : onContextMenu(e, r, c)
                }
                onTouchStart={(e) => !over && onTouchStart(e, r, c)}
                onTouchMove={onTouchMove}
                onTouchEnd={onTouchEnd}
                onTouchCancel={() => clearTimeout(touch.current.timer)}
                disabled={over || cell.isRevealed}
              >
                {content}
              </button>
            );
          })
        )}
      </div>

      <p className="hint">
        Left-click reveals · right-click flags (long-press on mobile). Or hand
        it to an agent above and watch it play — PAFG is the paper&apos;s fixed
        symbolic solver, PAFG-LLM is the mechanic-adapted one.
      </p>
    </div>
  );
}
