import React from "react";

// Box helper for the pipeline diagram. Uses currentColor so it adapts to
// light/dark theme automatically.
function Box({
  x,
  y,
  w,
  h,
  lines,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  lines: string[];
}) {
  const cx = x + w / 2;
  const cy = y + h / 2;
  const lh = 15;
  const startY = cy - ((lines.length - 1) * lh) / 2;
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={8}
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
      />
      <text
        x={cx}
        textAnchor="middle"
        fontSize={12.5}
        fill="currentColor"
        fontFamily="Times New Roman, serif"
      >
        {lines.map((l, i) => (
          <tspan
            key={i}
            x={cx}
            y={startY + i * lh}
            fontWeight={i === 0 ? 600 : 400}
            fontSize={i === 0 ? 12.5 : 11}
          >
            {l}
          </tspan>
        ))}
      </text>
    </g>
  );
}

export function PipelineFigure() {
  return (
    <div className="figure">
      <svg viewBox="0 0 940 340" role="img" aria-label="Mechanic-evolution pipeline diagram">
        <defs>
          <marker
            id="arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="7"
            markerHeight="7"
            orient="auto-start-reverse"
          >
            <path d="M0,0 L10,5 L0,10 z" fill="currentColor" />
          </marker>
        </defs>

        <g
          fill="none"
          stroke="currentColor"
          strokeWidth={1.5}
          markerEnd="url(#arrow)"
        >
          {/* row 1 */}
          <line x1="160" y1="170" x2="183" y2="170" />
          <line x1="315" y1="170" x2="330" y2="170" />
          {/* LLM Mutation -> two modes -> instantiate */}
          <path d="M470,158 L520,95" />
          <path d="M470,182 L520,245" />
          <path d="M672,95 L720,158" />
          <path d="M672,245 L720,182" />
          {/* instantiate down to eval */}
          <line x1="810" y1="198" x2="810" y2="270" />
          {/* eval -> fitness */}
          <line x1="608" y1="298" x2="532" y2="298" />
          {/* fitness -> archive (L path back up) */}
          <path d="M300,298 L90,298 L90,200" />
        </g>

        <Box x={20} y={142} w={140} h={56} lines={["Mechanic", "Archive"]} />
        <Box x={183} y={142} w={132} h={56} lines={["Select Parent", "mechanic / config"]} />
        <Box x={330} y={142} w={140} h={56} lines={["LLM Mutation"]} />
        <Box x={720} y={142} w={180} h={56} lines={["Instantiate", "Minesweeper variant"]} />

        <Box x={520} y={72} w={152} h={40} lines={["parameter changes"]} />
        <Box x={520} y={228} w={152} h={40} lines={["new mechanic code"]} />

        <Box x={610} y={272} w={290} h={52} lines={["Agent Evaluation", "Random · PAFG · PAFG-LLM"]} />
        <Box x={300} y={272} w={232} h={52} lines={["Fitness / Selection", "compare agent performance"]} />
      </svg>
      <div className="figcaption">
        Fig. 1 — Mechanic-evolution pipeline for Minesweeper-MORTAR.
      </div>
    </div>
  );
}

export function SkillSpreadFormula() {
  return (
    <div className="formula">
      <div className="eq">
        <span>
          P<span className="sub">a</span>
        </span>
        <span>=</span>
        <span className="frac">
          <span className="num">average safe cells revealed by a</span>
          <span className="den">S</span>
        </span>
      </div>
      <div className="eq">
        <span>skill_spread</span>
        <span>=</span>
        <span className="maxop">
          <span className="lab">max</span>
          <span className="cond">a ∈ A \ {"{random}"}</span>
        </span>
        <span>
          P<span className="sub">a</span> − P<span className="sub">random</span>
        </span>
      </div>
    </div>
  );
}
