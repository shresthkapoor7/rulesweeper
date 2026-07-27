import { ImageResponse } from "next/og";

// Large link-preview card (og:image / twitter summary_large_image).
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
export const alt = "RuleSweeper — Procedurally Generating Mechanics in Minesweeper";

export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          background: "#ffffff",
          padding: 80,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: 132,
            height: 132,
            borderRadius: 26,
            background: "#111111",
          }}
        >
          <svg width="88" height="88" viewBox="0 0 32 32">
            <g stroke="#ffffff" strokeWidth={2.4} strokeLinecap="round">
              <line x1="16" y1="5" x2="16" y2="27" />
              <line x1="5" y1="16" x2="27" y2="16" />
              <line x1="8.5" y1="8.5" x2="23.5" y2="23.5" />
              <line x1="23.5" y1="8.5" x2="8.5" y2="23.5" />
            </g>
            <circle cx="16" cy="16" r="6.5" fill="#ffffff" />
            <circle cx="13.6" cy="13.6" r="1.7" fill="#111111" />
          </svg>
        </div>
        <div
          style={{
            fontSize: 76,
            fontWeight: 700,
            color: "#111111",
            marginTop: 40,
          }}
        >
          RuleSweeper
        </div>
        <div
          style={{
            fontSize: 32,
            color: "#555555",
            marginTop: 18,
            textAlign: "center",
            maxWidth: 900,
          }}
        >
          Procedurally Generating Gameplay Mechanics in Minesweeper
        </div>
      </div>
    ),
    { ...size }
  );
}
