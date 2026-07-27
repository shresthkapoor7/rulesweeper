import { ImageResponse } from "next/og";

// PNG app icon (apple-touch-icon). Link-preview crawlers that ignore the SVG
// favicon pick this up as the square thumbnail.
export const size = { width: 180, height: 180 };
export const contentType = "image/png";

export default function AppleIcon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#111111",
        }}
      >
        <svg width="128" height="128" viewBox="0 0 32 32">
          <g
            stroke="#ffffff"
            strokeWidth={2.4}
            strokeLinecap="round"
          >
            <line x1="16" y1="5" x2="16" y2="27" />
            <line x1="5" y1="16" x2="27" y2="16" />
            <line x1="8.5" y1="8.5" x2="23.5" y2="23.5" />
            <line x1="23.5" y1="8.5" x2="8.5" y2="23.5" />
          </g>
          <circle cx="16" cy="16" r="6.5" fill="#ffffff" />
          <circle cx="13.6" cy="13.6" r="1.7" fill="#111111" />
        </svg>
      </div>
    ),
    { ...size }
  );
}
