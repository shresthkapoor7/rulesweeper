import React from "react";

// Minimal Python highlighter: bolds keywords and dims strings/comments so the
// structure is readable. Deliberately tiny — no dependency, no theme colors
// beyond the existing black/white palette.

const KEYWORDS = new Set([
  "class", "def", "return", "for", "in", "if", "elif", "else", "while",
  "break", "continue", "not", "and", "or", "is", "None", "True", "False",
  "import", "from", "as", "with", "lambda", "yield", "pass", "raise", "try",
  "except", "finally", "global", "nonlocal", "del", "assert", "self",
]);

type Seg = { text: string; kind: "code" | "string" | "comment" };

// Split source into code / string / comment runs so we never bold English
// words that happen to sit inside docstrings or comments.
function segment(src: string): Seg[] {
  const segs: Seg[] = [];
  const n = src.length;
  let i = 0;
  let buf = "";
  const flushCode = () => {
    if (buf) {
      segs.push({ text: buf, kind: "code" });
      buf = "";
    }
  };
  while (i < n) {
    const ch = src[i];
    if (ch === "#") {
      flushCode();
      let j = i;
      while (j < n && src[j] !== "\n") j++;
      segs.push({ text: src.slice(i, j), kind: "comment" });
      i = j;
      continue;
    }
    if (src.startsWith('"""', i) || src.startsWith("'''", i)) {
      flushCode();
      const q = src.slice(i, i + 3);
      let j = i + 3;
      while (j < n && !src.startsWith(q, j)) j++;
      j = Math.min(n, j + 3);
      segs.push({ text: src.slice(i, j), kind: "string" });
      i = j;
      continue;
    }
    if (ch === '"' || ch === "'") {
      flushCode();
      let j = i + 1;
      while (j < n && src[j] !== ch) {
        if (src[j] === "\\") j++;
        j++;
      }
      j = Math.min(n, j + 1);
      segs.push({ text: src.slice(i, j), kind: "string" });
      i = j;
      continue;
    }
    buf += ch;
    i++;
  }
  flushCode();
  return segs;
}

function renderCode(text: string, keyPrefix: string): React.ReactNode[] {
  // Split on word boundaries, bold the keywords.
  return text.split(/(\b\w+\b)/).map((tok, i) =>
    KEYWORDS.has(tok) ? (
      <strong key={`${keyPrefix}-${i}`}>{tok}</strong>
    ) : (
      <React.Fragment key={`${keyPrefix}-${i}`}>{tok}</React.Fragment>
    )
  );
}

export default function CodeBlock({ code }: { code: string }) {
  const segs = segment(code);
  return (
    <pre className="code">
      <code>
        {segs.map((seg, i) => {
          if (seg.kind === "comment")
            return (
              <span className="com" key={i}>
                {seg.text}
              </span>
            );
          if (seg.kind === "string")
            return (
              <span className="str" key={i}>
                {seg.text}
              </span>
            );
          return (
            <React.Fragment key={i}>{renderCode(seg.text, String(i))}</React.Fragment>
          );
        })}
      </code>
    </pre>
  );
}
