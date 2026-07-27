import type { Metadata } from "next";
import "./globals.css";
import ThemeToggle from "@/components/ThemeToggle";

const DESCRIPTION =
  "A research project adapting the MORTAR pipeline to generate and adapt gameplay mechanics in Minesweeper. Read the paper and play evolved mechanics.";

export const metadata: Metadata = {
  metadataBase: new URL("https://rulesweeper.vercel.app"),
  title: "rulesweeper",
  description: DESCRIPTION,
  openGraph: {
    title: "RuleSweeper",
    description: DESCRIPTION,
    url: "https://rulesweeper.vercel.app",
    siteName: "RuleSweeper",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "RuleSweeper",
    description: DESCRIPTION,
  },
};

// Set the theme before first paint to avoid a flash of the wrong theme.
const noFlashScript = `
(function () {
  try {
    var t = localStorage.getItem('theme');
    if (!t) {
      t = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    document.documentElement.dataset.theme = t;
  } catch (e) {}
})();
`;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: noFlashScript }} />
      </head>
      <body>
        <ThemeToggle />
        {children}
      </body>
    </html>
  );
}
