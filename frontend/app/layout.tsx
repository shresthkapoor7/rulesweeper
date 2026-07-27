import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RuleSweeper — Procedurally Generating Mechanics in Minesweeper",
  description:
    "A research project adapting the MORTAR pipeline to generate and adapt gameplay mechanics in Minesweeper. Read the paper and play two evolved mechanics.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
