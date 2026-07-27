import Link from "next/link";
import { notFound } from "next/navigation";
import { GAMES, getGame } from "@/lib/games";
import GameBoard from "@/components/GameBoard";

export function generateStaticParams() {
  return GAMES.map((g) => ({ slug: g.slug }));
}

export default async function PlayPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const game = getGame(slug);
  if (!game) notFound();

  return (
    <main className="container">
      <a href="/#play" className="backlink">
        Back to the paper
      </a>

      <div className="center">
        <h1 style={{ fontSize: 28 }}>{game.title}</h1>
        <p className="muted" style={{ marginTop: -8 }}>
          {game.lift} · {game.tagline}
        </p>
      </div>

      <GameBoard config={game.config} />

      <h2>The mechanic</h2>
      <p style={{ textAlign: "left" }}>{game.description}</p>

      <h3>How to read the board</h3>
      <ul className="tight">
        {game.howToRead.map((line, i) => (
          <li key={i}>{line}</li>
        ))}
      </ul>

      <hr />
      <h3 className="center">More mechanics to play</h3>
      <div className="more-grid">
        {GAMES.filter((g) => g.slug !== game.slug).map((g) => (
          <Link key={g.slug} href={`/play/${g.slug}`} className="more-card">
            <span className="more-title">{g.title}</span>
            <span className="more-tag">{g.tagline}</span>
          </Link>
        ))}
      </div>
    </main>
  );
}
