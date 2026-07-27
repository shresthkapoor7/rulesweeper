import Link from "next/link";
import { GAMES } from "@/lib/games";
import { PipelineFigure, SkillSpreadFormula } from "@/components/Figures";

export default function Home() {
  return (
    <main className="container">
      <div className="center">
        <h1>
          RuleSweeper: Procedurally Generating
          <br />
          Gameplay Mechanics in Minesweeper
        </h1>
        <div className="authors">
          Ryan Fleishman · Teddy Clark · Shresth Kapoor · Jan Borowski · Tim
          Merino · Julian Togelius
        </div>
        <div className="affil">New York University · IEEE CoG 2026</div>

        <div className="linkrow">
          <a className="btn" href="#play">
            Play the mechanics
          </a>
          <a className="btn" href="#method">
            Read the summary
          </a>
        </div>
      </div>

      <section className="abstract">
        <p>
          <span className="label">Abstract — </span>
          Procedural content generation in games has traditionally focused on
          producing new levels within a fixed set of game mechanics, but there
          is much less work around generating the mechanics themselves. Mechanic
          generation introduces a central challenge absent from standard PCG:
          the evaluation environment becomes dynamic, so fixed game-playing
          agents may fail as the rules evolve. We present an adaptation of the
          MORTAR pipeline that addresses this problem by both generating
          mechanics and adapting a solver agent for the puzzle game Minesweeper.
          Our system mutates mechanic components within a structured game
          configuration using a large language model, evaluates each candidate
          through play with both fixed and LLM-adapted agents, and lets the
          search process continue into mechanic spaces that standard evaluators
          cannot reliably handle. The resulting pipeline produces playable,
          mechanically distinct puzzle variants, while the LLM-boosted solver
          achieves consistent gains over a fixed symbolic baseline across the
          generated mechanic archive.
        </p>
      </section>

      <h2 id="method">What the paper does</h2>
      <p>
        Most procedural generation makes new <em>levels</em> for a fixed rule
        set. RuleSweeper instead generates new <em>rules</em>. We adapt MORTAR
        to a fixed Minesweeper engine whose every mechanic — board size, mine
        count, health, clue encoding, adjacency (neighborhood), reveal behavior,
        mine behavior, and win condition — lives behind a single structured{" "}
        <code>GameConfig</code> object. An LLM mutates that configuration, and
        for deeper changes it authors brand-new Python subclasses for one of
        five mechanic families (mine behavior, reveal strategy, clue/info
        strategy, neighborhood, and win condition).
      </p>

      <PipelineFigure />
      <p className="small muted center" style={{ marginTop: -4 }}>
        Each candidate is instantiated, played, scored, and — if it separates
        skilled from random play — folded back into the archive to seed future
        mutations.
      </p>

      <h3>The core challenge: a moving evaluation target</h3>
      <p>
        When you mutate mechanics, the environment used to judge a game keeps
        changing. A solver tuned to standard Minesweeper stops being a reliable
        yardstick once clue meaning, adjacency, or the win condition drift away
        from canonical. Perfect-information search like MCTS breaks down because
        Minesweeper is only partially observable and the forward model keeps
        shifting.
      </p>

      <h3>The method: a solver in the loop</h3>
      <p>
        Each candidate mechanic is scored by a panel of agents playing many
        seeded games:
      </p>
      <ul className="tight">
        <li>
          <b>Random</b> — reveals random cells, never flags; the unskilled
          floor.
        </li>
        <li>
          <b>PAFG</b> — a symbolic Minesweeper solver (First, Primary, Advanced,
          Guess) that reaches expert-level win rates on the base game.
        </li>
        <li>
          <b>pafg-llm</b> — an LLM writes a small subclass of PAFG tailored to
          the specific mechanic being evaluated, adapting opening move, adjacency
          handling, clue interpretation, and guessing.
        </li>
      </ul>
      <p>
        A mechanic is interesting when a skilled agent extracts substantially
        more progress than random play. We measure each agent&apos;s progress
        fraction P<sub>a</sub> — the average safe cells it reveals out of the S
        = r·c − m safe cells on the board — and define the selection signal,{" "}
        <b>skill spread</b>, as the best skilled agent&apos;s progress minus the
        random agent&apos;s:
      </p>

      <SkillSpreadFormula />

      <p>
        Candidates are admitted to a persistent archive only when skill spread ≥
        0.10, and future mutations are sampled from that archive with a
        MAP-Elites-style procedure.
      </p>

      <h2>Results</h2>
      <p>
        Over a 100-iteration run on Claude Sonnet, 51 mechanics were admitted to
        the archive. The LLM-adapted solver (pafg-llm) beat the fixed symbolic
        PAFG baseline on higher progress for 36 of 51 mechanics (70.6%) and on
        win rate for 40 of 51 (78.4%), with a mean win-rate gain of +0.41. The
        largest improvements came exactly where a fixed solver&apos;s
        assumptions break — mechanics that obfuscate clue meaning or change
        adjacency.
      </p>
      <table className="results">
        <thead>
          <tr>
            <th>Agent</th>
            <th>Wins</th>
            <th>Progress</th>
            <th>Turns</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Random</td>
            <td>0</td>
            <td>32.6%</td>
            <td>5.8</td>
          </tr>
          <tr>
            <td>PAFG</td>
            <td>80</td>
            <td>91.9%</td>
            <td>104.4</td>
          </tr>
          <tr>
            <td>PAFG-LLM</td>
            <td>86</td>
            <td>97.7%</td>
            <td>99.1</td>
          </tr>
        </tbody>
      </table>
      <p className="small muted">
        Table I from the paper — agent performance on canonical 16×16
        Minesweeper, averaged over 100 games on the same seed.
      </p>

      <h2 id="play">Play two evolved mechanics</h2>
      <p>
        These are two of the highest-lift mechanics the pipeline produced — the
        same examples highlighted in the paper. Both run entirely in your
        browser.
      </p>
      <div className="cards">
        {GAMES.map((g) => (
          <Link key={g.slug} href={`/play/${g.slug}`} className="card">
            <span className="lift">LLM lift {g.lift}</span>
            <h3>{g.title}</h3>
            <p>{g.tagline}</p>
          </Link>
        ))}
      </div>

      <hr />
      <p className="small muted center">
        Built on the{" "}
        <a href="https://github.com/shresthkapoor7/rulesweeper">rulesweeper</a>{" "}
        research engine. Every mechanic here is a configuration of that engine.
      </p>
    </main>
  );
}
