"""
mortar.py — MORTAR mutation engine for Minesweeper.

Evolves GameConfig variants by asking an LLM to suggest parameter mutations,
evaluating them with RandomAgent, and maintaining a flat archive of accepted configs.

Usage:
    OPENROUTER_API_KEY=... python mortar.py [--iterations N] [--games N] [--archive PATH]

Requires:
    pip install openai
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import re
import time

from openai import OpenAI

from agents import evaluate_config
from agents.random_agent import RandomAgent


def _load_dotenv(path: str = ".env") -> None:
    """Load key=value pairs from a .env file into os.environ (no-op if missing)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()
from config import GameConfig
from mechanics_archive import MECHANICS


# ---------------------------------------------------------------------------
# Field constraints — valid ranges / choices for each GameConfig field.
# Used to validate LLM-proposed mutations before applying them.
# ---------------------------------------------------------------------------

FIELD_CONSTRAINTS: dict[str, tuple | list] = {
    "rows":             (5, 30),
    "cols":             (5, 30),
    "mine_count":       (1, None),       # upper bound computed from board size
    "starting_health":  (1, 10),
    "mine_damage":      (1, 5),
    "reveal_strategy":  ["cascade", "single"],
    "flag_limit":       (0, None),       # None means unlimited; 0 means no flagging
    "safe_first_click": [True, False],
}

FIELD_DESCRIPTIONS: dict[str, str] = {
    "rows":             "board height (5–30)",
    "cols":             "board width (5–30)",
    "mine_count":       "total mines placed (1 to rows*cols - 9)",
    "starting_health":  "lives before game over (1–10)",
    "mine_damage":      "HP lost per mine hit (1–5)",
    "reveal_strategy":  '"cascade" (flood-fill) or "single" (one cell only)',
    "flag_limit":       "max flags allowed (null = unlimited, 0 = no flagging)",
    "safe_first_click": "true/false — guarantee first click is safe",
}


# ---------------------------------------------------------------------------
# Archive — flat dict of evaluated configs.
# ---------------------------------------------------------------------------

ARCHIVE: dict[str, dict] = {}


def _config_key(snapshot: dict) -> str:
    """Stable string key from a config snapshot dict."""
    return json.dumps(snapshot, sort_keys=True)


def _init_archive() -> None:
    """Pre-populate archive with the standard seed config."""
    seed_config = MECHANICS["standard"]()
    snapshot = dataclasses.asdict(seed_config)
    key = _config_key(snapshot)
    if key not in ARCHIVE:
        ARCHIVE[key] = {
            "config_snapshot":  snapshot,
            "description":      "Canonical 16×16 standard minesweeper",
            "fitness":          None,   # evaluated lazily on first use as parent
            "parent_snapshot":  None,
            "generation":       0,
        }


def save_archive(path: str = "archive.json") -> None:
    """Serialize ARCHIVE to JSON."""
    with open(path, "w") as f:
        json.dump(list(ARCHIVE.values()), f, indent=2)


def load_archive(path: str = "archive.json") -> None:
    """Load archive entries from JSON, merging into ARCHIVE."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        entries = json.load(f)
    for entry in entries:
        key = _config_key(entry["config_snapshot"])
        ARCHIVE[key] = entry


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Mutation prompt + response parsing
# ---------------------------------------------------------------------------

def build_mutation_prompt(
    config: GameConfig,
    fitness: dict,
    archive_entries: list[dict],
) -> str:
    snapshot = dataclasses.asdict(config)
    field_ref = "\n".join(
        f"  {k}: {desc}" for k, desc in FIELD_DESCRIPTIONS.items()
    )
    archive_summary = "\n".join(
        f"  - {e['description']} {e['config_snapshot']}"
        for e in archive_entries
        if e.get("description")
    ) or "  (empty — this is the first entry)"

    win_pct   = f"{fitness['win_rate'] * 100:.1f}%"
    prog_pct  = f"{fitness['avg_progress_fraction'] * 100:.1f}%"

    return f"""You are mutating a Minesweeper GameConfig to discover novel, playable variants.

Current config:
{json.dumps(snapshot, indent=2)}

Current fitness (RandomAgent, {fitness['n_games']} games):
  win_rate:         {win_pct}
  progress_fraction:{prog_pct}  (fraction of safe cells revealed on average)

Field reference:
{field_ref}

Already in archive:
{archive_summary}

Propose ONE mutation that changes 1–3 fields to create an interesting gameplay variant.
Aim for configs that are neither trivially easy nor trivially unwinnable.
Respond with JSON only — no explanation, no markdown:
{{"changes": {{"field": value}}, "description": "one sentence describing the gameplay effect"}}"""


def parse_mutation_response(
    response_text: str,
    base_config: GameConfig,
) -> GameConfig | None:
    """
    Parse the LLM's JSON response and return a validated GameConfig, or None on failure.
    """
    # Strip markdown code fences if the model wraps the JSON
    text = re.sub(r"```[a-z]*\n?", "", response_text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    changes = data.get("changes")
    if not isinstance(changes, dict) or not changes:
        return None

    # Validate each proposed change
    validated: dict = {}
    current = dataclasses.asdict(base_config)
    rows = changes.get("rows", current["rows"])
    cols = changes.get("cols", current["cols"])

    for field, value in changes.items():
        if field not in FIELD_CONSTRAINTS:
            return None
        constraint = FIELD_CONSTRAINTS[field]

        if isinstance(constraint, list):
            if value not in constraint:
                return None
        else:
            lo, hi = constraint
            if value is None and field == "flag_limit":
                pass  # None is valid for flag_limit
            else:
                if not isinstance(value, (int, float)):
                    return None
                if lo is not None and value < lo:
                    return None
                if hi is not None and value > hi:
                    return None

        validated[field] = value

    # Cross-field validation: mine_count must leave at least 9 safe cells
    new_mine_count = validated.get("mine_count", current["mine_count"])
    if new_mine_count >= rows * cols - 8:
        return None

    try:
        return dataclasses.replace(base_config, **validated)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Mutation loop
# ---------------------------------------------------------------------------

def run_mortar_step(
    base_entry: dict,
    archive: dict,
    n_games: int = 50,
) -> dict | None:
    """
    Run one MORTAR iteration: mutate base_entry's config via LLM, evaluate, return result.
    Returns None if the LLM fails to produce a valid config after retries, or the config
    fails admission criteria.
    """
    snapshot = base_entry["config_snapshot"]
    base_config = GameConfig(**snapshot)

    # Evaluate base config if not yet done (lazy init for seed entry)
    fitness = base_entry.get("fitness")
    if fitness is None:
        print("  Evaluating seed config...")
        fitness = evaluate_config(RandomAgent, base_config, n_games=n_games)
        base_entry["fitness"] = fitness

    client = _get_client()
    archive_entries = list(archive.values())

    new_config: GameConfig | None = None
    description: str = ""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="google/gemini-2.5-flash-lite",
                messages=[{"role": "user", "content": build_mutation_prompt(base_config, fitness, archive_entries)}],
                max_tokens=256,
                temperature=0.8,
            )
            text = response.choices[0].message.content
        except Exception as e:
            wait = 20 * (2 ** attempt)
            is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
            if is_rate_limit:
                print(f"  Rate limited — waiting {wait}s before retry {attempt + 1}/3...")
                time.sleep(wait)
            else:
                print(f"  LLM call failed (attempt {attempt + 1}): {e}")
            continue

        # Extract description before parsing (in case parse fails)
        try:
            raw = re.sub(r"```[a-z]*\n?", "", text).strip()
            description = json.loads(raw).get("description", "")
        except Exception:
            pass

        new_config = parse_mutation_response(text, base_config)
        if new_config is not None:
            break
        print(f"  Parse failed (attempt {attempt + 1}), retrying...")

    if new_config is None:
        print("  Failed to produce a valid config after 3 attempts.")
        return None

    # Evaluate the new config
    new_fitness = evaluate_config(RandomAgent, new_config, n_games=n_games)
    new_snapshot = dataclasses.asdict(new_config)

    # Admission criteria — use progress_fraction, not win_rate.
    # RandomAgent wins ~1% of games so win_rate is near-zero and noisy over 50 runs.
    # progress_fraction (cells_revealed / safe_cells) is stable and meaningful.
    pf = new_fitness["avg_progress_fraction"]
    if pf < 0.05:
        print(f"  Rejected: progress_fraction too low ({pf:.3f}) — config is nearly unplayable")
        return None
    if pf > 0.98:
        print(f"  Rejected: progress_fraction too high ({pf:.3f}) — config is trivially easy")
        return None

    generation = (base_entry.get("generation") or 0) + 1
    return {
        "config_snapshot":  new_snapshot,
        "description":      description,
        "fitness":          new_fitness,
        "parent_snapshot":  snapshot,
        "generation":       generation,
    }


def run_mortar_loop(
    n_iterations: int = 10,
    n_games_per_eval: int = 50,
    archive_path: str = "archive.json",
    delay: float = 10.0,
) -> None:
    """
    Run the MORTAR evolution loop for n_iterations steps.
    Picks a random archive entry as parent each step.
    Saves the archive after each accepted config.
    delay: seconds to wait between iterations (respects free-tier rate limits).
    """
    _init_archive()
    load_archive(archive_path)
    save_archive(archive_path)  # persist seed entry immediately

    accepted = 0
    for i in range(n_iterations):
        if i > 0 and delay > 0:
            time.sleep(delay)

        parent_entry = random.choice(list(ARCHIVE.values()))
        desc = parent_entry.get("description", "?")
        print(f"\n[{i+1}/{n_iterations}] Mutating: {desc}")

        result = run_mortar_step(parent_entry, ARCHIVE, n_games=n_games_per_eval)
        if result is None:
            print("  Skipped.")
            continue

        key = _config_key(result["config_snapshot"])
        if key in ARCHIVE:
            print("  Duplicate config — skipped.")
            continue

        ARCHIVE[key] = result
        accepted += 1
        f = result["fitness"]
        print(f"  Accepted: {result['description']}")
        print(f"  win={f['win_rate']*100:.1f}%  progress={f['avg_progress_fraction']*100:.1f}%  gen={result['generation']}")
        save_archive(archive_path)

    print(f"\nDone. {accepted}/{n_iterations} configs accepted. Archive size: {len(ARCHIVE)}.")
    print(f"Archive saved to {archive_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MORTAR — Minesweeper mechanic evolution")
    p.add_argument("--iterations", type=int,   default=10,            help="Number of mutation steps (default: 10)")
    p.add_argument("--games",      type=int,   default=50,            help="Games per config evaluation (default: 50)")
    p.add_argument("--archive",    default="archive.json",            help="Archive file path (default: archive.json)")
    p.add_argument("--delay",      type=float, default=10.0,          help="Seconds between iterations for rate limiting (default: 10)")
    args = p.parse_args()

    run_mortar_loop(
        n_iterations=args.iterations,
        n_games_per_eval=args.games,
        archive_path=args.archive,
        delay=args.delay,
    )
