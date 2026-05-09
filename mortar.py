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
import inspect
import json
import math
import os
import random
import re
import signal
import sys
import time

from openai import OpenAI

from agents import AGENTS, evaluate_config
from agents.pafg_archive_agent import (
    compile_agent_candidate,
    generate_agent_candidate,
    repair_agent_candidate,
)


# Default agent panel — resolved against AGENTS at run time so the panel
# automatically degrades when optional agents (e.g. neural, which needs torch)
# aren't installed. `pafg-llm` writes a fresh per-mechanic PAFG subclass via
# LLM and is the fairest agent for non-default mechanics — it stays in the
# default panel. `neural` is omitted by default because it consistently
# under-performs random + PAFG on the mechanic variants this pipeline targets.
DEFAULT_AGENTS = ["random", "pafg", "pafg-llm"]

# Special panel name: when included, mortar generates a per-mechanic
# LLM-tuned PAFG subclass before each panel evaluation rather than looking it
# up in the AGENTS registry. Opt-in only — see `--agents pafg-llm`.
PAFG_LLM_AGENT_NAME = "pafg-llm"
_PAFG_LLM_SENTINEL = "<pafg-llm-sentinel>"


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
from mine_behaviors import DriftingMines, ChainReactionMines
from reveal_strategies import CascadeReveal, SingleReveal
from info_strategies import (
    INFO_STRATEGIES,
    CountFlagsInfo,
    ParityInfo,
    DistanceInfo,
)
from neighborhoods import (
    NEIGHBORHOODS,
    VonNeumannNeighborhood,
    KnightNeighborhood,
)
from win_conditions import (
    WIN_CONDITIONS,
    RevealQuotaWin,
    FlagAllMinesWin,
)
from code_mutations import (
    KIND_SPEC,
    CodeValidationError,
    compile_and_register,
    register_all_from_archive,
)


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
    "mine_behavior":    ["static", "drifting", "chain-reaction"],
    "info_strategy":    ["count-mines", "count-flags", "parity", "distance", "direction", "noisy-count"],
    "neighborhood":     ["moore", "von-neumann", "diagonal", "knight", "radius-2-moore"],
    "win_condition":    ["standard", "reveal-quota", "flag-all-mines", "survival"],
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
    "mine_behavior":    '"static" (canonical), "drifting" (mines wander into adjacent unrevealed cells each turn; numbers update), or "chain-reaction" (hitting a mine cascades to all adjacent mines; pair with extra health)',
    "info_strategy":    'symbol shown on a revealed safe cell — "count-mines" (canonical mine count), "count-flags" (count flagged neighbors instead), "parity" (only E/O of mine count), "distance" (Chebyshev distance to nearest mine on board), "direction" (arrow toward nearest mine), "noisy-count" (true count with random ±1 lies)',
    "neighborhood":     'what counts as "adjacent" for adjacency counts, cascades, drift, and chain — "moore" (canonical 8-connected), "von-neumann" (4 orthogonal), "diagonal" (4 diagonal), "knight" (chess knight moves; cascade jumps non-locally), "radius-2-moore" (24 cells in 5×5 box)',
    "win_condition":    'objective definition — "standard" (reveal every safe cell), "reveal-quota" (reveal half of safe cells; partial win), "flag-all-mines" (every mine flagged AND only mines flagged), "survival" (act 20 turns without dying)',
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


# Default skill_spread imputed for entries that have not been evaluated yet.
# Picked to roughly match the median admitted spread so unevaluated parents
# (e.g. the seed before its first use) compete on roughly even footing with
# evaluated peers — they get selected, evaluated lazily, and then float to
# their real position next iteration.
_UNEVALUATED_SPREAD = 0.30


def _entry_spread(entry: dict) -> float:
    """Skill spread for an archive entry; falls back to a neutral default."""
    fitness = entry.get("fitness")
    if not fitness or "skill_spread" not in fitness:
        return _UNEVALUATED_SPREAD
    return float(fitness["skill_spread"])


def _archive_reverse_index(archive: dict[str, dict]) -> dict[str, dict]:
    """Map config_key(snapshot) -> entry, for lineage walks."""
    return {_config_key(e["config_snapshot"]): e for e in archive.values()}


def _lineage_trace(
    entry: dict,
    archive: dict[str, dict],
    *,
    max_depth: int = 5,
) -> list[dict]:
    """
    Walk ``entry``'s parent chain. Returns the chain ordered oldest -> newest
    (entry itself last). Capped at ``max_depth`` most recent steps so deep
    lineages don't bloat the prompt.

    Each step is matched against the archive via the parent's full config
    snapshot — that's the only handle we have, since entries don't carry an
    explicit parent_id.
    """
    by_key = _archive_reverse_index(archive)
    chain = [entry]
    cur = entry
    while True:
        parent_snapshot = cur.get("parent_snapshot")
        if not parent_snapshot:
            break
        parent_entry = by_key.get(_config_key(parent_snapshot))
        if parent_entry is None:
            break
        chain.append(parent_entry)
        cur = parent_entry
        if len(chain) > 32:  # safety
            break
    chain.reverse()
    if len(chain) > max_depth:
        chain = chain[-max_depth:]
    return chain


def _select_peers(
    archive: dict[str, dict],
    exclude_keys: set[str],
    *,
    top_k: int = 8,
    random_k: int = 3,
    rng: random.Random | None = None,
) -> list[dict]:
    """
    Return up to ``top_k`` highest-spread peers plus up to ``random_k`` random
    peers (without overlap), excluding any entry whose config_key is in
    ``exclude_keys`` (typically the parent's lineage).
    """
    rng = rng or random
    candidates = [
        e for e in archive.values()
        if _config_key(e["config_snapshot"]) not in exclude_keys
    ]
    if not candidates:
        return []
    by_spread = sorted(candidates, key=_entry_spread, reverse=True)
    top = by_spread[:top_k]
    top_keys = {_config_key(e["config_snapshot"]) for e in top}
    remaining = [
        e for e in candidates
        if _config_key(e["config_snapshot"]) not in top_keys
    ]
    rng.shuffle(remaining)
    return top + remaining[:random_k]


def _format_lineage_block(chain: list[dict]) -> str:
    if len(chain) <= 1:
        return "  (no ancestors — this parent is a seed)"
    lines = []
    for entry in chain:
        spread = _entry_spread(entry)
        gen = entry.get("generation", 0)
        desc = entry.get("description") or "(no description)"
        lines.append(f"  Gen {gen} | spread={spread:+.2f} | {desc}")
    return "\n".join(lines)


def _format_peers_block(peers: list[dict]) -> str:
    if not peers:
        return "  (no other entries in archive)"
    lines = []
    for entry in peers:
        spread = _entry_spread(entry)
        gen = entry.get("generation", 0)
        desc = entry.get("description") or "(no description)"
        lines.append(f"  Gen {gen} | spread={spread:+.2f} | {desc}")
    return "\n".join(lines)


_GEN_FIELDS: tuple[str, ...] = (
    "mine_behavior", "reveal_strategy", "info_strategy",
    "neighborhood", "win_condition",
)


def _format_active_gen_mechanics(
    parent_entry: dict,
    lineage_chain: list[dict],
) -> str:
    """
    For each field on the parent whose value starts with ``gen-``, find the
    lineage entry that introduced that mechanic and emit its description and
    full source. Empty string if no generated mechanics are active.

    The lookup walks the lineage chain (parent included) and matches on
    ``code_kind == field`` and ``code_key == value`` — that's the contract
    ``run_mortar_step`` writes when admitting a code-mode mutation.
    """
    snapshot = parent_entry.get("config_snapshot", {})
    blocks: list[str] = []
    for field in _GEN_FIELDS:
        value = snapshot.get(field)
        if not (isinstance(value, str) and value.startswith("gen-")):
            continue
        introducer = next(
            (
                e for e in lineage_chain
                if e.get("code_kind") == field and e.get("code_key") == value
            ),
            None,
        )
        if introducer is None:
            blocks.append(
                f"# Active mechanic: {field}={value}\n"
                f"# (source unavailable — introducing entry not in lineage cap)"
            )
            continue
        desc = introducer.get("description") or "(no description)"
        source = introducer.get("code_source") or "# (no source recorded)"
        blocks.append(
            f"# Active mechanic: {field}={value} — {desc}\n{source}"
        )
    return "\n\n".join(blocks)


def _format_archive_for_prompt(
    parent_entry: dict,
    archive: dict[str, dict],
    *,
    lineage_depth: int = 5,
    top_k: int = 8,
    random_k: int = 3,
    rng: random.Random | None = None,
) -> tuple[str, str, list[dict]]:
    """
    Build the lineage and peers sections for mutation prompts.

    Returns ``(lineage_block, peers_block, lineage_chain)``. The chain is
    returned so callers can reuse it for related prompt sections (e.g. inlining
    active gen-* code in the code-mode prompt).
    """
    chain = _lineage_trace(parent_entry, archive, max_depth=lineage_depth)
    exclude_keys = {_config_key(e["config_snapshot"]) for e in chain}
    peers = _select_peers(
        archive, exclude_keys, top_k=top_k, random_k=random_k, rng=rng,
    )
    return _format_lineage_block(chain), _format_peers_block(peers), chain


_COOLDOWN_PER_ATTEMPT = 0.03


def _entry_score(entry: dict) -> float:
    """
    Score an entry for parent selection: skill_spread minus a small
    cool-down penalty per attempted child. Keeps high-spread parents at the
    top while gently rotating through the lineage so we don't hammer the same
    parent every iteration.
    """
    spread = _entry_spread(entry)
    attempts = int(entry.get("n_children_attempted") or 0)
    return spread - _COOLDOWN_PER_ATTEMPT * attempts


def _select_parent(
    archive: dict[str, dict],
    *,
    temperature: float = 0.2,
    rng: random.Random | None = None,
) -> dict:
    """
    Sample a parent entry weighted by ``softmax(score / temperature)``, where
    score = skill_spread − cooldown × n_children_attempted.

    Lower temperature -> more elitist (high-score parents dominate). Higher
    temperature -> closer to uniform. ``temperature <= 0`` collapses to
    argmax (deterministic best). Entries with no fitness yet are treated as
    having ``_UNEVALUATED_SPREAD`` so they still get a chance to be evaluated
    lazily.
    """
    rng = rng or random
    entries = list(archive.values())
    if not entries:
        raise RuntimeError("Cannot select parent from empty archive.")
    if len(entries) == 1:
        return entries[0]

    scores = [_entry_score(e) for e in entries]
    if temperature <= 0:
        best_idx = max(range(len(entries)), key=scores.__getitem__)
        return entries[best_idx]

    # Subtract max for numerical stability before exp.
    max_score = max(scores)
    weights = [math.exp((s - max_score) / temperature) for s in scores]
    total = sum(weights)
    if total <= 0:
        return rng.choice(entries)
    pick = rng.uniform(0.0, total)
    cumulative = 0.0
    for entry, w in zip(entries, weights):
        cumulative += w
        if pick <= cumulative:
            return entry
    return entries[-1]


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------

# Per-LLM-call wall clock. Caps the OpenAI client's network wait so a stuck
# upstream can't hang the whole loop. The retry block already handles raised
# timeouts as one failed attempt.
LLM_TIMEOUT_SECONDS = 60.0

# Per-panel-evaluation wall clock. Caps each panel evaluation (parent or
# new-config) via SIGALRM. Triggered when a code mutation produces a behavior
# that infinite-loops or just makes games extremely slow. Generous enough that
# legitimate slow evals (e.g. PAFG on a tough board) finish well under it.
EVAL_TIMEOUT_SECONDS = 90


class EvalTimeout(Exception):
    """Raised when a panel evaluation exceeds EVAL_TIMEOUT_SECONDS."""


def _evaluate_with_timeout(
    panel: dict[str, type],
    config: GameConfig,
    n_games: int,
    timeout: int = EVAL_TIMEOUT_SECONDS,
) -> tuple[dict, list[str], dict[str, str]]:
    """
    Wall-clock-bounded panel evaluation with per-agent error tolerance.

    Returns (per_agent_results, crashed_agent_names, per_agent_errors).
    Individual agents that raise during evaluation are logged and excluded
    from the results — the rest of the panel still reports. The errors dict
    maps crashed agent names to a "Type: message" string for downstream
    repair attempts. Raises EvalTimeout if the *total* wall clock exceeds
    `timeout` (POSIX-only via SIGALRM; no-op elsewhere).
    """
    has_alarm = hasattr(signal, "SIGALRM")

    def _on_alarm(signum, frame):
        raise EvalTimeout(f"panel evaluation exceeded {timeout}s")

    prev_handler = None
    if has_alarm:
        prev_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(timeout)
    try:
        per_agent: dict = {}
        crashed: list[str] = []
        errors: dict[str, str] = {}
        for name, cls in panel.items():
            try:
                per_agent[name] = evaluate_config(cls, config, n_games=n_games)
            except EvalTimeout:
                raise
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                print(f"  Agent {name!r} crashed during eval ({msg}); dropping from this panel.")
                crashed.append(name)
                errors[name] = msg
        return per_agent, crashed, errors
    finally:
        if has_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)


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
    parent_entry: dict,
    archive: dict[str, dict],
    recent_failures: str = "",
) -> str:
    snapshot = dataclasses.asdict(config)
    field_ref = "\n".join(
        f"  {k}: {desc}" for k, desc in FIELD_DESCRIPTIONS.items()
    )
    lineage_block, peers_block, _chain = _format_archive_for_prompt(
        parent_entry, archive,
    )

    per_agent = fitness["per_agent"]
    n_games = next(iter(per_agent.values()))["n_games"]
    agent_lines = "\n".join(
        f"  {name:<9} win={f['win_rate']*100:5.1f}%  progress={f['avg_progress_fraction']*100:5.1f}%  turns={f['avg_turns']:5.1f}"
        for name, f in per_agent.items()
    )
    spread_pct = f"{fitness['skill_spread'] * 100:+.1f}%"
    failures_section = (
        f"\nRecent rejected proposals from this parent (do not repeat the same mistake):\n{recent_failures}\n"
        if recent_failures else ""
    )

    return f"""You are mutating a Minesweeper GameConfig to discover novel, playable variants where SKILL MATTERS.

A skilled agent (PAFG constraint solver) should clearly outperform a random agent on these configs. Configs where everyone scores the same are boring — they reward luck, not strategy.

Average turns is the mean number of actions per game — a useful secondary signal, especially for mechanics where progress is bounded or partial wins are common (a skilled agent often takes fewer turns to win, or survives more turns under aggressive mechanics).

Current config:
{json.dumps(snapshot, indent=2)}

Current fitness ({n_games} games per agent):
{agent_lines}
  skill_spread (best non-random agent − random progress): {spread_pct}

Field reference:
{field_ref}

Lineage (oldest → this parent; spread is the admitted skill_spread for each step):
{lineage_block}

Other archive entries (top by skill_spread + a few random):
{peers_block}
{failures_section}
Complementarity over compounding:
- If the lineage already has obfuscation (info_strategy != "count-mines", parity, noisy-count, distance, direction), prefer adding *informative* mechanics (extra clues, larger neighborhoods, extra health, lower mine_count).
- If the lineage already has aggression (chain-reaction, drifting mines, low health, high damage, single reveal, no safe-first-click), prefer adding *clarity* (lower damage, extra health, count-mines info, cascade reveal).
- Stacking two punishing mechanics on the same lineage usually produces unplayable configs.

Propose ONE mutation that changes 1–3 fields to create an interesting gameplay variant.
Aim for a LARGE skill spread: configs that are playable for the constraint solver but punishing for random play.
Respond with JSON only — no explanation, no markdown:
{{"changes": {{"field": value}}, "description": "one sentence describing the gameplay effect"}}"""


def _validate_changes(
    changes: dict,
    base_config: GameConfig,
) -> dict | None:
    """
    Validate a {field: value} mutation dict against FIELD_CONSTRAINTS.
    Returns the validated dict, or None if any value is out of bounds /
    of wrong type / for an unknown field. Shared by param mode (`changes`)
    and code mode (`config_overrides`).
    """
    if not isinstance(changes, dict):
        return None

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

    # Cross-field: mine_count must leave at least 9 safe cells
    new_mine_count = validated.get("mine_count", current["mine_count"])
    if new_mine_count >= rows * cols - 8:
        return None

    return validated


# Fields whose default value MORTAR treats as "vanilla". Used to count how
# many non-default mechanics are simultaneously active on a config — three or
# more is a strong predictor of unplayability.
_DEFAULT_FIELD_VALUES: dict[str, object] = {
    "info_strategy":   "count-mines",
    "reveal_strategy": "cascade",
    "mine_behavior":   "static",
    "neighborhood":    "moore",
    "win_condition":   "standard",
    "safe_first_click": True,
}


def _describe_config_warnings(snapshot: dict) -> list[str]:
    """
    Return human-readable warnings for known-bad mechanic combinations.

    Per the audit decision, these are warnings only — the +0.10 skill_spread
    gate still decides admission. The warnings are printed to stdout and fed
    into the per-parent failure-feedback ring buffer (P6) so subsequent LLM
    calls see them.
    """
    warnings: list[str] = []
    win_condition = snapshot.get("win_condition")
    flag_limit = snapshot.get("flag_limit")
    mine_count = snapshot.get("mine_count", 0)
    mine_behavior = snapshot.get("mine_behavior")
    starting_health = snapshot.get("starting_health", 1)
    mine_damage = snapshot.get("mine_damage", 1)
    safe_first_click = snapshot.get("safe_first_click", True)
    rows = snapshot.get("rows", 16)
    cols = snapshot.get("cols", 16)

    if (
        win_condition == "flag-all-mines"
        and flag_limit is not None
        and flag_limit < mine_count
    ):
        warnings.append(
            f"flag-all-mines win requires every mine flagged but flag_limit={flag_limit} "
            f"< mine_count={mine_count} — winning is mathematically impossible."
        )

    if mine_behavior == "chain-reaction" and starting_health <= mine_damage:
        warnings.append(
            f"chain-reaction with starting_health={starting_health} and mine_damage={mine_damage} "
            f"means hitting any mine triggers an unsurvivable cluster cascade."
        )

    if not safe_first_click and rows * cols > 0:
        density = mine_count / (rows * cols)
        if density >= 0.5:
            warnings.append(
                f"safe_first_click=False with mine density {density:.2f} (>= 0.5) "
                f"makes random play almost certainly die on its first click."
            )

    nondefault_count = sum(
        1 for field, default in _DEFAULT_FIELD_VALUES.items()
        if snapshot.get(field) != default
    )
    if nondefault_count >= 3:
        nondefault_fields = [
            f"{field}={snapshot.get(field)!r}"
            for field, default in _DEFAULT_FIELD_VALUES.items()
            if snapshot.get(field) != default
        ]
        warnings.append(
            f"{nondefault_count} non-default mechanics simultaneously active "
            f"({', '.join(nondefault_fields)}) — compound mechanic stacks "
            f"frequently fall below the playability gate."
        )

    return warnings


# Per-parent rolling buffer of rejected proposals. Lifetime: one
# run_mortar_loop invocation. Helps the LLM avoid re-proposing the same
# bad mutation when its parent is sampled repeatedly.
_FAILURE_RING_SIZE = 3
FailureBuffer = dict[str, list[dict]]


def _record_failure(
    buffer: FailureBuffer,
    parent_key: str,
    description: str,
    reason: str,
    warnings: list[str] | None = None,
) -> None:
    """Append a rejection to the per-parent ring; trim to most recent N."""
    record = {"description": description or "(no description)", "reason": reason}
    if warnings:
        record["warnings"] = warnings
    bucket = buffer.setdefault(parent_key, [])
    bucket.append(record)
    if len(bucket) > _FAILURE_RING_SIZE:
        del bucket[: len(bucket) - _FAILURE_RING_SIZE]


def _format_recent_failures(buffer: FailureBuffer, parent_key: str) -> str:
    """Render the rejection ring for a parent as a prompt block; empty if none."""
    bucket = buffer.get(parent_key) or []
    if not bucket:
        return ""
    lines = []
    for i, rec in enumerate(bucket, 1):
        lines.append(f"  {i}. \"{rec['description']}\" — REJECTED: {rec['reason']}")
        for w in rec.get("warnings") or []:
            lines.append(f"     warning: {w}")
    return "\n".join(lines)


def parse_mutation_response(
    response_text: str,
    base_config: GameConfig,
) -> GameConfig | None:
    """
    Parse the param-mode LLM JSON response and return a validated GameConfig,
    or None on failure.
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

    validated = _validate_changes(changes, base_config)
    if validated is None:
        return None

    try:
        return dataclasses.replace(base_config, **validated)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Code-mutation prompt + response parsing
# ---------------------------------------------------------------------------

# Public API surface exposed to LLM-authored mechanics. The prompt embeds this
# verbatim so the model knows what's available — keep it in sync with the
# Board / Player public methods (board.py, player.py).
_BOARD_PLAYER_CHEATSHEET = """\
Board (passed to your method):
  board.config              — GameConfig (rows, cols, mine_count, ...)
  board.grid[r][c]          — Cell with .is_mine .is_revealed .is_flagged .adjacent_mines
  board.neighbors(r, c)     — list of valid (r, c) neighbors per the configured Neighborhood (default: 8-Moore; may be Von Neumann / knight / etc.)
  board.in_bounds(r, c)     — bool
  board.move_mine(src, dst) — bool; refuses dst that's revealed/flagged/already-mine
  board.recompute_adjacency() — call ONCE after relocating any mines
  board.reveal_mine(r, c)   — mark a mine cell as revealed

Player (mine_behavior only):
  game.player.take_damage()
  game.player.is_alive() -> bool

Cell mutation: directly assign to board.grid[r][c].is_mine / .is_revealed /
.is_flagged / .adjacent_mines as needed."""


_ACTION_DICT_SCHEMA = """\
The `action` arg is a dict with:
  "action":         "reveal" | "flag"
  "coords":         (row, col) — what the player just clicked
  "hit_mine":       bool — did the player hit a mine this turn
  "newly_revealed": list[(row, col)] — MUTABLE; append cells you reveal\
"""


_CONSTRUCTOR_RULES = {
    "mine_behavior": (
        "If you override __init__, accept `seed: int | None = None` and call "
        "`super().__init__(seed)` — that initializes self._rng. If you don't "
        "need extra args, just don't override __init__."
    ),
    "reveal_strategy": (
        "Your class must be default-constructible (no required constructor "
        "args). Board instantiates strategies with no arguments."
    ),
    "info_strategy": (
        "If you override __init__, accept `seed: int | None = None` and call "
        "`super().__init__(seed)` — that stores self._seed. The Game "
        "instantiates info strategies with the game seed; use it for any "
        "per-cell deterministic noise (e.g. `random.Random(hash((self._seed, r, c)))`)."
    ),
    "neighborhood": (
        "Your class must be default-constructible (no required constructor "
        "args). Board instantiates neighborhoods with no arguments. The only "
        "method to implement is `offsets() -> list[tuple[int, int]]` returning "
        "(dr, dc) pairs that exclude (0, 0)."
    ),
    "win_condition": (
        "If you override __init__, accept `seed: int | None = None` and call "
        "`super().__init__(seed)` — that initializes self._rng. The Game "
        "instantiates win conditions with the game seed. evaluate() must "
        "return the literal string \"won\", the literal string \"lost\", or "
        "None to continue the game (do not return any other value)."
    ),
}


_KIND_EXEMPLARS: dict[str, list[type]] = {
    "mine_behavior":   [DriftingMines, ChainReactionMines],
    "reveal_strategy": [CascadeReveal, SingleReveal],
    "info_strategy":   [CountFlagsInfo, ParityInfo, DistanceInfo],
    "neighborhood":    [VonNeumannNeighborhood, KnightNeighborhood],
    "win_condition":   [RevealQuotaWin, FlagAllMinesWin],
}


def _exemplar_block(kind: str) -> str:
    parts = []
    for cls in _KIND_EXEMPLARS[kind]:
        try:
            parts.append(inspect.getsource(cls))
        except OSError:
            parts.append(f"# (source for {cls.__name__} unavailable)")
    return "\n\n".join(parts)


def build_code_mutation_prompt(
    base_config: GameConfig,
    fitness: dict,
    parent_entry: dict,
    archive: dict[str, dict],
    kind: str,
    recent_failures: str = "",
) -> str:
    """
    Build the prompt for code-mode mutation. `kind` is one of
    "mine_behavior", "reveal_strategy", "info_strategy", "neighborhood",
    "win_condition".
    """
    abc_cls, _, _ = KIND_SPEC[kind]
    snapshot = dataclasses.asdict(base_config)

    lineage_block, peers_block, chain = _format_archive_for_prompt(
        parent_entry, archive,
    )

    active_gen_block = _format_active_gen_mechanics(parent_entry, chain)

    per_agent = fitness["per_agent"]
    n_games = next(iter(per_agent.values()))["n_games"]
    agent_lines = "\n".join(
        f"  {name:<9} win={f['win_rate']*100:5.1f}%  progress={f['avg_progress_fraction']*100:5.1f}%  turns={f['avg_turns']:5.1f}"
        for name, f in per_agent.items()
    )
    spread_pct = f"{fitness['skill_spread'] * 100:+.1f}%"

    abc_source = inspect.getsource(abc_cls)

    action_section = ""
    if kind == "mine_behavior":
        action_section = f"\n== Action dict schema ==\n\n{_ACTION_DICT_SCHEMA}\n"

    active_gen_section = (
        f"\n== Active generated mechanics on this parent ==\n\n{active_gen_block}\n"
        if active_gen_block else ""
    )
    failures_section = (
        f"\n== Recent rejected proposals from this parent (do not repeat) ==\n\n{recent_failures}\n"
        if recent_failures else ""
    )

    return f"""You are mutating Minesweeper by writing a NEW {kind} subclass in Python.

Goal: discover novel, playable variants where SKILL MATTERS — a constraint solver should clearly outperform random play. Configs where everyone scores the same are boring.

Parent config:
{json.dumps(snapshot, indent=2)}

Parent fitness ({n_games} games per agent):
{agent_lines}
  skill_spread (best non-random agent − random progress): {spread_pct}

Lineage (oldest → this parent; spread is the admitted skill_spread for each step):
{lineage_block}

Other archive entries (top by skill_spread + a few random; do not re-create):
{peers_block}
{active_gen_section}{failures_section}
== Interface to implement ==

{abc_source}
{action_section}
== Public API your class may use ==

{_BOARD_PLAYER_CHEATSHEET}

== Hard constraints ==

- Define EXACTLY ONE subclass of {abc_cls.__name__}.
- No `import` statements. The runtime exposes: random, deque, {abc_cls.__name__}.
- {_CONSTRUCTOR_RULES[kind]}
- Do not call game.reveal() or game.flag() — the framework re-checks state after your hook returns.
- Do not perform I/O or read external state.
- Class name should be CamelCase and descriptive of the mechanic.

== Complementarity over compounding ==

- If the lineage already has obfuscation (info_strategy != "count-mines", parity, noisy-count, distance, direction), prefer authoring a mechanic that adds *clarity* or *survivability*.
- If the lineage already has aggression (chain-reaction, drifting, low health, high damage, single reveal), prefer authoring a mechanic that gives the player more *information* or *agency*.
- Stacking another punishing mechanic on a lineage that's already brutal will likely fall below the playability gate.

== Examples (existing implementations) ==

{_exemplar_block(kind)}

== Output ==

Respond with JSON only — no markdown, no prose:
{{
  "kind": "{kind}",
  "name": "DescriptiveName",
  "code": "<full Python source for your class — newlines as \\n>",
  "config_overrides": {{}},
  "description": "one sentence about the gameplay effect"
}}

config_overrides may be empty {{}} or include any of: rows, cols, mine_count, starting_health, mine_damage, safe_first_click, flag_limit. Use it to pair your mechanic with parameters that make it survivable (e.g. extra health for chain-style behaviors)."""


def parse_code_mutation_response(
    response_text: str,
    base_config: GameConfig,
) -> tuple[str, str, str, GameConfig, str] | None:
    """
    Parse a code-mode response into (kind, key, source, new_config, description).

    On success the generated class is registered in the appropriate registry
    via compile_and_register. On failure returns None. Note: registered
    classes are intentionally never unregistered — leaking a class costs one
    dict entry, while removing one risks dropping a class another archive
    entry depends on.
    """
    text = re.sub(r"```[a-z]*\n?", "", response_text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    kind = data.get("kind")
    source = data.get("code")
    description = data.get("description") or ""
    if kind not in KIND_SPEC or not isinstance(source, str) or not source.strip():
        return None

    overrides = data.get("config_overrides") or {}
    if not isinstance(overrides, dict):
        return None
    # The kind's own field is set from the generated key — drop it from overrides
    # if the model accidentally included it.
    overrides.pop("mine_behavior", None)
    overrides.pop("reveal_strategy", None)
    overrides.pop("win_condition", None)

    if overrides:
        validated = _validate_changes(overrides, base_config)
        if validated is None:
            return None
    else:
        validated = {}

    try:
        key, _cls = compile_and_register(source, kind)
    except CodeValidationError as e:
        first_line = str(e).splitlines()[0] if str(e) else "unknown"
        print(f"  Code validation failed: {first_line}")
        return None

    field = KIND_SPEC[kind][2]
    try:
        new_config = dataclasses.replace(base_config, **validated, **{field: key})
    except (TypeError, ValueError):
        return None

    return kind, key, source, new_config, description


# ---------------------------------------------------------------------------
# Mutation loop
# ---------------------------------------------------------------------------

def _build_panel(agent_names: list[str]) -> dict:
    """Resolve agent names against the AGENTS registry. Skip names not present
    (e.g. 'neural' when torch isn't installed) with a one-line warning. The
    `pafg-llm` slot is stored as a sentinel; `_materialize_panel` swaps in a
    per-mechanic LLM-generated class right before each evaluation."""
    panel: dict = {}
    for name in agent_names:
        if name == PAFG_LLM_AGENT_NAME:
            panel[name] = _PAFG_LLM_SENTINEL
            continue
        if name in AGENTS:
            panel[name] = AGENTS[name]
        else:
            print(f"  Warning: agent '{name}' unavailable — skipping.")
    if not panel:
        raise RuntimeError("Empty agent panel — at least one agent must be available.")
    return panel


def _build_pafg_llm_entry(
    snapshot: dict,
    description: str,
    code_meta: dict,
    parent_snapshot: dict | None = None,
    generation: int = 0,
) -> dict:
    """Synthesize the entry dict the pafg-llm prompt expects.

    ``parent_snapshot`` is what makes the lineage walk effective for stacked
    gen-* mechanics: passing the real parent lets ``_active_generated_mechanics``
    follow the chain through ARCHIVE and surface every ancestor's code source.
    """
    return {
        "config_snapshot": snapshot,
        "description":     description or "",
        "generation":      generation,
        "parent_snapshot": parent_snapshot,
        "code_kind":       code_meta.get("code_kind"),
        "code_source":     code_meta.get("code_source"),
        "code_key":        code_meta.get("code_key"),
    }


def _generate_pafg_llm_class(
    snapshot: dict,
    description: str,
    code_meta: dict,
    parent_snapshot: dict | None = None,
    archive: dict | None = None,
) -> tuple[type | None, dict | None]:
    """
    Generate a fresh per-mechanic pafg-llm subclass via LLM.

    Returns (compiled_class, candidate_dict) on success, (None, None) on
    failure. Never raises — mortar continues with the rest of the panel.
    """
    entry = _build_pafg_llm_entry(
        snapshot, description, code_meta, parent_snapshot=parent_snapshot,
    )
    try:
        candidate, _raw = generate_agent_candidate(entry, archive=archive)
    except Exception as e:
        print(f"  pafg-llm generation failed ({type(e).__name__}: {e}); dropping slot.")
        return None, None
    if candidate is None:
        print("  pafg-llm response unparseable; dropping slot.")
        return None, None
    try:
        cls = compile_agent_candidate(candidate["code"], candidate["name"])
    except Exception as e:
        print(f"  pafg-llm compile failed ({type(e).__name__}: {e}); dropping slot.")
        return None, None
    return cls, candidate


def _attempt_pafg_llm_repair(
    snapshot: dict,
    description: str,
    code_meta: dict,
    candidate: dict | None,
    error: str,
    config: GameConfig,
    n_games: int,
    parent_snapshot: dict | None = None,
    archive: dict | None = None,
) -> tuple[dict | None, dict | None]:
    """
    One-shot repair: ask the LLM to fix a crashing pafg-llm candidate, then
    re-run just that agent's evaluation.

    Returns (repaired_candidate, stats) on success, (None, None) otherwise.
    Never raises.
    """
    if not candidate:
        return None, None
    entry = _build_pafg_llm_entry(
        snapshot, description, code_meta, parent_snapshot=parent_snapshot,
    )
    print(f"  pafg-llm crashed ({error}); attempting one repair...")
    try:
        repaired = repair_agent_candidate(entry, candidate, error, archive=archive)
    except Exception as e:
        print(f"  pafg-llm repair LLM call failed ({type(e).__name__}: {e}); dropping slot.")
        return None, None
    if not repaired:
        print("  pafg-llm repair response unparseable; dropping slot.")
        return None, None
    try:
        cls = compile_agent_candidate(repaired["code"], repaired["name"])
    except Exception as e:
        print(f"  pafg-llm repair compile failed ({type(e).__name__}: {e}); dropping slot.")
        return None, None
    try:
        stats = evaluate_config(cls, config, n_games=n_games)
    except Exception as e:
        print(f"  pafg-llm repair retry crashed ({type(e).__name__}: {e}); dropping slot.")
        return None, None
    print("  pafg-llm repair succeeded; merging stats into panel.")
    return repaired, stats


def _recompile_cached_pafg_llm(cached: dict | None) -> type | None:
    """Recompile a cached pafg-llm agent source. None if missing or invalid."""
    if not cached:
        return None
    name = cached.get("name")
    code = cached.get("code")
    if not name or not code:
        return None
    try:
        return compile_agent_candidate(code, name)
    except Exception as e:
        print(f"  Cached pafg-llm class {name!r} failed to recompile "
              f"({type(e).__name__}: {e}); regenerating.")
        return None


def _materialize_panel(
    panel: dict,
    snapshot: dict,
    description: str,
    code_meta: dict,
    cached_agent: dict | None = None,
    parent_snapshot: dict | None = None,
    archive: dict | None = None,
) -> tuple[dict[str, type], dict | None]:
    """
    Replace the pafg-llm sentinel with a live class.

    If a cached agent source recompiles cleanly, reuse it (no LLM call).
    Otherwise generate a fresh subclass. On any failure, drop the slot for
    this evaluation rather than crashing the whole step.

    ``parent_snapshot`` and ``archive`` are forwarded to the pafg-llm prompt
    so the lineage walk can surface every active gen-* mechanic, not just
    whatever this entry directly introduced.

    Returns (live_panel, candidate_dict_to_persist_or_None). The second
    element is None on cache hit or when pafg-llm is not in the panel.
    """
    if PAFG_LLM_AGENT_NAME not in panel:
        return dict(panel), None

    cached_cls = _recompile_cached_pafg_llm(cached_agent)
    if cached_cls is not None:
        live_panel = {
            name: (cached_cls if value is _PAFG_LLM_SENTINEL else value)
            for name, value in panel.items()
        }
        return live_panel, None

    cls, candidate = _generate_pafg_llm_class(
        snapshot, description, code_meta,
        parent_snapshot=parent_snapshot,
        archive=archive,
    )
    if cls is None:
        live_panel = {
            name: value for name, value in panel.items()
            if value is not _PAFG_LLM_SENTINEL
        }
        return live_panel, None

    live_panel = {
        name: (cls if value is _PAFG_LLM_SENTINEL else value)
        for name, value in panel.items()
    }
    return live_panel, candidate


def _multi_fitness(per_agent: dict[str, dict]) -> dict:
    """Build the archive `fitness` dict from per-agent eval results.

    `skill_spread` is the gap between the best-performing non-random agent and
    the random baseline. This generalizes the original pafg-vs-random spread to
    panels that include other skilled agents (neural, pafg-llm).
    """
    random_pf = per_agent.get("random", {}).get("avg_progress_fraction", 0.0)
    non_random_pfs = [
        f.get("avg_progress_fraction", 0.0)
        for name, f in per_agent.items()
        if name != "random"
    ]
    best_skill_pf = max(non_random_pfs) if non_random_pfs else 0.0
    return {
        "per_agent":    per_agent,
        "skill_spread": best_skill_pf - random_pf,
        "n_games":      next(iter(per_agent.values()))["n_games"],
    }


# ---- Output formatting (ANSI color, gated on isatty) ----

_USE_COLOR = sys.stdout.isatty()
_C_RESET   = "\033[0m"  if _USE_COLOR else ""
_C_BOLD    = "\033[1m"  if _USE_COLOR else ""
_C_DIM     = "\033[2m"  if _USE_COLOR else ""
_C_GREEN   = "\033[32m" if _USE_COLOR else ""
_C_YELLOW  = "\033[33m" if _USE_COLOR else ""
_C_CYAN    = "\033[36m" if _USE_COLOR else ""


def _tag(label: str, color: str) -> str:
    return f"{color}{_C_BOLD}[{label}]{_C_RESET}"


_TAG_ACCEPTED = _tag("ACCEPTED", _C_GREEN)
_TAG_REJECTED = _tag("REJECTED", _C_YELLOW)
_TAG_SKIPPED  = _tag("SKIPPED",  _C_DIM)
_TAG_WARN     = _tag("WARN",     _C_CYAN)


def _format_per_agent(per_agent: dict[str, dict]) -> str:
    """Aligned per-agent stats, one row per agent."""
    if not per_agent:
        return ""
    name_w = max(len(name) for name in per_agent)
    return "\n".join(
        f"      {name:<{name_w}}  win {f['win_rate']*100:5.1f}%   prog {f['avg_progress_fraction']*100:5.1f}%   turns {f['avg_turns']:5.1f}"
        for name, f in per_agent.items()
    )


def run_mortar_step(
    base_entry: dict,
    archive: dict,
    panel: dict[str, type],
    n_games: int = 20,
    mode: str = "param",
    code_kind: str | None = None,
    admit_all: bool = False,
    failure_buffer: FailureBuffer | None = None,
) -> dict | None:
    """
    Run one MORTAR iteration. `mode` is "param" or "code"; "code" requires
    `code_kind` in {"mine_behavior", "reveal_strategy"}.

    Returns a new archive entry on success, or None if the LLM fails to
    produce a valid config after retries or the config fails admission.
    """
    snapshot = base_entry["config_snapshot"]
    base_config = GameConfig(**snapshot)
    parent_key = _config_key(snapshot)
    if failure_buffer is None:
        failure_buffer = {}

    # Lazy: evaluate the parent if it's never been measured.
    fitness = base_entry.get("fitness")
    if fitness is None or "per_agent" not in fitness:
        parent_code_meta = {
            k: base_entry.get(k) for k in ("code_kind", "code_source", "code_key")
        }
        live_panel, generated_agent = _materialize_panel(
            panel,
            snapshot,
            base_entry.get("description", ""),
            parent_code_meta,
            cached_agent=base_entry.get("pafg_llm_agent"),
            parent_snapshot=base_entry.get("parent_snapshot"),
            archive=archive,
        )
        if generated_agent is not None:
            base_entry["pafg_llm_agent"] = generated_agent

        print("  Evaluating base config across agent panel...")
        try:
            per_agent, crashed, errors = _evaluate_with_timeout(
                live_panel, base_config, n_games=n_games
            )
        except EvalTimeout as e:
            print(f"  {_TAG_SKIPPED} parent eval timed out ({e})")
            return None
        except Exception as e:
            print(f"  {_TAG_SKIPPED} parent eval crashed ({type(e).__name__}: {e})")
            return None
        if PAFG_LLM_AGENT_NAME in crashed:
            crashed_candidate = (
                generated_agent if generated_agent is not None
                else base_entry.get("pafg_llm_agent")
            )
            repaired, repaired_stats = _attempt_pafg_llm_repair(
                snapshot,
                base_entry.get("description", ""),
                parent_code_meta,
                crashed_candidate,
                errors.get(PAFG_LLM_AGENT_NAME, "unknown"),
                base_config,
                n_games,
                parent_snapshot=base_entry.get("parent_snapshot"),
                archive=archive,
            )
            if repaired_stats is not None:
                per_agent[PAFG_LLM_AGENT_NAME] = repaired_stats
                crashed.remove(PAFG_LLM_AGENT_NAME)
                base_entry["pafg_llm_agent"] = repaired
            else:
                base_entry.pop("pafg_llm_agent", None)
        if not per_agent:
            print(f"  {_TAG_SKIPPED} every agent in the panel crashed")
            return None
        fitness = _multi_fitness(per_agent)
        base_entry["fitness"] = fitness
        print(_format_per_agent(per_agent))

    if mode == "code":
        if code_kind not in KIND_SPEC:
            print(f"  Invalid code_kind: {code_kind!r}")
            return None
        max_tokens = 2000
        temperature = 0.9
    else:
        max_tokens = 256
        temperature = 0.8

    client = _get_client()

    new_config: GameConfig | None = None
    description: str = ""
    code_meta: dict = {"code_kind": None, "code_source": None, "code_key": None}

    recent_failures = _format_recent_failures(failure_buffer, parent_key)

    for attempt in range(3):
        if mode == "code":
            prompt = build_code_mutation_prompt(
                base_config, fitness, base_entry, archive, code_kind,
                recent_failures=recent_failures,
            )
        else:
            prompt = build_mutation_prompt(
                base_config, fitness, base_entry, archive,
                recent_failures=recent_failures,
            )

        try:
            response = client.chat.completions.create(
                model="anthropic/claude-sonnet-4.6",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=LLM_TIMEOUT_SECONDS,
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

        if mode == "code":
            parsed = parse_code_mutation_response(text, base_config)
            if parsed is not None:
                kind, key, source, new_config, description = parsed
                code_meta = {
                    "code_kind":   kind,
                    "code_source": source,
                    "code_key":    key,
                }
                break
        else:
            # Pull description before parse can fail (for diagnostics).
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
        _record_failure(
            failure_buffer, parent_key,
            description=description,
            reason="LLM produced no parseable / valid mutation after 3 attempts",
        )
        return None

    if description:
        print(f"  {_C_CYAN}Testing:{_C_RESET} {description}")

    # Surface known-bad combos as warnings (warn-and-admit-all). The +0.10
    # skill_spread gate still decides admission below; warnings are fed into
    # the per-parent rejection-feedback buffer if this mutation is rejected.
    new_snapshot = dataclasses.asdict(new_config)
    config_warnings = _describe_config_warnings(new_snapshot)
    for w in config_warnings:
        print(f"  {_TAG_WARN} {w}")

    # Evaluate the new config with the full panel. For pafg-llm we generate
    # a fresh per-mechanic subclass before evaluating. The new config's
    # parent IS base_entry, so its lineage walks through base_entry into the
    # archive — that's how multi-stack gen-* mechanics get surfaced.
    live_panel, generated_agent = _materialize_panel(
        panel, new_snapshot, description, code_meta,
        parent_snapshot=snapshot,
        archive=archive,
    )
    try:
        per_agent, crashed, errors = _evaluate_with_timeout(
            live_panel, new_config, n_games=n_games
        )
    except EvalTimeout as e:
        print(f"  {_TAG_REJECTED} eval timed out ({e})")
        _record_failure(
            failure_buffer, parent_key,
            description=description,
            reason=f"evaluation exceeded {EVAL_TIMEOUT_SECONDS}s wall clock",
            warnings=config_warnings,
        )
        return None
    except Exception as e:
        # Generated mechanic crashed at runtime — smoke test missed it.
        # Reject the mutation and continue the loop.
        print(f"  {_TAG_REJECTED} eval crashed ({type(e).__name__}: {e})")
        _record_failure(
            failure_buffer, parent_key,
            description=description,
            reason=f"evaluation crashed ({type(e).__name__}: {e})",
            warnings=config_warnings,
        )
        return None
    if PAFG_LLM_AGENT_NAME in crashed:
        repaired, repaired_stats = _attempt_pafg_llm_repair(
            new_snapshot,
            description,
            code_meta,
            generated_agent,
            errors.get(PAFG_LLM_AGENT_NAME, "unknown"),
            new_config,
            n_games,
            parent_snapshot=snapshot,
            archive=archive,
        )
        if repaired_stats is not None:
            per_agent[PAFG_LLM_AGENT_NAME] = repaired_stats
            crashed.remove(PAFG_LLM_AGENT_NAME)
            generated_agent = repaired
        else:
            generated_agent = None
    if not per_agent:
        print(f"  {_TAG_REJECTED} every agent in the panel crashed on this config")
        _record_failure(
            failure_buffer, parent_key,
            description=description,
            reason="every agent in the panel crashed on this config",
            warnings=config_warnings,
        )
        return None
    new_fitness = _multi_fitness(per_agent)

    print(_format_per_agent(per_agent))

    # Admission criterion: at least one skilled agent must clear the random
    # baseline by `spread_threshold`. Trivially-easy and trivially-hard configs
    # are both kept — the only thing that disqualifies a config is "no agent
    # outperforms random."
    spread = new_fitness["skill_spread"]
    spread_threshold = 0.10

    if spread < spread_threshold:
        msg = (
            f"skill spread too small ({spread:+.3f}) — "
            f"no agent in the panel beat random by {spread_threshold:+.2f}"
        )
        if not admit_all:
            print(f"  {_TAG_REJECTED} {msg}")
            _record_failure(
                failure_buffer, parent_key,
                description=description,
                reason=f"skill_spread={spread:+.3f} below playability gate {spread_threshold:+.2f}",
                warnings=config_warnings,
            )
            return None
        print(f"  {_C_DIM}[admit-all]{_C_RESET} {msg}")

    generation = (base_entry.get("generation") or 0) + 1
    result = {
        "config_snapshot":  new_snapshot,
        "description":      description,
        "fitness":          new_fitness,
        "parent_snapshot":  snapshot,
        "generation":       generation,
        **code_meta,
    }
    if generated_agent is not None:
        result["pafg_llm_agent"] = generated_agent
    return result


_CODE_KINDS = ["mine_behavior", "reveal_strategy", "info_strategy", "neighborhood", "win_condition"]


def _resolve_iter_mode(mode: str) -> tuple[str, str | None]:
    """
    Map the user-facing mode (param/code/mixed) to a concrete iteration mode
    and (for code mode) which surface to mutate.
    """
    if mode == "param":
        return "param", None
    if mode == "code":
        return "code", random.choice(_CODE_KINDS)
    if mode == "mixed":
        if random.random() < 0.5:
            return "code", random.choice(_CODE_KINDS)
        return "param", None
    raise ValueError(f"Unknown mode: {mode!r}")


def run_mortar_loop(
    n_iterations: int = 10,
    n_games_per_eval: int = 20,
    archive_path: str = "archive.json",
    delay: float = 10.0,
    agent_names: list[str] | None = None,
    mode: str = "mixed",
    admit_all: bool = False,
    parent_temperature: float = 0.2,
) -> None:
    """
    Run the MORTAR evolution loop for n_iterations steps.
    Picks a random archive entry as parent each step.
    Saves the archive after each accepted config.
    delay: seconds to wait between iterations (respects free-tier rate limits).
    mode: 'param' tunes GameConfig fields only; 'code' asks the LLM to write
          new MineBehavior/RevealStrategy classes; 'mixed' alternates 50/50.
    """
    panel = _build_panel(agent_names or DEFAULT_AGENTS)
    print(f"Agent panel: {', '.join(panel)}")
    print(f"Mutation mode: {mode}")
    print(f"Parent selection: softmax(skill_spread / {parent_temperature:.2f})")
    if admit_all:
        print("Admission gates: DISABLED (--admit-all) — every parseable mutation will be admitted.")

    _init_archive()
    load_archive(archive_path)
    n_loaded = register_all_from_archive(ARCHIVE)
    if n_loaded:
        print(f"Reregistered {n_loaded} generated mechanic(s) from archive.")
    save_archive(archive_path)  # persist seed entry + drop any unloadable code entries

    failure_buffer: FailureBuffer = {}
    accepted = 0
    for i in range(n_iterations):
        if i > 0 and delay > 0:
            time.sleep(delay)

        parent_entry = _select_parent(ARCHIVE, temperature=parent_temperature)
        desc = parent_entry.get("description", "?")
        parent_spread = _entry_spread(parent_entry)
        parent_entry["n_children_attempted"] = int(
            parent_entry.get("n_children_attempted") or 0
        ) + 1

        iter_mode, code_kind = _resolve_iter_mode(mode)
        kind_label = f"+{code_kind}" if iter_mode == "code" else ""
        header = (
            f"{_C_BOLD}[{i+1}/{n_iterations}]{_C_RESET} {iter_mode}{kind_label}  "
            f"parent (gen={parent_entry.get('generation', 0)}, "
            f"spread={parent_spread:+.2f}): {desc}"
        )
        print(f"\n{header}")

        result = run_mortar_step(
            parent_entry, ARCHIVE, panel=panel,
            n_games=n_games_per_eval,
            mode=iter_mode,
            code_kind=code_kind,
            admit_all=admit_all,
            failure_buffer=failure_buffer,
        )
        if result is None:
            # run_mortar_step has already printed a tagged reason.
            save_archive(archive_path)  # persist the bumped attempt counter
            continue

        parent_entry["n_children_admitted"] = int(
            parent_entry.get("n_children_admitted") or 0
        ) + 1
        key = _config_key(result["config_snapshot"])
        if key in ARCHIVE:
            print(f"  {_TAG_SKIPPED} duplicate config")
            continue

        ARCHIVE[key] = result
        accepted += 1
        f = result["fitness"]
        suffix = f"  code:{result['code_key']}" if result.get("code_key") else ""
        print(f"  {_TAG_ACCEPTED} skill_spread={f['skill_spread']*100:+.1f}%  gen={result['generation']}{suffix}")
        save_archive(archive_path)

    print(f"\n{_C_BOLD}Done.{_C_RESET} {accepted}/{n_iterations} configs accepted. Archive size: {len(ARCHIVE)}.")
    print(f"Archive saved to {archive_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MORTAR — Minesweeper mechanic evolution")
    p.add_argument("--iterations", type=int,   default=10,            help="Number of mutation steps (default: 10)")
    p.add_argument("--games",      type=int,   default=10,            help="Games per agent per config (default: 20)")
    p.add_argument("--archive",    default="archive.json",            help="Archive file path (default: archive.json)")
    p.add_argument("--delay",      type=float, default=10.0,          help="Seconds between iterations for rate limiting (default: 10)")
    p.add_argument("--agents",     nargs="+",  default=DEFAULT_AGENTS,
                   help=f"Agents in evaluation panel (default: {' '.join(DEFAULT_AGENTS)})")
    p.add_argument("--mode",       default="mixed", choices=["param", "code", "mixed"],
                   help="Mutation mode: 'param' tunes GameConfig fields only, "
                        "'code' generates new MineBehavior/RevealStrategy classes, "
                        "'mixed' alternates (default: mixed)")
    p.add_argument("--admit-all",  action="store_true",
                   help="Skip the skill-spread admission gate; admit every parseable "
                        "mutation regardless of whether any agent beats random. Use to "
                        "explore the wild edge of the mechanic space.")
    p.add_argument("--parent-temperature", type=float, default=0.2,
                   help="Softmax temperature for parent selection over skill_spread. "
                        "Lower = more elitist (high-spread parents dominate); higher = "
                        "closer to uniform random. Set to 0 for argmax (deterministic "
                        "best). Default: 0.2.")
    args = p.parse_args()

    run_mortar_loop(
        n_iterations=args.iterations,
        n_games_per_eval=args.games,
        archive_path=args.archive,
        delay=args.delay,
        agent_names=args.agents,
        mode=args.mode,
        admit_all=args.admit_all,
        parent_temperature=args.parent_temperature,
    )
