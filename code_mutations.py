"""
code_mutations.py — Runtime compilation, validation, and registration of
LLM-generated MineBehavior / RevealStrategy classes.

This module lets MORTAR extend the mechanic registries with classes the LLM
authors as Python source, rather than only mutating GameConfig parameters.

Sandbox posture: in-process exec() with curated globals + AST denylist + a
SIGALRM-based smoke-test timeout. This is NOT a security boundary — a
determined snippet can still escape via attribute introspection. Treat
archive.json with a non-null code_source field as executable code; only run
MORTAR against trusted models. Subprocess isolation is the correct follow-up.

POSIX-only: the smoke-test wallclock guard uses signal.SIGALRM. On non-POSIX
platforms the timeout is silently skipped (a runaway snippet hangs the loop).
"""

from __future__ import annotations
import ast
import builtins as _builtins_mod
import dataclasses
import hashlib
import json
import os
import random as _random_mod
import signal
import traceback
from collections import deque as _deque

from agents.random_agent import RandomAgent
from config import GameConfig
from game import MinesweeperGame, GameState
from mine_behaviors import MineBehavior, MINE_BEHAVIORS
from reveal_strategies import RevealStrategy, REVEAL_STRATEGIES


# Per-kind: (abc_class, registry_dict, GameConfig field name)
KIND_SPEC: dict[str, tuple[type, dict, str]] = {
    "mine_behavior":   (MineBehavior,   MINE_BEHAVIORS,    "mine_behavior"),
    "reveal_strategy": (RevealStrategy, REVEAL_STRATEGIES, "reveal_strategy"),
}


class CodeValidationError(Exception):
    """Raised when generated source fails any validation step."""


# ---------------------------------------------------------------------------
# Static analysis (AST denylist)
# ---------------------------------------------------------------------------

_FORBIDDEN_NAMES = {
    "__import__", "eval", "exec", "compile", "open",
    "globals", "locals", "vars", "__builtins__",
}

# Attribute access that enables introspection escapes. Class definitions still
# work because `def __init__` is a FunctionDef, not an Attribute access.
_FORBIDDEN_ATTRS = {
    "__class__", "__subclasses__", "__bases__", "__mro__",
    "__globals__", "__dict__", "__builtins__", "__import__",
    "__getattribute__", "__setattr__", "__delattr__",
    "__class_getitem__", "__reduce__", "__reduce_ex__",
}


def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise CodeValidationError(
                f"import statements are not permitted "
                f"(line {getattr(node, 'lineno', '?')})"
            )
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise CodeValidationError(
                f"forbidden name {node.id!r} "
                f"(line {getattr(node, 'lineno', '?')})"
            )
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRS:
            raise CodeValidationError(
                f"forbidden attribute {'.' + node.attr!r} "
                f"(line {getattr(node, 'lineno', '?')})"
            )


# ---------------------------------------------------------------------------
# Curated execution environment
# ---------------------------------------------------------------------------

_SAFE_BUILTIN_NAMES = (
    "range", "len", "min", "max", "abs", "sum", "all", "any",
    "enumerate", "zip", "reversed", "sorted", "map", "filter",
    "list", "tuple", "dict", "set", "frozenset",
    "int", "float", "bool", "str",
    "isinstance", "hasattr", "getattr", "callable",
    "True", "False", "None", "print", "repr",
    # Class machinery + standard exceptions.
    "__build_class__", "__name__", "object", "type", "super",
    "staticmethod", "classmethod", "property",
    "Exception", "ValueError", "TypeError", "RuntimeError", "KeyError",
    "IndexError", "AttributeError", "ZeroDivisionError", "StopIteration",
)

_SAFE_BUILTINS = {
    name: getattr(_builtins_mod, name)
    for name in _SAFE_BUILTIN_NAMES
    if hasattr(_builtins_mod, name)
}


def _build_globals(kind: str) -> dict:
    abc_cls, _, _ = KIND_SPEC[kind]
    return {
        "__builtins__":   _SAFE_BUILTINS,
        "__name__":       f"<{kind}>",
        "random":         _random_mod,
        "deque":          _deque,
        abc_cls.__name__: abc_cls,
    }


# ---------------------------------------------------------------------------
# Compilation, registration, smoke test
# ---------------------------------------------------------------------------

def code_key(source: str) -> str:
    """Stable key derived from source bytes — same source always maps to same key."""
    return "gen-" + hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]


def _find_subclass(local_ns: dict, abc_cls: type) -> type:
    candidates = [
        v for v in local_ns.values()
        if isinstance(v, type) and issubclass(v, abc_cls) and v is not abc_cls
    ]
    if len(candidates) == 0:
        raise CodeValidationError(
            f"no subclass of {abc_cls.__name__} found in generated source"
        )
    if len(candidates) > 1:
        names = ", ".join(c.__name__ for c in candidates)
        raise CodeValidationError(
            f"expected exactly one subclass of {abc_cls.__name__}, "
            f"found {len(candidates)}: {names}"
        )
    return candidates[0]


_SMOKE_BOARD = dict(rows=8, cols=8, mine_count=10)
_SMOKE_TURN_CAP = 30
_SMOKE_TIMEOUT_SECONDS = 5


def _smoke_test(cls: type, kind: str) -> None:
    """
    Run two short games with the candidate class temporarily registered. Raises
    CodeValidationError on crash or timeout. Restores registry state on the
    way out so a failed smoke test never leaks a temp registration.
    """
    _, registry, field = KIND_SPEC[kind]
    temp_key = "__smoketest__"
    registry[temp_key] = cls
    try:
        cfg = dataclasses.replace(
            GameConfig(**_SMOKE_BOARD, safe_first_click=True),
            **{field: temp_key},
        )
        for seed in (0, 1):
            _run_one_smoke_game(cfg, seed)
    finally:
        registry.pop(temp_key, None)


def _run_one_smoke_game(cfg: GameConfig, seed: int) -> None:
    has_alarm = hasattr(signal, "SIGALRM")

    def _on_alarm(signum, frame):
        raise TimeoutError(f"smoke test exceeded {_SMOKE_TIMEOUT_SECONDS}s")

    prev_handler = None
    if has_alarm:
        prev_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(_SMOKE_TIMEOUT_SECONDS)
    try:
        agent = RandomAgent(seed=seed)
        game = MinesweeperGame(cfg, seed=seed)
        turns = 0
        while (
            game.get_state() in (GameState.PENDING, GameState.ACTIVE)
            and turns < _SMOKE_TURN_CAP
        ):
            action, r, c = agent.choose_action(game)
            if action == "reveal":
                game.reveal(r, c)
            else:
                game.flag(r, c)
            turns += 1
    finally:
        if has_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)


def compile_and_register(source: str, kind: str) -> tuple[str, type]:
    """
    Compile → AST-validate → exec → locate subclass → smoke test → register.

    Idempotent: if `code_key(source)` is already in the registry, returns the
    existing entry without re-execing. Raises CodeValidationError on any
    validation failure.
    """
    if kind not in KIND_SPEC:
        raise CodeValidationError(f"unknown kind: {kind!r}")
    abc_cls, registry, _ = KIND_SPEC[kind]

    key = code_key(source)
    if key in registry:
        return key, registry[key]

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise CodeValidationError(f"syntax error: {e}") from e

    _validate_ast(tree)

    code_obj = compile(tree, f"<{key}>", "exec")
    g = _build_globals(kind)
    local_ns: dict = {}
    try:
        exec(code_obj, g, local_ns)
    except Exception as e:
        raise CodeValidationError(
            f"exec failed: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        ) from e

    cls = _find_subclass(local_ns, abc_cls)

    try:
        _smoke_test(cls, kind)
    except CodeValidationError:
        raise
    except Exception as e:
        raise CodeValidationError(
            f"smoke test crashed: {type(e).__name__}: {e}"
        ) from e

    registry[key] = cls
    return key, cls


def unregister(key: str, kind: str) -> None:
    """Drop a previously-registered generated class. No-op if absent or for
    builtin keys (we never want to wipe e.g. 'static')."""
    if kind not in KIND_SPEC:
        return
    if not key.startswith("gen-"):
        return
    _, registry, _ = KIND_SPEC[kind]
    registry.pop(key, None)


def register_all_from_archive(archive: dict) -> int:
    """
    Recompile-and-register every entry whose `code_kind` is set. Entries that
    fail to recompile (e.g. ABC was renamed, smoke test now crashes) are
    dropped from `archive` in-place with a logged warning.

    Returns the count successfully reregistered.
    """
    registered = 0
    drop_keys: list[str] = []
    for ak, entry in archive.items():
        kind = entry.get("code_kind")
        source = entry.get("code_source")
        if not kind or not source:
            continue
        try:
            key, _ = compile_and_register(source, kind)
        except CodeValidationError as e:
            desc = entry.get("description", "?")
            print(f"  Warning: dropping archive entry {desc!r} — {e}")
            drop_keys.append(ak)
            continue
        # Sanity: stored code_key should match the recomputed one.
        if entry.get("code_key") != key:
            entry["code_key"] = key
        registered += 1
    for ak in drop_keys:
        archive.pop(ak, None)
    return registered


def load_archive_file(path: str = "archive.json") -> list[dict]:
    """Read archive.json and return the raw list of entries. Returns [] if missing."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)
