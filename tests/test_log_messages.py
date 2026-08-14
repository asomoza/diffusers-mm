"""Every runtime message string must be ASCII-only.

``logging`` encodes with the stream's encoding, so a non-ASCII character in a
message raises ``UnicodeEncodeError`` inside the handler on a legacy-code-page
console and the record is dropped, not mangled — the message is simply lost.

Docstrings and comments are deliberately not covered: they never reach a stream.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "diffusers_mm"

# Attribute calls whose string arguments reach a stream handler or a user.
_EMITTING_ATTRS = frozenset({"debug", "info", "warning", "error", "exception", "critical", "warn"})


def _docstring_ids(tree: ast.Module) -> set[int]:
    """``id()`` of every docstring Constant, so they can be excluded."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                if isinstance(first.value.value, str):
                    ids.add(id(first.value))
    return ids


def _is_emitting(node: ast.AST) -> bool:
    """True for ``raise``, ``logger.<level>(...)`` and ``SomeError/Warning(...)`` calls."""
    if isinstance(node, ast.Raise):
        return True
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr in _EMITTING_ATTRS
    if isinstance(func, ast.Name):
        return func.id.endswith("Error") or func.id.endswith("Warning")
    return False


def _non_ascii_runtime_strings(path: pathlib.Path) -> list[tuple[int, str]]:
    """``(lineno, offending characters)`` for every emitted string in *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = _docstring_ids(tree)
    found: dict[int, str] = {}
    for node in ast.walk(tree):
        if not _is_emitting(node):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Constant) or not isinstance(sub.value, str):
                continue
            if id(sub) in skip:
                continue
            bad = sorted({c for c in sub.value if ord(c) > 127})
            if bad:
                found[sub.lineno] = "".join(bad)
    return sorted(found.items())


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: p.name)
def test_runtime_messages_are_ascii(path: pathlib.Path) -> None:
    offenders = _non_ascii_runtime_strings(path)
    assert not offenders, "non-ASCII in runtime message(s), breaks logging on legacy consoles: " + "; ".join(
        f"{path.name}:{lineno} contains {chars.encode('unicode_escape').decode()}" for lineno, chars in offenders
    )


def test_detector_catches_a_planted_offender(tmp_path: pathlib.Path) -> None:
    """The AST walk must flag a log call and ignore a docstring."""
    planted = tmp_path / "planted.py"
    planted.write_text(
        '"""Docstring with an em dash — allowed."""\n'
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        "def f():\n"
        '    """Another docstring → allowed."""\n'
        '    logger.info("pinned %d → %d", 1, 2)\n',
        encoding="utf-8",
    )
    assert _non_ascii_runtime_strings(planted) == [(6, "→")]
