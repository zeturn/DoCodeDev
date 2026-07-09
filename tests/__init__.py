"""Test package helpers."""

from __future__ import annotations

from pathlib import Path


def _remove_local_fixture_metadata() -> None:
    """Remove local metadata/binary files from diagnostic fixtures.

    Real DoBox fixture seeding uploads every file under each fixture directory as
    UTF-8 text. Developer machines may create binary metadata files there
    (.DS_Store, Icon\r, editor caches, etc.); those files are not part of the
    fixture and should not make real diagnostic tests fail before the agent loop
    starts.
    """

    root = Path(__file__).resolve().parent
    fixture_roots = [root / "fixtures" / "repos" / "diagnostic"]
    for fixture_root in fixture_roots:
        if not fixture_root.exists():
            continue
        for path in fixture_root.rglob("*"):
            if not path.is_file():
                continue
            if path.name.startswith(".") or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                _unlink_quietly(path)
                continue
            try:
                path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                _unlink_quietly(path)


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


_remove_local_fixture_metadata()
