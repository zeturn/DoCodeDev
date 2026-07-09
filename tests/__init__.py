"""Test package helpers."""

from __future__ import annotations

from pathlib import Path


def _remove_local_finder_metadata() -> None:
    """Remove local macOS Finder metadata from diagnostic fixtures.

    Real DoBox fixture seeding uploads every file under each fixture directory as
    UTF-8 text. Developer machines may create binary .DS_Store files there; those
    files are not part of the fixture and should not make real diagnostic tests
    fail before the agent loop starts.
    """

    root = Path(__file__).resolve().parent
    for junk in root.rglob(".DS_Store"):
        try:
            junk.unlink()
        except OSError:
            pass


_remove_local_finder_metadata()
