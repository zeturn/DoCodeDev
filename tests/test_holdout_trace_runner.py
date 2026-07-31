from __future__ import annotations

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

# The runner script is not importable as a package; use its functions directly.
import importlib.util as _iu

_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_holdout_case.py"
_spec = _iu.spec_from_file_location("run_holdout_case", _RUNNER)
_runner = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_runner)


class WorkspaceDiffRegressionTests(TestCase):
    """The trace runner must produce a ``workspace.diff`` that is safe to
    write as strict UTF-8 even when the workspace contains binary VCS
    internals (``.git/index``), and must never include ``.git`` content."""

    def _make_git_init(self, root: Path) -> None:
        for args in (
            ["git", "init", "-q"],
            ["git", "config", "user.email", "test@test.test"],
            ["git", "config", "user.name", "Test"],
        ):
            subprocess.run(args, cwd=root, check=True, capture_output=True)

    def test_diff_excludes_git_and_passes_utf8_write_roundtrip(self) -> None:
        with TemporaryDirectory() as tmp:
            before = Path(tmp) / "before"
            before.mkdir()
            (before / "guidebook.md").write_text("before", encoding="utf-8")

            after = Path(tmp) / "after"
            after.mkdir()
            # Simulate git init producing binary .git/index
            self._make_git_init(after)
            # Also a real file edit
            (after / "guidebook.md").write_text("after", encoding="utf-8")

            diff = _runner._workspace_diff(before, after)

            # Must not raise UnicodeEncodeError when writing as utf-8
            out = Path(tmp) / "workspace.diff"
            out.write_text(diff, encoding="utf-8")

            self.assertGreater(out.stat().st_size, 0, "diff file is empty")
            self.assertIn("guidebook.md", diff)
            self.assertNotIn(".git", diff)
            self.assertIn("-before", diff)
            self.assertIn("+after", diff)

    def test_git_only_changes_produce_empty_diff(self) -> None:
        with TemporaryDirectory() as tmp:
            before = Path(tmp) / "before"
            before.mkdir()
            (before / "guidebook.md").write_text("same", encoding="utf-8")
            self._make_git_init(before)

            after = Path(tmp) / "after"
            after.mkdir()
            (after / "guidebook.md").write_text("same", encoding="utf-8")
            self._make_git_init(after)

            diff = _runner._workspace_diff(before, after)

            out = Path(tmp) / "diff.txt"
            out.write_text(diff, encoding="utf-8")

            self.assertEqual(out.stat().st_size, 0, "diff must be empty when only .git changed")

    def test_snapshot_excludes_vcs_and_cache(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            root.mkdir()
            (root / "guidebook.md").write_text("content", encoding="utf-8")
            dotgit = root / ".git"
            dotgit.mkdir()
            (dotgit / "index").write_bytes(b"\x00\x01\xff")
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "x.pyc").write_bytes(b"\xaa")
            (root / "main.pyc").write_bytes(b"\xbb")

            snap = _runner._snapshot_workspace(root)

            self.assertEqual(list(snap), ["guidebook.md"])

    def test_binary_file_becomes_placeholder_not_unicode_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            root.mkdir()
            (root / "readme.txt").write_text("hello", encoding="utf-8")
            (root / "data.bin").write_bytes(b"\x80\x81\xff")

            snap = _runner._snapshot_workspace(root)

            self.assertIn("readme.txt", snap)
            self.assertIn("data.bin", snap)
            self.assertIn("<binary", snap["data.bin"])
            self.assertIn("sha256=", snap["data.bin"])
            # Round-trip must succeed
            out = Path(tmp) / "out.json"
            import json

            out.write_text(json.dumps(snap), encoding="utf-8")
            self.assertGreater(out.stat().st_size, 0)
