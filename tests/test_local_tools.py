from __future__ import annotations

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from tests.support.local_tools import DiagnosticLocalTools


class TestLocalToolsGitExclusion(IsolatedAsyncioTestCase):
    """After ``git init`` the workspace contains a ``.git`` directory with
    hook templates that include TODO comments. The deterministic test double
    must exclude those from every filesystem-facing tool."""

    async def test_snapshot_excludes_dotgit_after_git_init(self) -> None:
        with TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            _init_git(ws)
            (ws / "guidebook.md").write_text("# Guidebook", encoding="utf-8")
            tools = DiagnosticLocalTools(ws, test_command="echo ok")
            snap = tools.snapshot_files()
            self.assertIn("guidebook.md", snap)
            self.assertFalse(
                any(k == ".git" or k.startswith(".git/") for k in snap),
                f"snapshot_files leaked .git paths: {sorted(snap)}",
            )

    async def test_list_files_excludes_dotgit(self) -> None:
        with TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            _init_git(ws)
            (ws / "guidebook.md").write_text("# Guidebook", encoding="utf-8")
            tools = DiagnosticLocalTools(ws, test_command="echo ok")
            result = await tools.list_files(".")
            self.assertEqual(result.exit_code, 0)
            self.assertNotIn(".git/", result.output)
            self.assertNotIn(".git\n", result.output)

    async def test_search_excludes_dotgit(self) -> None:
        with TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            _init_git(ws)
            (ws / "guidebook.md").write_text("# Guidebook\nNeed to TODO later.\n", encoding="utf-8")
            tools = DiagnosticLocalTools(ws, test_command="echo ok")
            result = await tools.search("TODO")
            self.assertEqual(result.exit_code, 0)
            self.assertIn("guidebook.md", result.output)
            self.assertNotIn(".git", result.output)

    async def test_git_status_excludes_dotgit_after_real_file_edit(self) -> None:
        with TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            _init_git(ws)
            (ws / "guidebook.md").write_text("before", encoding="utf-8")
            tools = DiagnosticLocalTools(ws, test_command="echo ok")
            # simulate an edit
            (ws / "guidebook.md").write_text("after", encoding="utf-8")
            result = await tools.git_status()
            self.assertIn("guidebook.md", result.output)
            self.assertNotIn(".git", result.output)

    async def test_git_diff_excludes_dotgit_after_real_file_edit(self) -> None:
        with TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            _init_git(ws)
            (ws / "guidebook.md").write_text("before", encoding="utf-8")
            tools = DiagnosticLocalTools(ws, test_command="echo ok")
            (ws / "guidebook.md").write_text("after", encoding="utf-8")
            result = await tools.git_diff()
            self.assertIn("guidebook.md", result.output)
            self.assertNotIn(".git", result.output)

    async def test_snapshot_excludes_dotgit_when_not_a_git_repo(self) -> None:
        with TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            (ws / ".git").mkdir()
            (ws / ".git" / "nested").write_text("secret", encoding="utf-8")
            (ws / "guidebook.md").write_text("# Guidebook", encoding="utf-8")
            tools = DiagnosticLocalTools(ws, test_command="echo ok")
            snap = tools.snapshot_files()
            self.assertIn("guidebook.md", snap)
            self.assertFalse(
                any(k == ".git" or k.startswith(".git/") for k in snap),
                f"snapshot_files leaked .git paths: {sorted(snap)}",
            )


def _init_git(workspace: Path) -> None:
    subprocess.run(
        ["git", "init", "-q"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.test"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
