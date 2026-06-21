from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from docode.artifacts.github import CommandResult, GitHubExportRequest, GitHubExporter


class GitHubExporterTests(IsolatedAsyncioTestCase):
    async def test_skips_when_not_configured(self) -> None:
        exporter = GitHubExporter(enabled=False)
        result = await exporter.export_pull_request(
            GitHubExportRequest(
                repo="zeturn/example",
                branch="docode/job-1",
                base_branch="main",
                title="DoCode job",
                body="summary",
                patch_path="/tmp/patch.diff",
            )
        )
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.reason, "github_export_not_configured")

    async def test_enabled_exporter_runs_gh_flow(self) -> None:
        with TemporaryDirectory() as tmp:
            patch_path = Path(tmp) / "patch.diff"
            patch_path.write_text("diff --git a/a b/a\n+change\n", encoding="utf-8")
            calls: list[list[str]] = []

            async def fake_runner(command: list[str], cwd: Path | None) -> CommandResult:
                _ = cwd
                calls.append(command)
                if command[:3] == ["git", "status", "--short"]:
                    return CommandResult(" M a\n", "", 0)
                if command[:3] == ["gh", "pr", "create"]:
                    return CommandResult("https://github.com/zeturn/example/pull/1\n", "", 0)
                return CommandResult("", "", 0)

            exporter = GitHubExporter(enabled=True, work_dir=Path(tmp) / "gh", command_runner=fake_runner)
            result = await exporter.export_pull_request(
                GitHubExportRequest(
                    repo="zeturn/example",
                    branch="docode/job-1",
                    base_branch="main",
                    title="DoCode job",
                    body="summary",
                    patch_path=str(patch_path),
                )
            )

            self.assertEqual(result.status, "created")
            self.assertEqual(result.pull_request_url, "https://github.com/zeturn/example/pull/1")
            self.assertEqual(calls[0][:3], ["gh", "repo", "clone"])
            self.assertIn(["git", "apply", str(patch_path)], calls)
            self.assertEqual(calls[-1][:3], ["gh", "pr", "create"])
