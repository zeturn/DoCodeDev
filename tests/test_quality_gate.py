from __future__ import annotations

import json
from unittest import IsolatedAsyncioTestCase

from docode.agent.quality_gate import QualityGate
from docode.agent.task_contract import TaskContract
from docode.dobox.types import ToolResult


class QualityGateTools:
    def __init__(self, *, artifact: object, diff: str | None = None) -> None:
        self.artifact = artifact
        self.diff = diff or (
            "diff --git a/crawler.py b/crawler.py\n+print('crawler')\n"
            "diff --git a/data/github_trending.json b/data/github_trending.json\n+[]\n"
        )

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        _ = command, cwd
        return ToolResult(tool="run_command", output="ok")

    async def git_status(self) -> ToolResult:
        return ToolResult(tool="git_status", output=" A crawler.py\n A data/github_trending.json\n")

    async def git_diff(self) -> ToolResult:
        return ToolResult(tool="git_diff", output=self.diff)

    async def read_file(self, path: str) -> ToolResult:
        if path == "data/github_trending.json":
            return ToolResult(tool="read_file", output=json.dumps(self.artifact), metadata={"path": path})
        return ToolResult(tool="read_file", output="", exit_code=1, metadata={"path": path})


class QualityGateTests(IsolatedAsyncioTestCase):
    async def test_blocks_dirty_github_repository_field(self) -> None:
        result = await QualityGate().run(
            tools=QualityGateTools(
                artifact=[
                    {
                        "repository": "owner/repo\n\n owner /",
                        "url": "https://github.com/owner/repo",
                    }
                ]
            ),
            task_contract=TaskContract(must_modify_files=["data/github_trending.json"]),
            instruction="Build a GitHub Trending crawler that writes data/github_trending.json",
        )

        self.assertFalse(result.passed)
        self.assertTrue(any(issue.code == "json_required_field_dirty" for issue in result.issues))
        self.assertTrue(any(issue.code == "json_repository_invalid_format" for issue in result.issues))
        self.assertEqual(result.samples[0].path, "data/github_trending.json")

    async def test_blocks_github_repository_url_mismatch(self) -> None:
        result = await QualityGate().run(
            tools=QualityGateTools(
                artifact=[
                    {
                        "repository": "owner/repo",
                        "url": "https://github.com/other/repo",
                    }
                ]
            ),
            task_contract=TaskContract(must_modify_files=["data/github_trending.json"]),
            instruction="Build a GitHub Trending crawler that writes data/github_trending.json",
        )

        self.assertFalse(result.passed)
        self.assertTrue(any(issue.code == "json_repository_url_mismatch" for issue in result.issues))

    async def test_passes_clean_github_repository_records(self) -> None:
        result = await QualityGate().run(
            tools=QualityGateTools(
                artifact=[
                    {
                        "repository": "owner/repo",
                        "url": "https://github.com/owner/repo",
                    }
                ]
            ),
            task_contract=TaskContract(must_modify_files=["data/github_trending.json"]),
            instruction="Build a GitHub Trending crawler that writes data/github_trending.json",
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.blockers(), [])

    async def test_blocks_undeclared_third_party_dependency(self) -> None:
        result = await QualityGate().run(
            tools=QualityGateTools(
                artifact=[{"repository": "owner/repo", "url": "https://github.com/owner/repo"}],
                diff="diff --git a/crawler.py b/crawler.py\n+import requests\n",
            ),
            task_contract=TaskContract(must_modify_files=["crawler.py", "data/github_trending.json"]),
            instruction="Build a GitHub Trending crawler that writes data/github_trending.json",
        )

        self.assertFalse(result.passed)
        self.assertTrue(any(issue.code == "undeclared_third_party_dependency" for issue in result.issues))
