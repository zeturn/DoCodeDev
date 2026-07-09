from __future__ import annotations

import json
from unittest import IsolatedAsyncioTestCase

from docode.agent.quality_gate import QualityGate
from docode.agent.quality_gate import detect_empty_markdown_sections
from docode.agent.quality_gate import detect_duplicate_python_implementations
from docode.agent.task_contract import TaskContract
from docode.dobox.types import ToolResult


class QualityGateTools:
    def __init__(self, *, artifact: object, diff: str | None = None, artifact_path: str = "data/github_trending.json") -> None:
        self.artifact = artifact
        self.artifact_path = artifact_path
        self.diff = diff or (
            "diff --git a/crawler.py b/crawler.py\n+print('crawler')\n"
            f"diff --git a/{artifact_path} b/{artifact_path}\n+[]\n"
        )

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        _ = command, cwd
        return ToolResult(tool="run_command", output="ok")

    async def git_status(self) -> ToolResult:
        return ToolResult(tool="git_status", output=f" A crawler.py\n A {self.artifact_path}\n")

    async def git_diff(self) -> ToolResult:
        return ToolResult(tool="git_diff", output=self.diff)

    async def read_file(self, path: str) -> ToolResult:
        if path == self.artifact_path:
            output = self.artifact if isinstance(self.artifact, str) else json.dumps(self.artifact)
            return ToolResult(tool="read_file", output=output, metadata={"path": path})
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

    async def test_prefers_full_repository_over_repository_name(self) -> None:
        result = await QualityGate().run(
            tools=QualityGateTools(
                artifact=[
                    {
                        "owner": "owner",
                        "repository_name": "repo",
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

    async def test_generic_json_records_do_not_require_github_fields(self) -> None:
        result = await QualityGate().run(
            tools=QualityGateTools(
                artifact=[{"id": 1, "name": "Ada"}, {"id": 2, "name": "Grace"}],
                artifact_path="out.json",
                diff="diff --git a/crawler.py b/crawler.py\n+print('crawler')\n"
                "diff --git a/out.json b/out.json\n+[{\"id\": 1, \"name\": \"Ada\"}]\n",
            ),
            task_contract=TaskContract(must_modify_files=["crawler.py"]),
            instruction="Fix crawler.py so it parses records and the CLI writes JSON to --output out.json.",
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.blockers(), [])

    async def test_github_crawler_still_requires_repository_fields(self) -> None:
        result = await QualityGate().run(
            tools=QualityGateTools(
                artifact=[{"id": 1, "name": "Ada"}],
                artifact_path="data/github_trending.json",
            ),
            task_contract=TaskContract(must_modify_files=["crawler.py", "data/github_trending.json"]),
            instruction="Build a GitHub Trending repository crawler that writes data/github_trending.json",
        )

        self.assertFalse(result.passed)
        self.assertTrue(any(issue.code == "json_required_field_empty" for issue in result.issues))

    async def test_missing_requested_json_artifact_fails(self) -> None:
        result = await QualityGate().run(
            tools=QualityGateTools(
                artifact=[{"id": 1, "name": "Ada"}],
                artifact_path="other.json",
                diff="diff --git a/crawler.py b/crawler.py\n+print('crawler')\n",
            ),
            task_contract=TaskContract(must_modify_files=["crawler.py"]),
            instruction="Fix crawler.py so it writes JSON to --output out.json.",
        )

        self.assertFalse(result.passed)
        self.assertTrue(any(issue.code == "json_artifact_missing" and issue.path == "out.json" for issue in result.issues))

    async def test_invalid_requested_json_artifact_fails(self) -> None:
        result = await QualityGate().run(
            tools=QualityGateTools(
                artifact="{not valid json",
                artifact_path="out.json",
                diff="diff --git a/crawler.py b/crawler.py\n+print('crawler')\n"
                "diff --git a/out.json b/out.json\n+{not valid json\n",
            ),
            task_contract=TaskContract(must_modify_files=["crawler.py"]),
            instruction="Fix crawler.py so it writes JSON to --output out.json.",
        )

        self.assertFalse(result.passed)
        self.assertTrue(any(issue.code == "json_artifact_invalid" for issue in result.issues))

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

    def test_markdown_parent_section_counts_child_heading_content(self) -> None:
        issues = detect_empty_markdown_sections(
            "README.md",
            "# GitHub Trends Crawler\n\n"
            "## Usage\n\n"
            "### Basic usage\n\n"
            "```bash\n"
            "python3 crawler.py --source fixtures/sample.html --output data/output.json --dry-run\n"
            "```\n",
        )

        self.assertEqual(issues, [])

    def test_markdown_duplicate_section_passes_when_one_has_content(self) -> None:
        issues = detect_empty_markdown_sections(
            "README.md",
            "# GitHub Trends Crawler\n\n"
            "## Usage\n\n"
            "## Output\n\n"
            "Writes structured GitHub trending repository records.\n\n"
            "## Usage\n\n"
            "### Basic usage\n\n"
            "```bash\n"
            "python3 crawler.py --source fixtures/sample.html --output data/output.json --dry-run\n"
            "python3 crawler.py --preflight\n"
            "```\n",
        )

        self.assertEqual(issues, [])

    def test_markdown_duplicate_section_blocks_when_all_are_thin(self) -> None:
        issues = detect_empty_markdown_sections(
            "README.md",
            "# GitHub Trends Crawler\n\n"
            "## Usage\n\n"
            "## Output\n\n"
            "Writes structured output.\n\n"
            "## Usage\n\n",
        )

        self.assertTrue(any(issue.code == "markdown_section_empty" and issue.path == "README.md" for issue in issues))

    def test_duplicate_entrypoint_ignores_unchanged_diff_context(self) -> None:
        issues = detect_duplicate_python_implementations(
            "diff --git a/crawler.py b/crawler.py\n"
            "@@\n"
            " if __name__ == \"__main__\":\n"
            "-    old_main()\n"
            "+    main()\n"
        )

        self.assertEqual(issues, [])

    def test_duplicate_entrypoint_blocks_multiple_added_entrypoints(self) -> None:
        issues = detect_duplicate_python_implementations(
            "diff --git a/crawler.py b/crawler.py\n"
            "+if __name__ == \"__main__\":\n"
            "+    main()\n"
            "+if __name__ == \"__main__\":\n"
            "+    other_main()\n"
        )

        self.assertTrue(any(issue.code == "duplicate_python_entrypoint" for issue in issues))
