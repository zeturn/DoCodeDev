from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from docode.agent.verifier import VerificationResult
from docode.artifacts.exporter import ArtifactExporter
from docode.artifacts.github import GitHubExportRequest, GitHubExportResult, GitHubExporter
from docode.dobox.types import ToolResult
from docode.storage.models import CodingJob, new_id
from docode.storage.repository import InMemoryJobRepository


class RecordingGitHubExporter(GitHubExporter):
    def __init__(self) -> None:
        super().__init__(enabled=True)
        self.request: GitHubExportRequest | None = None

    async def export_pull_request(self, request: GitHubExportRequest) -> GitHubExportResult:
        self.request = request
        return GitHubExportResult(
            status="created",
            branch_url="https://github.com/zeturn/example/tree/docode/job",
            pull_request_url="https://github.com/zeturn/example/pull/1",
        )


class ArtifactExporterTests(IsolatedAsyncioTestCase):
    async def test_pr_body_uses_actual_verification_report(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            github = RecordingGitHubExporter()
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="user-1",
                    instruction="update readme",
                    artifact_mode="pr",
                    github_repo="zeturn/example",
                )
            )
            verification = VerificationResult(
                passed=True,
                confidence=0.91,
                reason="Verified with tests.",
                required_fixes=[],
                git_status=" M README.md\n",
                git_diff="diff --git a/README.md b/README.md\n+done\n",
                status_result=ToolResult(tool="git_status", output=" M README.md\n"),
                test_result=ToolResult(tool="run_tests", output="ok", metadata={"detected": True, "command": "go test ./..."}),
                build_result=ToolResult(tool="run_build", output="ok", metadata={"detected": True, "command": "go build ./..."}),
                lint_result=ToolResult(tool="run_lint", output="not detected", metadata={"detected": False}),
            )

            artifacts = await ArtifactExporter(Path(tmp), repo, github_exporter=github).export_success(job, verification, "Updated README.")

            assert github.request is not None
            self.assertIn("- `README.md`", github.request.body)
            self.assertIn("Verification: Verified with tests.", github.request.body)
            self.assertIn("tests `go test ./...`: exit 0", github.request.body)
            self.assertIn("build `go build ./...`: exit 0", github.request.body)
            self.assertIn("pull_request", {artifact.kind for artifact in artifacts})

    async def test_failure_step_log_redacts_full_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="user-1", instruction="fix tests"))
            await repo.add_step(
                job.id,
                "tool",
                {
                    "type": "tool_result",
                    "tool": "run_command",
                    "exit_code": 1,
                    "summary": "tests failed",
                    "output": "failure\nSECRET_TOKEN=do-not-export",
                },
            )
            await repo.add_step(
                job.id,
                "verifier",
                {
                    "passed": False,
                    "git_status": " M SECRET_FILE\n",
                    "git_diff": "diff --git a/a b/a\n+SECRET_TOKEN=do-not-export\n",
                    "test": {"tool": "run_tests", "exit_code": 1, "output": "failed\nSECRET_TOKEN=do-not-export"},
                },
            )

            artifacts = await ArtifactExporter(Path(tmp), repo).export_failure(
                job,
                "max_iterations_exceeded",
                steps=await repo.list_steps(job.id),
            )

            log_artifact = next(artifact for artifact in artifacts if artifact.kind == "log")
            log = Path(log_artifact.path).read_text(encoding="utf-8")
            self.assertIn('"output_bytes"', log)
            self.assertIn('"git_diff_bytes"', log)
            self.assertIn('"git_status_bytes"', log)
            self.assertNotIn("SECRET_TOKEN", log)
            self.assertNotIn("SECRET_FILE", log)
            self.assertNotIn('"git_diff":', log)
            self.assertNotIn('"output":', log)

    async def test_success_without_git_diff_omits_patch_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="user-1", instruction="create result"))
            verification = VerificationResult(
                passed=True,
                confidence=0.86,
                reason="Workspace artifact exists.",
                required_fixes=[],
                git_status="fatal: not a git repository",
                git_diff="",
                status_result=ToolResult(tool="git_status", output="fatal: not a git repository", exit_code=128),
                workspace_result=ToolResult(tool="list_files", output="DOCODE_RESULT.md\n", exit_code=0),
            )

            artifacts = await ArtifactExporter(Path(tmp), repo).export_success(job, verification, "Created result file.")

            self.assertNotIn("patch", {artifact.kind for artifact in artifacts})
            self.assertFalse((Path(tmp) / job.id / "patch.diff").exists())
            result_payload = json.loads((Path(tmp) / job.id / "result.json").read_text(encoding="utf-8"))
            self.assertNotIn("patch", result_payload["artifacts"])

    async def test_terminal_truncated_diff_omits_patch_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="user-1", instruction="fix tests"))
            truncated_diff = "diff --git a/app.py b/app.py\n+partial\n"

            artifacts = await ArtifactExporter(Path(tmp), repo).export_failure(
                job,
                "sandbox_error",
                git_diff=truncated_diff,
                git_diff_truncated=True,
            )

            self.assertNotIn("patch", {artifact.kind for artifact in artifacts})
            self.assertFalse((Path(tmp) / job.id / "patch.diff").exists())
            report = (Path(tmp) / job.id / "failure_report.md").read_text(encoding="utf-8")
            self.assertIn("Diff collection was truncated; patch artifact omitted.", report)
            result_payload = json.loads((Path(tmp) / job.id / "result.json").read_text(encoding="utf-8"))
            self.assertNotIn("patch", result_payload["artifacts"])
            self.assertTrue(result_payload["git_diff"]["truncated"])
