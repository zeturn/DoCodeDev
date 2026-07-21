from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from docode.agent.loop import CodingAgentLoop
from docode.agent.quality_gate import QualityGate
from docode.agent.stop_policy import StopPolicy
from docode.agent.verifier import VerificationResult
from docode.artifacts.exporter import ArtifactExporter
from docode.dobox.types import ToolResult
from docode.llm.runtime import AgentDecision
from docode.storage.models import CodingJob, JobStatus, new_id
from docode.storage.repository import InMemoryJobRepository
from tests.support.repository import RecordingRepository


FORBIDDEN_SMOKE_STRINGS = (
    "GitHub Trends",
    "GitHub Trending",
    "owner/repo",
    "Box-row",
    "stars today",
    "crawler.py",
)


class ReadmeSmokeLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, *, system, messages, tools, context):
        _ = system, messages, tools, context
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(
                type="tool_call",
                tool_name="write_file",
                args={
                    "path": "README.md",
                    "content": "# Readme Smoke Fixture\n\nThis fixture starts with a short project note.\n\nThe smoke job added this sentence.\n",
                },
            )
        return AgentDecision(type="final_candidate", summary="Updated README.md with one sentence.")


class FixtureReadmeTools:
    def __init__(self, fixture_root: Path) -> None:
        self.files = {
            path.relative_to(fixture_root).as_posix(): path.read_text(encoding="utf-8")
            for path in fixture_root.rglob("*")
            if path.is_file()
        }
        self.initial_files = dict(self.files)
        self.call_count = 0

    def definitions(self):
        return []

    def set_detected_command(self, name: str, command: str | None) -> None:
        _ = name, command

    async def call(self, tool_name: str, args: dict[str, object]) -> ToolResult:
        self.call_count += 1
        if tool_name == "write_file":
            path = normalize_path(str(args["path"]))
            self.files[path] = str(args["content"])
            return ToolResult(tool="write_file", output=f"wrote {path}", metadata={"path": path})
        if tool_name == "read_file":
            return await self.read_file(str(args["path"]))
        if tool_name == "list_files":
            return await self.list_files(str(args.get("path") or "."))
        if tool_name == "run_command":
            return ToolResult(tool="run_command", output="ok\n", metadata={"command": str(args.get("command") or "")})
        raise AssertionError(tool_name)

    async def list_files(self, path: str = ".") -> ToolResult:
        _ = path
        return ToolResult(tool="list_files", output="\n".join(sorted(self.files)) + "\n")

    async def read_file(self, path: str) -> ToolResult:
        normalized = normalize_path(path)
        if normalized not in self.files:
            return ToolResult(tool="read_file", output=f"{normalized} not found", exit_code=1, metadata={"path": normalized})
        return ToolResult(tool="read_file", output=self.files[normalized], metadata={"path": normalized})

    async def git_status(self) -> ToolResult:
        output = " M README.md\n" if self.files.get("README.md") != self.initial_files.get("README.md") else ""
        return ToolResult(tool="git_status", output=output)

    async def git_diff(self) -> ToolResult:
        if self.files.get("README.md") == self.initial_files.get("README.md"):
            return ToolResult(tool="git_diff", output="")
        return ToolResult(
            tool="git_diff",
            output=(
                "diff --git a/README.md b/README.md\n"
                "--- a/README.md\n"
                "+++ b/README.md\n"
                "@@\n"
                "+The smoke job added this sentence.\n"
            ),
        )

    async def run_tests(self) -> ToolResult:
        return ToolResult(tool="run_tests", output="no test command detected", metadata={"detected": False})

    async def run_build(self) -> ToolResult:
        return ToolResult(tool="run_build", output="no build command detected", metadata={"detected": False})

    async def run_lint(self) -> ToolResult:
        return ToolResult(tool="run_lint", output="no lint command detected", metadata={"detected": False})

    async def detect_test_command(self):
        return None

    async def detect_build_command(self):
        return None

    async def detect_lint_command(self):
        return None


class DiffAcceptingVerifier:
    async def verify(self, job, tools, evidence=None):
        _ = job, evidence
        status = await tools.git_status()
        diff = await tools.git_diff()
        return VerificationResult(
            passed=bool(diff.output.strip()),
            confidence=0.95,
            reason="README smoke diff verified.",
            required_fixes=[],
            git_status=status.output,
            git_diff=diff.output,
            status_result=status,
        )


class ReadmeSmokeJobTests(IsolatedAsyncioTestCase):
    async def test_readme_edit_job_runs_loop_records_steps_and_exports_result(self) -> None:
        fixture_root = Path(__file__).resolve().parent / "fixtures" / "repos" / "readme_edit"
        with TemporaryDirectory() as tmp:
            repo = RecordingRepository()
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="smoke",
                    instruction="Update README.md by adding one sentence.",
                )
            )
            tools = FixtureReadmeTools(fixture_root)

            loop = CodingAgentLoop(
                llm=ReadmeSmokeLLM(),
                tools=tools,
                verifier=DiffAcceptingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo, workspace_file_reader=tools.read_file),
                stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60),
                quality_gate=QualityGate(),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertIn(JobStatus.RUNNING, repo.status_updates)
            self.assertEqual(result.result_summary, "Updated README.md with one sentence.")
            self.assertIn("The smoke job added this sentence.", tools.files["README.md"])

            steps = await repo.list_steps(job.id)
            self.assertTrue(any(step.content.get("type") == "tool_call" and step.content.get("tool") == "write_file" for step in steps))
            self.assertTrue(any(step.content.get("type") == "tool_result" and step.content.get("tool") == "write_file" for step in steps))
            self.assertTrue(any(step.kind == "verifier" for step in steps))

            artifacts = await repo.list_artifacts(job.id)
            artifact_kinds = {artifact.kind for artifact in artifacts}
            self.assertIn("report", artifact_kinds)
            self.assertIn("result", artifact_kinds)
            self.assertIsNotNone(result.artifact_id)

            combined_steps = "\n".join(str(step.content) for step in steps)
            combined_files = "\n".join(tools.files.values())
            for forbidden in FORBIDDEN_SMOKE_STRINGS:
                self.assertNotIn(forbidden, combined_steps)
                self.assertNotIn(forbidden, combined_files)
            self.assertFalse(any(step.content.get("reason") == "active_repair_controller_forced_target_edit" for step in steps))


def normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    if normalized.startswith("/workspace/"):
        normalized = normalized[len("/workspace/") :]
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized
