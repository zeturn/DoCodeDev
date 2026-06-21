from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from docode.agent.loop import CodingAgentLoop
from docode.agent.stop_policy import StopPolicy
from docode.agent.verifier import CodingVerifier
from docode.artifacts.exporter import ArtifactExporter
from docode.dobox.types import ToolResult
from docode.llm.runtime import AgentDecision, LLMUsageMeter
from docode.storage.models import CodingJob, JobStatus, new_id
from docode.storage.repository import InMemoryJobRepository


class ScriptedLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.contexts: list[str] = []

    async def decide(self, *, system, messages, tools, context):
        self.contexts.append(context)
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "done\n"})
        return AgentDecision(type="final_candidate", summary="Updated README.")


class CancellingLLM:
    def __init__(self, repo: InMemoryJobRepository, job_id: str) -> None:
        self.repo = repo
        self.job_id = job_id

    async def decide(self, *, system, messages, tools, context):
        _ = system, messages, tools, context
        await self.repo.update_job(self.job_id, status=JobStatus.STOPPED, failure_reason="cancelled")
        return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "should not write\n"})


class BudgetBurningLLM:
    def __init__(self, meter: LLMUsageMeter, *, cost: float = 0.0) -> None:
        self.meter = meter
        self.cost = cost

    async def decide(self, *, system, messages, tools, context):
        _ = system, messages, tools, context
        self.meter.record_text_call(prompt="x" * 100, response='{"type":"tool_call"}', cost=self.cost)
        return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "over budget\n"})


class FlakyLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, *, system, messages, tools, context):
        _ = system, messages, tools, context
        self.calls += 1
        if self.calls == 1:
            raise ValueError("invalid json from provider")
        if self.calls == 2:
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "recovered\n"})
        return AgentDecision(type="final_candidate", summary="Recovered from a model error and updated README.")


class MissingSummaryLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.observed_summary_feedback = False

    async def decide(self, *, system, messages, tools, context):
        _ = system, tools, context
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "done\n"})
        if self.calls == 2:
            return AgentDecision(type="final_candidate", summary="")
        self.observed_summary_feedback = any(
            message.get("kind") == "feedback" and "final_summary_missing" in str(message.get("content"))
            for message in messages
        )
        return AgentDecision(type="final_candidate", summary="Updated README after providing a final summary.")


class ToolRepairLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.observed_tool_error = False

    async def decide(self, *, system, messages, tools, context):
        _ = system, tools, context
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "first attempt\n"})
        if self.calls == 2:
            self.observed_tool_error = any(
                message.get("role") == "tool" and message.get("exit_code") == 1 and "sandbox write unavailable" in str(message.get("output"))
                for message in messages
            )
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "repaired\n"})
        return AgentDecision(type="final_candidate", summary="Recovered from a transient tool error and updated README.")


class UnusableDecisionLLM:
    async def decide(self, *, system, messages, tools, context):
        _ = system, messages, tools, context
        return AgentDecision(type="nonsense")


class FakeTools:
    def __init__(self, *, fail_first_write: bool = False) -> None:
        self.files: dict[str, str] = {}
        self.call_count = 0
        self.fail_first_write = fail_first_write

    def definitions(self):
        return []

    async def call(self, tool_name: str, args: dict[str, object]) -> ToolResult:
        self.call_count += 1
        if tool_name == "write_file":
            if self.fail_first_write and self.call_count == 1:
                raise RuntimeError("sandbox write unavailable")
            self.files[str(args["path"])] = str(args["content"])
            return ToolResult(tool="write_file", output="wrote file")
        raise AssertionError(tool_name)

    async def list_files(self, path: str = ".") -> ToolResult:
        return ToolResult(tool="list_files", output="README.md\ngo.mod\n")

    async def read_file(self, path: str) -> ToolResult:
        if path == "README.md":
            return ToolResult(tool="read_file", output="# Example\n")
        if path == "go.mod":
            return ToolResult(tool="read_file", output="module example\n")
        return ToolResult(tool="read_file", output="", exit_code=1)

    async def git_status(self) -> ToolResult:
        return ToolResult(tool="git_status", output=" M README.md\n" if self.files else "")

    async def git_diff(self) -> ToolResult:
        output = "diff --git a/README.md b/README.md\n+done\n" if self.files else ""
        return ToolResult(tool="git_diff", output=output)

    async def run_tests(self) -> ToolResult:
        return ToolResult(tool="run_tests", output="ok", metadata={"detected": True, "command": "go test ./..."})

    async def run_build(self) -> ToolResult:
        return ToolResult(tool="run_build", output="ok", metadata={"detected": True, "command": "go build ./..."})

    async def run_lint(self) -> ToolResult:
        return ToolResult(tool="run_lint", output="no lint command detected", metadata={"detected": False})

    async def detect_test_command(self) -> str:
        return "go test ./..."

    async def detect_build_command(self) -> str:
        return "go build ./..."

    async def detect_lint_command(self):
        return None


class AgentLoopTests(IsolatedAsyncioTestCase):
    async def test_agent_loop_tools_verifies_and_exports_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme"))
            tools = FakeTools()
            llm = ScriptedLLM()

            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(tmp_path, repo),
                stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60),
            )
            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertEqual(result.result_summary, "Updated README.")
            artifacts = await repo.list_artifacts(job.id)
            self.assertEqual({artifact.kind for artifact in artifacts}, {"patch", "report", "log", "result", "zip"})
            self.assertEqual(result.artifact_id, next(artifact.id for artifact in artifacts if artifact.kind == "result"))
            self.assertTrue((tmp_path / job.id / "patch.diff").read_text(encoding="utf-8").startswith("diff --git"))
            self.assertTrue((tmp_path / job.id / "workspace.zip").exists())
            result_payload = json.loads((tmp_path / job.id / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result_payload["status"], "succeeded")
            self.assertEqual(result_payload["changed_files"], ["README.md"])
            self.assertEqual(
                result_payload["artifacts"],
                {
                    "patch": "patch.diff",
                    "report": "final_report.md",
                    "result": "result.json",
                    "zip": "workspace.zip",
                    "log": "test_log.txt",
                },
            )
            checks_by_name = {check["name"]: check for check in result_payload["checks"]}
            self.assertIn("git_status", checks_by_name)
            self.assertEqual(checks_by_name["test"]["command"], "go test ./...")
            self.assertIn("Detected commands: test=go test ./..., build=go build ./..., lint=not detected", llm.contexts[0])

            steps = await repo.list_steps(job.id)
            bootstrap = next(step for step in steps if step.content.get("type") == "bootstrap")
            self.assertEqual(bootstrap.content["detected_commands"]["test"], "go test ./...")
            self.assertIn("`go build ./...` exits successfully.", bootstrap.content["acceptance_criteria"])
            decisions = [step for step in steps if step.content.get("type") == "llm_decision"]
            self.assertEqual(decisions[0].kind, "llm")
            self.assertEqual(decisions[0].content["tool"], "write_file")
            tool_call = next(step for step in steps if step.content.get("type") == "tool_call")
            self.assertEqual(tool_call.content["tool"], "write_file")
            self.assertEqual(tool_call.content["args"]["content"]["bytes"], len("done\n".encode("utf-8")))
            tool_result = next(step for step in steps if step.content.get("type") == "tool_result")
            self.assertEqual(tool_result.content["summary"], "wrote file")

    async def test_agent_loop_observes_cancel_before_tool_call(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme"))
            tools = FakeTools()

            loop = CodingAgentLoop(
                llm=CancellingLLM(repo, job.id),
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.STOPPED)
            self.assertEqual(tools.call_count, 0)
            self.assertIsNotNone(result.artifact_id)
            artifacts = await repo.list_artifacts(job.id)
            self.assertEqual({artifact.kind for artifact in artifacts}, {"report", "log", "result", "zip"})
            self.assertEqual(result.artifact_id, next(artifact.id for artifact in artifacts if artifact.kind == "result"))
            result_payload = json.loads((Path(tmp) / job.id / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result_payload["status"], "stopped")
            self.assertEqual(result_payload["stopped_reason"], "cancelled")
            steps = await repo.list_steps(job.id)
            self.assertTrue(any(step.content.get("type") == "cancelled_observed" for step in steps))
            self.assertTrue(any(step.content.get("type") == "stopped_artifacts_exported" for step in steps))

    async def test_agent_loop_stops_before_tool_when_llm_budget_is_exhausted(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme", max_llm_tokens=1))
            tools = FakeTools()
            usage = LLMUsageMeter()

            loop = CodingAgentLoop(
                llm=BudgetBurningLLM(usage),
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60, max_llm_tokens=1),
                usage_meter=usage,
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.FAILED)
            self.assertEqual(result.failure_reason, "max_llm_tokens_exceeded")
            self.assertEqual(tools.call_count, 0)
            steps = await repo.list_steps(job.id)
            llm_step = next(step for step in steps if step.kind == "llm")
            self.assertGreater(llm_step.content["usage"]["total_tokens"], 1)

    async def test_agent_loop_stops_before_tool_when_llm_cost_budget_is_exhausted(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme", max_llm_cost=0.01))
            tools = FakeTools()
            usage = LLMUsageMeter()

            loop = CodingAgentLoop(
                llm=BudgetBurningLLM(usage, cost=0.02),
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60, max_llm_cost=0.01),
                usage_meter=usage,
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.FAILED)
            self.assertEqual(result.failure_reason, "max_llm_cost_exceeded")
            self.assertEqual(tools.call_count, 0)
            steps = await repo.list_steps(job.id)
            llm_step = next(step for step in steps if step.kind == "llm")
            self.assertEqual(llm_step.content["usage"]["cost"], 0.02)

    async def test_agent_loop_recovers_from_transient_llm_error(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme"))
            tools = FakeTools()

            loop = CodingAgentLoop(
                llm=FlakyLLM(),
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertEqual(result.result_summary, "Recovered from a model error and updated README.")
            self.assertEqual(tools.files["README.md"], "recovered\n")
            steps = await repo.list_steps(job.id)
            llm_error = next(step for step in steps if step.content.get("type") == "llm_error")
            self.assertEqual(llm_error.content["reason"], "llm_decision_failed")
            self.assertIn("invalid json", llm_error.content["detail"])

    async def test_agent_loop_requires_final_candidate_summary_before_success(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme"))
            tools = FakeTools()
            llm = MissingSummaryLLM()

            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertEqual(result.result_summary, "Updated README after providing a final summary.")
            self.assertTrue(llm.observed_summary_feedback)
            steps = await repo.list_steps(job.id)
            llm_errors = [step for step in steps if step.content.get("type") == "llm_error"]
            self.assertEqual(llm_errors[0].content["reason"], "final_summary_missing")
            verifier_steps = [step for step in steps if step.kind == "verifier"]
            self.assertEqual(len(verifier_steps), 1)
            result_payload = json.loads((Path(tmp) / job.id / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result_payload["summary"], "Updated README after providing a final summary.")

    async def test_agent_loop_recovers_from_transient_tool_error(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme"))
            tools = FakeTools(fail_first_write=True)
            llm = ToolRepairLLM()

            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertEqual(result.result_summary, "Recovered from a transient tool error and updated README.")
            self.assertEqual(tools.files["README.md"], "repaired\n")
            self.assertTrue(llm.observed_tool_error)
            steps = await repo.list_steps(job.id)
            tool_results = [step for step in steps if step.content.get("type") == "tool_result"]
            self.assertEqual(len(tool_results), 2)
            self.assertEqual(tool_results[0].content["exit_code"], 1)
            self.assertIn("sandbox write unavailable", tool_results[0].content["output"])
            self.assertEqual(tool_results[0].content["metadata"]["exception_type"], "RuntimeError")
            self.assertEqual(tool_results[1].content["exit_code"], 0)

    async def test_agent_loop_stops_after_repeated_unusable_model_output(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme"))
            tools = FakeTools()

            loop = CodingAgentLoop(
                llm=UnusableDecisionLLM(),
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=10, max_runtime_seconds=60, max_consecutive_failures=2),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.FAILED)
            self.assertEqual(result.failure_reason, "max_consecutive_failures_exceeded")
            self.assertEqual(tools.call_count, 0)
            self.assertIsNotNone(result.artifact_id)
            artifacts = await repo.list_artifacts(job.id)
            self.assertEqual({artifact.kind for artifact in artifacts}, {"report", "log", "result", "zip"})
            self.assertEqual(result.artifact_id, next(artifact.id for artifact in artifacts if artifact.kind == "result"))
            self.assertTrue((Path(tmp) / job.id / "failure_report.md").exists())
            result_payload = json.loads((Path(tmp) / job.id / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result_payload["status"], "failed")
            self.assertEqual(result_payload["failure_reason"], "max_consecutive_failures_exceeded")
            self.assertEqual(result_payload["changed_files"], [])
            steps = await repo.list_steps(job.id)
            llm_errors = [step for step in steps if step.content.get("type") == "llm_error"]
            self.assertEqual(len(llm_errors), 2)
            self.assertEqual(llm_errors[0].content["reason"], "model_returned_unusable_decision")
