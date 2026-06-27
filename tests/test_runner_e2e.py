from __future__ import annotations

import asyncio
import json
from zipfile import ZipFile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from docode.config import DocodeConfig
from docode.dobox.types import AgentSession, CommandResult, ProjectSandbox
from docode.llm.credentials import APICredCredentialResolver, RuntimeAuthorization
from docode.storage.models import CodingJob, JobStatus, new_id
from docode.storage.repository import InMemoryJobRepository
from docode.worker.runner import JobRunnerService, runtime_budget


class StatefulFakeDoBoxClient:
    def __init__(self) -> None:
        self.project_id = "project-1"
        self.files: dict[str, str] = {"README.md": "# Example\n"}
        self.commits: list[str] = []
        self.create_calls = 0
        self.session_calls = 0
        self.agent_session_ids: list[str | None] = []
        self.create_network_modes: list[str | None] = []
        self.deleted_projects: list[str] = []

    async def create_project(
        self,
        *,
        name: str,
        repo_url: str | None = None,
        branch: str | None = None,
        image: str | None = None,
        network_mode: str | None = None,
    ) -> ProjectSandbox:
        self.create_calls += 1
        self.create_network_modes.append(network_mode)
        return ProjectSandbox(
            project_id=self.project_id,
            sandbox_id="sandbox-1",
            raw={"name": name, "repo_url": repo_url, "branch": branch, "image": image, "network_mode": network_mode},
        )

    async def delete_project(self, project_id: str) -> None:
        self.deleted_projects.append(project_id)

    async def create_agent_session(self, project_id: str, name: str) -> AgentSession:
        self.session_calls += 1
        return AgentSession(session_id="7", raw={"id": 7, "project_id": project_id, "name": name})

    async def run_command(
        self,
        project_id: str,
        command,
        cwd: str = "/workspace",
        timeout_sec: int = 120,
        output_limit: int = 1_000_000,
        agent_session_id: str | None = None,
    ) -> CommandResult:
        self.agent_session_ids.append(agent_session_id)
        command_text = " ".join(command) if isinstance(command, list) else str(command)
        if "test -f" in command_text:
            return CommandResult("", 1)
        return CommandResult(f"ran {command_text}", 0)

    async def read_file(self, project_id: str, path: str, agent_session_id: str | None = None) -> str:
        self.agent_session_ids.append(agent_session_id)
        return self.files[path]

    async def write_file(self, project_id: str, path: str, content: str, agent_session_id: str | None = None) -> None:
        self.agent_session_ids.append(agent_session_id)
        self.files[path] = content

    async def list_files(self, project_id: str, path: str = ".", agent_session_id: str | None = None) -> CommandResult:
        self.agent_session_ids.append(agent_session_id)
        return CommandResult("\n".join(sorted(self.files)), 0)

    async def search(self, project_id: str, query: str, path: str = ".", agent_session_id: str | None = None) -> CommandResult:
        self.agent_session_ids.append(agent_session_id)
        matches = [f"{name}:1:{query}" for name, content in self.files.items() if query in content]
        return CommandResult("\n".join(matches), 0 if matches else 1)

    async def git_status(self, project_id: str, agent_session_id: str | None = None) -> CommandResult:
        self.agent_session_ids.append(agent_session_id)
        return CommandResult(" M DOCODE_RESULT.md\n" if "DOCODE_RESULT.md" in self.files else "", 0)

    async def git_diff(self, project_id: str, agent_session_id: str | None = None) -> str:
        self.agent_session_ids.append(agent_session_id)
        if "DOCODE_RESULT.md" not in self.files:
            return ""
        return "diff --git a/DOCODE_RESULT.md b/DOCODE_RESULT.md\nnew file mode 100644\n+scripted development agent\n"

    async def git_commit(self, project_id: str, message: str, agent_session_id: str | None = None) -> CommandResult:
        self.agent_session_ids.append(agent_session_id)
        self.commits.append(message)
        return CommandResult(f"[main abc123] {message}\n", 0)

    async def archive_workspace(self, project_id: str, agent_session_id: str | None = None) -> bytes:
        self.agent_session_ids.append(agent_session_id)
        return b"fake-tar-archive"


class TruncatedDiffDoBoxClient(StatefulFakeDoBoxClient):
    async def git_diff_result(self, project_id: str, agent_session_id: str | None = None) -> CommandResult:
        diff = await self.git_diff(project_id, agent_session_id=agent_session_id)
        return CommandResult(diff, 0, truncated=True)


class SlowCreateDoBoxClient(StatefulFakeDoBoxClient):
    async def create_project(
        self,
        *,
        name: str,
        repo_url: str | None = None,
        branch: str | None = None,
        image: str | None = None,
        network_mode: str | None = None,
    ) -> ProjectSandbox:
        await asyncio.sleep(0.01)
        return await super().create_project(name=name, repo_url=repo_url, branch=branch, image=image, network_mode=network_mode)


class CancellingAfterCreateDoBoxClient(StatefulFakeDoBoxClient):
    def __init__(self, repo: InMemoryJobRepository, job_id: str) -> None:
        super().__init__()
        self.repo = repo
        self.job_id = job_id

    async def create_project(
        self,
        *,
        name: str,
        repo_url: str | None = None,
        branch: str | None = None,
        image: str | None = None,
        network_mode: str | None = None,
    ) -> ProjectSandbox:
        project = await super().create_project(name=name, repo_url=repo_url, branch=branch, image=image, network_mode=network_mode)
        await self.repo.update_job(self.job_id, status=JobStatus.STOPPED, failure_reason="cancelled")
        return project


class FakeCredentialResolver(APICredCredentialResolver):
    def __init__(self) -> None:
        super().__init__("http://apicred.invalid/v1", "secret-token")
        self.authorize_calls = 0
        self.usage_calls = 0

    async def authorize(
        self,
        *,
        user_id: str,
        provider: str,
        model: str,
        job_id: str,
        max_iterations: int,
        max_runtime_seconds: int | None = None,
        max_tool_calls: int | None = None,
        max_llm_tokens: int | None = None,
        max_llm_cost: float | None = None,
        sandbox_network_mode: str | None = None,
        artifact_mode: str | None = None,
    ) -> RuntimeAuthorization:
        self.authorize_calls += 1
        self.calls.append(
            {
                "method": "POST",
                "path": "/runtime/authorize",
                "payload": {
                    "user_id": user_id,
                    "provider": provider,
                    "model": model,
                    "job_id": job_id,
                    "max_iterations": max_iterations,
                    "max_runtime_seconds": max_runtime_seconds,
                    "max_tool_calls": max_tool_calls,
                    "max_llm_tokens": max_llm_tokens,
                    "max_llm_cost": max_llm_cost,
                    "sandbox_network_mode": sandbox_network_mode,
                    "artifact_mode": artifact_mode,
                },
            }
        )
        return RuntimeAuthorization(allowed=True, reason="ok", budget_tokens=1000, budget_cost=1.0)

    async def report_usage(self, *, user_id: str, provider: str, model: str, tokens: int = 0, cost: float = 0.0) -> None:
        self.usage_calls += 1
        self.calls.append(
            {
                "method": "POST",
                "path": "/runtime/usage/report",
                "payload": {"user_id": user_id, "provider": provider, "model": model, "tokens": tokens, "cost": cost},
            }
        )


class DenyingCredentialResolver(FakeCredentialResolver):
    async def authorize(self, **kwargs) -> RuntimeAuthorization:
        await super().authorize(**kwargs)
        return RuntimeAuthorization(allowed=False, reason="budget_exhausted")


class AuthorizationFailingCredentialResolver(FakeCredentialResolver):
    async def authorize(self, **kwargs) -> RuntimeAuthorization:
        _ = kwargs
        self.authorize_calls += 1
        raise RuntimeError("apicred unavailable")


class UsageFailingCredentialResolver(FakeCredentialResolver):
    async def report_usage(self, *, user_id: str, provider: str, model: str, tokens: int = 0, cost: float = 0.0) -> None:
        self.usage_calls += 1
        raise RuntimeError("usage endpoint unavailable")


class RunnerE2ETests(IsolatedAsyncioTestCase):
    async def test_runner_completes_scripted_job_and_exports_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            config = DocodeConfig(artifact_dir=Path(tmp))
            fake_dobox = StatefulFakeDoBoxClient()
            fake_apicred = FakeCredentialResolver()
            runner = JobRunnerService(
                config=config,
                repository=repo,
                dobox_client_factory=lambda: fake_dobox,
                credential_resolver_factory=lambda: fake_apicred,
            )
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="user-1",
                    instruction="create a result file",
                    provider="scripted",
                    model="scripted",
                    apicred_access_token="bp_xat_runner",
                    max_iterations=5,
                    max_tool_calls=10,
                    artifact_mode="pr",
                    github_repo="zeturn/example",
                )
            )

            await runner.run_job(job.id)

            completed = await repo.get_job(job.id)
            assert completed is not None
            self.assertEqual(completed.status, JobStatus.SUCCEEDED)
            self.assertEqual(completed.dobox_project_id, "project-1")
            self.assertEqual(completed.dobox_agent_session_id, "7")
            self.assertIn("DOCODE_RESULT.md", fake_dobox.files)
            artifacts = await repo.list_artifacts(job.id)
            self.assertEqual({artifact.kind for artifact in artifacts}, {"patch", "report", "log", "result", "zip", "archive", "commit", "pull_request"})
            self.assertEqual(completed.artifact_id, next(artifact.id for artifact in artifacts if artifact.kind == "result"))
            self.assertTrue((Path(tmp) / job.id / "workspace.tar").exists())
            self.assertTrue((Path(tmp) / job.id / "workspace.zip").exists())
            with ZipFile(Path(tmp) / job.id / "workspace.zip") as archive:
                self.assertIn("workspace.tar", archive.namelist())
            result_payload = json.loads((Path(tmp) / job.id / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result_payload["status"], "succeeded")
            self.assertEqual(result_payload["changed_files"], ["DOCODE_RESULT.md"])
            self.assertIn("git_status", {check["name"] for check in result_payload["checks"]})
            self.assertEqual(
                result_payload["artifacts"],
                {
                    "patch": "patch.diff",
                    "report": "final_report.md",
                    "result": "result.json",
                    "zip": "workspace.zip",
                    "log": "test_log.txt",
                    "commit": "commit.txt",
                    "pull_request": "pull_request.txt",
                    "archive": "workspace.tar",
                },
            )
            self.assertIn("github_export_not_configured", (Path(tmp) / job.id / "pull_request.txt").read_text(encoding="utf-8"))
            self.assertEqual(len(fake_dobox.commits), 1)
            self.assertEqual(fake_dobox.create_calls, 1)
            self.assertEqual(fake_dobox.session_calls, 1)
            self.assertEqual(fake_dobox.create_network_modes, ["project"])
            self.assertTrue(fake_dobox.agent_session_ids)
            self.assertTrue(all(session_id == "7" for session_id in fake_dobox.agent_session_ids))
            self.assertEqual(fake_dobox.deleted_projects, [])
            self.assertEqual(fake_apicred.authorize_calls, 1)
            self.assertEqual(fake_apicred.access_token, "bp_xat_runner")
            authorize_payload = next(call["payload"] for call in fake_apicred.calls if call["path"] == "/runtime/authorize")
            self.assertEqual(authorize_payload["max_iterations"], 5)
            self.assertEqual(authorize_payload["max_runtime_seconds"], 1800)
            self.assertEqual(authorize_payload["max_tool_calls"], 10)
            self.assertEqual(authorize_payload["max_llm_tokens"], 100_000)
            self.assertEqual(authorize_payload["max_llm_cost"], None)
            self.assertEqual(authorize_payload["sandbox_network_mode"], "project")
            self.assertEqual(authorize_payload["artifact_mode"], "pr")
            self.assertEqual(fake_apicred.usage_calls, 1)
            steps = await repo.list_steps(job.id)
            self.assertTrue(any(step.content.get("type") == "apicred_authorize" for step in steps))
            runtime_step = next(step for step in steps if step.content.get("type") == "runtime_assembly")
            self.assertEqual(runtime_step.content["provider"], "scripted")
            self.assertEqual(runtime_step.content["sandbox_network_mode"], "project")
            self.assertEqual(runtime_step.content["dobox_agent_session_id"], "7")
            self.assertIn("run_command", runtime_step.content["tools"])
            self.assertTrue(any(step.content.get("type") == "apicred_usage_report" for step in steps))
            usage_step = next(step for step in steps if step.content.get("type") == "apicred_usage_report")
            self.assertEqual(usage_step.content["status"], "reported")
            self.assertEqual(usage_step.content["tokens"], 0)
            self.assertEqual(usage_step.content["budget_tokens"], 1000)
            self.assertEqual(usage_step.content["budget_cost"], 1.0)
            cleanup_step = next(step for step in steps if step.content.get("type") == "sandbox_cleanup")
            self.assertEqual(cleanup_step.content["status"], "kept")
            self.assertNotIn("secret-token", repr(completed))

    async def test_runner_keeps_success_when_usage_report_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            fake_dobox = StatefulFakeDoBoxClient()
            fake_apicred = UsageFailingCredentialResolver()
            runner = JobRunnerService(
                config=DocodeConfig(artifact_dir=Path(tmp), sandbox_retention="delete_on_success"),
                repository=repo,
                dobox_client_factory=lambda: fake_dobox,
                credential_resolver_factory=lambda: fake_apicred,
            )
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="user-1",
                    instruction="create a result file",
                    provider="scripted",
                    model="scripted",
                    max_iterations=5,
                    max_tool_calls=10,
                )
            )

            await runner.run_job(job.id)

            completed = await repo.get_job(job.id)
            assert completed is not None
            self.assertEqual(completed.status, JobStatus.SUCCEEDED)
            self.assertIsNone(completed.failure_reason)
            self.assertIsNotNone(completed.artifact_id)
            self.assertEqual(fake_apicred.usage_calls, 1)
            self.assertEqual(fake_dobox.deleted_projects, ["project-1"])
            result_payload = json.loads((Path(tmp) / job.id / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result_payload["status"], "succeeded")
            steps = await repo.list_steps(job.id)
            usage_step = next(step for step in steps if step.content.get("type") == "apicred_usage_report")
            self.assertEqual(usage_step.content["status"], "failed")
            self.assertIn("usage endpoint unavailable", usage_step.content["error"])
            cleanup_step = next(step for step in steps if step.content.get("type") == "sandbox_cleanup")
            self.assertEqual(cleanup_step.content["status"], "deleted")

    async def test_runner_passes_no_internet_sandbox_network_policy_to_dobox(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            fake_dobox = StatefulFakeDoBoxClient()
            runner = JobRunnerService(
                config=DocodeConfig(artifact_dir=Path(tmp)),
                repository=repo,
                dobox_client_factory=lambda: fake_dobox,
                credential_resolver_factory=FakeCredentialResolver,
            )
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="user-1",
                    instruction="create a result file",
                    provider="scripted",
                    model="scripted",
                    sandbox_network_mode="no_internet",
                    max_iterations=5,
                    max_tool_calls=10,
                )
            )

            await runner.run_job(job.id)

            self.assertEqual(fake_dobox.create_network_modes, ["no_internet"])
            steps = await repo.list_steps(job.id)
            runtime_step = next(step for step in steps if step.content.get("type") == "runtime_assembly")
            self.assertEqual(runtime_step.content["sandbox_network_mode"], "no_internet")

    async def test_runner_claims_job_once_for_duplicate_delivery(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            fake_dobox = SlowCreateDoBoxClient()
            runner = JobRunnerService(
                config=DocodeConfig(artifact_dir=Path(tmp)),
                repository=repo,
                dobox_client_factory=lambda: fake_dobox,
                credential_resolver_factory=FakeCredentialResolver,
            )
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="user-1",
                    instruction="create a result file",
                    provider="scripted",
                    model="scripted",
                    max_iterations=5,
                    max_tool_calls=10,
                )
            )

            await asyncio.gather(runner.run_job(job.id), runner.run_job(job.id))

            completed = await repo.get_job(job.id)
            assert completed is not None
            self.assertEqual(completed.status, JobStatus.SUCCEEDED)
            self.assertEqual(fake_dobox.create_calls, 1)

    async def test_runner_exports_failure_artifacts_when_authorization_is_denied(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            fake_dobox = StatefulFakeDoBoxClient()
            runner = JobRunnerService(
                config=DocodeConfig(artifact_dir=Path(tmp)),
                repository=repo,
                dobox_client_factory=lambda: fake_dobox,
                credential_resolver_factory=DenyingCredentialResolver,
            )
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="user-1",
                    instruction="create a result file",
                    provider="scripted",
                    model="scripted",
                )
            )

            await runner.run_job(job.id)

            completed = await repo.get_job(job.id)
            assert completed is not None
            self.assertEqual(completed.status, JobStatus.FAILED)
            self.assertEqual(completed.failure_reason, "budget_exhausted")
            self.assertIsNotNone(completed.artifact_id)
            self.assertEqual(fake_dobox.create_calls, 0)
            artifacts = await repo.list_artifacts(job.id)
            self.assertEqual({artifact.kind for artifact in artifacts}, {"report", "log", "result", "zip"})
            self.assertEqual(completed.artifact_id, next(artifact.id for artifact in artifacts if artifact.kind == "result"))
            result_payload = json.loads((Path(tmp) / job.id / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result_payload["status"], "failed")
            self.assertEqual(result_payload["failure_reason"], "budget_exhausted")
            self.assertEqual(result_payload["changed_files"], [])

    async def test_runner_exports_failure_artifacts_when_authorization_transport_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            fake_dobox = StatefulFakeDoBoxClient()
            fake_apicred = AuthorizationFailingCredentialResolver()
            runner = JobRunnerService(
                config=DocodeConfig(artifact_dir=Path(tmp)),
                repository=repo,
                dobox_client_factory=lambda: fake_dobox,
                credential_resolver_factory=lambda: fake_apicred,
            )
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="user-1",
                    instruction="create a result file",
                    provider="scripted",
                    model="scripted",
                )
            )

            await runner.run_job(job.id)

            completed = await repo.get_job(job.id)
            assert completed is not None
            self.assertEqual(completed.status, JobStatus.FAILED)
            self.assertEqual(completed.failure_reason, "apicred_authorize_failed:apicred unavailable")
            self.assertIsNotNone(completed.artifact_id)
            self.assertEqual(fake_apicred.authorize_calls, 1)
            self.assertEqual(fake_dobox.create_calls, 0)
            self.assertIsNone(completed.dobox_project_id)
            artifacts = await repo.list_artifacts(job.id)
            self.assertEqual({artifact.kind for artifact in artifacts}, {"report", "log", "result", "zip"})
            self.assertEqual(completed.artifact_id, next(artifact.id for artifact in artifacts if artifact.kind == "result"))
            result_payload = json.loads((Path(tmp) / job.id / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result_payload["status"], "failed")
            self.assertEqual(result_payload["failure_reason"], "apicred_authorize_failed:apicred unavailable")
            self.assertEqual(result_payload["changed_files"], [])
            steps = await repo.list_steps(job.id)
            auth_step = next(step for step in steps if step.content.get("type") == "apicred_authorize")
            self.assertEqual(auth_step.content["status"], "failed")
            self.assertEqual(auth_step.content["error"], "apicred unavailable")

    async def test_runner_deletes_sandbox_when_policy_requests_success_cleanup(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            config = DocodeConfig(artifact_dir=Path(tmp), sandbox_retention="delete_on_success")
            fake_dobox = StatefulFakeDoBoxClient()
            runner = JobRunnerService(
                config=config,
                repository=repo,
                dobox_client_factory=lambda: fake_dobox,
                credential_resolver_factory=FakeCredentialResolver,
            )
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="user-1",
                    instruction="create a result file",
                    provider="scripted",
                    model="scripted",
                    max_iterations=5,
                    max_tool_calls=10,
                )
            )

            await runner.run_job(job.id)

            self.assertEqual(fake_dobox.deleted_projects, ["project-1"])
            steps = await repo.list_steps(job.id)
            cleanup_step = next(step for step in steps if step.content.get("type") == "sandbox_cleanup")
            self.assertEqual(cleanup_step.content["status"], "deleted")
            self.assertEqual(cleanup_step.content["policy"], "delete_on_success")

    async def test_runner_does_not_start_stopped_job(self) -> None:
        repo = InMemoryJobRepository()
        fake_dobox = StatefulFakeDoBoxClient()
        runner = JobRunnerService(
            config=DocodeConfig(),
            repository=repo,
            dobox_client_factory=lambda: fake_dobox,
            credential_resolver_factory=FakeCredentialResolver,
        )
        job = await repo.create_job(
            CodingJob(
                id=new_id("job"),
                user_id="user-1",
                instruction="do not run",
                status=JobStatus.STOPPED,
                failure_reason="cancelled",
            )
        )

        await runner.run_job(job.id)

        completed = await repo.get_job(job.id)
        assert completed is not None
        self.assertEqual(completed.status, JobStatus.STOPPED)
        self.assertEqual(fake_dobox.create_calls, 0)

    async def test_runner_finalizes_stopped_sandbox_job_with_patch_and_archive(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            fake_dobox = StatefulFakeDoBoxClient()
            fake_dobox.files["DOCODE_RESULT.md"] = "scripted development agent\n"
            runner = JobRunnerService(
                config=DocodeConfig(artifact_dir=Path(tmp), sandbox_retention="keep"),
                repository=repo,
                dobox_client_factory=lambda: fake_dobox,
                credential_resolver_factory=FakeCredentialResolver,
            )
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="user-1",
                    instruction="cancel after edits",
                    status=JobStatus.STOPPED,
                    failure_reason="cancelled",
                    dobox_project_id="project-1",
                    dobox_sandbox_id="sandbox-1",
                    dobox_agent_session_id="7",
                )
            )
            await repo.add_step(job.id, "system", {"type": "cancelled", "reason": "user_requested_cancel"})

            await runner.run_job(job.id)

            completed = await repo.get_job(job.id)
            assert completed is not None
            self.assertEqual(completed.status, JobStatus.STOPPED)
            self.assertIsNotNone(completed.artifact_id)
            artifacts = await repo.list_artifacts(job.id)
            self.assertEqual({artifact.kind for artifact in artifacts}, {"patch", "report", "log", "archive", "result", "zip"})
            self.assertEqual(completed.artifact_id, next(artifact.id for artifact in artifacts if artifact.kind == "result"))
            self.assertEqual(fake_dobox.create_calls, 0)
            self.assertEqual(fake_dobox.agent_session_ids[:2], ["7", "7"])
            result_payload = json.loads((Path(tmp) / job.id / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result_payload["status"], "stopped")
            self.assertEqual(result_payload["artifacts"]["patch"], "patch.diff")
            self.assertEqual(result_payload["artifacts"]["archive"], "workspace.tar")
            with ZipFile(Path(tmp) / job.id / "workspace.zip") as archive:
                self.assertIn("workspace.tar", archive.namelist())
            steps = await repo.list_steps(job.id)
            cleanup_step = next(step for step in steps if step.content.get("type") == "sandbox_cleanup")
            self.assertEqual(cleanup_step.content["status"], "kept")

    async def test_runner_omits_patch_when_terminal_diff_is_truncated(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            fake_dobox = TruncatedDiffDoBoxClient()
            fake_dobox.files["DOCODE_RESULT.md"] = "scripted development agent\n"
            runner = JobRunnerService(
                config=DocodeConfig(artifact_dir=Path(tmp), sandbox_retention="keep"),
                repository=repo,
                dobox_client_factory=lambda: fake_dobox,
                credential_resolver_factory=FakeCredentialResolver,
            )
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="user-1",
                    instruction="cancel after edits",
                    status=JobStatus.STOPPED,
                    failure_reason="cancelled",
                    dobox_project_id="project-1",
                    dobox_sandbox_id="sandbox-1",
                    dobox_agent_session_id="7",
                )
            )

            await runner.run_job(job.id)

            completed = await repo.get_job(job.id)
            assert completed is not None
            self.assertEqual(completed.status, JobStatus.STOPPED)
            artifacts = await repo.list_artifacts(job.id)
            self.assertNotIn("patch", {artifact.kind for artifact in artifacts})
            self.assertFalse((Path(tmp) / job.id / "patch.diff").exists())
            result_payload = json.loads((Path(tmp) / job.id / "result.json").read_text(encoding="utf-8"))
            self.assertNotIn("patch", result_payload["artifacts"])
            self.assertTrue(result_payload["git_diff"]["truncated"])

    async def test_runner_deletes_sandbox_when_cancelled_after_project_create_and_policy_is_delete_always(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="user-1",
                    instruction="cancel during setup",
                    provider="scripted",
                    model="scripted",
                )
            )
            fake_dobox = CancellingAfterCreateDoBoxClient(repo, job.id)
            runner = JobRunnerService(
                config=DocodeConfig(artifact_dir=Path(tmp), sandbox_retention="delete_always"),
                repository=repo,
                dobox_client_factory=lambda: fake_dobox,
                credential_resolver_factory=FakeCredentialResolver,
            )

            await runner.run_job(job.id)

            completed = await repo.get_job(job.id)
            assert completed is not None
            self.assertEqual(completed.status, JobStatus.STOPPED)
            self.assertIsNotNone(completed.artifact_id)
            self.assertEqual(fake_dobox.deleted_projects, ["project-1"])
            self.assertEqual(fake_dobox.session_calls, 0)
            artifacts = await repo.list_artifacts(job.id)
            self.assertEqual(completed.artifact_id, next(artifact.id for artifact in artifacts if artifact.kind == "result"))
            result_payload = json.loads((Path(tmp) / job.id / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result_payload["status"], "stopped")
            self.assertEqual(result_payload["stopped_reason"], "cancelled")
            steps = await repo.list_steps(job.id)
            cleanup_step = next(step for step in steps if step.content.get("type") == "sandbox_cleanup")
            self.assertEqual(cleanup_step.content["status"], "deleted")
            self.assertTrue(any(step.content.get("type") == "stopped_artifacts_exported" for step in steps))

    async def test_runtime_budget_uses_strictest_positive_budget(self) -> None:
        self.assertEqual(runtime_budget(1000, 500), 500)
        self.assertEqual(runtime_budget(1000, None), 1000)
        self.assertEqual(runtime_budget(None, 500), 500)
        self.assertIsNone(runtime_budget(None, None))
        self.assertEqual(runtime_budget(2.5, 1.25), 1.25)
