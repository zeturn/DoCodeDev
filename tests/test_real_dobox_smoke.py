from __future__ import annotations

import os
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, skipUnless

from docode.agent.loop import CodingAgentLoop
from docode.agent.stop_policy import StopPolicy
from docode.agent.verifier import CodingVerifier
from docode.artifacts.exporter import ArtifactExporter
from docode.config import load_config
from docode.dobox.client import DoBoxClient
from docode.dobox.tools import DoBoxTools
from docode.llm.runtime import AgentDecision
from docode.runtime.smoke import ensure_dobox_smoke_token
from docode.storage.models import CodingJob, JobStatus, new_id
from docode.storage.repository import InMemoryJobRepository


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "repos" / "readme_edit"
SMOKE_SENTENCE = "This project has a smoke test."


class ReadmeEditLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, *, system, messages, tools, context):
        _ = system, messages, tools, context
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(type="tool_call", tool_name="read_file", args={"path": "README.md"})
        if self.calls == 2:
            return AgentDecision(
                type="tool_call",
                tool_name="edit_file",
                args={
                    "path": "README.md",
                    "old_text": "This fixture starts with a short project note.\n",
                    "new_text": (
                        "This fixture starts with a short project note.\n\n"
                        f"{SMOKE_SENTENCE}\n"
                    ),
                },
            )
        return AgentDecision(type="final_candidate", summary="Updated README.md with the smoke test sentence.")


@skipUnless(os.getenv("DOCODE_REAL_DOBOX_SMOKE") == "1", "set DOCODE_REAL_DOBOX_SMOKE=1 to run the real DoBox smoke")
class RealDoBoxSmokeTests(IsolatedAsyncioTestCase):
    async def test_readme_edit_runs_through_real_dobox(self) -> None:
        with TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "artifacts"
            config = load_config()
            config.artifact_dir = artifact_dir
            config.sandbox_network_mode = "no_internet"
            token, token_check = await ensure_dobox_smoke_token(config)
            if token_check.status != "passed" or not token:
                self.fail(f"DoBox auth failed: {token_check.detail}")
            config.dobox_token = token

            client = DoBoxClient(config.dobox_base_url, config.dobox_token)
            project = await client.create_project(
                name=f"docode-real-dobox-readme-{new_id('smoke')}",
                network_mode=config.sandbox_network_mode,
            )
            session = await client.create_agent_session(project.project_id, name="docode-real-dobox-readme")
            try:
                await self._seed_fixture(client, project.project_id, session.session_id)
                tools = DoBoxTools(
                    client,
                    project.project_id,
                    agent_session_id=session.session_id,
                    command_timeout_seconds=30,
                    output_limit_bytes=200_000,
                )
                repo = InMemoryJobRepository()
                job = await repo.create_job(
                    CodingJob(
                        id=new_id("job"),
                        user_id="real-dobox-smoke",
                        instruction="Update README.md by adding one sentence that says this project has a smoke test.",
                        provider="scripted",
                        model="scripted",
                        max_iterations=5,
                        max_runtime_seconds=120,
                        max_tool_calls=12,
                        sandbox_network_mode=config.sandbox_network_mode,
                        dobox_project_id=project.project_id,
                        dobox_sandbox_id=project.sandbox_id,
                        dobox_agent_session_id=session.session_id,
                    )
                )
                loop = CodingAgentLoop(
                    llm=ReadmeEditLLM(),
                    tools=tools,
                    verifier=CodingVerifier(),
                    repository=repo,
                    exporter=ArtifactExporter(
                        artifact_dir,
                        repo,
                        workspace_archive_provider=lambda: client.archive_workspace(project.project_id, agent_session_id=session.session_id),
                        workspace_file_reader=lambda path: client.read_file(project.project_id, path, agent_session_id=session.session_id),
                    ),
                    stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=120, max_tool_calls=12),
                    llm_retry_delays=(),
                )

                completed = await loop.run(job)

                self.assertEqual(completed.status, JobStatus.SUCCEEDED)
                readme = await client.read_file(project.project_id, "README.md", agent_session_id=session.session_id)
                self.assertIn(SMOKE_SENTENCE, readme.content)
                status = await client.git_status(project.project_id, agent_session_id=session.session_id)
                diff = await client.git_diff_result(project.project_id, agent_session_id=session.session_id)
                self.assertIn("README.md", status.output)
                self.assertIn(SMOKE_SENTENCE, diff.output)
                artifacts = await repo.list_artifacts(job.id)
                self.assertIn("report", {artifact.kind for artifact in artifacts})
                self.assertIn("result", {artifact.kind for artifact in artifacts})
                self.assertTrue((artifact_dir / job.id / "final_report.md").exists())
                self.assertTrue((artifact_dir / job.id / "result.json").exists())
                self.assertTrue((artifact_dir / job.id / "workspace.tar").exists())
                with tarfile.open(artifact_dir / job.id / "workspace.tar") as archive:
                    names = set(archive.getnames())
                self.assertTrue(any(name.endswith("README.md") for name in names))
                steps = await repo.list_steps(job.id)
                self.assertTrue(steps)
                self.assertTrue(any(step.content.get("type") == "bootstrap" for step in steps))
            finally:
                await client.delete_project(project.project_id)

    async def _seed_fixture(self, client: DoBoxClient, project_id: str, session_id: str) -> None:
        for path in sorted(FIXTURE_ROOT.rglob("*")):
            if path.is_file():
                relative = path.relative_to(FIXTURE_ROOT).as_posix()
                await client.write_file(project_id, relative, path.read_text(encoding="utf-8"), agent_session_id=session_id)
        result = await client.run_command(
            project_id,
            [
                "sh",
                "-lc",
                "git init -b main && git config user.email smoke@example.test && "
                "git config user.name 'DoCode Smoke' && git add . && git commit -m 'Initial fixture'",
            ],
            cwd="/workspace",
            timeout_sec=30,
            agent_session_id=session_id,
        )
        if result.exit_code != 0:
            self.fail(f"failed to initialize fixture git repository:\n{result.output}")
