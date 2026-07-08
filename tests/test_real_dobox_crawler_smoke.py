from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, skipUnless

from docode.agent.loop import CodingAgentLoop
from docode.agent.quality_gate import QualityGate
from docode.agent.stop_policy import StopPolicy
from docode.agent.verifier import CodingVerifier
from docode.artifacts.exporter import ArtifactExporter
from docode.config import load_config
from docode.dobox.client import DoBoxClient
from docode.dobox.tools import DoBoxTools
from docode.runtime.smoke import check_http_health, ensure_dobox_smoke_token
from docode.storage.models import CodingJob, JobStatus, new_id
from docode.storage.repository import InMemoryJobRepository

from tests.test_real_dobox_smoke import final_candidate_step_index, summarize_command_results
from tests.test_real_llm_smoke import build_real_llm_or_skip, summarize_job_steps


CRAWLER_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "repos" / "generic_crawler_cli"
REAL_LLM_SMOKE_ENABLED = os.getenv("DOCODE_REAL_LLM_SMOKE", "").lower() in {"1", "true", "yes", "on"}
CRAWLER_UNITTEST_COMMAND = "python -m unittest discover -s tests"
CRAWLER_CLI_COMMAND = "python crawler.py fixtures/products.html --output out.json"
CRAWLER_INSTRUCTION = f"""Implement crawler.py so it parses fixtures/products.html and writes product records to JSON.

Verification commands:
1. {CRAWLER_UNITTEST_COMMAND}
2. {CRAWLER_CLI_COMMAND}"""
EXPECTED_PRODUCTS = [
    {
        "sku": "lamp-001",
        "name": "Desk Lamp",
        "url": "/catalog/desk-lamp",
        "price": 24.99,
        "category": "Lighting",
        "in_stock": True,
    },
    {
        "sku": "mug-002",
        "name": "Travel Mug",
        "url": "/catalog/travel-mug",
        "price": 12.5,
        "category": "Kitchen",
        "in_stock": False,
    },
]
FORBIDDEN_STRINGS = (
    "GitHub Trends",
    "GitHub Trending",
    "owner/repo",
    "stars today",
    "Box-row",
)


@skipUnless(os.getenv("DOCODE_REAL_DOBOX_SMOKE") == "1", "set DOCODE_REAL_DOBOX_SMOKE=1 to run the real DoBox crawler smoke")
class RealDoBoxCrawlerSmokeTests(IsolatedAsyncioTestCase):
    async def test_real_llm_real_dobox_generic_crawler_cli_smoke(self) -> None:
        if not REAL_LLM_SMOKE_ENABLED:
            self.skipTest("set DOCODE_REAL_LLM_SMOKE=1 with DOCODE_REAL_DOBOX_SMOKE=1 to run the real LLM + real DoBox crawler smoke")
        with TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "artifacts"
            repo = InMemoryJobRepository()
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="real-llm-dobox-smoke",
                    instruction=CRAWLER_INSTRUCTION,
                    max_iterations=24,
                    max_runtime_seconds=300,
                    max_tool_calls=48,
                    sandbox_network_mode="no_internet",
                )
            )
            llm = await build_real_llm_or_skip(self, job)
            config = await self._real_dobox_config(artifact_dir)
            client = DoBoxClient(config.dobox_base_url, config.dobox_token)
            project = await client.create_project(
                name=f"docode-real-llm-dobox-crawler-{new_id('smoke')}",
                network_mode=config.sandbox_network_mode,
            )
            session = await client.create_agent_session(project.project_id, name="docode-real-llm-dobox-crawler")
            try:
                await self._seed_fixture(client, project.project_id, session.session_id, CRAWLER_FIXTURE_ROOT)
                await self._ensure_python_command(client, project.project_id, session.session_id)
                tools = DoBoxTools(
                    client,
                    project.project_id,
                    agent_session_id=session.session_id,
                    command_timeout_seconds=30,
                    output_limit_bytes=200_000,
                )
                job = await repo.update_job(
                    job.id,
                    dobox_project_id=project.project_id,
                    dobox_sandbox_id=project.sandbox_id,
                    dobox_agent_session_id=session.session_id,
                    sandbox_network_mode=config.sandbox_network_mode,
                )
                loop = CodingAgentLoop(
                    llm=llm,
                    tools=tools,
                    verifier=CodingVerifier(),
                    repository=repo,
                    exporter=ArtifactExporter(
                        artifact_dir,
                        repo,
                        workspace_archive_provider=lambda: client.archive_workspace(project.project_id, agent_session_id=session.session_id),
                        workspace_file_reader=lambda path: client.read_file(project.project_id, path, agent_session_id=session.session_id),
                    ),
                    stop_policy=StopPolicy(max_iterations=24, max_runtime_seconds=300, max_consecutive_failures=20, max_tool_calls=48),
                    quality_gate=QualityGate(),
                    llm_retry_delays=(),
                    llm_decision_timeout_seconds=30,
                )

                completed = await loop.run(job)
                steps = await repo.list_steps(job.id)
                if completed.status != JobStatus.SUCCEEDED:
                    status = await self._safe_git_output(
                        lambda: client.git_status(project.project_id, agent_session_id=session.session_id),
                        "git_status",
                    )
                    diff = await self._safe_git_output(
                        lambda: client.git_diff_result(project.project_id, agent_session_id=session.session_id),
                        "git_diff",
                    )
                    self.fail(
                        "real LLM + real DoBox crawler smoke failed "
                        f"with status={completed.status} reason={completed.failure_reason}\n\n"
                        f"Git status:\n{status}\n\nGit diff:\n{diff[:3000]}\n\n"
                        f"Command results:\n{summarize_command_results(steps)}\n\n"
                        f"Recent steps:\n{summarize_job_steps(steps)}"
                    )

                await self._assert_crawler_outputs(client, project.project_id, session.session_id, repo, job.id, artifact_dir, steps)
            finally:
                await client.delete_project(project.project_id)

    async def _real_dobox_config(self, artifact_dir: Path):
        config = load_config()
        config.artifact_dir = artifact_dir
        config.sandbox_network_mode = "no_internet"
        config.web_tools_enabled = False
        ok, detail = await check_http_health(config.dobox_base_url.rstrip("/") + "/health")
        if not ok:
            self.skipTest(f"DoBox is unavailable at {config.dobox_base_url}: {detail}")
        token, token_check = await ensure_dobox_smoke_token(config)
        if token_check.status != "passed" or not token:
            self.skipTest(f"DoBox auth failed: {token_check.detail}")
        config.dobox_token = token
        return config

    async def _safe_git_output(self, fetcher, label: str) -> str:
        try:
            result = await fetcher()
        except Exception as exc:  # pragma: no cover - exercised only when integration diagnostics fail.
            return f"<{label} unavailable: {type(exc).__name__}: {exc}>"
        return result.output

    def _step_git_output(self, steps, key: str) -> str:
        for step in reversed(steps):
            value = step.content.get(key)
            if isinstance(value, str) and value:
                return value
        return ""

    async def _assert_crawler_outputs(
        self,
        client: DoBoxClient,
        project_id: str,
        session_id: str,
        repo: InMemoryJobRepository,
        job_id: str,
        artifact_dir: Path,
        steps,
    ) -> None:
        source = await client.read_file(project_id, "crawler.py", agent_session_id=session_id)
        self.assertIn("parse_products", source.content)
        self.assertNotIn("return []", source.content)
        self.assertNotIn(json.dumps(EXPECTED_PRODUCTS), source.content)
        status = await self._safe_git_output(lambda: client.git_status(project_id, agent_session_id=session_id), "git_status")
        diff = await self._safe_git_output(lambda: client.git_diff_result(project_id, agent_session_id=session_id), "git_diff")
        if status.startswith("<git_status unavailable:"):
            status = self._step_git_output(steps, "git_status")
        if diff.startswith("<git_diff unavailable:"):
            diff = self._step_git_output(steps, "git_diff")
        self.assertIn("crawler.py", status)
        self.assertIn("crawler.py", diff)
        output = await client.read_file(project_id, "out.json", agent_session_id=session_id)
        self.assertEqual(json.loads(output.content), EXPECTED_PRODUCTS)
        unit_step = successful_command_step_index(steps, CRAWLER_UNITTEST_COMMAND)
        cli_step = successful_command_step_index(steps, CRAWLER_CLI_COMMAND)
        self.assertIsNotNone(unit_step, summarize_command_results(steps))
        self.assertIsNotNone(cli_step, summarize_command_results(steps))
        final_step = final_candidate_step_index(steps)
        self.assertIsNotNone(final_step, summarize_job_steps(steps))
        assert unit_step is not None and cli_step is not None and final_step is not None
        self.assertLess(unit_step, final_step, summarize_job_steps(steps))
        self.assertLess(cli_step, final_step, summarize_job_steps(steps))
        artifacts = await repo.list_artifacts(job_id)
        self.assertIn("report", {artifact.kind for artifact in artifacts})
        self.assertIn("result", {artifact.kind for artifact in artifacts})
        self.assertIn("archive", {artifact.kind for artifact in artifacts})
        self.assertTrue((artifact_dir / job_id / "final_report.md").exists())
        self.assertTrue((artifact_dir / job_id / "result.json").exists())
        self.assertTrue((artifact_dir / job_id / "workspace.tar").exists())
        with tarfile.open(artifact_dir / job_id / "workspace.tar") as archive:
            names = set(archive.getnames())
        self.assertTrue(any(name.endswith("out.json") for name in names))
        combined = source.content + "\n" + "\n".join(str(step.content) for step in steps)
        for forbidden in FORBIDDEN_STRINGS:
            self.assertNotIn(forbidden, combined)
        self.assertFalse(any(step.content.get("tool") in {"web_search", "fetch_url"} for step in steps), summarize_job_steps(steps))

    async def _seed_fixture(self, client: DoBoxClient, project_id: str, session_id: str, fixture_root: Path) -> None:
        for path in sorted(fixture_root.rglob("*")):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_file():
                relative = path.relative_to(fixture_root).as_posix()
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

    async def _ensure_python_command(self, client: DoBoxClient, project_id: str, session_id: str) -> None:
        result = await client.run_command(
            project_id,
            [
                "sh",
                "-lc",
                "command -v python >/dev/null 2>&1 || "
                "(mkdir -p /tmp/docode-bin && ln -sf \"$(command -v python3)\" /tmp/docode-bin/python && "
                "printf 'export PATH=/tmp/docode-bin:$PATH\\n' > \"$HOME/.bash_profile\") && "
                "bash -lc 'python --version'",
            ],
            cwd="/workspace",
            timeout_sec=30,
            agent_session_id=session_id,
        )
        if result.exit_code != 0:
            self.fail(f"failed to make python command available in crawler smoke sandbox:\n{result.output}")


def successful_command_step_index(steps, expected_command: str) -> int | None:
    expected = " ".join(expected_command.strip().split()).lower()
    for index, step in enumerate(steps):
        content = step.content
        if content.get("type") != "tool_result" or content.get("tool") != "run_command" or content.get("exit_code") != 0:
            continue
        command = " ".join(str((content.get("metadata") or {}).get("command") or "").strip().split()).lower()
        if command == expected:
            return index
    return None
