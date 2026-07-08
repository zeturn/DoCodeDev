from __future__ import annotations

import json
import os
import socket
import tarfile
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, skipUnless

from docode.agent.loop import CodingAgentLoop
from docode.agent.quality_gate import QualityGate
from docode.agent.stop_policy import StopPolicy
from docode.agent.tools import CompositeAgentTools
from docode.agent.verifier import CodingVerifier
from docode.artifacts.exporter import ArtifactExporter
from docode.config import load_config
from docode.dobox.client import DoBoxClient
from docode.dobox.tools import DoBoxTools
from docode.runtime.smoke import check_http_health, ensure_dobox_smoke_token
from docode.storage.models import CodingJob, JobStatus, new_id
from docode.storage.repository import InMemoryJobRepository
from docode.web.tools import WebTools, WebToolsConfig

from tests.test_real_dobox_crawler_smoke import successful_command_step_index
from tests.test_real_dobox_smoke import final_candidate_step_index, summarize_command_results
from tests.test_real_llm_smoke import build_real_llm_or_skip, summarize_job_steps


CRAWLER_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "repos" / "external_crawler_cli"
REAL_LLM_SMOKE_ENABLED = os.getenv("DOCODE_REAL_LLM_SMOKE", "").lower() in {"1", "true", "yes", "on"}
CRAWLER_UNITTEST_COMMAND = "python -m unittest discover -s tests"
EXPECTED_PRODUCTS = [
    {
        "id": "item-101",
        "name": "Trail Bottle",
        "url": "/shop/trail-bottle",
        "price": 18.75,
        "category": "Outdoors",
        "available": True,
    },
    {
        "id": "item-202",
        "name": "Canvas Tote",
        "url": "/shop/canvas-tote",
        "price": 14.5,
        "category": "Accessories",
        "available": False,
    },
]
FORBIDDEN_STRINGS = (
    "GitHub Trends",
    "GitHub Trending",
    "owner/repo",
    "stars today",
    "forks",
    "Box-row",
)


@skipUnless(os.getenv("DOCODE_REAL_DOBOX_SMOKE") == "1", "set DOCODE_REAL_DOBOX_SMOKE=1 to run the real DoBox external crawler smoke")
class RealDoBoxExternalCrawlerSmokeTests(IsolatedAsyncioTestCase):
    async def test_real_llm_real_dobox_external_crawler_cli_smoke(self) -> None:
        if not REAL_LLM_SMOKE_ENABLED:
            self.skipTest("set DOCODE_REAL_LLM_SMOKE=1 with DOCODE_REAL_DOBOX_SMOKE=1 to run the real LLM + real DoBox external crawler smoke")
        with TemporaryDirectory() as tmp, MockProductSource(EXPECTED_PRODUCTS) as source:
            artifact_dir = Path(tmp) / "artifacts"
            repo = InMemoryJobRepository()
            config = await self._real_dobox_config(artifact_dir)
            client = DoBoxClient(config.dobox_base_url, config.dobox_token)
            project = await client.create_project(
                name=f"docode-real-llm-dobox-external-crawler-{new_id('smoke')}",
                network_mode=config.sandbox_network_mode,
            )
            session = await client.create_agent_session(project.project_id, name="docode-real-llm-dobox-external-crawler")
            try:
                await self._ensure_python_command(client, project.project_id, session.session_id)
                source_url = await self._select_source_url(client, project.project_id, session.session_id, source.port)
                crawler_cli_command = f"python crawler.py {source_url} --output out.json"
                instruction = f"""Implement crawler.py so it fetches SOURCE_URL, parses the product records, and writes them to JSON.

Source URL:
{source_url}

Verification commands:
1. {CRAWLER_UNITTEST_COMMAND}
2. {crawler_cli_command}"""
                job = await repo.create_job(
                    CodingJob(
                        id=new_id("job"),
                        user_id="real-llm-dobox-smoke",
                        instruction=instruction,
                        max_iterations=30,
                        max_runtime_seconds=360,
                        max_tool_calls=64,
                        sandbox_network_mode=config.sandbox_network_mode,
                    )
                )
                llm = await build_real_llm_or_skip(self, job)
                await self._seed_fixture(client, project.project_id, session.session_id, CRAWLER_FIXTURE_ROOT)
                dobox_tools = DoBoxTools(
                    client,
                    project.project_id,
                    agent_session_id=session.session_id,
                    command_timeout_seconds=30,
                    output_limit_bytes=200_000,
                )
                tools = CompositeAgentTools(
                    dobox_tools,
                    WebTools(
                        WebToolsConfig(
                            fetch_timeout_seconds=10.0,
                            output_limit_bytes=200_000,
                            allow_private_hosts=True,
                        )
                    ),
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
                    stop_policy=StopPolicy(max_iterations=30, max_runtime_seconds=360, max_consecutive_failures=20, max_tool_calls=64),
                    quality_gate=QualityGate(),
                    llm_retry_delays=(),
                    llm_decision_timeout_seconds=30,
                )

                completed = await loop.run(job)
                steps = await repo.list_steps(job.id)
                if completed.status != JobStatus.SUCCEEDED:
                    status = await self._safe_git_output(lambda: client.git_status(project.project_id, agent_session_id=session.session_id), "git_status")
                    diff = await self._safe_git_output(lambda: client.git_diff_result(project.project_id, agent_session_id=session.session_id), "git_diff")
                    self.fail(
                        "real LLM + real DoBox external crawler smoke failed "
                        f"with status={completed.status} reason={completed.failure_reason}\n\n"
                        f"Source URL: {source_url}\n\n"
                        f"Git status:\n{status}\n\nGit diff:\n{diff[:3000]}\n\n"
                        f"Command results:\n{summarize_command_results(steps)}\n\n"
                        f"Recent steps:\n{summarize_job_steps(steps)}"
                    )

                await self._assert_crawler_outputs(
                    client,
                    project.project_id,
                    session.session_id,
                    repo,
                    job.id,
                    artifact_dir,
                    steps,
                    source_url,
                    crawler_cli_command,
                )
            finally:
                await client.delete_project(project.project_id)

    async def _real_dobox_config(self, artifact_dir: Path):
        config = load_config()
        config.artifact_dir = artifact_dir
        config.sandbox_network_mode = "project"
        config.web_tools_enabled = True
        config.web_fetch_allow_private_hosts = True
        ok, detail = await check_http_health(config.dobox_base_url.rstrip("/") + "/health")
        if not ok:
            self.skipTest(f"DoBox is unavailable at {config.dobox_base_url}: {detail}")
        token, token_check = await ensure_dobox_smoke_token(config)
        if token_check.status != "passed" or not token:
            self.skipTest(f"DoBox auth failed: {token_check.detail}")
        config.dobox_token = token
        return config

    async def _select_source_url(self, client: DoBoxClient, project_id: str, session_id: str, port: int) -> str:
        errors: list[str] = []
        for host in source_host_candidates():
            url = f"http://{host}:{port}/products.json"
            host_error = host_fetch_error(url)
            if host_error:
                errors.append(f"{url}: host fetch failed: {host_error}")
                continue
            result = await client.run_command(
                project_id,
                [
                    "bash",
                    "-lc",
                    "python - <<'PY'\n"
                    "import json, sys, urllib.request\n"
                    f"url = {url!r}\n"
                    "with urllib.request.urlopen(url, timeout=5) as response:\n"
                    "    data = json.load(response)\n"
                    "print(len(data))\n"
                    "PY",
                ],
                cwd="/workspace",
                timeout_sec=15,
                agent_session_id=session_id,
            )
            if result.exit_code == 0:
                return url
            errors.append(f"{url}: sandbox fetch failed: {result.output[:500]}")
        self.skipTest("mock source was not reachable from both host fetch_url and DoBox sandbox:\n" + "\n".join(errors))

    async def _safe_git_output(self, fetcher, label: str) -> str:
        try:
            result = await fetcher()
        except Exception as exc:  # pragma: no cover - integration diagnostics only.
            return f"<{label} unavailable: {type(exc).__name__}: {exc}>"
        return result.output

    async def _assert_crawler_outputs(
        self,
        client: DoBoxClient,
        project_id: str,
        session_id: str,
        repo: InMemoryJobRepository,
        job_id: str,
        artifact_dir: Path,
        steps,
        source_url: str,
        crawler_cli_command: str,
    ) -> None:
        source = await client.read_file(project_id, "crawler.py", agent_session_id=session_id)
        self.assertIn("fetch_and_parse", source.content)
        self.assertNotIn("return []", source.content)
        self.assertNotIn(json.dumps(EXPECTED_PRODUCTS), source.content)
        for product in EXPECTED_PRODUCTS:
            self.assertNotIn(product["id"], source.content)
            self.assertNotIn(product["name"], source.content)
        status = await self._safe_git_output(lambda: client.git_status(project_id, agent_session_id=session_id), "git_status")
        diff = await self._safe_git_output(lambda: client.git_diff_result(project_id, agent_session_id=session_id), "git_diff")
        self.assertIn("crawler.py", status)
        self.assertIn("crawler.py", diff)
        output = await client.read_file(project_id, "out.json", agent_session_id=session_id)
        self.assertEqual(json.loads(output.content), EXPECTED_PRODUCTS)
        unit_step = successful_command_step_index(steps, CRAWLER_UNITTEST_COMMAND)
        cli_step = successful_command_step_index(steps, crawler_cli_command)
        self.assertIsNotNone(unit_step, summarize_command_results(steps))
        self.assertIsNotNone(cli_step, summarize_command_results(steps))
        self.assertTrue(successful_fetch_url_step_exists(steps, source_url), summarize_job_steps(steps))
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
            self.fail(f"failed to make python command available in external crawler smoke sandbox:\n{result.output}")


class MockProductSource:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def __enter__(self):
        payload = json.dumps({"products": self.records}).encode("utf-8")

        class Handler(BaseHTTPRequestHandler):
            def do_GET(handler_self):
                if handler_self.path != "/products.json":
                    handler_self.send_error(404)
                    return
                handler_self.send_response(200)
                handler_self.send_header("Content-Type", "application/json")
                handler_self.send_header("Content-Length", str(len(payload)))
                handler_self.end_headers()
                handler_self.wfile.write(payload)

            def log_message(self, format, *args):
                return

        self.server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = int(self.server.server_port)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def source_host_candidates() -> list[str]:
    candidates: list[str] = []
    configured = os.getenv("DOCODE_EXTERNAL_CRAWLER_SOURCE_HOST")
    if configured:
        candidates.append(configured)
    candidates.extend(["host.docker.internal", "127.0.0.1"])
    try:
        hostname = socket.gethostname()
        candidates.append(socket.gethostbyname(hostname))
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            candidates.append(info[4][0])
    except OSError:
        pass
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def host_fetch_error(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.load(response)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    if data != {"products": EXPECTED_PRODUCTS}:
        return "unexpected mock source payload"
    return ""


def successful_fetch_url_step_exists(steps, expected_url: str) -> bool:
    for step in steps:
        content = step.content
        if content.get("type") != "tool_result" or content.get("tool") != "fetch_url" or content.get("exit_code") != 0:
            continue
        metadata = content.get("metadata") if isinstance(content.get("metadata"), dict) else {}
        if metadata.get("url") == expected_url:
            return True
    return False
