from __future__ import annotations

from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from docode.config import DocodeConfig
from docode.runtime.smoke import CommandProbe, SmokeCheck, run_scripted_smoke_job, run_smoke_check
from docode.storage.models import JobStatus
from docode.storage.repository import InMemoryJobRepository


class SmokeTests(IsolatedAsyncioTestCase):
    async def test_smoke_check_reports_dobox_failure(self) -> None:
        async def fake_health(url: str) -> tuple[bool, str]:
            if url.endswith("/health"):
                return False, "connection refused"
            return True, "HTTP 200"

        def fake_command(command: list[str], cwd, timeout: float) -> CommandProbe:
            return CommandProbe(True, "24.0.0")

        with patch("docode.runtime.smoke.importlib.util.find_spec", return_value=object()):
            report = await run_smoke_check(DocodeConfig(), health_checker=fake_health, command_runner=fake_command)

        self.assertEqual(report.status, "failed")
        checks = {check.name: check for check in report.checks}
        self.assertEqual(checks["dobox_health"].status, "failed")
        self.assertEqual(checks["apicred_models"].status, "passed")

    async def test_start_dobox_requires_startable_backend_dir(self) -> None:
        async def fake_health(url: str) -> tuple[bool, str]:
            if url.endswith("/health"):
                return False, "connection refused"
            return True, "HTTP 200"

        def fake_command(command: list[str], cwd, timeout: float) -> CommandProbe:
            return CommandProbe(True, "24.0.0")

        with patch("docode.runtime.smoke.importlib.util.find_spec", return_value=object()):
            report = await run_smoke_check(
                DocodeConfig(dobox_backend_dir=Path("/tmp/not-a-dobox-backend")),
                health_checker=fake_health,
                start_dobox=True,
                command_runner=fake_command,
            )

        checks = {check.name: check for check in report.checks}
        self.assertEqual(report.status, "failed")
        self.assertEqual(checks["dobox_backend_dir"].status, "warning")
        self.assertEqual(checks["dobox_autostart"].status, "failed")

    async def test_local_docker_diagnostics_are_advisory(self) -> None:
        async def fake_health(url: str) -> tuple[bool, str]:
            return True, "HTTP 200"

        def fake_command(command: list[str], cwd, timeout: float) -> CommandProbe:
            return CommandProbe(False, "daemon unavailable")

        with patch("docode.runtime.smoke.importlib.util.find_spec", return_value=object()):
            report = await run_smoke_check(DocodeConfig(), health_checker=fake_health, command_runner=fake_command)

        checks = {check.name: check for check in report.checks}
        self.assertEqual(report.status, "passed")
        self.assertEqual(checks["docker_daemon"].status, "warning")

    async def test_missing_sandbox_image_is_fatal_preflight(self) -> None:
        async def fake_health(url: str) -> tuple[bool, str]:
            return True, "HTTP 200"

        def fake_command(command: list[str], cwd, timeout: float) -> CommandProbe:
            if command[:3] == ["docker", "image", "inspect"]:
                return CommandProbe(False, "No such image")
            return CommandProbe(True, "24.0.0")

        with patch("docode.runtime.smoke.importlib.util.find_spec", return_value=object()):
            report = await run_smoke_check(DocodeConfig(), health_checker=fake_health, command_runner=fake_command)

        checks = {check.name: check for check in report.checks}
        self.assertEqual(report.status, "failed")
        self.assertEqual(checks["dobox_sandbox_image"].status, "failed")

    async def test_scripted_smoke_job_uses_shared_job_creation_policy(self) -> None:
        repo = InMemoryJobRepository()

        def fake_command(command: list[str], cwd, timeout: float) -> CommandProbe:
            return CommandProbe(True, "24.0.0")

        async def fake_health(url: str) -> tuple[bool, str]:
            return True, "HTTP 200"

        class FakeRunner:
            def __init__(self, *, config: DocodeConfig, repository: InMemoryJobRepository) -> None:
                self.config = config
                self.repository = repository

            async def run_job(self, job_id: str) -> None:
                await self.repository.update_job(job_id, status=JobStatus.SUCCEEDED, result_summary="smoke passed")

        async def fake_token(config: DocodeConfig) -> tuple[str | None, SmokeCheck]:
            return "token-1", SmokeCheck("dobox_auth", "passed", "test token")

        with (
            patch("docode.runtime.smoke.build_repository", return_value=repo),
            patch("docode.runtime.smoke.JobRunnerService", FakeRunner),
            patch("docode.runtime.smoke.check_http_health", fake_health),
            patch("docode.runtime.smoke.shutil.which", return_value="/usr/bin/mock"),
            patch("docode.runtime.smoke.importlib.util.find_spec", return_value=object()),
        ):
            report = await run_scripted_smoke_job(
                DocodeConfig(max_iterations=20, sandbox_network_mode="bridge"),
                instruction="create result",
                command_runner=fake_command,
                dobox_token_resolver=fake_token,
            )

        jobs = await repo.list_jobs()
        self.assertEqual(report.status, "passed")
        self.assertEqual(report.job_id, jobs[0].id)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].user_id, "smoke")
        self.assertEqual(jobs[0].provider, "scripted")
        self.assertEqual(jobs[0].max_iterations, 5)
        self.assertEqual(jobs[0].sandbox_network_mode, "project")

    async def test_scripted_smoke_job_fails_preflight_when_dobox_auth_fails(self) -> None:
        repo = InMemoryJobRepository()

        def fake_command(command: list[str], cwd, timeout: float) -> CommandProbe:
            return CommandProbe(True, "24.0.0")

        async def fake_token(config: DocodeConfig) -> tuple[str | None, SmokeCheck]:
            return None, SmokeCheck("dobox_auth", "failed", "HTTP 401")

        with (
            patch("docode.runtime.smoke.build_repository", return_value=repo),
            patch("docode.runtime.smoke.check_http_health", return_value=(True, "HTTP 200")),
            patch("docode.runtime.smoke.shutil.which", return_value="/usr/bin/mock"),
            patch("docode.runtime.smoke.importlib.util.find_spec", return_value=object()),
        ):
            report = await run_scripted_smoke_job(DocodeConfig(), command_runner=fake_command, dobox_token_resolver=fake_token)

        checks = {check.name: check for check in report.checks}
        self.assertEqual(report.status, "failed")
        self.assertEqual(checks["dobox_auth"].status, "failed")
        self.assertIsNone(report.job_id)
        self.assertEqual(await repo.list_jobs(), [])

    async def test_scripted_smoke_job_fails_preflight_when_python_dependency_is_missing(self) -> None:
        repo = InMemoryJobRepository()

        def fake_command(command: list[str], cwd, timeout: float) -> CommandProbe:
            return CommandProbe(True, "24.0.0")

        async def fake_health(url: str) -> tuple[bool, str]:
            return True, "HTTP 200"

        with (
            patch("docode.runtime.smoke.build_repository", return_value=repo),
            patch("docode.runtime.smoke.check_http_health", fake_health),
            patch("docode.runtime.smoke.shutil.which", return_value="/usr/bin/mock"),
            patch("docode.runtime.smoke.importlib.util.find_spec", return_value=None),
        ):
            report = await run_scripted_smoke_job(DocodeConfig(), command_runner=fake_command)

        checks = {check.name: check for check in report.checks}
        self.assertEqual(report.status, "failed")
        self.assertEqual(checks["python_dependency:httpx"].status, "failed")
        self.assertIsNone(report.job_id)
        self.assertEqual(await repo.list_jobs(), [])
