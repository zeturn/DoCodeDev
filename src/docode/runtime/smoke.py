from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import os
from contextlib import asynccontextmanager
import json
import shutil
import subprocess
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from docode.api.job_actions import CreateJobInput, create_coding_job
from docode.config import DocodeConfig
from docode.llm.credentials import APICredCredentialResolver
from docode.llm.model_policy import DocodeModelPolicy
from docode.storage.db import build_repository
from docode.worker.queue import AsyncJobQueue
from docode.worker.runner import JobRunnerService


@dataclass(frozen=True, slots=True)
class SmokeCheck:
    name: str
    status: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class SmokeReport:
    status: str
    checks: list[SmokeCheck]
    job_id: str | None = None
    artifacts: list[dict[str, Any]] | None = None


HealthChecker = Callable[[str], Awaitable[tuple[bool, str]]]
CommandRunner = Callable[[list[str], Path | None, float], "CommandProbe"]
DoboxTokenResolver = Callable[[DocodeConfig], Awaitable[tuple[str | None, "SmokeCheck"]]]
REQUIRED_PYTHON_MODULES = ("httpx", "fastapi", "pydantic", "uvicorn")


@dataclass(frozen=True, slots=True)
class CommandProbe:
    ok: bool
    detail: str


async def run_smoke_check(
    config: DocodeConfig,
    health_checker: HealthChecker | None = None,
    *,
    start_dobox: bool = False,
    command_runner: CommandRunner | None = None,
) -> SmokeReport:
    checker = health_checker or check_http_health
    checks: list[SmokeCheck] = []
    checks.extend(await local_dobox_checks(config, command_runner or run_command_probe))

    async with managed_local_dobox(config, checker, start_dobox, checks) as start_checks:
        checks.extend(start_checks)
        checks.extend(await dependency_checks(config, checker))

    status = "passed" if all(not is_fatal_smoke_failure(check) for check in checks) else "failed"
    return SmokeReport(status=status, checks=checks)


async def run_scripted_smoke_job(
    config: DocodeConfig,
    *,
    instruction: str = "create a result file",
    start_dobox: bool = False,
    command_runner: CommandRunner | None = None,
    dobox_token_resolver: DoboxTokenResolver | None = None,
) -> SmokeReport:
    checker = check_http_health
    checks = await local_dobox_checks(config, command_runner or run_command_probe)
    async with managed_local_dobox(config, checker, start_dobox, checks) as start_checks:
        checks.extend(start_checks)
        checks.extend(await dependency_checks(config, checker))
        preflight = SmokeReport(
            status="passed" if all(not is_fatal_smoke_failure(check) for check in checks) else "failed",
            checks=checks,
        )
        if preflight.status != "passed":
            return preflight

        token, token_check = await (dobox_token_resolver or ensure_dobox_smoke_token)(config)
        checks.append(token_check)
        preflight = SmokeReport(
            status="passed" if all(not is_fatal_smoke_failure(check) for check in checks) else "failed",
            checks=checks,
        )
        if preflight.status != "passed":
            return preflight
        config.dobox_token = token or config.dobox_token

        repository = build_repository(config)
        queue = AsyncJobQueue()
        model_policy = DocodeModelPolicy(
            config,
            APICredCredentialResolver(config.apicred_base_url, config.apicred_token, config.apicred_mode),
        )
        job = await create_coding_job(
            repository=repository,
            queue=queue,
            config=config,
            model_policy=model_policy,
            user_id="smoke",
            request=CreateJobInput(
                instruction=instruction,
                provider="scripted",
                model="scripted",
                max_iterations=5,
                artifact_mode="patch",
                sandbox_network_mode=config.sandbox_network_mode,
            ),
        )
        runner = JobRunnerService(config=config, repository=repository)
        await runner.run_job(job.id)
        completed = await repository.get_job(job.id)
        artifacts = [asdict(artifact) for artifact in await repository.list_artifacts(job.id)]
        checks = list(preflight.checks)
        checks.append(
            SmokeCheck(
                "scripted_job",
                completed.status.value if completed is not None else "failed",
                completed.failure_reason if completed else "job missing",
            )
        )
        status = "passed" if completed is not None and completed.status.value == "succeeded" else "failed"
        return SmokeReport(status=status, checks=checks, job_id=job.id, artifacts=artifacts)


async def dependency_checks(config: DocodeConfig, checker: HealthChecker) -> list[SmokeCheck]:
    checks: list[SmokeCheck] = []
    checks.extend(python_dependency_checks())

    dobox_ok, dobox_detail = await checker(config.dobox_base_url.rstrip("/") + "/health")
    checks.append(SmokeCheck("dobox_health", "passed" if dobox_ok else "failed", dobox_detail))

    apicred_ok, apicred_detail = await checker(config.apicred_base_url.rstrip("/") + "/models")
    checks.append(SmokeCheck("apicred_models", "passed" if apicred_ok else "warning", apicred_detail))

    checks.append(SmokeCheck("gh_cli", "passed" if shutil.which("gh") else "warning", shutil.which("gh") or "gh not found"))
    checks.append(SmokeCheck("artifact_dir", "passed", str(config.artifact_dir)))
    checks.append(SmokeCheck("database_path", "passed", config.database_path))
    return checks


def python_dependency_checks() -> list[SmokeCheck]:
    checks: list[SmokeCheck] = []
    for module in REQUIRED_PYTHON_MODULES:
        available = importlib.util.find_spec(module) is not None
        detail = "importable" if available else f"{module} is not importable; install project dependencies"
        checks.append(SmokeCheck(f"python_dependency:{module}", "passed" if available else "failed", detail))
    return checks


async def ensure_dobox_smoke_token(config: DocodeConfig) -> tuple[str | None, SmokeCheck]:
    if config.dobox_token:
        return config.dobox_token, SmokeCheck("dobox_auth", "passed", "configured token")

    try:
        import httpx
    except ModuleNotFoundError:
        return None, SmokeCheck("dobox_auth", "failed", "httpx is not importable")

    username = dobox_smoke_username()
    password = "docode-smoke-password"
    base_url = config.dobox_base_url.rstrip("/")
    register_payload = {
        "username": username,
        "email": f"{username}@local.invalid",
        "password": password,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(f"{base_url}/api/auth/register", json=register_payload)
            if response.status_code == 409:
                response = await client.post(f"{base_url}/api/auth/login", json={"username": username, "password": password})
            if response.status_code < 200 or response.status_code >= 300:
                return None, SmokeCheck("dobox_auth", "failed", f"HTTP {response.status_code}")
            token = response.json().get("token")
    except Exception as exc:
        return None, SmokeCheck("dobox_auth", "failed", str(exc))

    if not token:
        return None, SmokeCheck("dobox_auth", "failed", "DoBox auth response did not include a token")
    return str(token), SmokeCheck("dobox_auth", "passed", "smoke user token resolved")


def dobox_smoke_username() -> str:
    digest = hashlib.sha256(str(Path.cwd()).encode("utf-8")).hexdigest()[:12]
    return f"docode_smoke_{digest}"


async def local_dobox_checks(config: DocodeConfig, command_runner: CommandRunner) -> list[SmokeCheck]:
    checks: list[SmokeCheck] = []
    backend = config.dobox_backend_dir
    backend_status = "passed" if is_dobox_backend_dir(backend) else "warning"
    checks.append(SmokeCheck("dobox_backend_dir", backend_status, str(backend)))

    docker_path = shutil.which("docker")
    checks.append(SmokeCheck("docker_cli", "passed" if docker_path else "warning", docker_path or "docker not found"))
    if docker_path is None:
        checks.append(SmokeCheck("docker_daemon", "warning", "docker CLI unavailable"))
        return checks

    probe = await asyncio.to_thread(command_runner, ["docker", "version", "--format", "{{.Server.Version}}"], None, 5.0)
    checks.append(SmokeCheck("docker_daemon", "passed" if probe.ok else "warning", probe.detail))
    if not probe.ok:
        return checks

    image_probe = await asyncio.to_thread(command_runner, ["docker", "image", "inspect", "dobox/code-sandbox:latest"], None, 5.0)
    checks.append(
        SmokeCheck(
            "dobox_sandbox_image",
            "passed" if image_probe.ok else "failed",
            image_probe.detail if image_probe.ok else "dobox/code-sandbox:latest is missing; build DoBoxDev/sandbox first",
        )
    )
    return checks


@asynccontextmanager
async def managed_local_dobox(
    config: DocodeConfig,
    checker: HealthChecker,
    start_dobox: bool,
    existing_checks: list[SmokeCheck],
):
    if not start_dobox:
        yield []
        return

    health_url = config.dobox_base_url.rstrip("/") + "/health"
    health_ok, detail = await checker(health_url)
    if health_ok:
        yield [SmokeCheck("dobox_autostart", "skipped", f"already reachable: {detail}")]
        return

    if any(check.name == "dobox_backend_dir" and check.status != "passed" for check in existing_checks):
        yield [SmokeCheck("dobox_autostart", "failed", "DoBox backend directory is not startable")]
        return

    if any(check.name == "docker_daemon" and check.status != "passed" for check in existing_checks):
        yield [SmokeCheck("dobox_autostart", "failed", "Docker daemon is not reachable")]
        return

    process: subprocess.Popen[str] | None = None
    try:
        process = start_dobox_process(config)
        start_check = await wait_for_dobox(config, checker, process)
        yield [start_check]
    except Exception as exc:
        yield [SmokeCheck("dobox_autostart", "failed", str(exc))]
    finally:
        if process is not None:
            await asyncio.to_thread(stop_process, process)


def start_dobox_process(config: DocodeConfig) -> subprocess.Popen[str]:
    env = dict(os.environ)
    port = port_from_base_url(config.dobox_base_url)
    if port:
        env["PORT"] = port
    return subprocess.Popen(
        ["go", "run", "./cmd/server"],
        cwd=config.dobox_backend_dir,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


async def wait_for_dobox(config: DocodeConfig, checker: HealthChecker, process: subprocess.Popen[str]) -> SmokeCheck:
    deadline = time.monotonic() + config.dobox_start_timeout_seconds
    health_url = config.dobox_base_url.rstrip("/") + "/health"
    last_detail = "not checked"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return SmokeCheck("dobox_autostart", "failed", f"process exited with code {process.returncode}")
        ok, last_detail = await checker(health_url)
        if ok:
            return SmokeCheck("dobox_autostart", "passed", last_detail)
        await asyncio.sleep(0.5)
    return SmokeCheck("dobox_autostart", "failed", f"timed out waiting for {health_url}: {last_detail}")


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def is_dobox_backend_dir(path: Path) -> bool:
    return (path / "go.mod").exists() and (path / "cmd/server/main.go").exists()


def port_from_base_url(base_url: str) -> str | None:
    parsed = urlparse(base_url)
    if parsed.port is not None:
        return str(parsed.port)
    if parsed.scheme == "http":
        return "80"
    if parsed.scheme == "https":
        return "443"
    return None


def run_command_probe(command: list[str], cwd: Path | None, timeout: float) -> CommandProbe:
    try:
        completed = subprocess.run(command, cwd=cwd, timeout=timeout, capture_output=True, text=True, check=False)
    except Exception as exc:
        return CommandProbe(False, str(exc))
    output = (completed.stdout or completed.stderr or "").strip()
    detail = output.splitlines()[0] if output else f"exit code {completed.returncode}"
    return CommandProbe(completed.returncode == 0, detail)


def is_fatal_smoke_failure(check: SmokeCheck) -> bool:
    fatal_checks = {"dobox_health", "dobox_autostart", "dobox_auth", "dobox_sandbox_image", "artifact_dir", "database_path"}
    return check.status == "failed" and (check.name in fatal_checks or check.name.startswith("python_dependency:"))


async def check_http_health(url: str) -> tuple[bool, str]:
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(url)
        status_code = response.status_code
    except ModuleNotFoundError:
        status_code, detail = await asyncio.to_thread(check_http_health_urllib, url)
        if status_code is None:
            return False, detail
    except Exception as exc:
        return False, str(exc)
    if 200 <= status_code < 500:
        return True, f"HTTP {status_code}"
    return False, f"HTTP {status_code}"


def check_http_health_urllib(url: str) -> tuple[int | None, str]:
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    try:
        with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=5) as response:
            return response.status, f"HTTP {response.status}"
    except HTTPError as exc:
        return exc.code, f"HTTP {exc.code}"
    except URLError as exc:
        return None, str(exc.reason)
    except Exception as exc:
        return None, str(exc)


def write_smoke_report(report: SmokeReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
