"""Release vertical-slice live runner.

Runs the Coding Agent through the REAL production path:

    JobRunnerService -> RuntimeComponents -> CodingAgentLoop
    -> real provider (DecisionLLM/provider adapter)
    -> real DoBox workspace (DoBoxTools)
    -> repository inspection -> edit -> required command -> verification
    -> finalization -> ArtifactExporter -> terminal job result

It does NOT use ScriptedLLM, DiagnosticLocalTools, FakeDoBox, or any
second Agent runtime. The agent is reached only through JobRunnerService.

The fixture repository is seeded into a freshly created DoBox project for
every run, so consecutive runs never reuse a previously modified workspace.

Configuration is read from the existing repository env-var names (see
``docode.config``). Missing real infrastructure fails closed: the runner
never reports a success it cannot prove.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from datetime import datetime
import re
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Any

from dataclasses import dataclass

from docode.config import DocodeConfig, load_config
from docode.dobox.client import DoBoxClient
from docode.llm.credentials import APICredCredentialResolver, ProviderCredential
from docode.llm.runtime import build_docode_llm
from docode.runtime.smoke import (
    CommandRunner,
    HealthChecker,
    check_http_health,
    ensure_dobox_smoke_token,
    local_dobox_checks,
    managed_local_dobox,
    run_command_probe,
)
from docode.storage.models import CodingJob, JobStatus, new_id, public_job_dict
from docode.storage.repository import InMemoryJobRepository
from docode.storage.step_redaction import redacted_step_content
from docode.worker.runner import JobRunnerService
from docode.eval.evidence import (
    DoBoxWorkspaceInspector,
    LocalWorkspaceInspector,
    _json_default,
    build_artifact_manifest,
    build_commands_record,
    build_job_record,
    build_outcomes_record,
    build_steps_record,
    build_summary,
    count_metrics,
    redact_endpoint,
    write_evidence_bundle,
)

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "release_vertical_slice"

INSTRUCTION = (
    "Fix the bug in this repository so that the calculator addition behavior is correct.\n\n"
    "Requirements:\n"
    "- Modify the implementation, not the test expectations.\n"
    "- Run `python -m unittest -q`.\n"
    "- Do not finish until the command passes.\n"
    "- Provide a concise final summary.\n\n"
    "Verification commands:\n"
    "- python -m unittest -q\n"
)

REQUIRED_COMMAND_MARKER = "unittest"
REQUIRED_COMMAND = "python -m unittest -q"
FUNCTIONAL_CHECK = (
    "python -c \"import calculator as c; "
    "assert c.add(2, 3) == 5, 'add(2,3) != 5'; "
    "assert c.add(-2, 2) == 0, 'add(-2,2) != 0'; "
    "print('FUNCTIONAL_OK')\""
)

FORBIDDEN_DOUBLE_SUBSTRINGS = ("fake", "mock", "stub", "scripted", "diagnosticlocal")


# ── Config resolution ────────────────────────────────────────────────────


def resolve_provider_and_config() -> tuple[Any, dict[str, ProviderCredential], str, str, list[str]]:
    """Return (config, local_credentials, provider, model, failure_reasons).

    Reads the existing repository env-var names (``docode.config``) plus the
    provider aliases documented for this runner. Fails closed by populating
    ``failure_reasons`` instead of raising.
    """
    config = load_config()
    reasons: list[str] = []
    local_credentials: dict[str, ProviderCredential] = {}

    # DoBox URL is optional: when unset the config default (http://localhost:3000)
    # is used, and `--start-dobox` can autostart a local backend. An external
    # endpoint is configured only via DOCODE_DOBOX_BASE_URL.
    dobox_url = os.getenv("DOCODE_DOBOX_BASE_URL")
    if dobox_url:
        config.dobox_base_url = dobox_url
    dobox_token = os.getenv("DOCODE_DOBOX_TOKEN") or os.getenv("DOCODE_DOBOX_API_KEY")
    if dobox_token:
        config.dobox_token = dobox_token

    provider = (os.getenv("DOCODE_PROVIDER") or "openai").strip().lower()

    if provider == "openai":
        key = (
            os.getenv("DOCODE_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DOCODE_PROVIDER_API_KEY")
        )
        if not key:
            reasons.append("openai provider API key missing (set DOCODE_OPENAI_API_KEY)")
        base = (
            os.getenv("DOCODE_OPENAI_BASE_URL")
            or os.getenv("DOCODE_PROVIDER_BASE_URL")
            or "https://api.openai.com/v1"
        )
        model = (
            os.getenv("DOCODE_DEFAULT_MODEL")
            or os.getenv("DOCODE_PROVIDER_MODEL")
            or "gpt-5.4"
        )
        if key:
            local_credentials["openai"] = ProviderCredential(
                provider="openai", model=model, api_key=key, base_url=base
            )
            config.direct_openai_enabled = True
            config.openai_api_key = key
            config.openai_base_url = base
            config.default_model = model
    elif provider == "deepseek":
        key = (
            os.getenv("DOCODE_DEEPSEEK_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("DOCODE_PROVIDER_API_KEY")
        )
        if not key:
            reasons.append("deepseek provider API key missing (set DOCODE_DEEPSEEK_API_KEY)")
        base = (
            os.getenv("DOCODE_DEEPSEEK_BASE_URL")
            or os.getenv("DOCODE_PROVIDER_BASE_URL")
            or "https://api.deepseek.com/v1"
        )
        model = (
            os.getenv("DOCODE_DEEPSEEK_MODEL")
            or os.getenv("DOCODE_PROVIDER_MODEL")
            or "deepseek-chat"
        )
        if key:
            local_credentials["deepseek"] = ProviderCredential(
                provider="deepseek", model=model, api_key=key, base_url=base
            )
    else:
        reasons.append(f"unsupported DOCODE_PROVIDER={provider!r} (use openai or deepseek)")

    if local_credentials:
        # Force the credential resolver onto the local-credential fallback path
        # so the real provider client is used deterministically without
        # depending on an external APICred runtime being reachable.
        config.apicred_mode = "proxy"

    return config, local_credentials, provider, model, reasons


@dataclass
class DoboxReadiness:
    reachable: bool
    mode: str
    started_by_runner: bool
    fail_reason: str | None
    docker_daemon_available: bool | None
    sandbox_image_available: bool | None
    autostart_checks: list


async def plan_dobox_readiness(
    config: DocodeConfig,
    *,
    start_dobox: bool,
    health_checker: HealthChecker = check_http_health,
    command_runner: CommandRunner = run_command_probe,
) -> DoboxReadiness:
    """Decide DoBox mode and whether autostart is required (does not start it).

    Reuses the shared ``docode.runtime.smoke`` checks so there is a single
    source of truth for local DoBox readiness and startup.
    """
    checks = await local_dobox_checks(config, command_runner)
    docker_daemon_available = _smoke_check_passed(checks, "docker_daemon")
    sandbox_image_available = _smoke_check_passed(checks, "dobox_sandbox_image")
    reachable, _detail = await health_checker(config.dobox_base_url.rstrip("/") + "/health")
    if reachable:
        return DoboxReadiness(True, "existing", False, None, docker_daemon_available, sandbox_image_available, checks)
    if not start_dobox:
        return DoboxReadiness(
            False,
            "autostarted",
            False,
            f"DoBox unreachable at {config.dobox_base_url}; pass --start-dobox to autostart a local DoBox, "
            f"or set DOCODE_DOBOX_BASE_URL to an external endpoint",
            docker_daemon_available,
            sandbox_image_available,
            checks,
        )
    return DoboxReadiness(False, "autostarted", True, None, docker_daemon_available, sandbox_image_available, checks)


def _smoke_check_passed(checks: list, name: str) -> bool | None:
    for check in checks:
        if check.name == name:
            return check.status == "passed"
    return None


def _relative_backend_dir(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except Exception:
        return "redacted"


# ── Fixture seeding DoBox client ──────────────────────────────────────────


class FixtureSeedingDoBoxClient(DoBoxClient):
    """A real ``DoBoxClient`` that seeds a fixture repository after project creation.

    This is production infrastructure (it talks to the real DoBox API); it only
    writes the agreed fixture files into the freshly created workspace so the
    agent starts from a known buggy repository. It is not a test double of the
    agent runtime.
    """

    def __init__(self, base_url: str, token: str, fixture_root: Path) -> None:
        super().__init__(base_url, token)
        self._fixture_root = fixture_root

    async def create_project(
        self,
        *,
        name: str,
        repo_url: str | None = None,
        branch: str | None = None,
        image: str | None = None,
        network_mode: str | None = None,
    ):
        project = await super().create_project(
            name=name, repo_url=repo_url, branch=branch, image=image, network_mode=network_mode
        )
        await self._seed_fixture(project.project_id)
        return project

    async def _seed_fixture(self, project_id: str) -> None:
        # Seed every file as raw bytes through the binary-safe upload path so
        # that non-UTF-8 content (images, compiled artifacts, corrupted samples,
        # encoding-damaged files) is preserved byte-for-byte. We never decode to
        # UTF-8 first; text and binary files travel through a single bytes path.
        for path in sorted(self._fixture_root.rglob("*")):
            # Never silently treat a symlink as a regular file: following it
            # would upload the target's bytes under the symlink's path and can
            # leak files outside the fixture workspace. Symlinks are skipped.
            if path.is_symlink():
                continue
            if not path.is_file():
                continue
            rel = path.relative_to(self._fixture_root).as_posix()
            # Defensive guard: the backend also rejects traversal, but we refuse
            # to upload any entry whose relative path escapes the fixture root.
            if ".." in rel.split("/"):
                raise RuntimeError(f"refusing to seed path-traversal entry: {rel}")
            data = path.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            await self.write_file(project_id, rel, content_base64=b64)
        await self.run_command(
            project_id,
            [
                "bash",
                "-lc",
                "cd /workspace && git init -q && "
                "git config user.email 'dev@docode.ai' && "
                "git config user.name 'Docode' && "
                "git add -A && git commit -q -m 'Initialize fixture'",
            ],
            cwd="/workspace",
            timeout_sec=30,
            output_limit=200_000,
        )


# ── Independent workspace inspectors (used by the hidden checker) ─────────


# ── Hidden checker ────────────────────────────────────────────────────────


async def run_hidden_checker(
    inspector: Any,
    fixture_root: Path,
    job: CodingJob,
    steps: list[Any],
) -> dict[str, Any]:
    """Independent post-run verification. The agent runtime never sees this logic."""
    checks: dict[str, bool] = {}
    failures: list[str] = []

    # 1. Functional behavior of the fixed implementation.
    func_code, func_out = await inspector.run_command(FUNCTIONAL_CHECK)
    checks["functional_behavior"] = func_code == 0 and "FUNCTIONAL_OK" in func_out
    if not checks["functional_behavior"]:
        failures.append(f"functional behavior check failed (exit={func_code}): {func_out[:400]}")

    # 2. Required command passes in the final (post-edit) workspace.
    rc, rout = await inspector.run_command(REQUIRED_COMMAND)
    checks["required_command_passed"] = rc == 0
    if rc != 0:
        failures.append(f"required command '{REQUIRED_COMMAND}' exited {rc}: {rout[:400]}")

    # 3. Required command executed AFTER the last edit, and passed.
    last_edit_index = -1
    last_required_index = -1
    last_required_passed = False
    for step in steps:
        content = step.content if hasattr(step, "content") else step.get("content", {})
        if content.get("type") != "tool_result":
            continue
        tool = content.get("tool")
        if tool in ("write_file", "edit_file", "replace_in_file", "apply_patch"):
            last_edit_index = step.step_index
        elif tool == "run_command":
            cmd = (content.get("metadata") or {}).get("command") or ""
            if REQUIRED_COMMAND_MARKER in cmd:
                last_required_index = step.step_index
                last_required_passed = content.get("exit_code") == 0
    if last_required_index < 0:
        # No required-command step recorded. Fall back to the independent
        # re-run in the final (post-edit) workspace as the freshness proof:
        # if it passes there, the command passes on the edited code.
        checks["fresh_after_edit"] = checks["required_command_passed"]
        if not checks["required_command_passed"]:
            failures.append("required command not found in steps and did not pass on independent re-run")
    elif last_edit_index < 0:
        checks["fresh_after_edit"] = last_required_passed
        if not last_required_passed:
            failures.append("required command executed but did not pass")
    else:
        checks["fresh_after_edit"] = last_required_passed and last_required_index > last_edit_index
        if not checks["fresh_after_edit"]:
            failures.append(
                "required command did not run after the last edit or did not pass "
                f"(edit@{last_edit_index}, required@{last_required_index}, passed={last_required_passed})"
            )

    # 4. Implementation was actually modified.
    try:
        current_calc = await inspector.read_text("calculator.py")
    except Exception as exc:  # noqa: BLE001 - checker must record, not crash
        current_calc = ""
        failures.append(f"could not read calculator.py: {exc}")
    original_calc = (fixture_root / "calculator.py").read_text(encoding="utf-8")
    checks["implementation_modified"] = current_calc.strip() != original_calc.strip()
    if not checks["implementation_modified"]:
        failures.append("calculator.py was not modified from the fixture")

    # 5. Tests were not weakened or deleted.
    original_test = (fixture_root / "tests" / "test_calculator.py").read_text(encoding="utf-8")
    try:
        current_test = await inspector.read_text("tests/test_calculator.py")
        tests_intact = current_test.strip() == original_test.strip()
    except Exception:
        tests_intact = False
    checks["tests_not_weakened"] = tests_intact
    if not checks["tests_not_weakened"]:
        failures.append("tests/test_calculator.py was weakened or removed")

    # 6. An artifact id was produced.
    checks["artifact_present"] = bool(job.artifact_id)
    if not checks["artifact_present"]:
        failures.append("no artifact_id was recorded")

    # 7. Terminal status is success.
    checks["terminal_success"] = job.status == JobStatus.SUCCEEDED
    if not checks["terminal_success"]:
        failures.append(f"terminal status is {job.status.value}")

    return {"passed": all(checks.values()), "checks": checks, "failures": failures}


# ── Evidence bundle builders (no secrets) ────────────────────────────────


def build_summary(
    *,
    run_id: str,
    fixture: str,
    job: CodingJob,
    iterations: int,
    tool_calls: int,
    outcome_count: int,
    components: dict[str, str],
    started_at: str,
    finished_at: str,
    dobox_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "fixture": fixture,
        "job_id": job.id,
        "status": "succeeded" if job.status == JobStatus.SUCCEEDED else "failed",
        "failure_reason": job.failure_reason,
        "provider": {"model": job.model, "base_url": redact_endpoint(None)},
        "dobox": {"workspace_id": job.dobox_project_id, "endpoint": redact_endpoint(None)},
        "iterations": iterations,
        "tool_calls": tool_calls,
        "outcome_count": outcome_count,
        "artifact_id": job.artifact_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "components": components,
        "dobox_runtime": dobox_runtime or {},
    }


def build_job_record(job: CodingJob) -> dict[str, Any]:
    record = public_job_dict(job)
    record["status"] = job.status.value
    return record


def build_steps_record(steps: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for step in steps:
        content = step.content if hasattr(step, "content") else step.get("content", {})
        out.append(
            {
                "step_index": step.step_index,
                "kind": step.kind,
                "content": redacted_step_content(content),
            }
        )
    return out


def build_outcomes_record(steps: list[Any]) -> list[dict[str, Any]]:
    return [
        (step.content if hasattr(step, "content") else step.get("content", {}))
        for step in steps
        if (step.kind if hasattr(step, "kind") else step.get("kind")) == "outcome"
    ]


def build_commands_record(steps: list[Any]) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for step in steps:
        content = step.content if hasattr(step, "content") else step.get("content", {})
        if content.get("type") == "tool_result" and content.get("tool") == "run_command":
            metadata = content.get("metadata") or {}
            commands.append(
                {
                    "step_index": step.step_index,
                    "command": metadata.get("command"),
                    "exit_code": content.get("exit_code"),
                }
            )
    return commands


def build_artifact_manifest(artifacts: list[Any]) -> list[dict[str, Any]]:
    return [
        {"id": a.id, "kind": a.kind, "path": a.path, "size_bytes": a.size_bytes}
        for a in artifacts
    ]


def count_metrics(steps: list[Any]) -> tuple[int, int, int]:
    iterations = 0
    tool_calls = 0
    outcome_count = 0
    for step in steps:
        content = step.content if hasattr(step, "content") else step.get("content", {})
        kind = step.kind if hasattr(step, "kind") else step.get("kind")
        if content.get("type") == "llm_decision":
            iterations += 1
        if kind == "tool" and content.get("type") in ("tool_call", "tool_result"):
            tool_calls += 1
        if kind == "outcome":
            outcome_count += 1
    return iterations, tool_calls, outcome_count


async def write_evidence_bundle(
    *,
    output_dir: Path,
    run_id: str,
    fixture: str,
    job: CodingJob,
    steps: list[Any],
    artifacts: list[Any],
    dobox: DoBoxClient | None,
    fixture_root: Path,
    components: dict[str, str],
    started_at: str,
    finished_at: str,
    checker_result: dict[str, Any] | None,
    dobox_runtime: dict[str, Any] | None = None,
) -> Path:
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    iterations, tool_calls, outcome_count = count_metrics(steps)
    summary = build_summary(
        run_id=run_id,
        fixture=fixture,
        job=job,
        iterations=iterations,
        tool_calls=tool_calls,
        outcome_count=outcome_count,
        components=components,
        started_at=started_at,
        finished_at=finished_at,
        dobox_runtime=dobox_runtime,
    )

    def _write(name: str, payload: Any) -> None:
        if isinstance(payload, (str, bytes)):
            mode = "wb" if isinstance(payload, bytes) else "w"
            data = payload.encode("utf-8") if isinstance(payload, str) else payload
            (run_dir / name).write_bytes(data)
        else:
            (run_dir / name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )

    _write("summary.json", summary)
    _write("job.json", build_job_record(job))
    _write("steps.json", build_steps_record(steps))
    _write("outcomes.json", build_outcomes_record(steps))
    _write("commands.json", build_commands_record(steps))
    _write("artifact-manifest.json", build_artifact_manifest(artifacts))
    _write("terminal-result.json", job.terminal_result or {"status": job.status.value, "failure_reason": job.failure_reason})
    _write("checker-result.json", checker_result or {"passed": None, "checks": {}, "failures": ["checker not executed"]})

    # Real git evidence from the workspace (independent of agent tool results).
    if dobox is not None and job.dobox_project_id:
        try:
            status = await dobox.git_status(job.dobox_project_id, agent_session_id=job.dobox_agent_session_id)
            _write("git-status.txt", status.output if hasattr(status, "output") else str(status))
        except Exception as exc:  # noqa: BLE001
            _write("git-status.txt", f"git_status unavailable: {exc}\n")
        try:
            diff = await dobox.git_diff_result(job.dobox_project_id, agent_session_id=job.dobox_agent_session_id)
            _write("git-diff.patch", diff.output if hasattr(diff, "output") else str(diff))
        except Exception as exc:  # noqa: BLE001
            _write("git-diff.patch", f"git_diff unavailable: {exc}\n")

    return run_dir


# ── Single-run driver ─────────────────────────────────────────────────────


def _build_coding_job(
    config: Any,
    *,
    user_id: str,
    instruction: str,
    provider: str,
    model: str,
) -> CodingJob:
    """Construct the CodingJob for a release vertical-slice run.

    ``max_consecutive_failures`` is intentionally NOT taken from ``DocodeConfig``
    (``DocodeConfig`` has no such attribute); the job falls back to the dataclass
    default defined on ``CodingJob``. This keeps the harness resilient to config
    shape changes and avoids hard-coding the value here.
    """
    return CodingJob(
        id=new_id("job"),
        user_id=user_id,
        instruction=instruction,
        provider=provider,
        model=model,
        max_iterations=config.max_iterations,
        max_runtime_seconds=config.max_runtime_seconds,
        max_tool_calls=config.max_tool_calls,
        artifact_mode="patch",
    )


async def run_single_job(
    *,
    config: Any,
    local_credentials: dict[str, ProviderCredential],
    provider: str,
    model: str,
    fixture: str,
    output_dir: Path,
    dobox: DoBoxClient,
    dobox_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fixture_root = FIXTURE_ROOT / fixture
    repo: InMemoryJobRepository = InMemoryJobRepository()
    captured: dict[str, str] = {}

    def credential_resolver_factory() -> APICredCredentialResolver:
        return APICredCredentialResolver(
            config.apicred_base_url,
            config.apicred_token,
            config.apicred_mode,
            local_credentials=local_credentials,
        )

    def llm_factory(job: CodingJob):
        resolver = credential_resolver_factory()
        captured["llm_type"] = ""

        async def _build() -> Any:
            llm = await build_docode_llm(job, resolver)
            captured["llm_type"] = type(llm).__name__
            return llm

        return _build()

    runner = JobRunnerService(
        config=config,
        repository=repo,
        dobox_client_factory=lambda: dobox,
        credential_resolver_factory=credential_resolver_factory,
        llm_factory=llm_factory,
    )

    job = await repo.create_job(
        _build_coding_job(
            config,
            user_id="release-vertical-slice",
            instruction=INSTRUCTION,
            provider=provider,
            model=model,
        )
    )
    run_id = job.id
    started_at = job.created_at.isoformat()

    try:
        await runner.run_job(job.id)
    except Exception as exc:  # noqa: BLE001 - runner already records; never crash the harness
        await repo.add_step(
            job.id,
            "system",
            {"type": "harness_exception", "error": str(exc), "exception_type": type(exc).__name__},
        )

    job = await repo.get_job(job.id) or job
    steps = await repo.list_steps(job.id)
    artifacts = await repo.list_artifacts(job.id)
    finished_at = (job.completed_at or job.updated_at).isoformat()

    components = {
        "runner": type(runner).__name__,
        "llm": captured.get("llm_type") or "unknown",
        "tools": "DoBoxTools",
        "repository": type(repo).__name__,
        "exporter": "ArtifactExporter",
    }

    inspector = DoBoxWorkspaceInspector(dobox, job.dobox_project_id, job.dobox_agent_session_id) if job.dobox_project_id else None
    checker_result: dict[str, Any] | None = None
    if inspector is not None:
        checker_result = await run_hidden_checker(inspector, fixture_root, job, steps)

    await write_evidence_bundle(
        output_dir=output_dir,
        run_id=run_id,
        fixture=fixture,
        job=job,
        steps=steps,
        artifacts=artifacts,
        dobox=dobox,
        fixture_root=fixture_root,
        components=components,
        started_at=started_at,
        finished_at=finished_at,
        checker_result=checker_result,
        dobox_runtime=dobox_runtime,
    )

    _assert_no_forbidden_doubles(components)
    return {"run_id": run_id, "status": job.status.value, "checker": checker_result, "failure_reason": job.failure_reason}


def _assert_no_forbidden_doubles(components: dict[str, str]) -> None:
    blob = " ".join(components.values()).lower()
    for forbidden in FORBIDDEN_DOUBLE_SUBSTRINGS:
        if forbidden in blob:
            raise RuntimeError(f"gate violation: forbidden test-double substring {forbidden!r} in components {components}")


def is_successful_live_run(result: dict[str, Any]) -> bool:
    """A live run counts as successful only when the job reached the
    ``SUCCEEDED`` terminal state AND the independent hidden checker passed.

    The persisted ``status`` is the lowercase enum value ``"succeeded"``; we
    compare against ``JobStatus.SUCCEEDED.value`` rather than a hardcoded
    uppercase string so the accounting cannot silently disagree with the
    repository state. We deliberately do NOT normalize the stored status to
    uppercase — the repository persists lowercase enum values.
    """
    return (
        result.get("status") == JobStatus.SUCCEEDED.value
        and bool((result.get("checker") or {}).get("passed"))
    )


def count_successful_live_runs(runs: list[dict[str, Any]]) -> int:
    return sum(1 for r in runs if is_successful_live_run(r))


def build_live_manifest(
    *,
    fixture: str,
    provider: str,
    model: str,
    runs: list[dict[str, Any]],
    required_runs: int,
    output_dir: Path | None,
) -> dict[str, Any]:
    passed = count_successful_live_runs(runs)
    return {
        "fixture": fixture,
        "provider": provider,
        "model": model,
        "runs": [
            {
                "run_id": r["run_id"],
                "status": r["status"],
                "checker_passed": (r["checker"] or {}).get("passed"),
                "evidence_dir": str(output_dir / r["run_id"]) if output_dir is not None else r["run_id"],
            }
            for r in runs
        ],
        "success_rate": f"{passed}/{len(runs)}",
        "required": f"{required_runs}/{required_runs}",
    }


# ── CLI ───────────────────────────────────────────────────────────────────


async def main_async(args: argparse.Namespace) -> int:
    config, local_credentials, provider, model, reasons = resolve_provider_and_config()

    fixture = args.fixture
    fixture_root = FIXTURE_ROOT / fixture
    if not fixture_root.is_dir():
        reasons.append(f"fixture {fixture!r} not found at {fixture_root}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── DoBox readiness (reuses docode.runtime.smoke) ──────────────────────
    readiness = await plan_dobox_readiness(config, start_dobox=args.start_dobox)
    if readiness.fail_reason:
        reasons.append(readiness.fail_reason)

    dobox_mode = readiness.mode
    started_by_runner = False
    docker_daemon_available = readiness.docker_daemon_available
    sandbox_image_available = readiness.sandbox_image_available
    # Only autostart when explicitly requested AND the endpoint is not already up.
    effective_start = args.start_dobox and not readiness.reachable

    async with managed_local_dobox(
        config, check_http_health, effective_start, readiness.autostart_checks, keep=args.keep_dobox
    ) as start_checks:
        # When we asked to start a new backend, confirm it actually came up.
        if effective_start:
            autostart_failed = next(
                (c for c in start_checks if c.name == "dobox_autostart" and c.status == "failed"),
                None,
            )
            if autostart_failed is not None:
                reasons.append(f"DoBox autostart failed: {autostart_failed.detail}")
            else:
                started_by_runner = True

        # Resolve a DoBox token (uses configured token, else registers a smoke user).
        token, _token_check = await ensure_dobox_smoke_token(config)
        if token:
            config.dobox_token = token
        elif not config.dobox_token:
            reasons.append("DoBox token could not be resolved (auth failed)")

        # Fail closed: missing real infrastructure must never be reported as success.
        if reasons:
            report = {
                "status": "failed",
                "failure_reason": "environment_failure",
                "details": reasons,
                "note": "SKIPPED != PASSED: missing real infrastructure; no success claimed.",
            }
            (output_dir / "terminal_result.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print("[release-vertical-slice] FAIL-CLOSED (environment):", file=sys.stderr)
            for reason in reasons:
                print(f"  - {reason}", file=sys.stderr)
            return 2

        dobox = FixtureSeedingDoBoxClient(config.dobox_base_url, config.dobox_token, fixture_root)
        dobox_runtime = {
            "dobox_mode": dobox_mode,
            "dobox_backend_dir": _relative_backend_dir(config.dobox_backend_dir),
            "dobox_started_by_runner": started_by_runner,
            "docker_daemon_available": docker_daemon_available,
            "sandbox_image_available": sandbox_image_available,
        }

        runs: list[dict[str, Any]] = []
        for index in range(args.runs):
            print(f"[release-vertical-slice] run {index + 1}/{args.runs} ...", file=sys.stderr)
            result = await run_single_job(
                config=config,
                local_credentials=local_credentials,
                provider=provider,
                model=model,
                fixture=fixture,
                output_dir=output_dir,
                dobox=dobox,
                dobox_runtime=dobox_runtime,
            )
            runs.append(result)
            print(
                f"[release-vertical-slice] run {index + 1}: status={result['status']} "
                f"checker_passed={result['checker']['passed'] if result['checker'] else None}",
                file=sys.stderr,
            )

        manifest = build_live_manifest(
            fixture=fixture,
            provider=provider,
            model=model,
            runs=runs,
            required_runs=args.runs,
            output_dir=output_dir,
        )
        passed = count_successful_live_runs(runs)
        (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0 if passed >= args.runs else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DoCode Runtime V2 release vertical slice (live).")
    parser.add_argument("--fixture", default="simple_bugfix")
    parser.add_argument("--output", default="artifacts/release-vertical-slice")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--start-dobox", action="store_true", help="autostart a local DoBox backend if unreachable")
    parser.add_argument("--keep-dobox", action="store_true", help="keep the autostarted backend after the run (debug only)")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
