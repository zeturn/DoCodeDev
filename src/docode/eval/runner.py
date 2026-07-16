"""Generic evaluation case runner.

``run_case`` drives ONE fixture through the real production path:

    JobRunnerService -> RuntimeComponents -> CodingAgentLoop
    -> real provider -> real DoBox workspace (DoBoxTools)
    -> repository inspection -> edit -> required command -> verification
    -> finalization -> ArtifactExporter -> terminal job result
    -> independent hidden checker

It reuses the single ``JobRunnerService`` implementation (never a second
runtime) and the seeded DoBox project pattern from the vertical-slice runner.
It does NOT parse provider config, autostart DoBox, or redact secrets itself;
those are provided by the orchestrator (``scripts/run_release_eval_suite.py``).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from docode.eval.checker import CheckerContext, run_checker_module
from docode.eval.evidence import DoBoxWorkspaceInspector, write_evidence_bundle
from docode.eval.fixture import Fixture
from docode.eval.models import RunResult, classify_run_outcome, derive_false_flags
from docode.llm.credentials import APICredCredentialResolver, ProviderCredential
from docode.llm.runtime import build_docode_llm
from docode.storage.models import CodingJob, JobStatus, new_id
from docode.storage.repository import InMemoryJobRepository
from docode.worker.runner import JobRunnerService


def _build_coding_job(
    config: Any,
    *,
    user_id: str,
    instruction: str,
    provider: str,
    model: str,
) -> CodingJob:
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


def _latest_usage(steps: list[Any]) -> dict[str, Any]:
    for step in reversed(steps):
        content = step.content if hasattr(step, "content") else step.get("content", {})
        usage = content.get("usage") if isinstance(content, dict) else None
        if isinstance(usage, dict):
            return usage
    return {}


def _count_steps(steps: list[Any]) -> tuple[int, int, int, dict[str, int], int, int]:
    iterations = 0
    tool_calls = 0
    outcomes = 0
    by_type: dict[str, int] = {}
    no_progress = 0
    transport_errors = 0
    decision_parse_errors = 0
    for step in steps:
        content = step.content if hasattr(step, "content") else step.get("content", {})
        kind = step.kind if hasattr(step, "kind") else step.get("kind")
        if content.get("type") == "llm_decision":
            iterations += 1
        if kind == "tool" and content.get("type") in ("tool_call", "tool_result"):
            tool_calls += 1
            tool = content.get("tool")
            if tool:
                by_type[tool] = by_type.get(tool, 0) + 1
        if kind == "outcome":
            outcomes += 1
        etype = str(content.get("type") or "")
        if etype == "no_progress":
            no_progress += 1
        if etype == "transport_error" or "transport" in str(content.get("error") or "").lower():
            transport_errors += 1
        if "unsupported decision type" in str(content).lower():
            decision_parse_errors += 1
    return iterations, tool_calls, outcomes, by_type, no_progress, transport_errors


async def run_case(
    *,
    suite_run_id: str,
    case_id: str,
    run_index: int,
    config: Any,
    local_credentials: dict[str, ProviderCredential],
    provider: str,
    model: str,
    fixture: Fixture,
    dobox: Any,
    output_dir: Path,
    dobox_runtime: dict[str, Any] | None = None,
) -> RunResult:
    fixture_root = fixture.fixture_dir
    instruction = fixture.instruction_path.read_text(encoding="utf-8")
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
            user_id="release-eval-suite",
            instruction=instruction,
            provider=provider,
            model=model,
        )
    )
    run_id = job.id
    started_at = job.created_at.isoformat()

    harness_error = False
    try:
        await runner.run_job(job.id)
    except Exception as exc:  # noqa: BLE001 - runner records; never crash the suite
        harness_error = True
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

    inspector = (
        DoBoxWorkspaceInspector(dobox, job.dobox_project_id, job.dobox_agent_session_id)
        if job.dobox_project_id
        else None
    )
    checker_result: dict[str, Any] | None = None
    if inspector is not None:
        ctx = CheckerContext(
            inspector=inspector,
            fixture_root=fixture.workspace_dir,
            job=job,
            steps=steps,
            expected_terminal=fixture.manifest.expected_terminal,
            required_commands=fixture.manifest.required_commands,
        )
        result = await run_checker_module(fixture.checker_path, ctx, safe=True)
        checker_result = result.to_dict()

    await write_evidence_bundle(
        output_dir=output_dir,
        run_id=run_id,
        fixture=case_id,
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

    iterations, tool_calls, outcomes, by_type, no_progress, transport_errors = _count_steps(steps)
    usage = _latest_usage(steps)
    total_tokens = int(usage.get("total_tokens") or usage.get("tokens") or 0)
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    estimated_cost = float(usage.get("cost") or 0.0)

    checker_passed = bool((checker_result or {}).get("passed"))
    terminal_status = job.status.value if hasattr(job.status, "value") else str(job.status)
    outcome = classify_run_outcome(
        terminal_status=terminal_status,
        checker_passed=checker_passed,
        expected_terminal=fixture.manifest.expected_terminal,
        failure_reason=job.failure_reason,
        harness_error=harness_error,
    )
    false_failure, false_success = derive_false_flags(outcome, terminal_status=terminal_status, checker_passed=checker_passed)

    decision_parse_errors = sum(
        1
        for step in steps
        if "unsupported decision type" in str(getattr(step, "content", step)).lower()
    )

    return RunResult(
        suite_run_id=suite_run_id,
        case_id=case_id,
        run_index=run_index,
        job_id=job.id,
        project_id=job.dobox_project_id,
        sandbox_id=job.dobox_sandbox_id if hasattr(job, "dobox_sandbox_id") else None,
        agent_session_id=job.dobox_agent_session_id,
        artifact_id=job.artifact_id,
        provider=provider,
        model=model,
        terminal_status=terminal_status,
        outcome=outcome,
        checker_passed=checker_passed,
        expected_terminal=fixture.manifest.expected_terminal,
        required_commands=list(fixture.manifest.required_commands),
        iterations=iterations,
        llm_decision_count=iterations,
        tool_call_count=tool_calls,
        tool_calls_by_type=by_type,
        edit_count=by_type.get("write_file", 0) + by_type.get("edit_file", 0) + by_type.get("apply_patch", 0),
        command_count=by_type.get("run_command", 0),
        elapsed_seconds=_elapsed(started_at, finished_at),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost=estimated_cost,
        failure_reason=job.failure_reason,
        no_progress_count=no_progress,
        transport_errors=transport_errors,
        decision_parse_errors=decision_parse_errors,
        false_failure=false_failure,
        false_success=false_success,
    )


def _elapsed(started_at: str, finished_at: str) -> float:
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(finished_at)
        return max(0.0, (end - start).total_seconds())
    except Exception:  # noqa: BLE001
        return 0.0
