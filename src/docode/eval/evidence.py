"""Evidence bundle builders and workspace inspectors for the eval harness.

This is the single canonical implementation of the per-run evidence bundle
(used by the release vertical-slice runner and the eval-suite runner). The
script ``scripts/run_release_vertical_slice.py`` re-exports these names so its
existing tests keep passing; there is intentionally only one copy.

Secrets are never written: API keys, tokens and authorization headers are
never serialized, and provider base URLs are redacted via :func:`redact_endpoint`.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from docode.storage.models import CodingJob, JobStatus, public_job_dict
from docode.storage.step_redaction import redacted_step_content


def _json_default(value: Any) -> Any:
    """Serialization fallback for evidence JSON.

    Explicitly handles the value types the evidence bundle can carry. It does
    NOT fall back to a broad ``str()`` for unknown types, so a genuinely
    non-serializable object fails loudly instead of being silently mangled.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def redact_endpoint(url: str | None) -> str:
    """Never write a real endpoint/secret into artifacts."""
    if not url:
        return "redacted"
    return "redacted"


class DoBoxWorkspaceInspector:
    """Reads/executes against the real DoBox workspace, independent of the agent."""

    def __init__(self, dobox: Any, project_id: str, session_id: str | None) -> None:
        self.dobox = dobox
        self.project_id = project_id
        self.session_id = session_id

    async def read_text(self, path: str) -> str:
        result = await self.dobox.read_file(self.project_id, path, agent_session_id=self.session_id)
        return result.content if hasattr(result, "content") else str(result)

    async def run_command(self, command: str) -> tuple[int, str]:
        result = await self.dobox.run_command(
            self.project_id, command, cwd="/workspace", timeout_sec=120, output_limit=200_000
        )
        return result.exit_code, result.output


class LocalWorkspaceInspector:
    """Filesystem-backed inspector for unit tests (no DoBox, no provider)."""

    def __init__(self, workspace_root: Path) -> None:
        self.root = Path(workspace_root)

    async def read_text(self, path: str) -> str:
        return (self.root / path).read_text(encoding="utf-8")

    async def run_command(self, command: str) -> tuple[int, str]:
        import subprocess

        completed = subprocess.run(
            command,
            shell=True,
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.returncode, completed.stdout + completed.stderr


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
    dobox: Any,
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
