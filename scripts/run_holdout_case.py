"""Single-case deterministic holdout trace runner.

Runs ONE holdout case through the real Runtime V2 classes (no simplified loop) and
emits a structured trace that captures tool lifecycle, AgentState transitions,
TaskGraph transitions, scheduler freshness and finalization attempts. The trace is
designed to be diffed against a passing run (``scripts/compare_holdout_traces.py``) so
the first semantic divergence between environments can be located precisely.

This script intentionally reuses ``CodingAgentLoop``, ``CodingVerifier``,
``QualityGate``, ``ArtifactExporter``, ``AgentState``, ``TaskGraph``,
``VerificationScheduler`` and ``RepairCoordinator`` from the production runtime. It only
adds non-invasive tracing wrappers around the local tool implementation and the agent
state / finalization entry point.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import locale
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

# Ensure the repository root (containing the ``tests`` package) is importable even when
# only ``PYTHONPATH=src`` is provided by the caller (e.g. CI).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from docode.agent.loop import CodingAgentLoop
from docode.agent.quality_gate import QualityGate
from docode.agent.stop_policy import StopPolicy
from docode.agent.verifier import CodingVerifier
from docode.agent.workflow import final_candidate_gate
from docode.artifacts.exporter import ArtifactExporter
from docode.storage.models import CodingJob, JobStatus, new_id

import docode.agent.loop as loop_mod
from docode.agent.state import AgentState

from tests.holdout.definitions import CASE_BY_NAME
from tests.holdout.harness import (
    HoldoutLocalTools,
    ScriptedHoldoutLLM,
    materialize_fixture,
    summarize_steps,
    validate_workspace,
)
from tests.support.repository import RecordingRepository

TRACE: list[dict[str, Any]] = []


def _collect(event: dict[str, Any]) -> None:
    TRACE.append(event)


def _workspace_table(workspace: Path) -> dict[str, str]:
    table: dict[str, str] = {}
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.name.endswith(".pyc"):
            continue
        rel = path.relative_to(workspace).as_posix()
        table[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return table


def _changed_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    changed: list[str] = []
    for rel in set(before) | set(after):
        if before.get(rel) != after.get(rel):
            changed.append(rel)
    return sorted(changed)


class TracingLocalTools(HoldoutLocalTools):
    def __init__(self, workspace: Path, *, test_command: str = "python -m unittest discover -s tests", collector=None) -> None:
        super().__init__(workspace, test_command=test_command)
        if collector is None:
            self._collector = _collect
        elif callable(collector):
            self._collector = collector
        else:
            self._collector = collector.append

    async def call(self, tool_name: str, args: dict[str, object]) -> Any:
        before = _workspace_table(self.workspace)
        before_hash = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
        self._collector(
            {
                "event_type": "tool_started",
                "tool": tool_name,
                "cwd": str(self.workspace),
                "args_normalized": args,
                "workspace_hash_before": before_hash,
                "changed_paths_before": [],
            }
        )
        try:
            result = await super().call(tool_name, args)
        except Exception as exc:  # noqa: BLE001 - surface the exception as a trace event
            after = _workspace_table(self.workspace)
            after_hash = hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest()
            self._collector(
                {
                    "event_type": "tool_exception",
                    "tool": tool_name,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "workspace_changed_despite_exception": after_hash != before_hash,
                    "changed_paths_after": _changed_paths(before, after),
                }
            )
            raise
        after = _workspace_table(self.workspace)
        after_hash = hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest()
        self._collector(
            {
                "event_type": "tool_completed",
                "tool": tool_name,
                "exit_code": result.exit_code,
                "output": (result.output or "")[:4000],
                "metadata": result.metadata,
                "truncated": result.truncated,
                "workspace_hash_after": after_hash,
                "changed_paths_after": _changed_paths(before, after),
            }
        )
        return result


class TracingAgentState(AgentState):
    def _snapshot_state(self) -> dict[str, Any]:
        nodes: dict[str, Any] = {}
        tg = self.task_graph
        if tg is not None:
            for node_id, node in tg.nodes.items():
                status = node.status
                if hasattr(status, "value"):
                    status = status.value
                nodes[node_id] = {
                    "status": status,
                    "target_files": list(getattr(node, "target_files", []) or []),
                    "evidence_refs": list(getattr(node, "evidence_refs", []) or []),
                    "failure_reason": getattr(node, "failure_reason", None),
                    "attempt_count": getattr(node, "attempt_count", None),
                }
        return {
            "edit_epoch": self.edit_epoch,
            "consecutive_failures": self.consecutive_failures,
            "quality_gate_passed": self.quality_gate_passed,
            "repair_mode": self.repair_mode,
            "task_graph_nodes": nodes,
        }

    def _diff_task_graph(self, before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
        diffs: list[dict[str, Any]] = []
        bn = before["task_graph_nodes"]
        an = after["task_graph_nodes"]
        for node_id in set(bn) | set(an):
            b = bn.get(node_id)
            a = an.get(node_id)
            if b != a:
                diffs.append(
                    {
                        "node_id": node_id,
                        "from": b["status"] if b else None,
                        "to": a["status"] if a else None,
                        "evidence_refs": a["evidence_refs"] if a else None,
                        "failure_reason": a["failure_reason"] if a else None,
                    }
                )
        return diffs

    def add_tool_result(self, result: Any) -> None:
        before = self._snapshot_state()
        super().add_tool_result(result)
        after = self._snapshot_state()
        collector = getattr(type(self), "_collector", _collect)
        collector(
            {
                "event_type": "agent_state_after_tool",
                "tool": result.tool,
                "before": before,
                "after": after,
                "task_graph_transitions": self._diff_task_graph(before, after),
            }
        )


class TracingLoop(CodingAgentLoop):
    async def handle_final_candidate(self, state: AgentState, decision: Any) -> Any:
        attempt = getattr(self, "_fc_attempt", 0) + 1
        self._fc_attempt = attempt
        status = await self.tools.git_status()
        state.latest_git_status = status
        gate = final_candidate_gate(state, status.output)
        scheduler_ready = state.verification_scheduler is None or state.verification_scheduler.next_command() is None
        graph_complete = state.task_graph is None or state.task_graph.complete()
        cf_before = state.consecutive_failures
        await self.repository.add_step(
            state.job.id,
            "system",
            {
                "type": "finalization_attempt",
                "attempt": attempt,
                "task_graph_ready": graph_complete,
                "scheduler_ready": scheduler_ready,
                "gate_allowed": gate.allowed,
                "gate_reason": gate.reason,
                "consecutive_failures_before": cf_before,
            },
        )
        result = await super().handle_final_candidate(state, decision)
        await self.repository.add_step(
            state.job.id,
            "system",
            {
                "type": "finalization_attempt_result",
                "attempt": attempt,
                "outcome": "completed" if result is not None else "rejected",
                "consecutive_failures_after": state.consecutive_failures,
            },
        )
        return result


def capture_environment(workspace: Path, ref_type: str | None) -> dict[str, Any]:
    def _version(cmd: list[str]) -> str | None:
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30, shell=False)
            return (completed.stdout or completed.stderr).strip().splitlines()[0] if (completed.stdout or completed.stderr) else None
        except Exception:  # noqa: BLE001
            return None

    secret_names = ("DEEPSEEK_API_KEY", "DOCODE_DEEPSEEK_API_KEY", "DOCODE_DOBOX_TOKEN", "DOCODE_APICRED_TOKEN")
    secrets = {}
    for name in secret_names:
        value = os.getenv(name)
        if value:
            secrets[name] = {"present": True, "length": len(value), "hash_prefix": hashlib.sha256(value.encode()).hexdigest()[:8]}
        else:
            secrets[name] = {"present": False}

    commit = ""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False, cwd=str(workspace)).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""

    return {
        "os": os.name,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "sys_executable": sys.executable,
        "cwd": os.getcwd(),
        "workspace": str(workspace),
        "default_encoding": sys.getdefaultencoding(),
        "filesystem_encoding": sys.getfilesystemencoding(),
        "locale_preferred_encoding": locale.getpreferredencoding(False),
        "os_linesep": repr(os.linesep),
        "python_hash_seed": os.getenv("PYTHONHASHSEED"),
        "checkout_commit": commit,
        "checkout_ref_type": ref_type,
        "ci_env": {
            "GITHUB_EVENT_NAME": os.getenv("GITHUB_EVENT_NAME"),
            "GITHUB_REF": os.getenv("GITHUB_REF"),
            "GITHUB_SHA": os.getenv("GITHUB_SHA"),
            "GITHUB_BASE_REF": os.getenv("GITHUB_BASE_REF"),
        },
        "git_version": _version(["git", "--version"]),
        "node_version": _version(["node", "--version"]),
        "npm_version": _version(["npm", "--version"]),
        "go_version": _version(["go", "version"]),
        "secrets": secrets,
    }


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _copy_workspace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _workspace_diff(before: Path, after: Path) -> str:
    before_files = {p.relative_to(before).as_posix(): p.read_text(encoding="utf-8", errors="surrogateescape") for p in before.rglob("*") if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc")} if before.exists() else {}
    after_files = {p.relative_to(after).as_posix(): p.read_text(encoding="utf-8", errors="surrogateescape") for p in after.rglob("*") if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc")} if after.exists() else {}
    import difflib

    lines: list[str] = []
    for rel in sorted(set(before_files) | set(after_files)):
        b = before_files.get(rel, "").splitlines(keepends=True)
        a = after_files.get(rel, "").splitlines(keepends=True)
        if b == a:
            continue
        lines.append(f"diff --git a/{rel} b/{rel}\n")
        lines.extend(difflib.unified_diff(b, a, fromfile=f"a/{rel}", tofile=f"b/{rel}"))
    return "".join(lines)


async def run_case(case_name: str, trace_path: Path, workspace_artifacts: Path, keep_workspace: bool, print_events: bool, fail_on_unsuccessful: bool) -> int:
    case = CASE_BY_NAME[case_name]
    collector: list[dict[str, Any]] = []
    TracingAgentState._collector = collector.append

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        workspace = materialize_fixture(case, root / "workspace")
        workspace_before = workspace_artifacts / "workspace-before"
        _copy_workspace(workspace, workspace_before)

        repo = RecordingRepository()
        job = await repo.create_job(
            CodingJob(
                id=new_id("job"),
                user_id="holdout-deterministic",
                instruction=case.instruction,
                max_iterations=36,
                max_runtime_seconds=900,
                max_consecutive_failures=10,
                max_tool_calls=80,
            )
        )
        tools = TracingLocalTools(workspace, test_command=case.required_commands[0], collector=collector)
        loop = TracingLoop(
            llm=ScriptedHoldoutLLM(case),
            tools=tools,
            verifier=CodingVerifier(),
            repository=repo,
            exporter=ArtifactExporter(root / "artifacts", repo, workspace_file_reader=tools.read_file),
            stop_policy=StopPolicy(max_iterations=36, max_runtime_seconds=900, max_consecutive_failures=10, max_tool_calls=80),
            quality_gate=QualityGate(),
        )
        # Make the loop instantiate the tracing AgentState subclass.
        loop_mod.AgentState = TracingAgentState

        ref_type = os.getenv("HOLDOUT_REF_TYPE") or os.getenv("GITHUB_EVENT_NAME")
        environment = capture_environment(workspace, ref_type)

        result = await loop.run(job)
        steps = await repo.list_steps(job.id)
        summary = summarize_steps(steps)
        failures = validate_workspace(case, workspace)

        workspace_after = workspace_artifacts / "workspace-after"
        if keep_workspace:
            _copy_workspace(workspace, workspace_after)
        diff_text = _workspace_diff(workspace_before, workspace_after if keep_workspace else workspace)

        scheduler = None
        if result is not None and hasattr(result, "verification_scheduler"):
            pass
        state = None
        # Reconstruct scheduler/repair/task_graph snapshots from the final AgentState if available.
        # The loop does not expose the state, so derive from steps where possible.
        trace_payload = {
            "case": case_name,
            "environment": environment,
            "result_status": result.status.value if result is not None else None,
            "failure_reason": result.failure_reason if result is not None else None,
            "result_summary": result.result_summary if result is not None else None,
            "summary": summary,
            "validation_failures": failures,
            "events": collector,
        }
        steps_payload = [{"index": step.step_index, "kind": step.kind, "content": step.content} for step in steps]

        _dump_json(trace_path, trace_payload)
        _dump_json(workspace_artifacts / "steps.json", steps_payload)
        _dump_json(workspace_artifacts / "job.json", result.to_dict() if result is not None and hasattr(result, "to_dict") else str(result))
        _dump_json(workspace_artifacts / "environment.json", environment)
        _dump_json(workspace_artifacts / "tool-results.json", [s["content"] for s in steps_payload if s["kind"] == "tool" and s["content"].get("type") == "tool_result"])
        _dump_json(workspace_artifacts / "task-graph.json", _extract_task_graph(steps_payload))
        _dump_json(workspace_artifacts / "scheduler.json", _extract_scheduler(steps_payload))
        _dump_json(workspace_artifacts / "repair.json", _extract_repair(steps_payload))
        (workspace_artifacts / "workspace.diff").write_text(diff_text, encoding="utf-8")

        if print_events:
            for event in collector:
                print(json.dumps(event, default=str))
            print(json.dumps({"type": "run_summary", "status": trace_payload["result_status"], "failures": failures}, default=str))

        success = result is not None and result.status == JobStatus.SUCCEEDED and not failures
        if fail_on_unsuccessful and not success:
            return 1
        return 0


def _extract_task_graph(steps_payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for step in steps_payload:
        content = step["content"]
        if content.get("type") == "task_graph_transition":
            events.append(content)
    return events


def _extract_scheduler(steps_payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for step in steps_payload:
        content = step["content"]
        if content.get("type") in {"verification_scheduler", "scheduler_command"}:
            events.append(content)
    return events


def _extract_repair(steps_payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for step in steps_payload:
        content = step["content"]
        if content.get("type") in {"repair_action", "repair_coordinator"}:
            events.append(content)
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single deterministic holdout case with full tracing.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--workspace-artifacts", required=True)
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--print-events", action="store_true")
    parser.add_argument("--fail-on-unsuccessful", action="store_true")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    workspace_artifacts = Path(args.workspace_artifacts)
    workspace_artifacts.mkdir(parents=True, exist_ok=True)

    return asyncio.run(
        run_case(
            case_name=args.case,
            trace_path=trace_path,
            workspace_artifacts=workspace_artifacts,
            keep_workspace=args.keep_workspace,
            print_events=args.print_events,
            fail_on_unsuccessful=args.fail_on_unsuccessful,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
