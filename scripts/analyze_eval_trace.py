#!/usr/bin/env python3
"""Read-only causal forensics for DoCode evaluation evidence bundles.

The analyzer reconstructs, for every recorded run, the causal timeline that led
to its terminal result:

* first successful agent operation
* first causal failure (first non-zero tool result, rejected decision, or
  blocked action)
* whether the workspace was ever modified
* whether the task's required/verification commands were executed
* repeated-action fingerprints and no-progress signals
* verifier rejections and the evidence the verifier actually consumed
* terminal classification and hidden-checker outcome

It is intentionally fixture-agnostic: no case identifier, fixture name or
checker path is special-cased.  Evidence is only ever read, never mutated, and
all emitted text is passed through a conservative secret scrubber.

Usage::

    python scripts/analyze_eval_trace.py <suite-or-job-dir> [...] \
        --markdown docs/eval/baseline-v1-failure-analysis.md \
        --json artifacts/analysis/baseline-v1-failure-analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

MAX_SNIPPET = 220
MAX_FINGERPRINT_ARG = 160

SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"), "sk-<redacted>"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{8,}"), "gh_<redacted>"),
    (re.compile(r"\bey[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}"), "<jwt-redacted>"),
    (re.compile(r"(?i)\b(authorization|api[_-]?key|access[_-]?token|secret|password)\b\s*[:=]\s*\S+"), r"\1=<redacted>"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"), "bearer <redacted>"),
)

# Tool names that mutate the agent workspace.  Kept generic on purpose: the
# list mirrors the runtime tool registry rather than any particular fixture.
EDIT_TOOLS = frozenset(
    {"write_file", "edit_file", "replace_in_file", "apply_patch", "delete_file", "create_file", "move_file"}
)
COMMAND_TOOLS = frozenset({"run_command", "run_tests", "run_build", "run_lint", "run_smoke", "preview"})


def scrub(value: Any) -> Any:
    """Recursively remove credential-shaped substrings from evidence text."""

    if isinstance(value, str):
        text = value
        for pattern, replacement in SECRET_PATTERNS:
            text = pattern.sub(replacement, text)
        return text
    if isinstance(value, dict):
        return {key: scrub(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    return value


def snippet(value: Any, limit: int = MAX_SNIPPET) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(scrub(text)).replace("\r\n", "\n").replace("\n", " ⏎ ").strip()
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def load_json(path: Path) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return rows


@dataclass
class TimelineEvent:
    step_index: int
    kind: str
    label: str
    detail: str
    ok: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "kind": self.kind,
            "label": self.label,
            "detail": self.detail,
            "ok": self.ok,
        }


@dataclass
class RunAnalysis:
    run_id: str
    evidence_dir: str
    case_id: str | None = None
    job_id: str | None = None
    project_id: str | None = None
    sandbox_id: str | None = None
    agent_session_id: str | None = None
    artifact_id: str | None = None
    provider: str | None = None
    model: str | None = None
    terminal_status: str | None = None
    terminal_category: str | None = None
    failure_reason: str | None = None
    outcome: str | None = None
    expected_terminal: str | None = None
    checker_passed: bool | None = None
    checker_functional_passed: bool | None = None
    checker_failed_checks: list[str] = field(default_factory=list)
    iterations: int = 0
    decision_types: "Counter[str]" = field(default_factory=Counter)
    tool_calls: "Counter[str]" = field(default_factory=Counter)
    tool_failures: "Counter[str]" = field(default_factory=Counter)
    required_commands: list[str] = field(default_factory=list)
    required_commands_run: list[str] = field(default_factory=list)
    required_commands_passed: list[str] = field(default_factory=list)
    workspace_modified: bool = False
    first_success: TimelineEvent | None = None
    first_failure: TimelineEvent | None = None
    last_edit: TimelineEvent | None = None
    last_verification: TimelineEvent | None = None
    timeline: list[TimelineEvent] = field(default_factory=list)
    repeated_actions: list[dict[str, Any]] = field(default_factory=list)
    blocked_actions: list[dict[str, Any]] = field(default_factory=list)
    no_progress_events: list[dict[str, Any]] = field(default_factory=list)
    verifier_rejections: list[dict[str, Any]] = field(default_factory=list)
    verifier_reused_explicit: list[dict[str, Any]] = field(default_factory=list)
    detected_commands: dict[str, Any] = field(default_factory=dict)
    explicit_commands: list[str] = field(default_factory=list)
    diff_bytes: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def suspected_layer(self) -> str:
        """Coarse attribution of the terminal failure to a runtime layer."""

        if self.terminal_status in {"succeeded", "success"}:
            return "n/a"
        if self.verifier_rejections and self.workspace_modified and self.required_commands_passed:
            return "runtime/finalization-verifier"
        if self.first_failure is not None and self.first_failure.kind == "tool":
            return f"runtime/tool:{self.first_failure.label}"
        if self.blocked_actions:
            return "runtime/repeated-action-guard"
        if self.no_progress_events:
            return "runtime/no-progress-guard"
        if not self.workspace_modified:
            return "agent/no-edit"
        return "unclassified"

    @property
    def repetition_pattern(self) -> str:
        if not self.repeated_actions:
            return "none"
        top = Counter(item["fingerprint"] for item in self.repeated_actions).most_common(1)[0]
        return f"{top[1]}x {top[0]}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "evidence_dir": self.evidence_dir,
            "case_id": self.case_id,
            "job_id": self.job_id,
            "project_id": self.project_id,
            "sandbox_id": self.sandbox_id,
            "agent_session_id": self.agent_session_id,
            "artifact_id": self.artifact_id,
            "provider": self.provider,
            "model": self.model,
            "terminal_status": self.terminal_status,
            "terminal_category": self.terminal_category,
            "failure_reason": self.failure_reason,
            "outcome": self.outcome,
            "expected_terminal": self.expected_terminal,
            "checker_passed": self.checker_passed,
            "checker_functional_passed": self.checker_functional_passed,
            "checker_failed_checks": self.checker_failed_checks,
            "iterations": self.iterations,
            "decision_types": dict(self.decision_types),
            "tool_calls": dict(self.tool_calls),
            "tool_failures": dict(self.tool_failures),
            "required_commands": self.required_commands,
            "required_commands_run": self.required_commands_run,
            "required_commands_passed": self.required_commands_passed,
            "workspace_modified": self.workspace_modified,
            "diff_bytes": self.diff_bytes,
            "detected_commands": self.detected_commands,
            "explicit_commands": self.explicit_commands,
            "first_success": self.first_success.as_dict() if self.first_success else None,
            "first_failure": self.first_failure.as_dict() if self.first_failure else None,
            "last_edit": self.last_edit.as_dict() if self.last_edit else None,
            "last_verification": self.last_verification.as_dict() if self.last_verification else None,
            "repeated_actions": self.repeated_actions,
            "blocked_actions": self.blocked_actions,
            "no_progress_events": self.no_progress_events,
            "verifier_rejections": self.verifier_rejections,
            "verifier_reused_explicit": self.verifier_reused_explicit,
            "suspected_layer": self.suspected_layer,
            "repetition_pattern": self.repetition_pattern,
            "timeline": [event.as_dict() for event in self.timeline],
            "notes": self.notes,
        }


def normalise_command(command: str) -> str:
    return " ".join(str(command).split())


def action_fingerprint(tool: str, args: Any) -> str:
    return f"{tool}({snippet(args, MAX_FINGERPRINT_ARG)})"


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class JobEvidence:
    """Read-only view over a single exported job evidence directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.job = load_json(directory / "job.json") or {}
        self.summary = load_json(directory / "summary.json") or {}
        self.terminal = load_json(directory / "terminal-result.json") or {}
        self.checker = load_json(directory / "checker-result.json")
        self.commands = load_json(directory / "commands.json") or []
        self.steps = load_json(directory / "steps.json") or []
        self.outcomes = load_json(directory / "outcomes.json") or []
        diff_path = directory / "git-diff.patch"
        self.diff_bytes = diff_path.stat().st_size if diff_path.exists() else 0

    @property
    def exists(self) -> bool:
        return bool(self.steps) or bool(self.job) or bool(self.summary)


def analyse_job(directory: Path, result_row: dict[str, Any] | None = None) -> RunAnalysis:
    evidence = JobEvidence(directory)
    job = evidence.job
    row = result_row or {}

    analysis = RunAnalysis(
        run_id=str(row.get("job_id") or job.get("id") or directory.name),
        evidence_dir=directory.as_posix(),
        case_id=row.get("case_id") or evidence.summary.get("fixture"),
        job_id=job.get("id") or row.get("job_id"),
        project_id=str(row.get("project_id") or job.get("dobox_project_id") or "") or None,
        sandbox_id=str(row.get("sandbox_id") or job.get("dobox_sandbox_id") or "") or None,
        agent_session_id=str(row.get("agent_session_id") or job.get("dobox_agent_session_id") or "") or None,
        artifact_id=row.get("artifact_id") or evidence.summary.get("artifact_id"),
        provider=row.get("provider") or job.get("provider"),
        model=row.get("model") or job.get("model"),
        terminal_status=row.get("terminal_status") or evidence.terminal.get("status") or job.get("status"),
        terminal_category=evidence.terminal.get("category"),
        failure_reason=row.get("failure_reason") or evidence.terminal.get("failure_reason") or job.get("failure_reason"),
        outcome=row.get("outcome"),
        expected_terminal=row.get("expected_terminal"),
        checker_passed=row.get("checker_passed"),
        iterations=_as_int(row.get("iterations") or evidence.summary.get("iterations")),
        required_commands=[normalise_command(item) for item in (row.get("required_commands") or [])],
        diff_bytes=evidence.diff_bytes,
    )

    _apply_checker(analysis, evidence.checker)
    _walk_steps(analysis, evidence.steps)
    _apply_commands(analysis, evidence.commands)

    if analysis.diff_bytes > 0:
        analysis.workspace_modified = True
    return analysis


def _apply_checker(analysis: RunAnalysis, checker: Any) -> None:
    if not isinstance(checker, dict):
        return
    if analysis.checker_passed is None:
        analysis.checker_passed = bool(checker.get("passed"))
    checks = checker.get("checks")
    if not isinstance(checks, list):
        return
    failed = [str(check.get("name")) for check in checks if isinstance(check, dict) and not check.get("passed")]
    analysis.checker_failed_checks = failed
    # "functional" = every hidden/behavioural check except the terminal-status
    # bookkeeping check, which only mirrors the runtime's own verdict.
    functional = [
        check
        for check in checks
        if isinstance(check, dict) and str(check.get("name")) not in {"terminal_success", "artifact_present"}
    ]
    if functional:
        analysis.checker_functional_passed = all(bool(check.get("passed")) for check in functional)


def _apply_commands(analysis: RunAnalysis, commands: Any) -> None:
    if not isinstance(commands, list):
        return
    required = {normalise_command(command) for command in analysis.required_commands}
    for entry in commands:
        if not isinstance(entry, dict):
            continue
        command = normalise_command(str(entry.get("command") or ""))
        if command and command in required:
            if command not in analysis.required_commands_run:
                analysis.required_commands_run.append(command)
            if _as_int(entry.get("exit_code"), 1) == 0 and command not in analysis.required_commands_passed:
                analysis.required_commands_passed.append(command)


def _record(analysis: RunAnalysis, event: TimelineEvent) -> None:
    analysis.timeline.append(event)
    if event.ok is True and analysis.first_success is None:
        analysis.first_success = event
    if event.ok is False and analysis.first_failure is None:
        analysis.first_failure = event


def _walk_steps(analysis: RunAnalysis, steps: Sequence[Any]) -> None:
    seen_fingerprints: dict[str, int] = {}
    pending_call: dict[str, Any] | None = None

    for step in steps:
        if not isinstance(step, dict):
            continue
        index = _as_int(step.get("step_index"))
        kind = str(step.get("kind") or "")
        content = step.get("content")
        if not isinstance(content, dict):
            continue
        ctype = str(content.get("type") or "")

        if ctype == "bootstrap":
            detected = content.get("detected_commands")
            if isinstance(detected, dict):
                analysis.detected_commands = {key: value for key, value in detected.items()}
            explicit = content.get("explicit_commands")
            if isinstance(explicit, list):
                analysis.explicit_commands = [normalise_command(str(item)) for item in explicit]
                for command in analysis.explicit_commands:
                    if command not in analysis.required_commands:
                        analysis.required_commands.append(command)
            continue

        if ctype == "llm_decision":
            decision = str(content.get("decision_type") or content.get("decision") or "unknown")
            analysis.decision_types[decision] += 1
            continue

        if ctype == "tool_call":
            pending_call = content
            continue

        if ctype == "tool_result":
            tool = str(content.get("tool") or (pending_call or {}).get("tool") or "unknown")
            args = content.get("args")
            if args is None and pending_call is not None:
                args = pending_call.get("args")
            exit_code = _as_int(content.get("exit_code"), 0)
            ok = exit_code == 0 and not content.get("error")
            analysis.tool_calls[tool] += 1
            if not ok:
                analysis.tool_failures[tool] += 1
            detail = snippet(content.get("summary") or content.get("output") or content.get("error"))
            event = TimelineEvent(index, "tool", tool, f"exit={exit_code} {detail}".strip(), ok)
            _record(analysis, event)

            fingerprint = action_fingerprint(tool, args)
            previous = seen_fingerprints.get(fingerprint)
            if previous is not None:
                analysis.repeated_actions.append(
                    {
                        "fingerprint": fingerprint,
                        "step_index": index,
                        "previous_step_index": previous,
                        "exit_code": exit_code,
                    }
                )
            seen_fingerprints[fingerprint] = index

            if tool in EDIT_TOOLS and ok:
                analysis.workspace_modified = True
                analysis.last_edit = event
            if tool in COMMAND_TOOLS:
                analysis.last_verification = event
            pending_call = None
            continue

        if ctype == "step_outcome":
            status = str(content.get("status") or content.get("outcome") or "")
            if status and status.lower() not in {"ok", "success", "succeeded", "progress"}:
                _record(
                    analysis,
                    TimelineEvent(index, "outcome", status, snippet(content.get("reason") or content.get("detail")), False),
                )
            continue

        if ctype in {"no_progress", "no_progress_detected", "stuck"}:
            analysis.no_progress_events.append({"step_index": index, "detail": snippet(content)})
            continue

        if ctype in {"blocked_action", "repeated_action", "action_blocked"}:
            analysis.blocked_actions.append({"step_index": index, "detail": snippet(content)})
            continue

        if ctype == "quality_gate" and content.get("passed") is False:
            _record(analysis, TimelineEvent(index, "quality_gate", "quality_gate", snippet(content.get("reason")), False))
            continue

        if kind == "verifier" or ctype in {"verification", "final_verification"}:
            passed = content.get("passed")
            reason = snippet(content.get("reason"), 400)
            event = TimelineEvent(index, "verifier", "verification", reason, bool(passed))
            analysis.timeline.append(event)
            if passed is False:
                record: dict[str, Any] = {
                    "step_index": index,
                    "reason": reason,
                    "required_fixes": [snippet(fix) for fix in (content.get("required_fixes") or [])],
                }
                judgement = content.get("llm_judgement")
                if isinstance(judgement, dict):
                    record["llm_judgement_passed"] = judgement.get("passed")
                    record["llm_judgement_reason"] = snippet(judgement.get("reason"), 400)
                for key in ("test", "build", "lint"):
                    detail = content.get(key)
                    if isinstance(detail, dict):
                        record[f"{key}_exit_code"] = detail.get("exit_code")
                        record[f"{key}_output_bytes"] = detail.get("output_bytes")
                analysis.verifier_rejections.append(record)
                if analysis.first_failure is None:
                    analysis.first_failure = event
            explicit = content.get("explicit_commands")
            if isinstance(explicit, list):
                for item in explicit:
                    if not isinstance(item, dict):
                        continue
                    command = normalise_command(str(item.get("command") or ""))
                    exit_code = _as_int(item.get("exit_code"), 1)
                    analysis.verifier_reused_explicit.append(
                        {"step_index": index, "command": command, "exit_code": exit_code}
                    )
                    if command:
                        if command not in analysis.required_commands_run:
                            analysis.required_commands_run.append(command)
                        if exit_code == 0 and command not in analysis.required_commands_passed:
                            analysis.required_commands_passed.append(command)
            continue

        if ctype == "terminal_result":
            analysis.terminal_status = analysis.terminal_status or content.get("status")
            analysis.failure_reason = analysis.failure_reason or content.get("failure_reason")
            continue


def discover_runs(root: Path) -> list[tuple[Path, dict[str, Any] | None]]:
    """Locate job evidence directories under a suite directory or a job dir."""

    if not root.exists():
        return []
    rows_by_job: dict[str, dict[str, Any]] = {}
    results_path = root / "results.jsonl"
    if results_path.exists():
        for row in load_jsonl(results_path):
            job_id = str(row.get("job_id") or "")
            if job_id:
                rows_by_job[job_id] = row

    job_dirs: list[Path] = []
    if (root / "steps.json").exists() or (root / "job.json").exists():
        job_dirs.append(root)
    else:
        for child in sorted(root.iterdir()):
            if child.is_dir() and ((child / "steps.json").exists() or (child / "job.json").exists()):
                job_dirs.append(child)

    runs: list[tuple[Path, dict[str, Any] | None]] = []
    for directory in job_dirs:
        job = load_json(directory / "job.json") or {}
        job_id = str(job.get("id") or directory.name)
        runs.append((directory, rows_by_job.get(job_id)))
    return runs


def aggregate(analyses: Sequence[RunAnalysis]) -> dict[str, Any]:
    first_failure_tool: "Counter[str]" = Counter()
    first_failure_kind: "Counter[str]" = Counter()
    terminal_reasons: "Counter[str]" = Counter()
    suspected_layers: "Counter[str]" = Counter()
    tool_failures: "Counter[str]" = Counter()
    repeated: "Counter[str]" = Counter()
    verifier_reject_reasons: "Counter[str]" = Counter()

    for analysis in analyses:
        if analysis.first_failure is not None:
            first_failure_kind[analysis.first_failure.kind] += 1
            first_failure_tool[analysis.first_failure.label] += 1
        else:
            first_failure_kind["none"] += 1
        terminal_reasons[str(analysis.failure_reason or analysis.terminal_status or "unknown")] += 1
        suspected_layers[analysis.suspected_layer] += 1
        tool_failures.update(analysis.tool_failures)
        for item in analysis.repeated_actions:
            repeated[str(item["fingerprint"]).split("(")[0]] += 1
        for rejection in analysis.verifier_rejections:
            verifier_reject_reasons[str(rejection.get("reason", ""))[:120]] += 1

    return {
        "run_count": len(analyses),
        "first_failure_kind": dict(first_failure_kind.most_common()),
        "first_failure_label": dict(first_failure_tool.most_common()),
        "terminal_reasons": dict(terminal_reasons.most_common()),
        "suspected_layers": dict(suspected_layers.most_common()),
        "tool_failures": dict(tool_failures.most_common()),
        "repeated_action_tools": dict(repeated.most_common()),
        "verifier_rejection_reasons": dict(verifier_reject_reasons.most_common()),
        "workspace_modified": sum(1 for a in analyses if a.workspace_modified),
        "ran_required_commands": sum(1 for a in analyses if a.required_commands_run),
        "passed_required_commands": sum(1 for a in analyses if a.required_commands_passed),
        "verifier_rejected": sum(1 for a in analyses if a.verifier_rejections),
        "checker_functionally_correct": sum(1 for a in analyses if a.checker_functional_passed),
    }


def _bool_cell(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def render_markdown(analyses: Sequence[RunAnalysis], summary: dict[str, Any], sources: Sequence[str]) -> str:
    lines: list[str] = []
    lines.append("# Evaluation trace forensics")
    lines.append("")
    lines.append("Generated by `scripts/analyze_eval_trace.py` (read-only).")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    for source in sources:
        lines.append(f"- `{source}`")
    lines.append("")
    lines.append("## Causal matrix")
    lines.append("")
    lines.append(
        "| Case | First causal failure | First failed tool | Workspace modified | Required cmd run | Required cmd passed | Repetition pattern | Terminal reason | Suspected layer |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for analysis in analyses:
        failure = analysis.first_failure
        first_failure = f"step {failure.step_index}: {failure.label}" if failure else "none"
        failed_tool = failure.label if failure and failure.kind == "tool" else "-"
        lines.append(
            "| {case} | {first} | {tool} | {mod} | {run} | {passed} | {rep} | {terminal} | {layer} |".format(
                case=analysis.case_id or analysis.run_id,
                first=first_failure,
                tool=failed_tool,
                mod=_bool_cell(analysis.workspace_modified),
                run=_bool_cell(bool(analysis.required_commands_run)),
                passed=_bool_cell(bool(analysis.required_commands_passed)),
                rep=analysis.repetition_pattern,
                terminal=analysis.failure_reason or analysis.terminal_status or "unknown",
                layer=analysis.suspected_layer,
            )
        )
    lines.append("")
    lines.append("## Aggregate signals")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Per-run causal timelines")
    lines.append("")
    for analysis in analyses:
        lines.append(f"### {analysis.case_id or analysis.run_id}")
        lines.append("")
        lines.append(f"- job: `{analysis.job_id}`")
        lines.append(
            f"- project/sandbox/session: `{analysis.project_id}` / `{analysis.sandbox_id}` / `{analysis.agent_session_id}`"
        )
        lines.append(f"- artifact: `{analysis.artifact_id}`")
        lines.append(f"- provider/model: `{analysis.provider}` / `{analysis.model}`")
        lines.append(
            f"- terminal: `{analysis.terminal_status}` / category `{analysis.terminal_category}` / reason `{analysis.failure_reason}`"
        )
        lines.append(f"- outcome class: `{analysis.outcome}` (expected terminal `{analysis.expected_terminal}`)")
        lines.append(
            f"- checker: passed={_bool_cell(analysis.checker_passed)}, functional_checks_passed={_bool_cell(analysis.checker_functional_passed)}"
        )
        if analysis.checker_failed_checks:
            lines.append(f"- checker failed checks: {', '.join(f'`{name}`' for name in analysis.checker_failed_checks)}")
        lines.append(f"- iterations: {analysis.iterations}; decisions: {dict(analysis.decision_types)}")
        lines.append(f"- tool calls: {dict(analysis.tool_calls)}")
        lines.append(f"- tool failures: {dict(analysis.tool_failures) or '{}'}")
        lines.append(f"- detected commands: {analysis.detected_commands or '{}'}")
        lines.append(f"- required/explicit commands: {analysis.required_commands or []}")
        lines.append(f"- required commands executed: {analysis.required_commands_run or []}")
        lines.append(f"- required commands passing: {analysis.required_commands_passed or []}")
        lines.append(f"- workspace modified: {_bool_cell(analysis.workspace_modified)} (diff bytes {analysis.diff_bytes})")
        if analysis.first_success:
            lines.append(
                f"- first successful operation: step {analysis.first_success.step_index} `{analysis.first_success.label}`"
            )
        if analysis.first_failure:
            lines.append(
                f"- first causal failure: step {analysis.first_failure.step_index} `{analysis.first_failure.label}` — {analysis.first_failure.detail}"
            )
        if analysis.last_edit:
            lines.append(f"- last successful edit: step {analysis.last_edit.step_index} `{analysis.last_edit.label}`")
        if analysis.last_verification:
            lines.append(
                f"- last verification command: step {analysis.last_verification.step_index} `{analysis.last_verification.label}` ({analysis.last_verification.detail})"
            )
        if analysis.repeated_actions:
            lines.append(f"- repeated actions: {len(analysis.repeated_actions)} (top: {analysis.repetition_pattern})")
        if analysis.blocked_actions:
            lines.append(f"- blocked actions: {len(analysis.blocked_actions)}")
        if analysis.no_progress_events:
            lines.append(f"- no-progress events: {len(analysis.no_progress_events)}")
        if analysis.verifier_rejections:
            lines.append(f"- verifier rejections: {len(analysis.verifier_rejections)}")
            first = analysis.verifier_rejections[0]
            lines.append(f"  - first rejection reason: {first.get('reason')}")
            if first.get("llm_judgement_reason"):
                lines.append(f"  - verifier model verdict: {first.get('llm_judgement_reason')}")
            if "test_output_bytes" in first:
                lines.append(
                    f"  - evidence handed to judge: test exit={first.get('test_exit_code')} bytes={first.get('test_output_bytes')}"
                )
        if analysis.verifier_reused_explicit:
            lines.append(f"- explicit command results known to verifier: {analysis.verifier_reused_explicit}")
        lines.append("")
        lines.append("<details><summary>timeline</summary>")
        lines.append("")
        lines.append("| step | kind | label | ok | detail |")
        lines.append("| --- | --- | --- | --- | --- |")
        for event in analysis.timeline:
            lines.append(
                "| {i} | {k} | {l} | {ok} | {d} |".format(
                    i=event.step_index,
                    k=event.kind,
                    l=event.label,
                    ok=_bool_cell(event.ok),
                    d=event.detail.replace("|", "\\|"),
                )
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Causal forensics for DoCode evaluation evidence (read-only).")
    parser.add_argument("evidence", nargs="+", help="Suite evidence directory or single job evidence directory")
    parser.add_argument("--markdown", help="Write a Markdown report to this path")
    parser.add_argument("--json", dest="json_path", help="Write the structured analysis to this path")
    parser.add_argument("--quiet", action="store_true", help="Do not print the Markdown report to stdout")
    args = parser.parse_args(argv)

    analyses: list[RunAnalysis] = []
    sources: list[str] = []
    for raw in args.evidence:
        root = Path(raw).expanduser()
        runs = discover_runs(root)
        if not runs:
            print(f"warning: no job evidence found under {root}", file=sys.stderr)
            continue
        sources.append(root.as_posix())
        for directory, row in runs:
            analyses.append(analyse_job(directory, row))

    if not analyses:
        print("error: no evidence analysed", file=sys.stderr)
        return 2

    analyses.sort(key=lambda item: (item.case_id or "", item.run_id))
    summary = aggregate(analyses)
    payload = {
        "schema_version": 1,
        "sources": sources,
        "summary": summary,
        "runs": [analysis.as_dict() for analysis in analyses],
    }

    if args.json_path:
        json_path = Path(args.json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = render_markdown(analyses, summary, sources)
    if args.markdown:
        markdown_path = Path(args.markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown, encoding="utf-8")
    if not args.quiet:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
