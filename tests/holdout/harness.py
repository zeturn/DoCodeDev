from __future__ import annotations

import json
import os
import shutil
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from docode.dobox.types import ToolResult

from .definitions import HoldoutCase
from tests.support.local_tools import DiagnosticLocalTools, safe_workspace_path, python_portable_command
from tests.support.path_utils import normalize_path


EDIT_TOOLS = {"write_file", "edit_file", "replace_in_file", "apply_patch"}
READ_TOOLS = {"read_file", "read_file_range", "read_symbol", "search", "list_files"}


def materialize_fixture(case: HoldoutCase, destination: Path) -> Path:
    source = Path(__file__).resolve().parents[1] / "fixtures" / "holdout" / case.fixture
    shutil.copytree(source, destination)
    if case.name == "amber_depth":
        target = destination / "zephyr_lattice.py"
        current = target.read_text(encoding="utf-8")
        filler = "\n".join(f"ZE_{index:04d} = {index}" for index in range(1, 6001))
        target.write_text(current.replace("# HOLDOUT_LARGE_FILE_FILLER", filler), encoding="utf-8")
    return destination


class HoldoutLocalTools(DiagnosticLocalTools):
    def set_detected_command(self, name: str, command: str | None) -> None:
        _ = name, command

    async def read_file(self, path: str) -> ToolResult:
        result = await super().read_file(path)
        limit = 20_000
        if result.exit_code == 0 and len(result.output.encode("utf-8")) > limit:
            encoded = result.output.encode("utf-8")
            output = encoded[:limit].decode("utf-8", errors="ignore")
            return ToolResult(
                tool="read_file",
                output=output + "\n[output truncated; use read_file_range or search]\n",
                exit_code=0,
                metadata={**result.metadata, "original_bytes": len(encoded)},
                truncated=True,
            )
        return result

    async def read_file_range(self, path: str, start_line: int = 1, end_line: int = 120) -> ToolResult:
        normalized = normalize_path(path)
        target = safe_workspace_path(self.workspace, normalized)
        if not target.is_file():
            return ToolResult(tool="read_file_range", output=f"{normalized} not found", exit_code=1, metadata={"path": normalized})
        lines = target.read_text(encoding="utf-8").splitlines()
        start = max(1, int(start_line))
        end = min(len(lines), max(start, int(end_line)))
        output = "\n".join(f"{index}: {line}" for index, line in enumerate(lines[start - 1 : end], start=start))
        return ToolResult(tool="read_file_range", output=output, metadata={"path": normalized, "start_line": start, "end_line": end})

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        workspace_cwd = self.workspace if cwd in {"", ".", "/workspace"} else safe_workspace_path(self.workspace, normalize_path(cwd))
        if "<<'NODE'" not in command:
            executable_command = command
            node_probe = re.match(r"^command -v node >/dev/null 2>&1 &&\s*(.+)$", executable_command, flags=re.DOTALL)
            if node_probe:
                executable_command = node_probe.group(1)
            completed = subprocess.run(
                python_portable_command(executable_command),
                cwd=workspace_cwd,
                shell=True,
                text=True,
                capture_output=True,
                check=False,
                timeout=120,
            )
            result = ToolResult(
                tool="run_command",
                output=completed.stdout + completed.stderr,
                exit_code=completed.returncode,
                metadata={"command": command, "executed_command": python_portable_command(executable_command)},
            )
            self.command_results.append(result)
            return result
        lines = command.splitlines()
        marker_index = next((index for index, line in enumerate(lines) if "node <<'NODE'" in line), -1)
        if marker_index < 0 or lines[-1].strip() != "NODE":
            result = ToolResult(tool="run_command", output="invalid NODE heredoc", exit_code=2, metadata={"command": command})
            self.command_results.append(result)
            return result
        output_parts: list[str] = []
        exit_code = 0
        prefix = "\n".join(lines[: marker_index + 1]).split("node <<'NODE'", 1)[0]
        for pattern in (r"node --check\s+([^\s&]+)", r"node --test\s+([^\s&]+)"):
            match = re.search(pattern, prefix)
            if match:
                completed = subprocess.run(
                    ["node", "--check" if "--check" in pattern else "--test", match.group(1)],
                    cwd=workspace_cwd,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=60,
                )
                output_parts.append(completed.stdout + completed.stderr)
                if completed.returncode != 0:
                    exit_code = completed.returncode
                    break
        if exit_code == 0:
            completed = subprocess.run(
                ["node"],
                input="\n".join(lines[marker_index + 1 : -1]) + "\n",
                cwd=workspace_cwd,
                text=True,
                capture_output=True,
                check=False,
                timeout=45,
            )
            output_parts.append(completed.stdout + completed.stderr)
            exit_code = completed.returncode
        result = ToolResult(
            tool="run_command",
            output="".join(output_parts),
            exit_code=exit_code,
            metadata={"command": command, "executed_command": "node <atomic heredoc>"},
        )
        self.command_results.append(result)
        return result


class ScriptedHoldoutLLM:
    def __init__(self, case: HoldoutCase) -> None:
        actions: list[dict[str, Any]] = [{"tool": "list_files", "args": {"path": "."}}]
        actions.extend({"tool": "read_file", "args": {"path": path}} for path in case.read_paths)
        actions.extend(case.script)
        commands_in_script = {
            str(action.get("args", {}).get("command"))
            for action in case.script
            if action.get("tool") == "run_command"
        }
        actions.extend(
            {"tool": "run_command", "args": {"command": command}}
            for command in case.required_commands
            if command not in commands_in_script or any(action.get("expect_exit") for action in case.script if action.get("args", {}).get("command") == command)
        )
        self.actions = actions
        self.calls = 0

    async def decide(self, *, system, messages, tools, context):
        from docode.llm.runtime import AgentDecision

        _ = system, messages, tools, context
        self.calls += 1
        if self.actions:
            action = self.actions.pop(0)
            return AgentDecision(type="tool_call", tool_name=str(action["tool"]), args=dict(action.get("args") or {}))
        return AgentDecision(
            type="final_candidate",
            summary="Completed the frozen holdout task and ran every required verification command.",
            verification="All required commands passed.",
            remaining_risks=[],
        )


def run_independent_command(workspace: Path, command: str) -> subprocess.CompletedProcess[str]:
    if "<<'NODE'" in command:
        lines = command.splitlines()
        return subprocess.run(
            ["node"],
            input="\n".join(lines[1:-1]) + "\n",
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
    executable = command
    stripped = command.strip()
    if stripped == "python" or stripped.startswith("python "):
        executable = f'"{sys.executable}"{stripped[len("python"):]}'
    return subprocess.run(executable, cwd=workspace, shell=True, text=True, capture_output=True, check=False, timeout=60)


def validate_workspace(case: HoldoutCase, workspace: Path) -> list[str]:
    failures: list[str] = []
    for relative in case.expected_files:
        if not (workspace / relative).is_file():
            failures.append(f"missing expected file: {relative}")
    for command in case.required_commands:
        completed = run_independent_command(workspace, command)
        if completed.returncode != 0:
            failures.append(f"command failed ({command}): {(completed.stdout + completed.stderr)[-1000:]}")
    if case.name == "ivory_quill" and (workspace / "nexora/__main__.py").is_file():
        completed = run_independent_command(workspace, 'python -m nexora "mist river"')
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            failures.append(f"invalid nexora JSON: {exc}")
        else:
            if payload != {"token": "mist river", "segments": 2, "checksum": 997}:
                failures.append(f"wrong nexora payload: {payload!r}")
    if case.name == "silver_source" and (workspace / "mosaic-result.json").is_file():
        try:
            records = json.loads((workspace / "mosaic-result.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"invalid mosaic artifact: {exc}")
        else:
            if records != [
                {"ember_code": "E-17", "caption": "Aster Vale", "drift_index": 9},
                {"ember_code": "E-42", "caption": "Brass Willow", "drift_index": 14},
            ]:
                failures.append(f"wrong mosaic payload: {records!r}")
    if case.name == "sable_manual":
        source = (workspace / "engine/quiet_core.py").read_text(encoding="utf-8")
        if source != 'def stable_identifier(value: str) -> str:\n    return "-".join(value.lower().split())\n':
            failures.append("docs-only task modified engine/quiet_core.py")
    return failures


def summarize_steps(steps: list[Any]) -> dict[str, Any]:
    contents = [step.content for step in steps]
    tool_calls = [content for content in contents if content.get("type") == "tool_call"]
    tool_results = [content for content in contents if content.get("type") == "tool_result"]
    edits = [content for content in tool_results if content.get("tool") in EDIT_TOOLS and content.get("exit_code") == 0]
    reads = [content for content in tool_results if content.get("tool") in READ_TOOLS and content.get("exit_code") == 0]
    commands = [content for content in tool_results if content.get("tool") == "run_command"]
    verifier = [step.content for step in steps if step.kind == "verifier"]
    quality = [content for content in contents if content.get("type") == "quality_gate"]
    first_edit_index = min((step.step_index for step in steps if step.content in edits), default=None)
    first_read_index = min((step.step_index for step in steps if step.content in reads), default=None)
    whole_file_rewrite = any(content.get("tool") == "write_file" for content in edits)
    return {
        "iterations": len([content for content in contents if content.get("type") == "llm_decision"]),
        "llm_decisions": len([content for content in contents if content.get("type") == "llm_decision"]),
        "tool_calls": len(tool_calls),
        "commands_run": len(commands),
        "successful_commands": len([content for content in commands if content.get("exit_code") == 0]),
        "repair_actions": len([content for content in contents if content.get("type") == "repair_action"]),
        "final_candidate_attempted": any(
            (content.get("type") == "llm_decision" and content.get("decision_type") == "final_candidate")
            or content.get("type") == "auto_final_candidate"
            for content in contents
        ),
        "verifier_result": verifier[-1].get("passed") if verifier else None,
        "quality_gate_result": quality[-1].get("passed") if quality else None,
        "read_before_edit": first_read_index is not None and first_edit_index is not None and first_read_index < first_edit_index,
        "whole_file_rewrite": whole_file_rewrite,
    }


def temporary_workspace(case: HoldoutCase):
    holder = TemporaryDirectory()
    workspace = Path(holder.name) / case.name
    materialize_fixture(case, workspace)
    return holder, workspace


def secret_values() -> set[str]:
    values = set()
    for name in ("DEEPSEEK_API_KEY", "DOCODE_DEEPSEEK_API_KEY", "DOCODE_DOBOX_TOKEN", "DOCODE_APICRED_TOKEN"):
        value = os.getenv(name)
        if value:
            values.add(value)
    return values


def sanitize(value: Any) -> Any:
    secrets = secret_values()
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        redacted = value
        for secret in secrets:
            redacted = redacted.replace(secret, "[REDACTED]")
        return redacted
    return value
