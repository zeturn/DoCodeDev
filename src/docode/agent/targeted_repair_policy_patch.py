from __future__ import annotations

from typing import Any


def apply() -> None:
    """Install repair and workflow policy overrides into docode.agent.loop."""

    from docode.agent import loop as loop_module
    from docode.dobox import tools as dobox_tools_module

    if not hasattr(loop_module, "_docode_original_allowed_tool_definitions_for_state"):
        loop_module._docode_original_allowed_tool_definitions_for_state = loop_module.allowed_tool_definitions_for_state
    if not hasattr(loop_module, "_docode_original_required_test_tool_block"):
        loop_module._docode_original_required_test_tool_block = loop_module.required_test_tool_block
    if not hasattr(loop_module, "_docode_original_targeted_repair_targets"):
        loop_module._docode_original_targeted_repair_targets = loop_module.targeted_repair_targets
    if not hasattr(dobox_tools_module.DoBoxTools, "_docode_original_definitions"):
        dobox_tools_module.DoBoxTools._docode_original_definitions = dobox_tools_module.DoBoxTools.definitions

    loop_module.targeted_repair_targets = targeted_repair_targets
    loop_module.targeted_repair_forced_tool = targeted_repair_forced_tool
    loop_module.targeted_repair_allowed_tools_for_phase = targeted_repair_allowed_tools_for_phase
    loop_module.review_repair_target_files = review_repair_target_files
    loop_module.repair_action_from_quality_gate = repair_action_from_quality_gate
    loop_module.allowed_tool_definitions_for_state = allowed_tool_definitions_for_state
    loop_module.required_test_tool_block = required_test_tool_block

    dobox_tools_module.DoBoxTools.definitions = patched_dobox_definitions
    dobox_tools_module.DoBoxTools.read_file_range = read_file_range
    dobox_tools_module.DoBoxTools.read_symbol = read_symbol


def patched_dobox_definitions(self: Any) -> list[Any]:
    from docode.dobox import tools as dobox_tools_module

    original = getattr(dobox_tools_module.DoBoxTools, "_docode_original_definitions")
    definitions = list(original(self))
    existing = {getattr(definition, "name", "") for definition in definitions}
    if "read_file_range" not in existing:
        definitions.append(
            dobox_tools_module.ToolDefinition(
                "read_file_range",
                "Read a 1-based inclusive line range from a file under /workspace. Use this when read_file output is too long or truncated.",
                {"path": "string", "start_line": "integer", "end_line": "integer"},
                self.read_file_range,
            )
        )
    if "read_symbol" not in existing:
        definitions.append(
            dobox_tools_module.ToolDefinition(
                "read_symbol",
                "Read the definition body for a Python function or class symbol from a file under /workspace, with nearby context lines.",
                {"path": "string", "symbol": "string", "context_lines": "integer"},
                self.read_symbol,
            )
        )
    return definitions


async def read_file_range(self: Any, path: str, start_line: int = 1, end_line: int = 120):
    from docode.dobox import tools as dobox_tools_module
    from docode.dobox.file_readers import read_line_range
    from docode.dobox.types import FileResult

    path_error = dobox_tools_module.workspace_path_error(path)
    if path_error:
        return dobox_tools_module.rejected_tool_result("read_file_range", path_error, {"path": path})
    file_result = await self.client.read_file(self.project_id, path, agent_session_id=self.agent_session_id)
    text = file_result.content if isinstance(file_result, FileResult) else str(file_result)
    output, metadata = read_line_range(text, start_line, end_line)
    metadata = {"path": path, **metadata, "source_truncated": bool(getattr(file_result, "truncated", False))}
    return self._compress("read_file_range", output, 0, metadata, truncated=bool(getattr(file_result, "truncated", False)))


async def read_symbol(self: Any, path: str, symbol: str, context_lines: int = 5):
    from docode.dobox import tools as dobox_tools_module
    from docode.dobox.file_readers import read_python_symbol
    from docode.dobox.types import FileResult, ToolResult

    path_error = dobox_tools_module.workspace_path_error(path)
    if path_error:
        return dobox_tools_module.rejected_tool_result("read_symbol", path_error, {"path": path, "symbol": symbol})
    name = str(symbol or "").strip()
    if not name:
        return ToolResult(tool="read_symbol", output="symbol must be a non-empty string", exit_code=2, metadata={"path": path})
    file_result = await self.client.read_file(self.project_id, path, agent_session_id=self.agent_session_id)
    text = file_result.content if isinstance(file_result, FileResult) else str(file_result)
    output, metadata = read_python_symbol(text, name, context_lines)
    exit_code = 1 if output.startswith("symbol not found:") else 0
    metadata = {"path": path, **metadata, "source_truncated": bool(getattr(file_result, "truncated", False))}
    return self._compress("read_symbol", output, exit_code, metadata, truncated=bool(getattr(file_result, "truncated", False)))


def targeted_repair_targets(state: Any) -> set[str]:
    """Return active repair targets, expanding parser mismatches to consistency files."""

    from docode.agent import loop as loop_module

    original = getattr(loop_module, "_docode_original_targeted_repair_targets")
    targets = set(original(state))
    action = state.active_repair_action or {}
    category = str(action.get("category") or "")
    signature = str(action.get("signature") or "")
    reason = str(action.get("reason") or "")
    instruction = str(action.get("instruction") or "")
    combined = "\n".join([category, signature, reason, instruction]).lower()
    if category == "parsed_value_mismatch" or "parsed_value_mismatch" in signature:
        candidates = [
            "crawler.py",
            "tests/test_parser.py",
            "fixtures/sample.html",
            "fixtures/sample.csv",
        ]
        allowed = set(state.task_contract.must_modify_files) if state.task_contract is not None else set(candidates)
        for path in candidates:
            if path in allowed:
                targets.add(path)
    elif any(token in combined for token in ("fixture/test consistency", "fixture inconsistent", "test fixture")):
        allowed = set(state.task_contract.must_modify_files) if state.task_contract is not None else set()
        for path in ("tests/test_parser.py", "fixtures/sample.html", "fixtures/sample.csv"):
            if not allowed or path in allowed:
                targets.add(path)
    return {loop_module.normalize_workspace_relative_path(path) for path in targets if path}


def targeted_repair_forced_tool(
    state: Any,
    tool_name: str,
) -> tuple[str, dict[str, object], str] | None:
    """Force only deterministic targeted-repair actions."""

    from docode.agent import loop as loop_module

    if state.repair_mode != "targeted_repair" or not state.active_repair_action:
        return None

    targets = sorted(loop_module.targeted_repair_targets(state))
    if not targets:
        return None

    if loop_module.targeted_repair_modified_target(state):
        action = state.active_repair_action or {}
        commands = [str(command) for command in action.get("rerun_commands") or [] if str(command)]
        if not commands:
            return None
        return (
            "run_command",
            {"command": commands[0]},
            "active_repair_requires_exact_rerun",
        )

    target = targets[0]
    read_count = loop_module.targeted_repair_read_count(state)
    if read_count <= 0 and tool_name in {"run_command", "git_status", "git_diff", "search", "list_files"}:
        return (
            "read_file",
            {"path": target},
            "active_repair_requires_inspection_or_patch",
        )
    if read_count > 0 and tool_name == "read_file":
        symbol = _repair_symbol_hint(state)
        if symbol:
            return (
                "read_symbol",
                {"path": target, "symbol": symbol, "context_lines": 8},
                "active_repair_retarget_repeated_read_to_symbol",
            )
        return (
            "read_file_range",
            {"path": target, "start_line": 1, "end_line": 160},
            "active_repair_retarget_repeated_read_to_range",
        )
    return None


def _repair_symbol_hint(state: Any) -> str:
    action = state.active_repair_action or {}
    text = "\n".join(
        str(action.get(key) or "")
        for key in ("signature", "reason", "instruction")
    )
    lowered = text.lower()
    known_symbols = [
        "number_from_text",
        "parse_trending",
        "main",
        "write_events",
        "dry_run",
        "preflight",
    ]
    for symbol in known_symbols:
        if symbol.lower() in lowered:
            return symbol
    import re

    for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
        candidate = match.group(1)
        if candidate not in {"assertEqual", "assertTrue", "assertFalse", "print", "str", "int"}:
            return candidate
    return ""


def targeted_repair_allowed_tools_for_phase(state: Any) -> set[str]:
    from docode.agent import loop as loop_module

    if loop_module.targeted_repair_modified_target(state):
        return {"run_command", "git_status", "git_diff"}
    inspect_tools = {
        "read_file",
        "read_file_range",
        "read_symbol",
        "edit_file",
        "write_file",
        "replace_in_file",
        "apply_patch",
        "git_status",
        "git_diff",
    }
    if state.targeted_repair_phase == "inspect_allowed":
        return inspect_tools
    if state.targeted_repair_phase == "edit_forced":
        return {"edit_file", "write_file", "replace_in_file", "apply_patch"}
    return inspect_tools


def allowed_tool_definitions_for_state(definitions: list[Any], state: Any) -> list[Any]:
    """Hide non-command tools while an exact TEST_REQUIRED command is pending."""

    from docode.agent import loop as loop_module

    loop_module.refresh_targeted_repair_phase(state)
    status_output = state.latest_git_status.output if state.latest_git_status is not None else ""
    workflow = loop_module.workflow_snapshot(state, status_output)
    if workflow.phase == loop_module.WorkflowPhase.TEST_REQUIRED and not loop_module.missing_must_modify_targets(state):
        allowed = {"run_command"}
        return [definition for definition in definitions if getattr(definition, "name", None) in allowed]
    original = getattr(loop_module, "_docode_original_allowed_tool_definitions_for_state")
    return original(definitions, state)


def required_test_tool_block(state: Any, workflow: Any, tool_name: str, args: dict[str, object]) -> str:
    """Keep TEST_REQUIRED strict, but make repeated wrong choices non-fatal control feedback."""

    from docode.agent import loop as loop_module

    original = getattr(loop_module, "_docode_original_required_test_tool_block")
    detail = original(state, workflow, tool_name, args)
    if not detail:
        return ""
    if workflow.phase != loop_module.WorkflowPhase.TEST_REQUIRED:
        return detail
    command = ""
    missing = getattr(workflow, "missing_commands", None) or []
    if missing:
        command = str(missing[0])
    if command:
        return f"test_required_exact_command_control: run_command now with exactly: {command}"
    return "test_required_exact_command_control: run the exact required TEST_REQUIRED command now"


def repair_action_from_quality_gate(result: Any):
    """Turn quality-gate blockers into targeted repairs when paths are known."""

    blockers = list(result.blockers())
    if not blockers:
        return None

    target_files: list[str] = []
    lines = ["Quality gate blocked finalization. Modify the listed target file(s) before running commands again."]
    for issue in blockers:
        path = str(getattr(issue, "path", "") or "").strip()
        if path and path not in target_files:
            target_files.append(path)
        code = str(getattr(issue, "code", "") or "quality_blocker")
        message = str(getattr(issue, "message", "") or "Quality gate blocker")
        hint = str(getattr(issue, "repair_hint", "") or "").strip()
        where = f" ({path})" if path else ""
        lines.append(f"- [{code}] {message}{where}")
        if hint:
            lines.append(f"  Repair hint: {hint}")

    if not target_files:
        from docode.agent import loop as loop_module

        issue_text = "\n".join(f"{getattr(issue, 'code', '')}: {getattr(issue, 'message', '')}" for issue in blockers)
        return loop_module.plan_repair_from_tool_result(
            tool="run_command",
            output=issue_text,
            metadata={"command": "python3 crawler.py --dry-run"},
        )

    from docode.agent.repair_planner import RepairAction

    signature_source = "|".join(
        f"{getattr(issue, 'code', '')}:{getattr(issue, 'message', '')}:{getattr(issue, 'path', '')}"
        for issue in blockers
    )
    return RepairAction(
        category="quality_gate_repair",
        signature="quality:" + str(abs(hash(signature_source)))[:12],
        reason="quality_gate_blocked",
        target_files=target_files,
        allowed_tools=["read_file", "read_file_range", "read_symbol", "edit_file", "write_file", "replace_in_file", "apply_patch", "git_status", "git_diff"],
        forbidden_tools=["run_command", "web_search", "fetch_url", "preview", "logs"],
        instruction="\n".join(lines),
        rerun_commands=[],
        exploration_forbidden=True,
        initial_inspection_budget=1,
    )


def review_repair_target_files(task_contract: Any | None, issue_text: str = "") -> list[str]:
    if task_contract is None or not task_contract.must_modify_files:
        return []

    lower = issue_text.lower()
    preferred: list[str] = []
    parser_issue = any(
        token in lower
        for token in (
            "parser",
            "parse",
            "parsed",
            "output",
            "owner",
            "repository",
            "url",
            "language",
            "stars",
            "forks",
            "record",
            "field",
            "schema",
            "dry-run",
            "json",
            "events.jsonl",
        )
    )
    fixture_inconsistency = any(
        token in lower
        for token in (
            "fixture inconsistent",
            "inconsistent fixture",
            "fixture contradicts",
            "sample html contradicts",
            "test fixture contradicts",
            "fixture/test consistency",
            "assertionerror",
            "tests/test_parser.py",
        )
    )

    if parser_issue:
        preferred.append("crawler.py")
        preferred.append("tests/test_parser.py")
        preferred.append("fixtures/sample.html")
        preferred.append("fixtures/sample.csv")
    if fixture_inconsistency:
        preferred.append("tests/test_parser.py")
        preferred.append("fixtures/sample.html")
        preferred.append("fixtures/sample.csv")
    if "schema" in lower:
        preferred.append("schemas/output.schema.json")

    preferred.extend(
        [
            "crawler.py",
            "tests/test_parser.py",
            "fixtures/sample.html",
            "fixtures/sample.csv",
            "schemas/output.schema.json",
            "manifest.json",
            "sources.json",
        ]
    )
    available = set(task_contract.must_modify_files)
    ordered: list[str] = []
    for path in preferred:
        if path in available and path not in ordered:
            ordered.append(path)
    ordered.extend(path for path in task_contract.must_modify_files if path not in ordered)
    return ordered
