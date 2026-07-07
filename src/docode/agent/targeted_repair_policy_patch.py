from __future__ import annotations

import importlib.abc
import sys
from types import ModuleType
from typing import Any

_PATCHED_MODULE_IDS: set[int] = set()
_HOOK_INSTALLED = False

INSPECT_TOOLS = {"read_file", "read_file_range", "read_symbol", "edit_file", "write_file", "replace_in_file", "apply_patch", "git_status", "git_diff"}
EDIT_TOOLS = {"edit_file", "write_file", "replace_in_file", "apply_patch"}
RERUN_TOOLS = {"run_command", "git_status", "git_diff"}
BROAD_INSPECT_TOOLS = {"read_file", "search", "list_files"}


def install() -> None:
    """Install the targeted-repair policy for direct loop imports and workers.

    Unit tests often import docode.agent.loop directly, while the worker imports
    loop before calling apply(). This entrypoint supports both cases: patch an
    already-loaded loop module immediately, otherwise install a narrow import hook
    that patches docode.agent.loop after its real loader finishes executing.
    """

    module = sys.modules.get("docode.agent.loop")
    if isinstance(module, ModuleType) and hasattr(module, "allowed_tool_definitions_for_state"):
        patch_loop_module(module)
        return
    install_import_hook()


def apply() -> None:
    """Backward-compatible worker entrypoint."""

    install()


def install_import_hook() -> None:
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    sys.meta_path.insert(0, _LoopPatchFinder())
    _HOOK_INSTALLED = True


class _LoopPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object | None, target: object | None = None):
        if fullname != "docode.agent.loop":
            return None
        for finder in sys.meta_path:
            if finder is self or is_docode_repair_patch_finder(finder):
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                continue
            spec.loader = _LoopPatchLoader(spec.loader)
            return spec
        return None


def is_docode_repair_patch_finder(finder: object) -> bool:
    module = finder.__class__.__module__
    return module.startswith("docode.agent.targeted_repair_")


class _LoopPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self.wrapped = wrapped

    def create_module(self, spec):
        create_module = getattr(self.wrapped, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped.exec_module(module)
        patch_loop_module(module)


def patch_loop_module(loop_module: ModuleType) -> None:
    module_id = id(loop_module)
    if module_id in _PATCHED_MODULE_IDS:
        return

    _save_original(loop_module, "allowed_tool_definitions_for_state")
    _save_original(loop_module, "required_test_tool_block")
    _save_original(loop_module, "targeted_repair_targets")
    _save_original(loop_module, "repair_mode_tool_block")
    _save_original(loop_module, "targeted_repair_exploration_block")
    _save_original(loop_module, "note_targeted_repair_tool_result")
    _save_original(loop_module, "targeted_repair_rerun_satisfied")
    _save_original(loop_module, "targeted_repair_rerun_command_block")
    _save_original(loop_module, "targeted_repair_forced_tool")
    _save_original(loop_module, "targeted_repair_read_count")

    loop_module.targeted_repair_targets = targeted_repair_targets
    loop_module.targeted_repair_forced_tool = targeted_repair_forced_tool
    loop_module.targeted_repair_allowed_tools_for_phase = targeted_repair_allowed_tools_for_phase
    loop_module.targeted_repair_read_count = targeted_repair_read_count
    loop_module.targeted_repair_rerun_satisfied = targeted_repair_rerun_satisfied
    loop_module.targeted_repair_rerun_command_block = targeted_repair_rerun_command_block
    loop_module.repair_mode_tool_block = repair_mode_tool_block
    loop_module.targeted_repair_exploration_block = targeted_repair_exploration_block
    loop_module.note_targeted_repair_tool_result = note_targeted_repair_tool_result
    loop_module.review_repair_target_files = review_repair_target_files
    loop_module.repair_action_from_quality_gate = repair_action_from_quality_gate
    loop_module.allowed_tool_definitions_for_state = allowed_tool_definitions_for_state
    loop_module.required_test_tool_block = required_test_tool_block

    _install_dobox_tools()
    _PATCHED_MODULE_IDS.add(module_id)


def _save_original(module: ModuleType, name: str) -> None:
    original_name = f"_docode_original_{name}"
    if not hasattr(module, original_name) and hasattr(module, name):
        setattr(module, original_name, getattr(module, name))


def _install_dobox_tools() -> None:
    from docode.dobox import tools as dobox_tools_module

    if not hasattr(dobox_tools_module.DoBoxTools, "_docode_original_definitions"):
        dobox_tools_module.DoBoxTools._docode_original_definitions = dobox_tools_module.DoBoxTools.definitions
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


def targeted_repair_phase(state: Any) -> str:
    if state.repair_mode != "targeted_repair" or not state.active_repair_action:
        return "inactive"
    from docode.agent import loop as loop_module

    if loop_module.targeted_repair_modified_target(state):
        return "rerun_required"
    budget = max(0, int((state.active_repair_action or {}).get("initial_inspection_budget") or 0))
    inspected = targeted_repair_read_count(state)
    return "edit_forced" if inspected >= budget else "inspect_allowed"


def targeted_repair_allowed_tools_for_phase(state: Any) -> set[str]:
    phase = targeted_repair_phase(state)
    state.targeted_repair_phase = None if phase == "inactive" else phase
    if phase == "rerun_required":
        return set(RERUN_TOOLS)
    if phase == "edit_forced":
        return set(EDIT_TOOLS)
    if phase == "inspect_allowed":
        return set(INSPECT_TOOLS)
    return set()


def targeted_repair_targets(state: Any) -> set[str]:
    from docode.agent import loop as loop_module

    original = getattr(loop_module, "_docode_original_targeted_repair_targets")
    targets = set(original(state))
    action = state.active_repair_action or {}
    category = str(action.get("category") or "")
    signature = str(action.get("signature") or "")
    instruction = str(action.get("instruction") or "")
    combined = "\n".join([category, signature, instruction]).lower()
    if category == "parsed_value_mismatch" or "parsed_value_mismatch" in signature:
        candidates = ["crawler.py", "tests/test_parser.py", "fixtures/sample.html", "fixtures/sample.csv"]
        allowed = set(state.task_contract.must_modify_files) if state.task_contract is not None else set(candidates)
        targets.update(path for path in candidates if path in allowed)
    elif any(token in combined for token in ("fixture/test consistency", "fixture inconsistent", "test fixture")):
        allowed = set(state.task_contract.must_modify_files) if state.task_contract is not None else set()
        for path in ("tests/test_parser.py", "fixtures/sample.html", "fixtures/sample.csv"):
            if not allowed or path in allowed:
                targets.add(path)
    return {loop_module.normalize_workspace_relative_path(path) for path in targets if path}


def targeted_repair_forced_tool(state: Any, tool_name: str) -> tuple[str, dict[str, object], str] | None:
    from docode.agent import loop as loop_module

    if state.repair_mode != "targeted_repair" or not state.active_repair_action:
        return None

    targets = sorted(loop_module.targeted_repair_targets(state))
    if not targets:
        return None

    if loop_module.targeted_repair_modified_target(state):
        command = next_targeted_repair_rerun_command(state)
        if not command:
            return None
        if tool_name != "run_command":
            return "run_command", {"command": command}, "active_repair_requires_exact_rerun"
        return None

    phase = targeted_repair_phase(state)
    target = targets[0]

    if phase == "edit_forced" and tool_name not in EDIT_TOOLS:
        content = loop_module.default_crawler_artifact_file_content(target, state)
        if content is not None:
            return (
                "write_file",
                {"path": target, "content": content},
                "active_repair_controller_forced_target_edit",
            )
        return None

    read_count = targeted_repair_read_count(state)
    if read_count <= 0 and tool_name in {"run_command", "git_status", "git_diff", "search", "list_files", "web_search", "fetch_url"}:
        return "read_file", {"path": target}, "active_repair_requires_inspection_or_patch"

    if read_count > 0 and tool_name in BROAD_INSPECT_TOOLS:
        symbol = _repair_symbol_hint(state)
        if symbol:
            return "read_symbol", {"path": target, "symbol": symbol, "context_lines": 8}, "active_repair_retarget_repeated_read_to_symbol"
        return "read_file_range", {"path": target, "start_line": 1, "end_line": 160}, "active_repair_retarget_repeated_read_to_range"

    return None


def repair_mode_tool_block(state: Any, tool_name: str) -> str:
    from docode.agent import loop as loop_module

    if state.repair_mode != "targeted_repair" or not state.active_repair_action:
        original = getattr(loop_module, "_docode_original_repair_mode_tool_block")
        return original(state, tool_name)
    allowed = targeted_repair_allowed_tools_for_phase(state)
    if tool_name in allowed:
        return ""
    targets = ", ".join(sorted(loop_module.targeted_repair_targets(state))) or "the target file"
    phase = targeted_repair_phase(state)
    if phase == "edit_forced":
        return f"{tool_name} is blocked while repair_mode=targeted_repair. Inspection budget is exhausted; modify {targets} using edit_file, write_file, replace_in_file, or apply_patch."
    if phase == "rerun_required":
        command = next_targeted_repair_rerun_command(state) or "the repair rerun command"
        return f"{tool_name} is blocked while repair_mode=targeted_repair. Rerun exactly: {command}"
    return f"{tool_name} is blocked while repair_mode=targeted_repair. Allowed tools now: {', '.join(sorted(allowed))}."


def targeted_repair_exploration_block(state: Any, tool_name: str) -> str:
    if state.repair_mode != "targeted_repair" or not state.active_repair_action:
        return ""
    if targeted_repair_phase(state) != "edit_forced":
        return ""
    if tool_name in {"read_file", "read_file_range", "read_symbol", "search", "list_files", "run_command", "web_search", "fetch_url"}:
        targets = ", ".join(str(target) for target in (state.active_repair_action.get("target_files") or [])) or "the target file"
        return f"Targeted repair is in edit_forced. Modify {targets} now; do not inspect or run commands before editing."
    return ""


def note_targeted_repair_tool_result(state: Any, result: Any) -> None:
    from docode.agent import loop as loop_module

    if state.repair_mode != "targeted_repair" or not state.active_repair_action:
        return
    if not result.ok:
        state.targeted_repair_phase = targeted_repair_phase(state)
        return
    if result.tool in {"read_file", "read_file_range", "read_symbol", "search", "list_files"}:
        state.targeted_repair_inspections += 1
    elif result.tool in EDIT_TOOLS:
        state.targeted_repair_edits += 1
    loop_module.refresh_targeted_repair_phase(state)


def targeted_repair_read_count(state: Any) -> int:
    if state.active_repair_action is None:
        return 0
    return sum(
        1
        for message in state.messages[state.active_repair_started_at:]
        if message.get("role") == "tool" and message.get("tool") in {"read_file", "read_file_range", "read_symbol", "search", "list_files"}
    )


def targeted_repair_rerun_satisfied(state: Any) -> bool:
    from docode.agent import loop as loop_module

    action = state.active_repair_action or {}
    commands = [str(command) for command in action.get("rerun_commands") or [] if str(command)]
    if not commands:
        return bool(action) and loop_module.targeted_repair_modified_target(state)
    if not loop_module.targeted_repair_modified_target(state):
        return False
    observed = targeted_repair_successful_commands_after_latest_edit(state)
    return all(any(loop_module.commands_equivalent(seen, command) for seen in observed) for command in commands)


def targeted_repair_successful_commands_after_latest_edit(state: Any) -> list[str]:
    start = state.active_repair_started_at
    for index in range(len(state.messages) - 1, state.active_repair_started_at - 1, -1):
        message = state.messages[index]
        if message.get("role") == "tool" and message.get("tool") in EDIT_TOOLS and int(message.get("exit_code") or 0) == 0:
            start = index + 1
            break
    observed: list[str] = []
    for message in state.messages[start:]:
        if message.get("role") != "tool" or int(message.get("exit_code") or 0) != 0:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        command = " ".join(str(metadata.get("command") or "").split())
        if command:
            observed.append(command)
    return observed


def next_targeted_repair_rerun_command(state: Any) -> str:
    from docode.agent import loop as loop_module

    action = state.active_repair_action or {}
    commands = [str(command) for command in action.get("rerun_commands") or [] if str(command)]
    observed = targeted_repair_successful_commands_after_latest_edit(state)
    for command in commands:
        if not any(loop_module.commands_equivalent(seen, command) for seen in observed):
            return command
    return ""


def targeted_repair_rerun_command_block(state: Any, args: dict[str, object]) -> str:
    from docode.agent import loop as loop_module

    command = next_targeted_repair_rerun_command(state)
    if not command:
        return ""
    observed = " ".join(str(args.get("command") or "").strip().split())
    if loop_module.commands_equivalent(observed, command):
        return ""
    return f"Active targeted repair was modified; rerun this repair command now: {command}"


def allowed_tool_definitions_for_state(definitions: list[Any], state: Any) -> list[Any]:
    from docode.agent import loop as loop_module

    if state.repair_mode == "targeted_repair" and state.active_repair_action:
        allowed = targeted_repair_allowed_tools_for_phase(state)
        return [definition for definition in definitions if getattr(definition, "name", None) in allowed]
    status_output = state.latest_git_status.output if state.latest_git_status is not None else ""
    workflow = loop_module.workflow_snapshot(state, status_output)
    if workflow.phase == loop_module.WorkflowPhase.TEST_REQUIRED and not loop_module.missing_must_modify_targets(state):
        return [definition for definition in definitions if getattr(definition, "name", None) == "run_command"]
    original = getattr(loop_module, "_docode_original_allowed_tool_definitions_for_state")
    return original(definitions, state)


def required_test_tool_block(state: Any, workflow: Any, tool_name: str, args: dict[str, object]) -> str:
    from docode.agent import loop as loop_module

    original = getattr(loop_module, "_docode_original_required_test_tool_block")
    detail = original(state, workflow, tool_name, args)
    if not detail or workflow.phase != loop_module.WorkflowPhase.TEST_REQUIRED:
        return detail
    if "required target files are still missing" in detail:
        return detail
    if tool_name == "run_command":
        return detail
    missing = getattr(workflow, "missing_commands", None) or []
    command = str(missing[0]) if missing else ""
    return f"test_required_exact_command_control: run_command now with exactly: {command}" if command else "test_required_exact_command_control: run the exact required TEST_REQUIRED command now"


def repair_action_from_quality_gate(result: Any):
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
        return loop_module.plan_repair_from_tool_result(tool="run_command", output=issue_text, metadata={"command": "python3 crawler.py --dry-run"})
    from docode.agent.repair_planner import RepairAction

    signature_source = "|".join(f"{getattr(issue, 'code', '')}:{getattr(issue, 'message', '')}:{getattr(issue, 'path', '')}" for issue in blockers)
    return RepairAction(
        category="quality_gate_repair",
        signature="quality:" + str(abs(hash(signature_source)))[:12],
        reason="quality_gate_blocked",
        target_files=target_files,
        allowed_tools=sorted(INSPECT_TOOLS),
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
    parser_issue = any(token in lower for token in ("parser", "parse", "parsed", "output", "owner", "repository", "url", "language", "stars", "forks", "record", "field", "schema", "dry-run", "json", "events.jsonl"))
    fixture_issue = any(token in lower for token in ("fixture", "assertionerror", "tests/test_parser.py"))
    if parser_issue:
        preferred.extend(["crawler.py", "tests/test_parser.py", "fixtures/sample.html", "fixtures/sample.csv"])
    if fixture_issue:
        preferred.extend(["tests/test_parser.py", "fixtures/sample.html", "fixtures/sample.csv"])
    if "schema" in lower:
        preferred.append("schemas/output.schema.json")
    preferred.extend(["crawler.py", "tests/test_parser.py", "fixtures/sample.html", "fixtures/sample.csv", "schemas/output.schema.json", "manifest.json", "sources.json"])
    available = set(task_contract.must_modify_files)
    ordered: list[str] = []
    for path in preferred:
        if path in available and path not in ordered:
            ordered.append(path)
    ordered.extend(path for path in task_contract.must_modify_files if path not in ordered)
    return ordered


def _repair_symbol_hint(state: Any) -> str:
    action = state.active_repair_action or {}
    text = "\n".join(str(action.get(key) or "") for key in ("signature", "reason", "instruction"))
    lowered = text.lower()
    for symbol in ("number_from_text", "parse_trending", "parse_repositories", "parse_repos", "main", "write_events", "dry_run", "preflight"):
        if symbol.lower() in lowered:
            return symbol
    import re

    for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
        candidate = match.group(1)
        if candidate not in {"assertEqual", "assertTrue", "assertFalse", "print", "str", "int"}:
            return candidate
    return ""
