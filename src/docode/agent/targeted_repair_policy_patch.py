from __future__ import annotations

from typing import Any


def apply() -> None:
    """Install repair and workflow policy overrides into docode.agent.loop."""

    from docode.agent import loop as loop_module

    if not hasattr(loop_module, "_docode_original_allowed_tool_definitions_for_state"):
        loop_module._docode_original_allowed_tool_definitions_for_state = loop_module.allowed_tool_definitions_for_state
    if not hasattr(loop_module, "_docode_original_required_test_tool_block"):
        loop_module._docode_original_required_test_tool_block = loop_module.required_test_tool_block
    if not hasattr(loop_module, "_docode_original_targeted_repair_targets"):
        loop_module._docode_original_targeted_repair_targets = loop_module.targeted_repair_targets

    loop_module.targeted_repair_targets = targeted_repair_targets
    loop_module.targeted_repair_forced_tool = targeted_repair_forced_tool
    loop_module.targeted_repair_allowed_tools_for_phase = targeted_repair_allowed_tools_for_phase
    loop_module.review_repair_target_files = review_repair_target_files
    loop_module.repair_action_from_quality_gate = repair_action_from_quality_gate
    loop_module.allowed_tool_definitions_for_state = allowed_tool_definitions_for_state
    loop_module.required_test_tool_block = required_test_tool_block


def targeted_repair_targets(state: Any) -> set[str]:
    """Return active repair targets, expanding parser mismatches to consistency files.

    A parsed-value mismatch is not always a crawler.py bug. In generated artifact
    tasks, the model may have produced an inconsistent fixture or test expectation
    (for example fixture owner1 while the generated test expects owner). If the
    repair target is locked to crawler.py only, the loop can reject the correct
    fixture/test consistency repair until max_iterations. For this category, allow
    the parser, tests, and fixture files to be repaired as one consistency unit.
    """

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
    """Force the next valid targeted-repair action."""

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
    if read_count > 0 and tool_name in {"run_command", "read_file", "search", "list_files"}:
        content = loop_module.default_crawler_artifact_file_content(target, state)
        if content is not None:
            return (
                "write_file",
                {"path": target, "content": content},
                "active_repair_requires_target_patch",
            )
    return None


def targeted_repair_allowed_tools_for_phase(state: Any) -> set[str]:
    from docode.agent import loop as loop_module

    if loop_module.targeted_repair_modified_target(state):
        return {"run_command", "git_status", "git_diff"}
    if state.targeted_repair_phase == "inspect_allowed":
        return {
            "read_file",
            "edit_file",
            "write_file",
            "replace_in_file",
            "apply_patch",
            "git_status",
            "git_diff",
        }
    if state.targeted_repair_phase == "edit_forced":
        return {"edit_file", "write_file", "replace_in_file", "apply_patch"}
    return {"read_file", "edit_file", "write_file", "replace_in_file", "apply_patch"}


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
        allowed_tools=["read_file", "edit_file", "write_file", "replace_in_file", "apply_patch", "git_status", "git_diff"],
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
