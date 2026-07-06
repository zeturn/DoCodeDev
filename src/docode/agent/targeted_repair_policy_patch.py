from __future__ import annotations

from typing import Any


def apply() -> None:
    """Install targeted-repair policy overrides into docode.agent.loop.

    This module intentionally patches only policy helper functions. The loop calls
    these helpers by module-global name at runtime, so replacing them is enough to
    harden the repair path without rewriting the large agent loop module.
    """

    from docode.agent import loop as loop_module

    loop_module.targeted_repair_forced_tool = targeted_repair_forced_tool
    loop_module.targeted_repair_allowed_tools_for_phase = targeted_repair_allowed_tools_for_phase
    loop_module.review_repair_target_files = review_repair_target_files
    loop_module.repair_action_from_quality_gate = repair_action_from_quality_gate


def targeted_repair_forced_tool(
    state: Any,
    tool_name: str,
) -> tuple[str, dict[str, object], str] | None:
    """Force the next valid targeted-repair action.

    Before the target file changes, wrong exploration/actions are retargeted to an
    inspection or target rewrite. After the target file changes, any next action
    is retargeted to the exact rerun command recorded by the repair action.
    """

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


def repair_action_from_quality_gate(result: Any):
    """Turn quality-gate blockers into targeted repairs when paths are known.

    The previous fallback used repair_mode=quality_repair for markdown blockers
    such as `README.md Usage is empty`. That mode still allowed run_command and
    also prevented FINAL_READY auto-finalization, so the loop could dry-run until
    max_iterations. A path-specific targeted repair forces a real edit to the
    blocker path and then clears itself after that edit because no rerun command
    is required for documentation-only quality fixes.
    """

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
        )
    )

    if parser_issue:
        preferred.append("crawler.py")
    if fixture_inconsistency:
        preferred.append("fixtures/sample.html")
    if "tests/test_parser.py" in lower or "test file" in lower or "parser tests" in lower:
        preferred.append("tests/test_parser.py")
    if "schema" in lower:
        preferred.append("schemas/output.schema.json")

    preferred.extend(
        [
            "crawler.py",
            "fixtures/sample.html",
            "tests/test_parser.py",
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
