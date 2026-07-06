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


def targeted_repair_forced_tool(
    state: Any,
    tool_name: str,
) -> tuple[str, dict[str, object], str] | None:
    """Force the next valid targeted-repair action.

    Before the target file changes, wrong exploration/actions are retargeted to an
    inspection or target rewrite. After the target file changes, *any* wrong next
    action is retargeted to the exact rerun command recorded by the repair action.
    This prevents traces from getting stuck in reject loops such as:

      read_file -> rejected -> dry-run -> rejected -> cat file -> rejected

    after the repair target was already patched.
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
        expected = commands[0]
        if tool_name == "run_command":
            args = getattr(state, "_current_tool_args", {}) or {}
            observed = " ".join(str(args.get("command") or "").strip().split())
            if any(loop_module.commands_equivalent(observed, command) for command in commands):
                return None
        return (
            "run_command",
            {"command": expected},
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
