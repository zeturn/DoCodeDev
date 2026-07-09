from __future__ import annotations

import hashlib
from typing import Any

from docode.agent.repair_planner import (
    RepairAction,
    TARGETED_REPAIR_ALLOWED_TOOLS,
    TARGETED_REPAIR_FORBIDDEN_TOOLS,
    inferred_source_targets,
)
from docode.agent.workflow import commands_equivalent
from docode.dobox.types import ToolResult


def apply_loop_runtime_fixes(loop_module: Any) -> None:
    """Install small loop fixes that keep targeted repair advisory, not coercive.

    The connector used for remote edits only supports whole-file replacement.
    Keeping this small patch module separate avoids risky full rewrites of the
    large loop module while still fixing the runtime paths exercised by real
    LLM diagnostics.
    """

    if getattr(loop_module, "_runtime_fixes_applied", False):
        return
    setattr(loop_module, "_runtime_fixes_applied", True)

    original_plan = loop_module.CodingAgentLoop.plan_targeted_repair_from_failure

    async def patched_plan_targeted_repair_from_failure(self, state, result: ToolResult) -> None:
        previous_signature = str((state.active_repair_action or {}).get("signature") or "")
        await original_plan(self, state, result)
        current_signature = str((state.active_repair_action or {}).get("signature") or "")
        if current_signature and current_signature != previous_signature:
            return
        action = _fallback_required_command_repair(loop_module, state, result)
        if action is not None:
            await self.activate_targeted_repair(state, action, result=result)

    loop_module.CodingAgentLoop.plan_targeted_repair_from_failure = patched_plan_targeted_repair_from_failure

    def advisory_targeted_repair_action_block(state, tool_name: str, args: dict[str, object]) -> str:
        _ = state, tool_name, args
        return ""

    loop_module.targeted_repair_action_block = advisory_targeted_repair_action_block
    _patch_dobox_git_helpers(loop_module)


def _patch_dobox_git_helpers(loop_module: Any) -> None:
    tools_cls = getattr(loop_module, "DoBoxTools", None)
    if tools_cls is None or getattr(tools_cls, "_runtime_git_helpers_patched", False):
        return
    setattr(tools_cls, "_runtime_git_helpers_patched", True)
    original_git_status = tools_cls.git_status
    original_git_diff = tools_cls.git_diff

    async def safe_git_status(self):
        try:
            return await original_git_status(self)
        except Exception as exc:
            return ToolResult(
                tool="git_status",
                output=f"git_status unavailable: {type(exc).__name__}: {exc}",
                exit_code=124,
                metadata={"runtime_safe_fallback": True, "error_type": type(exc).__name__},
            )

    async def safe_git_diff(self):
        try:
            return await original_git_diff(self)
        except Exception as exc:
            return ToolResult(
                tool="git_diff",
                output=f"git_diff unavailable: {type(exc).__name__}: {exc}",
                exit_code=124,
                metadata={"runtime_safe_fallback": True, "error_type": type(exc).__name__},
            )

    tools_cls.git_status = safe_git_status
    tools_cls.git_diff = safe_git_diff


def _fallback_required_command_repair(loop_module: Any, state, result: ToolResult) -> RepairAction | None:
    if result.tool != "run_command" or result.ok:
        return None
    task_contract = state.task_contract
    if task_contract is None:
        return None
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    command = " ".join(str(metadata.get("command") or "").split())
    if not command:
        return None
    required_commands = [str(item) for item in task_contract.must_run_commands if str(item)]
    if not any(commands_equivalent(command, required) for required in required_commands):
        return None

    target_files = _fallback_target_files(task_contract, result.output, command)
    failure_summary = _truncate(result.output, 2400)
    signature_source = f"{command}\n{failure_summary}"
    signature = "failed_required_command:" + hashlib.sha1(signature_source.encode("utf-8", errors="ignore")).hexdigest()[:12]
    target_text = ", ".join(target_files) if target_files else "the relevant source file"
    instruction = (
        "A required verification command failed, but no specific repair plan matched the output.\n\n"
        f"Failed command:\n{command}\n\n"
        f"Failure output summary:\n{failure_summary}\n\n"
        f"Candidate target files: {target_text}.\n"
        "Inspect the failure output and edit a relevant source file before rerunning the command.\n"
        f"After editing, rerun exactly: {command}"
    )
    return RepairAction(
        category="failed_required_command",
        signature=signature,
        reason="required verification command failed",
        target_files=target_files,
        allowed_tools=list(TARGETED_REPAIR_ALLOWED_TOOLS),
        forbidden_tools=list(TARGETED_REPAIR_FORBIDDEN_TOOLS),
        instruction=instruction,
        rerun_commands=[command],
        exploration_forbidden=False,
        initial_inspection_budget=2,
    )


def _fallback_target_files(task_contract, output: str, command: str) -> list[str]:
    targets: list[str] = []
    for path in getattr(task_contract, "must_modify_files", []) or []:
        normalized = _normalize_workspace_path(str(path))
        if normalized and normalized not in targets:
            targets.append(normalized)
    if not targets:
        for path in inferred_source_targets(output, command):
            normalized = _normalize_workspace_path(str(path))
            if normalized and normalized not in targets:
                targets.append(normalized)
    return targets


def _normalize_workspace_path(path: str) -> str:
    value = str(path or "").replace("\\", "/").strip()
    if value.startswith("/workspace/"):
        value = value[len("/workspace/") :]
    elif value.startswith("/workspace"):
        value = value[len("/workspace") :].lstrip("/")
    while value.startswith("./"):
        value = value[2:]
    return value


def _truncate(text: str, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "\n...[truncated]"
