from __future__ import annotations

import hashlib
from typing import Any

from docode.agent.repair_planner import (
    RepairAction,
    TARGETED_REPAIR_ALLOWED_TOOLS,
    TARGETED_REPAIR_FORBIDDEN_TOOLS,
    inferred_source_targets,
)
from docode.agent.workflow import WorkflowPhase, commands_equivalent, workflow_snapshot
from docode.dobox.types import ToolResult

UNSAFE_TARGETED_REPAIR_TOOLS = {"web_search", "fetch_url", "preview", "logs"}
FINAL_READY_REPAIR_CATEGORIES_REQUIRING_EXPLICIT_SATISFACTION = {"quality_gate_repair", "review_repair"}


def apply_loop_runtime_fixes(loop_module: Any) -> None:
    """Install narrowly scoped loop fixes for the current branch.

    The long-term goal is still to keep this behavior in production modules, but
    this compatibility hook is already imported by ``docode.agent.__init__`` on
    this branch. Keep it small and idempotent.
    """

    if getattr(loop_module, "_runtime_fixes_applied", False):
        return
    setattr(loop_module, "_runtime_fixes_applied", True)

    _patch_required_command_fallback(loop_module)
    _patch_targeted_repair_action_block(loop_module)
    _patch_targeted_repair_rerun_satisfied(loop_module)
    _patch_stop_policy_for_final_ready(loop_module)
    _patch_maybe_auto_finalize_before_stop(loop_module)
    _patch_dobox_git_helpers(loop_module)


def _patch_required_command_fallback(loop_module: Any) -> None:
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


def _patch_targeted_repair_action_block(loop_module: Any) -> None:
    original_block = loop_module.targeted_repair_action_block

    def patched_targeted_repair_action_block(state, tool_name: str, args: dict[str, object]) -> str:
        if getattr(state, "repair_mode", None) == "targeted_repair" and getattr(state, "active_repair_action", None):
            action = state.active_repair_action or {}
            forbidden = {str(tool) for tool in action.get("forbidden_tools") or [] if str(tool)}
            if tool_name in UNSAFE_TARGETED_REPAIR_TOOLS and tool_name in forbidden:
                return (
                    f"{tool_name} is blocked by the active targeted repair action. "
                    "Use local read/edit/run_command/git tools for this repair instead."
                )
        return original_block(state, tool_name, args)

    loop_module.targeted_repair_action_block = patched_targeted_repair_action_block


def _patch_targeted_repair_rerun_satisfied(loop_module: Any) -> None:
    original_satisfied = loop_module.targeted_repair_rerun_satisfied

    def patched_targeted_repair_rerun_satisfied(state) -> bool:
        if original_satisfied(state):
            return True
        action = state.active_repair_action or {}
        commands = [str(command) for command in action.get("rerun_commands") or [] if str(command)]
        if not commands:
            return False
        start = int(getattr(state, "active_repair_started_at", 0) or 0)
        for message in reversed(state.messages[start:]):
            if message.get("role") != "tool" or str(message.get("tool") or "") != "run_command":
                continue
            if int(message.get("exit_code") or 0) != 0:
                continue
            metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
            observed = " ".join(str(metadata.get("command") or "").split())
            if observed and any(commands_equivalent(observed, command) for command in commands):
                return True
        return False

    loop_module.targeted_repair_rerun_satisfied = patched_targeted_repair_rerun_satisfied


def _patch_stop_policy_for_final_ready(loop_module: Any) -> None:
    stop_policy_cls = getattr(loop_module, "StopPolicy", None)
    if stop_policy_cls is None or getattr(stop_policy_cls, "_runtime_final_ready_stop_patched", False):
        return
    setattr(stop_policy_cls, "_runtime_final_ready_stop_patched", True)
    original_evaluate = stop_policy_cls.evaluate

    def patched_evaluate(self, state):
        decision = original_evaluate(self, state)
        if getattr(decision, "reason", None) != "max_iterations_exceeded":
            return decision
        status = getattr(getattr(state, "latest_git_status", None), "output", "") or ""
        try:
            current_workflow = workflow_snapshot(state, status)
        except Exception:
            return decision
        if current_workflow.phase == WorkflowPhase.FINAL_READY:
            # Give the loop one final pass so its existing FINAL_READY auto-final
            # path can submit a final_candidate instead of failing immediately.
            return type(decision)(False, None)
        return decision

    stop_policy_cls.evaluate = patched_evaluate


def _patch_maybe_auto_finalize_before_stop(loop_module: Any) -> None:
    async def patched_maybe_auto_finalize_before_stop(self, state, stop_reason: str):
        if stop_reason != "max_iterations_exceeded":
            return None
        status = await self.tools.git_status()
        state.latest_git_status = status
        current_workflow = workflow_snapshot(state, status.output)
        if current_workflow.phase != WorkflowPhase.FINAL_READY:
            return None

        if state.active_repair_action:
            active_category = str((state.active_repair_action or {}).get("category") or "")
            if active_category in FINAL_READY_REPAIR_CATEGORIES_REQUIRING_EXPLICIT_SATISFACTION:
                if not loop_module.targeted_repair_rerun_satisfied(state):
                    return None
            else:
                state.active_repair_action = None
                state.active_repair_started_at = 0
                state.targeted_repair_phase = None
                state.targeted_repair_inspections = 0
                state.targeted_repair_edits = 0
                if state.repair_mode == "targeted_repair":
                    state.repair_mode = None

        return await self.auto_finalize_ready_workflow(
            state,
            reason="final_ready_stop_policy_auto_finalized",
            detail="max_iterations_exceeded reached after the workspace became FINAL_READY; submitting final_candidate from workflow evidence.",
            workflow_state=current_workflow.to_dict(),
        )

    loop_module.CodingAgentLoop.maybe_auto_finalize_before_stop = patched_maybe_auto_finalize_before_stop


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
