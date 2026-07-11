from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import posixpath
import re
import shlex
from dataclasses import replace
from typing import Any
from urllib.parse import urlparse

from docode.agent.context import ContextManager, ContextPack, target_file_guidance
from docode.agent.artifact_validator import ExecutionEvidence
from docode.agent.inspector import ProjectInspector
from docode.agent.prompts import DOCODE_SYSTEM_PROMPT
from docode.agent.quality_gate import QualityGate, QualityGateResult
from docode.agent.repair_planner import (
    RepairAction,
    TARGETED_REPAIR_FORBIDDEN_TOOLS,
    format_repair_action,
    infer_named_fixture_files,
    infer_python_traceback_files,
    plan_repair_from_tool_result,
)
from docode.agent.reviewer import CodeReviewer, ReviewResult
from docode.agent.runtime_components import RuntimeComponents, build_runtime_components
from docode.agent.repository_index import build_remote_repository_index
from docode.agent.workspace_reader import DoBoxWorkspaceReader
from docode.agent.verification_scheduler import VerificationScheduler
from docode.agent.profiles import select_task_profile
from docode.agent.repair_coordinator import RepairPhase
from docode.agent.source_inspection import (
    attempted_source_urls,
    crawler_source_inspection_required,
    instruction_source_urls,
    source_inspection_evidence,
    successful_source_inspection,
)
from docode.agent.state import AgentState
from docode.agent.source_policy import continuation_allowed, source_progress_forced, source_tool_block
from docode.agent.stuck import NO_DIFF_EXPLORATION_BUDGET, REPAIR_ALLOWED_TOOLS, StuckDetector, git_status_clean
from docode.agent.stop_policy import StopPolicy
from docode.agent.task_contract import TaskContract, heredoc_delimiter_from_command, is_crawler_instruction, task_contract_from_instruction
from docode.agent.verifier import CodingVerifier, VerificationResult, changed_files_from_diff, verification_evidence_from_steps
from docode.agent.workflow import (
    WorkflowPhase,
    changed_paths_from_status,
    command_was_run,
    commands_equivalent,
    display_command,
    final_candidate_gate,
    normalize_command,
    successful_edit_tool_called,
    workflow_snapshot,
)
from docode.artifacts.exporter import ArtifactExporter, terminal_artifact_id
from docode.dobox.tools import DoBoxTools
from docode.dobox.types import ToolResult
from docode.llm.runtime import AgentDecision, DecisionLLM, LLMUsageMeter, ProviderUnavailableError
from docode.storage.models import CodingJob, JobStatus
from docode.storage.repository import JobRepository

INITIAL_NO_DIFF_EXPLORATION_BUDGET = NO_DIFF_EXPLORATION_BUDGET
LOCAL_INSPECTION_TOOLS = {"read_file", "read_file_range", "read_symbol", "list_files", "search", "git_status", "git_diff"}
EDIT_TOOLS = {"write_file", "edit_file", "replace_in_file", "apply_patch"}
FOCUSED_REPAIR_READ_TOOLS = {"read_file", "read_file_range", "read_symbol"}
TARGETED_REPAIR_GIT_TOOLS = {"git_status", "git_diff"}
CONTEXT_HEAVY_REPAIR_CATEGORIES = {
    "missing_required_field",
    "parsed_value_mismatch",
    "json_semantic_failure",
    "parser_records_empty",
    "parser_records_too_few",
    "parser_record_count_mismatch",
}
MIN_CONTEXT_REPAIR_INSPECTION_BUDGET = 3
FAILED_REQUIRED_COMMAND_ALLOWED_TOOLS = [
    "read_file",
    "read_file_range",
    "list_files",
    "search",
    "edit_file",
    "write_file",
    "replace_in_file",
    "apply_patch",
    "run_command",
    "git_status",
    "git_diff",
]


class CodingAgentLoop:
    def __init__(
        self,
        *,
        llm: DecisionLLM,
        tools: DoBoxTools,
        verifier: CodingVerifier,
        repository: JobRepository,
        exporter: ArtifactExporter,
        stop_policy: StopPolicy,
        inspector: ProjectInspector | None = None,
        context_manager: ContextManager | None = None,
        usage_meter: LLMUsageMeter | None = None,
        stuck_detector: StuckDetector | None = None,
        quality_gate: QualityGate | None = None,
        reviewer: CodeReviewer | None = None,
        llm_max_attempts: int = 3,
        llm_retry_delays: tuple[float, ...] = (2.0, 5.0),
        llm_decision_timeout_seconds: float = 45.0,
        runtime_components: RuntimeComponents | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.verifier = verifier
        self.repository = repository
        self.exporter = exporter
        self.stop_policy = stop_policy
        self.inspector = inspector or ProjectInspector()
        self.context_manager = context_manager or ContextManager()
        self.usage_meter = usage_meter
        self.stuck_detector = stuck_detector or StuckDetector()
        self.quality_gate = quality_gate or QualityGate()
        self.reviewer = reviewer
        self.llm_max_attempts = max(1, llm_max_attempts)
        self.llm_retry_delays = llm_retry_delays
        self.llm_decision_timeout_seconds = max(1.0, llm_decision_timeout_seconds)
        self.runtime_components = runtime_components

    async def run(self, job: CodingJob) -> CodingJob:
        job = await self.repository.update_job(job.id, status=JobStatus.RUNNING)
        state = AgentState(job=job)
        await self.bootstrap(state)

        while True:
            cancelled = await self.cancelled_job(job.id)
            if cancelled is not None:
                return cancelled
            if state.terminal_repair_reason:
                return await self.fail(job.id, state.terminal_repair_reason)

            self.sync_llm_usage(state)
            stop = self.stop_policy.evaluate(state)
            if stop.should_stop:
                finalized = await self.maybe_auto_finalize_before_stop(state, stop.reason or "stopped")
                if finalized is not None:
                    return finalized
                return await self.fail(job.id, stop.reason or "stopped")

            refresh_targeted_repair_phase(state)
            await self.maybe_execute_controller_source_inspection(state)
            context_pack = await self.collect_observation(state)
            observation = context_pack.render()
            await self.repository.add_step(job.id, "system", observation_step(context_pack))
            current_workflow = workflow_snapshot(state, state.latest_git_status.output if state.latest_git_status else "")
            if await self.maybe_execute_controller_required_command(state, current_workflow):
                completed_workflow = workflow_snapshot(state, state.latest_git_status.output if state.latest_git_status else "")
                if completed_workflow.phase == WorkflowPhase.FINAL_READY and state.repair_mode is None and not state.active_repair_action:
                    finalized = await self.auto_finalize_ready_workflow(
                        state,
                        reason="controller_required_commands_satisfied",
                        detail="The controller executed the final multiline verification command successfully; submitting from complete workflow evidence.",
                        workflow_state=completed_workflow.to_dict(),
                    )
                    if finalized is not None:
                        return finalized
                continue
            stuck = self.stuck_detector.evaluate(state=state, latest_git_status=state.latest_git_status.output if state.latest_git_status else "")
            if stuck.stuck:
                state.stuck_count += 1
                state.add_feedback(f"{stuck.reason}: {stuck.repair_instruction}")
                if state.stuck_count >= 2:
                    state.repair_mode = "must_edit"
                await self.repository.add_step(
                    job.id,
                    "system",
                    {
                        "type": "stuck_detector",
                        "reason": stuck.reason,
                        "repair_instruction": stuck.repair_instruction,
                        "stuck_count": state.stuck_count,
                        "repair_mode": state.repair_mode,
                    },
                )
            try:
                decision = await self.decide_with_transient_retries(state, observation)
            except Exception as exc:
                provider_failure = provider_failure_reason(exc)
                if provider_failure == "llm_auth_failed":
                    await self.record_model_failure(state, provider_failure, str(exc))
                    return await self.fail(job.id, provider_failure)
                if provider_failure is not None:
                    await self.record_model_failure(state, provider_failure, str(exc))
                    return await self.fail(job.id, provider_failure)
                if non_retryable_llm_error(exc):
                    await self.record_model_failure(state, "llm_auth_failed", str(exc))
                    return await self.fail(job.id, "llm_auth_failed")
                current_workflow = workflow_snapshot(state, state.latest_git_status.output if state.latest_git_status else "")
                if current_workflow.phase == WorkflowPhase.FINAL_READY and not state.active_repair_action and state.repair_mode is None:
                    finalized = await self.auto_finalize_ready_workflow(
                        state,
                        reason="final_ready_llm_decision_failed",
                        detail=str(exc),
                        workflow_state=current_workflow.to_dict(),
                    )
                    if finalized is not None:
                        return finalized
                    continue
                await self.record_model_failure(state, "llm_decision_failed", str(exc))
                continue
            self.sync_llm_usage(state)
            await self.repository.add_step(job.id, "llm", decision_to_step(decision, self.usage_meter))
            stop = self.stop_policy.evaluate(state)
            if stop.should_stop:
                return await self.fail(job.id, stop.reason or "stopped")

            if decision.type == "tool_call" and decision.tool_name:
                current_workflow = workflow_snapshot(state, state.latest_git_status.output if state.latest_git_status else "")
                if await self.maybe_activate_required_command_repair(state, current_workflow):
                    continue
                tool_name = decision.tool_name
                tool_args = decision.args or {}
                forced_repair = targeted_repair_forced_tool(state, tool_name, tool_args)
                if forced_repair is not None:
                    forced_tool_name, forced_tool_args, forced_reason = forced_repair
                    await self.repository.add_step(
                        job.id,
                        "system",
                        {
                            "type": "decision_retargeted",
                            "reason": forced_reason,
                            "from_tool": tool_name,
                            "to_tool": forced_tool_name,
                            "target": forced_tool_args.get("path"),
                        },
                    )
                    tool_name = forced_tool_name
                    tool_args = forced_tool_args
                review_retargeted_args = None
                if review_retargeted_args is not None:
                    await self.repository.add_step(
                        job.id,
                        "system",
                        {
                            "type": "decision_retargeted",
                            "reason": "review_repair_requires_artifact_edit",
                            "from_tool": tool_name,
                            "to_tool": "write_file",
                            "target": review_retargeted_args.get("path"),
                        },
                    )
                    tool_name = "write_file"
                    tool_args = review_retargeted_args
                current_tools = allowed_tool_definitions_for_state(self.tools.definitions(), state)
                current_tool_names = {str(getattr(definition, "name", "")) for definition in current_tools}
                if current_tool_names and tool_name not in current_tool_names:
                    same_turn_retry = (
                        is_crawler_instruction(state.job.instruction)
                        and tool_name in {"inspect_source", "fetch_url", "web_search", *LOCAL_INSPECTION_TOOLS}
                    )
                    await self.record_unavailable_tool_requested(
                        state,
                        requested_tool=tool_name,
                        requested_args=tool_args,
                        available_tools=sorted(current_tool_names),
                        workflow_state=current_workflow.to_dict(),
                        increment_iteration=not same_turn_retry,
                        reason="source_inspection_complete_edit_required" if tool_name == "inspect_source" else "tool_not_in_current_schema",
                    )
                    if not same_turn_retry:
                        continue
                    await self.repository.add_step(
                        job.id,
                        "system",
                        {
                            "type": "llm_schema_repair_retry",
                            "reason": "invalid_tool_same_turn_retry",
                            "requested_tool": tool_name,
                            "available_tools": sorted(current_tool_names),
                        },
                    )
                    retry_observation = (
                        observation
                        + "\n\nThe requested tool is not available in this workflow phase. "
                        "The source evidence is already present in Source Inspection memory. "
                        "Choose exactly one available tool now. Read the target file once if needed, otherwise edit it. "
                        "Do not request inspect_source, fetch_url, or web_search."
                    )
                    try:
                        decision = await self.decide_with_transient_retries(state, retry_observation)
                    except Exception as exc:
                        await self.record_model_failure(state, "llm_schema_repair_retry_failed", str(exc))
                        continue
                    self.sync_llm_usage(state)
                    await self.repository.add_step(job.id, "llm", decision_to_step(decision, self.usage_meter))
                    if decision.type != "tool_call" or not decision.tool_name:
                        await self.record_rejected_decision(
                            state,
                            reason="invalid_tool_same_turn_retry_unusable",
                            detail="Schema-repair retry must choose exactly one currently available tool.",
                            workflow_state=current_workflow.to_dict(),
                        )
                        continue
                    tool_name = decision.tool_name
                    tool_args = decision.args or {}
                    current_tools = allowed_tool_definitions_for_state(self.tools.definitions(), state)
                    current_tool_names = {str(getattr(definition, "name", "")) for definition in current_tools}
                    if current_tool_names and tool_name not in current_tool_names:
                        await self.record_unavailable_tool_requested(
                            state,
                            requested_tool=tool_name,
                            requested_args=tool_args,
                            available_tools=sorted(current_tool_names),
                            workflow_state=current_workflow.to_dict(),
                            reason="invalid_tool_same_turn_retry_exhausted",
                        )
                        continue
                source_tool_block = source_inspection_required_tool_block(state, tool_name)
                if source_tool_block:
                    await self.record_rejected_decision(
                        state,
                        reason="source_inspection_required_tool_forbidden",
                        detail=source_tool_block,
                        workflow_state=current_workflow.to_dict(),
                    )
                    continue
                repair_tool_block = repair_mode_tool_block(state, tool_name)
                if repair_tool_block:
                    await self.record_rejected_decision(
                        state,
                        reason=f"{state.repair_mode}_tool_forbidden" if state.repair_mode else "repair_mode_tool_forbidden",
                        detail=repair_tool_block,
                    )
                    continue
                repair_read_result = targeted_repair_read_policy_result(state, tool_name, tool_args)
                if repair_read_result is not None:
                    state.add_tool_result(repair_read_result)
                    await self.repository.add_step(
                        job.id,
                        "tool",
                        {
                            "type": "tool_result",
                            "tool": repair_read_result.tool,
                            "exit_code": repair_read_result.exit_code,
                            "summary": summarize_output(repair_read_result.output),
                            "output": repair_read_result.output,
                            "truncated": repair_read_result.truncated,
                            "metadata": repair_read_result.metadata or {},
                        },
                    )
                    state.iteration += 1
                    continue
                targeted_action_block = targeted_repair_action_block(state, tool_name, tool_args)
                if targeted_action_block:
                    await self.record_rejected_decision(
                        state,
                        reason="targeted_repair_wrong_action",
                        detail=targeted_action_block,
                    )
                    continue
                targeted_exploration_block = targeted_repair_exploration_block(state, tool_name)
                if targeted_exploration_block:
                    await self.record_rejected_decision(
                        state,
                        reason="targeted_repair_exploration_limit",
                        detail=targeted_exploration_block,
                    )
                    continue
                edit_command_block = edit_required_tool_block(state, current_workflow, tool_name, tool_args)
                if edit_command_block:
                    await self.record_rejected_decision(
                        state,
                        reason="edit_required_tool_forbidden",
                        detail=edit_command_block,
                        workflow_state=current_workflow.to_dict(),
                    )
                    continue
                duplicate_read_block = duplicate_read_file_block(state, current_workflow, tool_name, tool_args)
                if duplicate_read_block:
                    retarget_command = duplicate_inspection_required_command_retarget(state, current_workflow)
                    if retarget_command:
                        await self.repository.add_step(
                            job.id,
                            "system",
                            {
                                "type": "decision_retargeted",
                                "reason": "inspection_loop_requires_test_evidence",
                                "from_tool": tool_name,
                                "to_tool": "run_command",
                                "command": retarget_command,
                                "workflow_state": current_workflow.to_dict(),
                            },
                        )
                        tool_name = "run_command"
                        tool_args = {"command": retarget_command, "cwd": "/workspace"}
                    else:
                        result = cached_duplicate_read_result(state, tool_args, duplicate_read_block)
                        state.add_tool_result(result)
                        await self.repository.add_step(
                            job.id,
                            "tool",
                            {
                                "type": "tool_result",
                                "tool": result.tool,
                                "exit_code": result.exit_code,
                                "summary": summarize_output(result.output),
                                "output": result.output,
                                "truncated": result.truncated,
                                "metadata": result.metadata or {},
                            },
                        )
                        state.iteration += 1
                        continue
                test_command_block = required_test_tool_block(state, current_workflow, tool_name, tool_args)
                if test_command_block:
                    if "required target files are still missing" in test_command_block:
                        state.repair_mode = "must_edit"
                    await self.record_rejected_decision(
                        state,
                        reason="test_required_tool_forbidden",
                        detail=test_command_block,
                        workflow_state=current_workflow.to_dict(),
                    )
                    continue
                tool_args = crawler_corrected_fetch_url_args(state, tool_name, tool_args) or tool_args
                external_source_block = crawler_external_source_tool_block(state, tool_name, tool_args)
                if external_source_block:
                    await self.record_rejected_decision(
                        state,
                        reason="crawler_source_domain_forbidden",
                        detail=external_source_block,
                        workflow_state=current_workflow.to_dict(),
                    )
                    continue
                if current_workflow.phase == WorkflowPhase.FINAL_READY:
                    active_category = str((state.active_repair_action or {}).get("category") or "")
                    stale_workflow_repair = state.active_repair_action and active_category not in {
                        "quality_gate_repair",
                        "review_repair",
                    }
                    if stale_workflow_repair:
                        state.active_repair_action = None
                        state.active_repair_started_at = 0
                        state.targeted_repair_phase = None
                        state.targeted_repair_inspections = 0
                        state.targeted_repair_edits = 0
                        if state.repair_mode == "targeted_repair":
                            state.repair_mode = None
                        await self.repository.add_step(
                            job.id,
                            "system",
                            {
                                "type": "repair_action_cleared",
                                "reason": "workflow_final_ready",
                                "detail": "Cleared stale targeted repair state because required workflow commands are already satisfied.",
                            },
                        )
                    if state.repair_mode is None and not state.active_repair_action:
                        finalized = await self.auto_finalize_ready_workflow(
                            state,
                            reason="final_ready_tool_auto_finalized",
                            detail=(
                                f"{decision.tool_name} was requested after the workflow reached FINAL_READY. "
                                "The loop is submitting a final_candidate from workflow evidence instead."
                            ),
                            workflow_state=current_workflow.to_dict(),
                        )
                        if finalized is not None:
                            return finalized
                        continue
                cancelled = await self.cancelled_job(job.id)
                if cancelled is not None:
                    return cancelled
                await self.repository.add_step(
                    job.id,
                    "tool",
                    {
                        "type": "tool_call",
                        "tool": tool_name,
                        "args": sanitize_tool_args(tool_args),
                    },
                )
                try:
                    result = await self.tools.call(tool_name, tool_args)
                except Exception as exc:
                    result = tool_exception_result(tool_name, exc)
                result = enrich_tool_result_metadata(tool_name, tool_args, result)
                state.add_tool_result(result)
                reset_parser_mismatch_convergence(state, result)
                note_targeted_repair_tool_result(state, result)
                await self.repository.add_step(
                    job.id,
                    "tool",
                    {
                        "type": "tool_result",
                        "tool": result.tool,
                        "exit_code": result.exit_code,
                        "summary": summarize_output(result.output),
                        "output": result.output,
                        "truncated": result.truncated,
                        "metadata": result.metadata or {},
                    },
                )
                if not result.ok:
                    await self.plan_targeted_repair_from_failure(state, result)
                elif targeted_repair_rerun_satisfied(state):
                    state.active_repair_action = None
                    state.active_repair_started_at = 0
                    state.targeted_repair_phase = None
                    state.targeted_repair_inspections = 0
                    state.targeted_repair_edits = 0
                    if state.repair_mode == "targeted_repair":
                        state.repair_mode = None
                state.iteration += 1
                continue

            if decision.type == "final_candidate":
                completed = await self.handle_final_candidate(state, decision)
                if completed is not None:
                    return completed
                continue

            await self.record_model_failure(state, "model_returned_unusable_decision", f"decision_type={decision.type}")
            continue

    async def plan_targeted_repair_from_failure(self, state: AgentState, result: ToolResult) -> None:
        action = parser_source_mismatch_repair(state, result)
        if action is None:
            action = plan_repair_from_tool_result(tool=result.tool, output=result.output, metadata=result.metadata or {})
        if action is None:
            action = fallback_required_command_repair(state, result)
        if action is None:
            return
        action = refine_repair_action_targets(action, state.task_contract)
        await self.activate_targeted_repair(state, action, result=result)

    async def maybe_execute_controller_required_command(self, state: AgentState, workflow: Any) -> bool:
        command = controller_owned_required_command(state, workflow)
        if not command:
            return False
        args: dict[str, object] = {"command": command, "cwd": "/workspace"}
        await self.repository.add_step(
            state.job.id,
            "system",
            {
                "type": "required_command_auto_execution",
                "reason": "controller_owned_multiline_verification",
                "command": command,
                "command_summary": display_command(command),
            },
        )
        await self.repository.add_step(
            state.job.id,
            "tool",
            {
                "type": "tool_call",
                "tool": "run_command",
                "args": sanitize_tool_args(args),
            },
        )
        try:
            result = await self.tools.call("run_command", args)
        except Exception as exc:
            result = tool_exception_result("run_command", exc)
        result = enrich_tool_result_metadata("run_command", args, result)
        state.add_tool_result(result)
        reset_parser_mismatch_convergence(state, result)
        note_targeted_repair_tool_result(state, result)
        await self.repository.add_step(
            state.job.id,
            "tool",
            {
                "type": "tool_result",
                "tool": result.tool,
                "exit_code": result.exit_code,
                "summary": summarize_output(result.output),
                "output": result.output,
                "truncated": result.truncated,
                "metadata": result.metadata or {},
            },
        )
        if not result.ok:
            await self.plan_targeted_repair_from_failure(state, result)
        elif targeted_repair_rerun_satisfied(state):
            state.active_repair_action = None
            state.active_repair_started_at = 0
            state.targeted_repair_phase = None
            state.targeted_repair_inspections = 0
            state.targeted_repair_edits = 0
            if state.repair_mode == "targeted_repair":
                state.repair_mode = None
        state.iteration += 1
        return True

    async def maybe_execute_controller_source_inspection(self, state: AgentState) -> bool:
        if not crawler_source_inspection_required(state.job.instruction):
            return False
        if successful_source_inspection(state.messages, state.job.instruction) is not None:
            return False
        if successful_edit_tool_called(state):
            return False
        source_urls = instruction_source_urls(state.job.instruction)
        if not source_urls:
            return False
        selected_url = source_urls[0]
        attempted = attempted_source_urls(state.messages) | state.source_inspection_auto_attempted_urls
        if selected_url in attempted:
            return False
        state.source_inspection_auto_attempted_urls.add(selected_url)
        args: dict[str, object] = {"url": selected_url, "mode": "raw", "max_bytes": 50_000, "timeout": 15}
        await self.repository.add_step(
            state.job.id,
            "system",
            {
                "type": "source_inspection_auto_execution",
                "reason": "crawler_literal_source_requires_sandbox_inspection_before_edit",
                "url": selected_url,
                "execution_scope": "sandbox",
            },
        )
        await self.repository.add_step(
            state.job.id,
            "tool",
            {"type": "tool_call", "tool": "inspect_source", "args": sanitize_tool_args(args)},
        )
        try:
            result = await self.tools.call("inspect_source", args)
        except Exception as exc:
            result = tool_exception_result("inspect_source", exc)
        result = enrich_tool_result_metadata("inspect_source", args, result)
        result = ToolResult(
            tool=result.tool,
            output=result.output,
            exit_code=result.exit_code,
            metadata={**(result.metadata or {}), "controller_owned": True},
            truncated=result.truncated,
        )
        state.add_tool_result(result)
        await self.repository.add_step(
            state.job.id,
            "tool",
            {
                "type": "tool_result",
                "tool": result.tool,
                "exit_code": result.exit_code,
                "summary": summarize_output(result.output),
                "output": result.output,
                "truncated": result.truncated,
                "metadata": result.metadata or {},
            },
        )
        evidence = source_inspection_evidence(state.messages, state.job.instruction)[-1]
        await self.repository.add_step(
            state.job.id,
            "system",
            {"type": "source_inspection_evidence", **evidence.to_dict()},
        )
        return True

    async def maybe_activate_required_command_repair(self, state: AgentState, workflow: Any) -> bool:
        if workflow.phase != WorkflowPhase.REPAIR_REQUIRED:
            return False
        failed = latest_failed_required_command(state)
        if failed is None:
            return False
        action = plan_repair_from_tool_result(tool=failed.tool, output=failed.output, metadata=failed.metadata or {})
        if action is None:
            return False
        action = refine_repair_action_targets(action, state.task_contract)
        current_signature = str((state.active_repair_action or {}).get("signature") or "")
        if state.repair_mode == "targeted_repair" and current_signature == action.signature:
            return False
        await self.activate_targeted_repair(state, action, result=failed)
        return True

    async def auto_finalize_ready_workflow(
        self,
        state: AgentState,
        *,
        reason: str,
        detail: str,
        workflow_state: dict[str, object],
    ) -> CodingJob | None:
        await self.repository.add_step(
            state.job.id,
            "system",
            {
                "type": "auto_final_candidate",
                "reason": reason,
                "detail": truncate_text(detail, 1200),
                "workflow_state": workflow_state,
            },
        )
        return await self.handle_final_candidate(
            state,
            auto_final_candidate_decision(state, reason),
        )

    async def maybe_auto_finalize_before_stop(self, state: AgentState, stop_reason: str) -> CodingJob | None:
        if stop_reason != "max_iterations_exceeded":
            return None
        status = await self.tools.git_status()
        state.latest_git_status = status
        current_workflow = workflow_snapshot(state, status.output)
        if current_workflow.phase != WorkflowPhase.FINAL_READY:
            return None
        if state.active_repair_action and not targeted_repair_rerun_satisfied(state):
            return None
        return await self.auto_finalize_ready_workflow(
            state,
            reason="final_ready_stop_policy_auto_finalized",
            detail="max_iterations_exceeded reached after the workspace became FINAL_READY; submitting final_candidate from workflow evidence.",
            workflow_state=current_workflow.to_dict(),
        )

    async def handle_final_candidate(self, state: AgentState, decision: AgentDecision) -> CodingJob | None:
        job = state.job
        cancelled = await self.cancelled_job(job.id)
        if cancelled is not None:
            return cancelled
        if crawler_source_inspection_required(job.instruction) and successful_source_inspection(state.messages, job.instruction) is None:
            await self.record_rejected_decision(
                state,
                reason="source_inspection_required_before_final",
                detail=(
                    "A successful sandbox inspect_source result for a literal task source is required before final_candidate. "
                    "web_search and host fetch_url results do not satisfy this requirement."
                ),
            )
            return None
        status = await self.tools.git_status()
        state.latest_git_status = status
        final_summary = (decision.summary or "").strip()
        if not final_summary:
            await self.record_model_failure(
                state,
                "final_summary_missing",
                "final_candidate must include a non-empty summary before verification can complete",
            )
            return None
        gate = final_candidate_gate(state, status.output)
        if gate.allowed and state.active_repair_action and not targeted_repair_rerun_satisfied(state):
            state.active_repair_action = None
            state.active_repair_started_at = 0
            state.targeted_repair_phase = None
            state.targeted_repair_inspections = 0
            state.targeted_repair_edits = 0
            if state.repair_mode == "targeted_repair":
                state.repair_mode = None
            await self.repository.add_step(
                job.id,
                "system",
                {
                    "type": "repair_action_cleared",
                    "reason": "workflow_final_ready",
                    "detail": "Cleared stale targeted repair state because required workflow commands are already satisfied.",
                },
            )
        if not gate.allowed:
            if gate.repair_mode:
                state.repair_mode = gate.repair_mode
            await self.record_rejected_decision(
                state,
                reason=gate.reason,
                detail=gate.detail,
                workflow_state=gate.snapshot.to_dict(),
            )
            return None
        scheduled = state.verification_scheduler.next_command() if state.verification_scheduler is not None else None
        if scheduled:
            await self.record_rejected_decision(
                state,
                reason="verification_scheduler_stale",
                detail=f"Verification evidence is stale or incomplete. Run the scheduler-selected command exactly: {display_command(scheduled)}",
                workflow_state=gate.snapshot.to_dict(),
            )
            return None
        quality_kwargs: dict[str, Any] = {"tools": self.tools, "task_contract": state.task_contract, "instruction": job.instruction}
        quality_parameters = inspect.signature(self.quality_gate.run).parameters
        if "artifact_contract" in quality_parameters:
            quality_kwargs["artifact_contract"] = state.artifact_contract
        if "execution_evidence" in quality_parameters:
            quality_kwargs["execution_evidence"] = artifact_execution_evidence(state)
        quality = await self.quality_gate.run(**quality_kwargs)
        state.quality_gate_attempts += 1
        state.last_quality_gate = quality.to_dict()
        await self.repository.add_step(job.id, "system", quality.to_dict())
        if not quality.passed:
            state.quality_gate_passed = False
            quality_action = repair_action_from_quality_gate(quality)
            if quality_action is not None:
                await self.activate_targeted_repair(state, quality_action)
            else:
                state.repair_mode = "quality_repair"
                state.add_feedback(quality_repair_feedback(quality))
            state.iteration += 1
            return None
        state.quality_gate_passed = True
        if state.repair_mode == "quality_repair":
            state.repair_mode = None
        await self.repository.add_step(
            job.id,
            "system",
            {"type": "finalization_stage", "stage": "verification_start"},
        )
        await self.repository.update_job(job.id, status=JobStatus.VERIFYING)
        evidence = verification_evidence_from_steps(
            await self.repository.list_steps(job.id),
            explicit_commands=list(state.task_contract.must_run_commands) if state.task_contract else [],
        ).with_no_test_reason(decision.no_test_reason)
        try:
            verify_kwargs: dict[str, Any] = {"evidence": evidence}
            verify_parameters = inspect.signature(self.verifier.verify).parameters
            supports_context = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in verify_parameters.values())
            if supports_context or "task_contract" in verify_parameters:
                verify_kwargs["task_contract"] = state.task_contract
            if supports_context or "inspection" in verify_parameters:
                verify_kwargs["inspection"] = state.inspection
            verification = await asyncio.wait_for(
                self.verifier.verify(job, self.tools, **verify_kwargs),
                timeout=180,
            )
        except Exception as exc:
            await self.repository.add_step(
                job.id,
                "system",
                {"type": "finalization_stage_failed", "stage": "verification", "error": str(exc)},
            )
            raise
        self.sync_llm_usage(state)
        await self.repository.add_step(job.id, "verifier", verification_to_dict(verification))
        stop = self.stop_policy.evaluate(state)
        if stop.should_stop and stop.reason != "max_iterations_exceeded":
            return await self.fail(job.id, stop.reason or "stopped")
        if not verification.passed:
            if requires_non_empty_diff_repair(verification):
                state.repair_mode = "must_edit"
            elif verification.required_fixes:
                state.repair_mode = "quality_repair"
            state.add_feedback(verification_repair_feedback(verification, state.task_contract))
            state.iteration += 1
            return None
        await self.repository.add_step(
            job.id,
            "system",
            {"type": "finalization_stage", "stage": "review_start"},
        )
        try:
            review = await asyncio.wait_for(self.run_independent_review(state, quality, final_summary), timeout=120)
        except Exception as exc:
            await self.repository.add_step(
                job.id,
                "system",
                {"type": "finalization_stage_failed", "stage": "review", "error": str(exc)},
            )
            raise
        if review is not None:
            self.sync_llm_usage(state)
            await self.repository.add_step(job.id, "system", review.to_dict())
            if not review.passed:
                review_action = repair_action_from_review(review, state.task_contract)
                if review_action is not None:
                    state.add_feedback(review_repair_feedback(review))
                    await self.activate_targeted_repair(state, review_action)
                else:
                    state.repair_mode = "quality_repair"
                    state.add_feedback(review_repair_feedback(review))
                state.iteration += 1
                return None
        await self.repository.add_step(
            job.id,
            "system",
            {"type": "finalization_stage", "stage": "export_start"},
        )
        try:
            artifacts = await asyncio.wait_for(self.exporter.export_success(job, verification, final_summary), timeout=60)
        except Exception as exc:
            await self.repository.add_step(
                job.id,
                "system",
                {"type": "finalization_stage_failed", "stage": "export", "error": str(exc)},
            )
            raise
        artifact_id = terminal_artifact_id(artifacts)
        await self.repository.add_step(
            job.id,
            "system",
            {"type": "finalization_stage", "stage": "job_complete", "artifact_id": artifact_id},
        )
        return await self.repository.update_job(
            job.id,
            status=JobStatus.SUCCEEDED,
            result_summary=final_summary,
            artifact_id=artifact_id,
        )

    async def activate_targeted_repair(self, state: AgentState, action: RepairAction, result: ToolResult | None = None) -> None:
        command = str((result.metadata or {}).get("command") or "") if result is not None else ""
        if command:
            state.last_failed_command = command
        if state.repair_coordinator is None:
            components = getattr(self, "runtime_components", None)
            state.repair_coordinator = components.repair_coordinator if components is not None else build_runtime_components(state.job.instruction).repair_coordinator
        phase = state.repair_coordinator.activate(action)
        count = state.repair_coordinator.attempt_count(action.signature)
        state.failure_signatures = {action.signature: count}
        state.repair_action_attempts = count
        if phase == RepairPhase.NON_CONVERGENT:
            state.terminal_repair_reason = "repeated_zero_record_parser_failure" if action.failure_class == "parser_source_mismatch" else f"repair_non_convergent:{action.signature}"
            state.add_feedback(
                "non_convergent_repair: The same failure signature remained after the configured maximum attempts. "
                "Stopping instead of continuing blind rewrites."
            )
            await self.repository.add_step(
                state.job.id,
                "system",
                {
                    "type": "non_convergent_repair",
                    "reason": state.terminal_repair_reason,
                    "signature": action.signature,
                    "repeated_count": count,
                },
            )
            return
        action_payload = repair_action_contract(action, state)
        state.active_repair_action = action_payload
        state.active_repair_started_at = repair_action_start_index(state, result)
        state.targeted_repair_phase = phase.value
        state.targeted_repair_inspections = 0
        state.targeted_repair_edits = 0
        state.repair_mode = "targeted_repair"
        state.add_feedback(format_repair_action(action, repeated_count=count))
        state.consecutive_failures = 0
        await self.repository.add_step(
            state.job.id,
            "system",
            {
                "type": "repair_action",
                "repair_action": action_payload,
                "repeated_count": count,
            },
        )

    async def run_independent_review(self, state: AgentState, quality: QualityGateResult, final_summary: str) -> ReviewResult | None:
        if self.reviewer is None:
            return None
        return await self.reviewer.review(
            instruction=state.job.instruction,
            task_contract=state.task_contract,
            quality=quality,
            recent_tool_results=recent_tool_results(state),
            final_summary=final_summary,
        )

    async def cancelled_job(self, job_id: str) -> CodingJob | None:
        current = await self.repository.get_job(job_id)
        if current is not None and current.status == JobStatus.STOPPED:
            reason = current.failure_reason or "cancelled"
            await self.repository.add_step(job_id, "system", {"type": "cancelled_observed", "reason": reason})
            if current.artifact_id:
                return current
            artifact_id = None
            try:
                git_diff = ""
                git_diff_truncated = False
                try:
                    diff_result = await self.tools.git_diff()
                    git_diff = diff_result.output
                    git_diff_truncated = diff_result.truncated
                except Exception:
                    git_diff = ""
                    git_diff_truncated = False
                artifacts = await self.exporter.export_stopped(
                    current,
                    reason,
                    steps=await self.repository.list_steps(job_id),
                    git_diff=git_diff,
                    git_diff_truncated=git_diff_truncated,
                )
                artifact_id = terminal_artifact_id(artifacts)
                await self.repository.add_step(job_id, "system", {"type": "stopped_artifacts_exported", "reason": reason, "artifact_id": artifact_id})
            except Exception as exc:
                await self.repository.add_step(job_id, "system", {"type": "stopped_export_failed", "reason": reason, "error": str(exc)})
            return await self.repository.update_job(job_id, artifact_id=artifact_id)
        return None

    async def bootstrap(self, state: AgentState) -> None:
        components = self.runtime_components or build_runtime_components(state.job.instruction)
        self.runtime_components = components
        state.profile = components.profile
        state.task_contract = components.task_contract
        state.artifact_contract = components.artifact_contract
        state.verification_scheduler = components.verification_scheduler
        state.repair_coordinator = components.repair_coordinator
        state.repository_context = components.repository_context
        state.task_graph = components.task_graph
        state.finalization_controller = components.finalization_controller
        if state.profile.context_policy.use_repository_index and all(hasattr(self.tools, name) for name in ("list_files", "read_file", "search")):
            state.repository_context = await build_remote_repository_index(DoBoxWorkspaceReader(self.tools))
            components.repository_context = state.repository_context
        inspection = await self.inspector.inspect(state.job.instruction, self.tools, state.task_contract)
        state.inspection = inspection
        await self.repository.add_step(
            state.job.id,
            "system",
            {
                "type": "bootstrap",
                "listing": inspection.listing,
                "important_files": list(inspection.important_files),
                "detected_commands": inspection.detected_commands,
                "explicit_commands": inspection.explicit_commands,
                "plan": inspection.plan,
                "acceptance_criteria": inspection.acceptance_criteria,
                "task_contract": {
                    "must_modify_files": state.task_contract.must_modify_files,
                    "must_run_commands": state.task_contract.must_run_commands,
                    "forbidden_finish_conditions": state.task_contract.forbidden_finish_conditions,
                },
                "runtime_components": {
                    "profile": state.profile.name,
                    "repository_files": len(state.repository_context.files) if state.repository_context is not None else 0,
                    "repository_symbols": len(state.repository_context.symbols) if state.repository_context is not None else 0,
                    "task_nodes": list(state.task_graph.nodes) if state.task_graph is not None else [],
                    "verification_commands": [node.command for node in state.verification_scheduler.commands],
                },
            },
        )

    async def collect_observation(self, state: AgentState) -> ContextPack:
        refresh_targeted_repair_phase(state)
        status = await self.tools.git_status()
        state.latest_git_status = status
        if state.repair_mode == "must_edit" and not git_status_clean(status.output):
            state.repair_mode = None
            state.stuck_count = 0
        await self.repository.add_step(
            state.job.id,
            "system",
            {
                "type": "workflow_state",
                **workflow_snapshot(state, status.output).to_dict(),
            },
        )
        workflow = workflow_snapshot(state, status.output)
        include_source_body = not successful_edit_tool_called(state)
        context_pack = self.context_manager.build_pack(
            job=state.job,
            inspection=state.inspection,
            messages=state.messages,
            git_status=status,
            iteration=state.iteration,
            tool_calls_count=state.tool_calls_count,
            llm_tokens_used=state.llm_tokens_used,
            llm_cost_used=state.llm_cost_used,
            task_contract=state.task_contract,
            repair_mode=state.repair_mode,
            active_repair_action=state.active_repair_action,
            targeted_repair_phase=state.targeted_repair_phase,
            workflow_phase=workflow.phase.value if hasattr(workflow.phase, "value") else str(workflow.phase),
            include_source_body=include_source_body,
            profile=state.profile,
            repository_context=state.repository_context,
            task_graph=state.task_graph,
        )
        return context_pack

    async def fail(self, job_id: str, reason: str) -> CodingJob:
        current = await self.repository.get_job(job_id)
        if current is None:
            return await self.repository.update_job(job_id, status=JobStatus.FAILED, failure_reason=reason)
        artifact_id = None
        try:
            git_diff = ""
            git_diff_truncated = False
            try:
                diff_result = await self.tools.git_diff()
                git_diff = diff_result.output
                git_diff_truncated = diff_result.truncated
            except Exception:
                git_diff = ""
                git_diff_truncated = False
            artifacts = await self.exporter.export_failure(
                current,
                reason,
                steps=await self.repository.list_steps(job_id),
                git_diff=git_diff,
                git_diff_truncated=git_diff_truncated,
            )
            artifact_id = terminal_artifact_id(artifacts)
            await self.repository.add_step(job_id, "system", {"type": "failure_artifacts_exported", "reason": reason, "artifact_id": artifact_id})
        except Exception as exc:
            await self.repository.add_step(job_id, "system", {"type": "failure_export_failed", "reason": reason, "error": str(exc)})
        return await self.repository.update_job(job_id, status=JobStatus.FAILED, failure_reason=reason, artifact_id=artifact_id)

    async def record_model_failure(self, state: AgentState, reason: str, detail: str) -> None:
        self.sync_llm_usage(state)
        await self.repository.add_step(
            state.job.id,
            "llm",
            {
                "type": "llm_error",
                "reason": reason,
                "detail": truncate_text(detail, 2000),
                "usage": self.usage_meter.snapshot() if self.usage_meter is not None else None,
            },
        )
        state.add_feedback(f"{reason}: {truncate_text(detail, 1000)}")
        state.iteration += 1

    async def decide_with_transient_retries(self, state: AgentState, observation: str):
        tools = allowed_tool_definitions_for_state(self.tools.definitions(), state)
        for attempt in range(1, self.llm_max_attempts + 1):
            try:
                return await asyncio.wait_for(
                    self.llm.decide(
                        system=DOCODE_SYSTEM_PROMPT,
                        messages=self._llm_messages(state),
                        tools=tools,
                        context=observation,
                    ),
                    timeout=self.llm_decision_timeout_seconds,
                )
            except Exception as exc:
                if isinstance(exc, asyncio.TimeoutError):
                    reason = "llm_provider_unavailable:provider_network_error"
                    if attempt >= self.llm_max_attempts:
                        raise RuntimeError(
                            f"llm_decision_timeout after {self.llm_decision_timeout_seconds:.0f}s"
                        ) from exc
                    delay = self.llm_retry_delays[min(attempt - 1, len(self.llm_retry_delays) - 1)] if self.llm_retry_delays else 0.0
                    await self.record_model_retry(
                        state,
                        reason,
                        f"llm_decision_timeout after {self.llm_decision_timeout_seconds:.0f}s",
                        attempt,
                        delay,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue
                reason = provider_failure_reason(exc)
                if reason == "llm_auth_failed" or not retryable_provider_failure(reason, exc) or attempt >= self.llm_max_attempts:
                    raise
                delay = self.llm_retry_delays[min(attempt - 1, len(self.llm_retry_delays) - 1)] if self.llm_retry_delays else 0.0
                await self.record_model_retry(state, reason or "llm_provider_unavailable", str(exc), attempt, delay)
                if delay > 0:
                    await asyncio.sleep(delay)
        raise RuntimeError("LLM retry loop exhausted unexpectedly")

    async def record_model_retry(self, state: AgentState, reason: str, detail: str, attempt: int, delay: float) -> None:
        self.sync_llm_usage(state)
        await self.repository.add_step(
            state.job.id,
            "llm",
            {
                "type": "llm_retry",
                "reason": reason,
                "attempt": attempt,
                "next_delay_seconds": delay,
                "detail": truncate_text(detail, 1200),
                "usage": self.usage_meter.snapshot() if self.usage_meter is not None else None,
            },
        )

    async def record_rejected_decision(
        self,
        state: AgentState,
        reason: str,
        detail: str,
        workflow_state: dict[str, object] | None = None,
    ) -> None:
        self.sync_llm_usage(state)
        payload = {
            "type": "decision_rejected",
            "reason": reason,
            "detail": truncate_text(detail, 2000),
            "repair_mode": state.repair_mode,
        }
        if workflow_state is not None:
            payload["workflow_state"] = workflow_state
        await self.repository.add_step(state.job.id, "system", payload)
        missing_commands = workflow_state.get("missing_commands") if workflow_state is not None else None
        next_command = ""
        if state.repair_mode != "targeted_repair" and isinstance(missing_commands, list) and missing_commands:
            next_command = f"\nNext required command: {missing_commands[0]}"
        if state.repair_mode == "targeted_repair" and state.active_repair_action:
            target = next(iter(sorted(targeted_repair_targets(state))), "the target file")
            action = state.active_repair_action
            commands = [str(command) for command in action.get("rerun_commands") or [] if str(command)]
            compact = (
                "Active repair:\n"
                f"- target file: {target}\n"
                "- suggested tools: read_file/read_file_range/edit_file/apply_patch/write_file/run_command/git_status/git_diff\n"
            )
            if commands:
                compact += f"- relevant rerun command: {commands[0]}\n"
            state.add_feedback(f"{compact}\n{reason}: {truncate_text(detail, 600)}")
        else:
            state.add_feedback(f"{reason}: {truncate_text(detail, 1000)}{next_command}")
        state.iteration += 1

    async def record_unavailable_tool_requested(
        self,
        state: AgentState,
        *,
        requested_tool: str,
        requested_args: dict[str, object],
        available_tools: list[str],
        workflow_state: dict[str, object] | None = None,
        increment_iteration: bool = True,
        reason: str = "tool_not_in_current_schema",
    ) -> None:
        self.sync_llm_usage(state)
        edit_pressure = duplicate_inspection_edit_forced(
            state,
            workflow_snapshot(state, state.latest_git_status.output if state.latest_git_status else ""),
        )
        payload: dict[str, object] = {
            "type": "unavailable_tool_requested",
            "requested_tool": requested_tool,
            "available_tools": available_tools,
            "requested_args": sanitize_tool_args(requested_args),
            "reason": reason,
        }
        if workflow_state is not None:
            payload["workflow_state"] = workflow_state
        await self.repository.add_step(state.job.id, "system", payload)

        if requested_tool == "inspect_source" and reason == "source_inspection_complete_edit_required":
            edit_tools = ", ".join(tool for tool in ("write_file", "edit_file", "replace_in_file", "apply_patch") if tool in available_tools)
            target_files = target_candidates_for_edit_pressure(state)
            target = target_files[0] if target_files else "the most likely target file"
            feedback = (
                "source_inspection_complete_edit_required: Source evidence is already retained in Source Inspection memory.\n"
                f"Available editing tools: {edit_tools or ', '.join(available_tools)}.\n"
                f"Read {target} once if necessary, otherwise you must edit it now."
            )
        elif requested_tool in {"inspect_source", *LOCAL_INSPECTION_TOOLS} and edit_pressure:
            edit_tools = ", ".join(tool for tool in ("write_file", "edit_file", "replace_in_file", "apply_patch") if tool in available_tools)
            target_files = target_candidates_for_edit_pressure(state)
            target = target_files[0] if target_files else "the most likely target file"
            feedback = (
                f"unavailable_tool_requested: The requested tool `{requested_tool}` is not available in this turn.\n"
                f"The available editing tools are: {edit_tools or ', '.join(available_tools)}.\n"
                f"You already inspected the relevant files. You must edit {target} now."
            )
        else:
            feedback = (
                f"unavailable_tool_requested: The requested tool `{requested_tool}` is not present in the current tool schema.\n"
                f"Available tools: {', '.join(available_tools)}."
            )
        state.add_feedback(feedback)
        if increment_iteration:
            state.iteration += 1

    def sync_llm_usage(self, state: AgentState) -> None:
        if self.usage_meter is not None:
            state.llm_tokens_used = self.usage_meter.total_tokens
            state.llm_cost_used = self.usage_meter.cost

    def _llm_messages(self, state: AgentState) -> list[dict[str, Any]]:
        if not state.messages:
            return []
        return [compact_llm_message(message) for message in state.messages[-4:]]


def artifact_execution_evidence(state: AgentState) -> ExecutionEvidence:
    scheduler = state.verification_scheduler
    producer = None
    validator = None
    if scheduler is not None:
        for node in scheduler.commands:
            evidence = scheduler.evidence.get(node.command)
            if evidence is None or not evidence.passed:
                continue
            if node.kind == "producer":
                producer = evidence
            elif node.kind == "validator":
                validator = evidence
    request_paths: list[str] = []
    for message in state.messages:
        if message.get("role") != "tool" or message.get("tool") != "inspect_source" or int(message.get("exit_code") or 0) != 0:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        if metadata.get("cached"):
            continue
        parsed = urlparse(str(metadata.get("final_url") or metadata.get("requested_url") or ""))
        request_paths.append(parsed.path + (f"?{parsed.query}" if parsed.query else ""))
    return ExecutionEvidence(
        edit_epoch=state.edit_epoch,
        producer_epoch=producer.edit_epoch if producer else None,
        producer_sequence=producer.sequence if producer else None,
        validator_epoch=validator.edit_epoch if validator else None,
        validator_sequence=validator.sequence if validator else None,
        request_paths=tuple(request_paths),
    )


def verification_to_dict(result: VerificationResult) -> dict[str, object]:
    return {
        "passed": result.passed,
        "confidence": result.confidence,
        "reason": result.reason,
        "required_fixes": result.required_fixes,
        "git_status": result.git_status,
        "git_diff": result.git_diff,
        "status": {
            "tool": result.status_result.tool if result.status_result else None,
            "exit_code": result.status_result.exit_code if result.status_result else None,
            "output": result.status_result.output if result.status_result else None,
        },
        "test": {
            "tool": result.test_result.tool if result.test_result else None,
            "exit_code": result.test_result.exit_code if result.test_result else None,
            "output": result.test_result.output if result.test_result else None,
        },
        "build": {
            "tool": result.build_result.tool if result.build_result else None,
            "exit_code": result.build_result.exit_code if result.build_result else None,
            "output": result.build_result.output if result.build_result else None,
        },
        "lint": {
            "tool": result.lint_result.tool if result.lint_result else None,
            "exit_code": result.lint_result.exit_code if result.lint_result else None,
            "output": result.lint_result.output if result.lint_result else None,
        },
        "smoke": {
            "tool": result.smoke_result.tool if result.smoke_result else None,
            "exit_code": result.smoke_result.exit_code if result.smoke_result else None,
            "output": result.smoke_result.output if result.smoke_result else None,
            "metadata": result.smoke_result.metadata if result.smoke_result else None,
        },
        "explicit_commands": [
            {
                "command": (item.metadata or {}).get("command"),
                "exit_code": item.exit_code,
                "output": item.output,
                "metadata": item.metadata or {},
            }
            for item in (result.explicit_results or [])
        ],
        "llm_judgement": {
            "passed": result.llm_judgement.passed,
            "confidence": result.llm_judgement.confidence,
            "reason": result.llm_judgement.reason,
            "required_fixes": result.llm_judgement.required_fixes,
        }
        if result.llm_judgement
        else None,
        "verification_plan": {
            "required_commands": result.verification_plan.required_commands,
            "explicit_commands": result.verification_plan.explicit_commands,
            "smoke_commands": result.verification_plan.smoke_commands,
            "require_test_change": result.verification_plan.require_test_change,
            "require_entrypoint_run": result.verification_plan.require_entrypoint_run,
            "require_no_placeholder": result.verification_plan.require_no_placeholder,
            "require_external_source_verified": result.verification_plan.require_external_source_verified,
            "artifact_export": result.verification_plan.artifact_export,
            "docs_only": result.verification_plan.docs_only,
            "external_source_repair": result.verification_plan.external_source_repair,
        }
        if result.verification_plan
        else None,
        "evidence": {
            "successful_fetch_urls": result.evidence.successful_fetch_urls,
            "successful_web_search_queries": result.evidence.successful_web_search_queries,
            "successful_source_inspections": result.evidence.successful_source_inspections or [],
            "relevant_fetch_urls": result.evidence.relevant_fetch_urls or [],
            "latest_edit_step_index": result.evidence.latest_edit_step_index,
            "latest_edit_epoch": result.evidence.latest_edit_epoch,
            "command_runs": [
                {
                    "command": run.command,
                    "output": run.output,
                    "exit_code": run.exit_code,
                    "step_index": run.step_index,
                    "edit_epoch": run.edit_epoch,
                    "explicit": run.explicit,
                }
                for run in (result.evidence.command_runs or [])
            ],
            "no_test_reason": result.evidence.no_test_reason,
        }
        if result.evidence
        else None,
    }


def auto_final_candidate_decision(state: AgentState, reason: str) -> AgentDecision:
    status_output = state.latest_git_status.output if state.latest_git_status is not None else ""
    changed = [path for path in changed_paths_from_status(status_output) if path not in {".docode_probe", ".docode_probe_api"}]
    target = ", ".join(changed[:5]) if changed else "the requested workspace changes"
    return AgentDecision(
        type="final_candidate",
        summary=f"Completed the requested changes in {target}.",
        verification=f"Auto-finalized from FINAL_READY workflow evidence after {reason}.",
        no_test_reason="No additional automated test was requested at finalization time; using workflow evidence and verifier checks.",
    )


def observation_step(context_pack: ContextPack) -> dict[str, object]:
    return {
        "type": "observation",
        "content": context_pack.render(),
        "task_contract": context_pack.task_contract,
        "repo_map": context_pack.repo_map,
        "working_memory": context_pack.working_memory,
        "file_memory": context_pack.file_memory,
        "source_inspection": context_pack.source_inspection,
        "latest_evidence": context_pack.latest_evidence,
        "recent_messages": context_pack.recent_messages,
    }


def enrich_tool_result_metadata(tool_name: str, args: dict[str, object], result: ToolResult) -> ToolResult:
    metadata = dict(result.metadata or {})
    if "path" not in metadata and tool_name in {"write_file", "edit_file", "replace_in_file", *FOCUSED_REPAIR_READ_TOOLS}:
        path = args.get("path")
        if path:
            metadata["path"] = str(path)
    if tool_name == "read_file_range":
        metadata.setdefault("start_line", args.get("start_line", 1))
        metadata.setdefault("end_line", args.get("end_line", 120))
    elif tool_name == "read_symbol":
        metadata.setdefault("symbol", args.get("symbol", ""))
        metadata.setdefault("context_lines", args.get("context_lines", 5))
    if "command" not in metadata and tool_name == "run_command":
        command = args.get("command")
        if command:
            metadata["command"] = str(command)
    if tool_name == "inspect_source":
        metadata.setdefault("requested_url", str(args.get("url") or ""))
        metadata.setdefault("mode", str(args.get("mode") or "raw"))
        metadata.setdefault("execution_scope", "sandbox")
    if metadata == (result.metadata or {}):
        return result
    return ToolResult(
        tool=result.tool,
        output=result.output,
        exit_code=result.exit_code,
        metadata=metadata,
        truncated=result.truncated,
    )


def compact_llm_message(message: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("role", "kind", "tool", "exit_code", "truncated"):
        if key in message:
            compact[key] = message[key]
    if "content" in message:
        compact["content"] = truncate_text(str(message["content"]), 500)
    if "output" in message:
        compact["output"] = (
            "<source body represented in Source Inspection>"
            if message.get("tool") == "inspect_source"
            else truncate_text(str(message["output"]), 500)
        )
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        keep = {}
        for key in ("path", "command", "reason", "url", "status_code", "content_type", "prompt_output_truncated"):
            if key in metadata:
                keep[key] = display_command(str(metadata[key])) if key == "command" else metadata[key]
        if keep:
            compact["metadata"] = keep
    return compact


def verification_repair_feedback(result: VerificationResult, task_contract: TaskContract | None = None) -> str:
    parts = [result.reason]
    if result.required_fixes:
        parts.append("Required fixes:\n" + "\n".join(f"- {fix}" for fix in result.required_fixes))
    if requires_non_empty_diff_repair(result):
        parts.append(
            "Mandatory next step:\n"
            "- Call edit_file, write_file, replace_in_file, or apply_patch to change the target file.\n"
            "- Do not call final_candidate until git_status shows a modified file.\n"
            "- Inspect the target file, replace the wrong implementation, run the relevant tests or smoke command, run git_diff, then final_candidate."
        )
    if task_contract is not None:
        changed = set(changed_files_from_diff(result.git_diff))
        missing = [path for path in task_contract.must_modify_files if path not in changed and f"b/{path}" not in changed]
        if missing:
            parts.append("Required file missing from diff:\n" + "\n".join(f"- {path}" for path in missing))
        hints = task_specific_repair_hints(task_contract.must_modify_files)
        if hints:
            parts.append("Task-specific repair sequence:\n" + "\n".join(f"- {hint}" for hint in hints))
    if result.smoke_result is not None and result.smoke_result.exit_code != 0:
        command = result.smoke_result.metadata.get("command") if result.smoke_result.metadata else None
        if command:
            parts.append("Smoke command:\n" + command)
        parts.append("Smoke output:\n" + truncate_text(result.smoke_result.output, 4000))
    return "\n\n".join(part for part in parts if part)


def quality_repair_feedback(result: QualityGateResult) -> str:
    lines = ["Pre-build quality gate failed. Fix these blockers before final_candidate:"]
    for issue in result.blockers():
        line = f"- [{issue.code}] {issue.message}"
        if issue.path:
            line += f" ({issue.path})"
        lines.append(line)
        if issue.repair_hint:
            lines.append(f"  Repair hint: {issue.repair_hint}")
    if result.samples:
        lines.append("Artifact samples inspected:")
        for sample in result.samples[:3]:
            lines.append(f"- {sample.path}: {sample.summary}")
    lines.append("After repairing, rerun the relevant command(s), inspect the output artifact, then submit final_candidate again.")
    return "\n".join(lines)


def review_repair_feedback(result: ReviewResult) -> str:
    lines = ["Independent reviewer found blocking quality issues. Repair these before final_candidate:"]
    for issue in result.blocking_issues:
        lines.append(f"- {issue}")
    if result.repair_plan:
        lines.append("Repair plan:")
        for item in result.repair_plan:
            lines.append(f"- {item}")
    if result.warnings:
        lines.append("Warnings:")
        for warning in result.warnings[:5]:
            lines.append(f"- {warning}")
    lines.append("After repairing, rerun the relevant verification command(s), inspect artifacts, then submit final_candidate again.")
    return "\n".join(lines)


def refine_repair_action_targets(action: RepairAction, task_contract: TaskContract | None) -> RepairAction:
    if action.category != "parsed_value_mismatch":
        return action
    source_targets = task_contract_source_targets(task_contract)
    if not source_targets:
        return action
    action_targets = list(action.target_files)
    if "main.py" in action_targets and "main.py" not in source_targets:
        action_targets = [path for path in action_targets if path != "main.py"]
    target_files = unique_preserving_paths([*source_targets, *action_targets])
    if target_files == action.target_files:
        return action
    return replace(action, target_files=target_files)


def fallback_required_command_repair(state: AgentState, result: ToolResult) -> RepairAction | None:
    if result.tool != "run_command" or result.ok:
        return None
    task_contract = state.task_contract
    if task_contract is None:
        return None
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    command = normalize_command(str(metadata.get("command") or ""))
    if not command:
        return None
    required_commands = [str(item) for item in task_contract.must_run_commands if str(item).strip()]
    if not any(commands_equivalent(command, required) for required in required_commands):
        return None

    failure_summary = truncate_text(str(result.output or ""), 2400)
    signature = "failed_required_command:" + hashlib.sha1(f"{command}\n{failure_summary}".encode("utf-8", errors="ignore")).hexdigest()[:12]
    target_files = fallback_required_command_target_files(state, output=str(result.output or ""), command=command)
    target_text = ", ".join(target_files) if target_files else "the relevant source file"
    command_summary = display_command(command)
    instruction = (
        "A required verification command failed, but no specific repair plan matched the output.\n\n"
        f"Failed command:\n{command_summary}\n\n"
        f"Failure output summary:\n{failure_summary}\n\n"
        f"Candidate target files: {target_text}.\n"
        "Edit a relevant source file before rerunning the command.\n"
        f"After editing, the controller will rerun exactly: {command_summary}"
    )
    return RepairAction(
        category="failed_required_command",
        signature=signature,
        reason="required verification command failed",
        target_files=target_files,
        allowed_tools=list(FAILED_REQUIRED_COMMAND_ALLOWED_TOOLS),
        forbidden_tools=list(TARGETED_REPAIR_FORBIDDEN_TOOLS),
        instruction=instruction,
        rerun_commands=[command],
        exploration_forbidden=False,
        initial_inspection_budget=2,
    )


ZERO_RECORD_RE = re.compile(r"\b(?:wrote|collected|parsed|produced|found)?\s*0\s+(?:records?|cards?|entries?|items?|rows?)\b", re.IGNORECASE)


def parser_source_mismatch_repair(state: AgentState, result: ToolResult) -> RepairAction | None:
    if not is_crawler_instruction(state.job.instruction) or result.tool != "run_command":
        return None
    if not re.search(r"AssertionError:\s*0\b|\bempty\s+(?:payload|records?|list)\b", result.output, re.IGNORECASE):
        return None
    commands = list(state.task_contract.must_run_commands if state.task_contract is not None else [])
    if len(commands) < 2:
        return None
    producer = str(commands[0])
    producer_result = next(
        (
            message
            for message in reversed(state.messages)
            if message.get("role") == "tool"
            and message.get("tool") == "run_command"
            and int(message.get("exit_code") or 0) == 0
            and commands_equivalent(str((message.get("metadata") or {}).get("command") or ""), producer)
        ),
        None,
    )
    if producer_result is None or not ZERO_RECORD_RE.search(str(producer_result.get("output") or "")):
        return None
    targets = list(state.task_contract.must_modify_files if state.task_contract is not None else [])
    if not targets:
        targets = [token.strip("'\"") for token in producer.split() if token.strip("'\"").endswith(".py")][:1]
    target = targets[0] if targets else "main.py"
    diagnosis = source_parser_diagnosis(state, target)
    signature = f"parser_source_mismatch:{normalize_workspace_relative_path(target)}:zero_records"
    return RepairAction(
        category="parser_source_mismatch",
        signature=signature,
        reason="The collector executed successfully but matched zero records.",
        target_files=[target],
        allowed_tools=["read_file", "read_file_range", "read_symbol", "edit_file", "write_file", "replace_in_file", "apply_patch", "run_command", "git_status", "git_diff"],
        forbidden_tools=TARGETED_REPAIR_FORBIDDEN_TOOLS,
        rerun_commands=[producer, *[str(command) for command in commands[1:]]],
        instruction=(
            "The collector executed successfully but matched zero records. The current parser assumptions likely do not match "
            "the inspected source structure.\n\n"
            f"Latest producer output:\n{truncate_text(str(producer_result.get('output') or ''), 800)}\n\n"
            f"Validator assertion:\n{truncate_text(result.output, 800)}\n\n"
            f"Evidence-backed source/parser diagnosis:\n{diagnosis}\n\n"
            "Re-read the retained representative source excerpt, update the parser generically, then rerun the producer before the validator."
        ),
        initial_inspection_budget=1,
        failure_class="parser_source_mismatch",
        producer_semantic_result="zero_records",
    )


def source_parser_diagnosis(state: AgentState, target: str) -> str:
    evidence = [item for item in source_inspection_evidence(state.messages, state.job.instruction) if item.usable]
    observed: list[str] = []
    summaries: list[str] = []
    for item in evidence:
        summary = item.structure_summary or {}
        summaries.append(json.dumps(summary, ensure_ascii=False))
        for key in ("top_level_keys", "sample_keys", "class_names", "data_attributes"):
            observed.extend(str(value) for value in summary.get(key) or [])
        observed.extend(str(entry.get("tag")) for entry in summary.get("repeated_tags") or [] if isinstance(entry, dict))
        observed.extend(str(key) for key in (summary.get("list_fields") or {}))
        observed.extend(str(key) for key in (summary.get("pagination_fields") or {}))
    target_normalized = normalize_workspace_relative_path(target)
    code_text = ""
    for message in reversed(state.messages):
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        path = normalize_workspace_relative_path(str(metadata.get("path") or ""))
        if path == target_normalized and message.get("tool") in {"read_file", "read_file_range", "read_symbol", "write_file", "edit_file"}:
            code_text = str(message.get("output") or "")
            if code_text:
                break
    references = sorted(
        {
            match.group(1)
            for match in re.finditer(r"['\"]([A-Za-z_][A-Za-z0-9_.:-]{1,48})['\"]", code_text)
        }
    )[:30]
    observed_set = {value.lower() for value in observed}
    absent = [value for value in references if value.lower() not in observed_set][:15]
    excerpt = next((item.body for item in evidence if item.body), "")
    return (
        f"Usable source count: {len(evidence)}\n"
        f"Observed source summaries: {truncate_text(' | '.join(summaries), 1800)}\n"
        f"Current parser references: {', '.join(references) or '<not recovered from prior reads/edits>'}\n"
        f"Parser identifiers absent from observed structure: {', '.join(absent) or '<none detected>'}\n"
        f"Representative source excerpt: {truncate_text(excerpt, 1500)}"
    )


def reset_parser_mismatch_convergence(state: AgentState, result: ToolResult) -> None:
    if result.tool != "run_command" or result.exit_code != 0 or ZERO_RECORD_RE.search(result.output):
        return
    commands = list(state.task_contract.must_run_commands if state.task_contract is not None else [])
    observed = str((result.metadata or {}).get("command") or "")
    if not commands or not commands_equivalent(observed, str(commands[0])):
        return
    for signature in list(state.failure_signatures):
        if signature.startswith("parser_source_mismatch:"):
            state.failure_signatures.pop(signature, None)


SOURCE_FILE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".rb",
    ".php",
    ".sh",
}


def fallback_required_command_target_files(state: AgentState, *, output: str, command: str) -> list[str]:
    task_contract = state.task_contract
    candidates: list[str] = []
    if task_contract is not None:
        candidates.extend(str(path) for path in task_contract.must_modify_files if str(path).strip())
    candidates.extend(infer_python_traceback_files(output))
    candidates.extend(infer_named_fixture_files(output, command))
    candidates.extend(source_files_implied_by_command(command))
    status_output = state.latest_git_status.output if state.latest_git_status is not None else ""
    candidates.extend(changed_source_files_from_status(status_output))
    candidates.extend(successfully_inspected_source_files(state))
    return unique_preserving_paths(candidates)


def successfully_inspected_source_files(state: AgentState) -> list[str]:
    paths: list[str] = []
    for message in state.messages:
        if (
            message.get("role") != "tool"
            or message.get("tool") not in FOCUSED_REPAIR_READ_TOOLS
            or int(message.get("exit_code") or 0) != 0
        ):
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        path = normalize_workspace_relative_path(str(metadata.get("path") or ""))
        suffix = posixpath.splitext(path)[1].lower()
        if path and suffix in SOURCE_FILE_EXTENSIONS:
            paths.append(path)
    return unique_preserving_paths(paths)


def source_files_implied_by_command(command: str) -> list[str]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    candidates: list[str] = []
    for token in tokens:
        cleaned = normalize_command_path_token(token)
        if is_source_file_candidate(cleaned):
            candidates.append(cleaned)
    return candidates


def changed_source_files_from_status(status_output: str) -> list[str]:
    return [path for path in changed_paths_from_status(status_output) if is_source_file_candidate(path)]


def normalize_command_path_token(token: str) -> str:
    cleaned = str(token or "").strip().strip("'\"`")
    cleaned = cleaned.split("::", 1)[0]
    cleaned = cleaned.split(":", 1)[0] if re.search(r"\.[A-Za-z0-9]+:\d+$", cleaned) else cleaned
    return normalize_workspace_relative_path(cleaned)


def is_source_file_candidate(path: str) -> bool:
    normalized = normalize_workspace_relative_path(path)
    if not normalized or normalized.startswith("-") or generated_artifact_target(normalized):
        return False
    suffix = posixpath.splitext(normalized)[1].lower()
    return suffix in SOURCE_FILE_EXTENSIONS


def task_contract_source_targets(task_contract: TaskContract | None) -> list[str]:
    if task_contract is None:
        return []
    targets: list[str] = []
    for path in task_contract.must_modify_files:
        normalized = normalize_workspace_relative_path(path)
        if not normalized or normalized.startswith(("tests/", "test_")):
            continue
        if "/fixtures/" in f"/{normalized}" or "/fixture/" in f"/{normalized}":
            continue
        if normalized.endswith(".py"):
            targets.append(normalized)
    return unique_preserving_paths(targets)


def unique_preserving_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for path in paths:
        normalized = normalize_workspace_relative_path(path)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def repair_action_from_quality_gate(result: QualityGateResult) -> RepairAction | None:
    blockers = list(result.blockers())
    if not blockers:
        return None
    if all(issue.code == "json_artifact_missing" for issue in blockers):
        return None
    target_files: list[str] = []
    rerun_commands: list[str] = []
    lines = ["Quality gate blocked finalization. Modify the listed target file(s) before running commands again."]
    for issue in blockers:
        path = str(issue.path or "").strip()
        code = str(issue.code or "quality_blocker")
        repair_path = quality_gate_repair_target(path, code)
        if repair_path and repair_path not in target_files:
            target_files.append(repair_path)
        where = f" ({path})" if path else ""
        lines.append(f"- [{code}] {issue.message}{where}")
        if issue.repair_hint:
            lines.append(f"  Repair hint: {issue.repair_hint}")
    if not target_files:
        issue_text = "\n".join(f"{issue.code}: {issue.message}" for issue in blockers)
        return plan_repair_from_tool_result(
            tool="run_command",
            output=issue_text,
            metadata={},
        )
    signature_source = "|".join(f"{issue.code}:{issue.message}:{issue.path}" for issue in blockers)
    return RepairAction(
        category="quality_gate_repair",
        signature="quality:" + str(abs(hash(signature_source)))[:12],
        reason="quality_gate_blocked",
        target_files=target_files,
        allowed_tools=[
            "apply_patch",
            "edit_file",
            "git_diff",
            "git_status",
            "read_file",
            "read_file_range",
            "read_symbol",
            "replace_in_file",
            "write_file",
        ],
        forbidden_tools=["run_command", "web_search", "fetch_url", "preview", "logs"],
        instruction="\n".join(lines),
        rerun_commands=rerun_commands,
        exploration_forbidden=True,
        initial_inspection_budget=1,
    )


def quality_gate_repair_target(path: str, code: str = "") -> str:
    normalized = normalize_workspace_relative_path(path)
    if str(code).startswith("json_") and normalized:
        return normalized
    return normalized


def repair_action_from_review(result: ReviewResult, task_contract: TaskContract | None) -> RepairAction | None:
    issue_text = "\n".join(str(issue) for issue in result.blocking_issues if str(issue).strip())
    if not issue_text:
        return None
    target_files = review_repair_target_files(task_contract, issue_text)
    if not target_files:
        return None
    repair_plan = "\n".join(f"- {item}" for item in result.repair_plan if str(item).strip())
    instruction = (
        "Independent review blocked finalization. Modify the target artifact files before running commands again.\n"
        "Blocking issues:\n"
        f"{issue_text}\n"
    )
    if repair_plan:
        instruction += f"Repair plan:\n{repair_plan}\n"
    instruction += "Edit the implicated production artifact or source file, then rerun the required verification command."
    return RepairAction(
        category="review_repair",
        signature="review:" + str(abs(hash(issue_text)))[:12],
        reason="independent_review_blocked",
        target_files=target_files,
        allowed_tools=TARGETED_REPAIR_ALLOWED_TOOLS_FOR_REVIEW,
        forbidden_tools=["web_search", "fetch_url", "preview", "logs"],
        instruction=instruction,
        rerun_commands=list(task_contract.must_run_commands) if task_contract is not None else [],
        exploration_forbidden=True,
        initial_inspection_budget=0,
    )


TARGETED_REPAIR_ALLOWED_TOOLS_FOR_REVIEW = [
    "read_file",
    "edit_file",
    "write_file",
    "replace_in_file",
    "apply_patch",
    "run_command",
    "git_status",
    "git_diff",
]


def review_repair_target_files(task_contract: TaskContract | None, issue_text: str = "") -> list[str]:
    if task_contract is not None and task_contract.must_modify_files:
        lower = issue_text.lower()
        preferred: list[str] = []
        available = set(task_contract.must_modify_files)
        for path in task_contract.must_modify_files:
            name = path.rsplit("/", 1)[-1].lower()
            if path.lower() in lower or name in lower:
                preferred.append(path)
        preferred.extend(path for path in task_contract.must_modify_files if not path.startswith(("tests/", "test_")))
        preferred.extend(task_contract.must_modify_files)
        ordered = []
        for path in preferred:
            if path in available and path not in ordered:
                ordered.append(path)
        return ordered
    return []


def repair_action_contract(action: RepairAction, state: AgentState) -> dict[str, Any]:
    payload = action.to_dict()
    target_files = [str(path) for path in payload.get("target_files") or [] if str(path)]
    target_file = target_files[0] if target_files else None
    rerun_commands = [str(command) for command in payload.get("rerun_commands") or [] if str(command)]
    instruction = str(payload.get("instruction") or "")
    category = str(payload.get("category") or "")
    raw_budget = payload.get("initial_inspection_budget")
    try:
        inspection_budget = int(raw_budget if raw_budget is not None else 2)
    except (TypeError, ValueError):
        inspection_budget = 2
    if category in CONTEXT_HEAVY_REPAIR_CATEGORIES and inspection_budget != 0:
        inspection_budget = max(inspection_budget, MIN_CONTEXT_REPAIR_INSPECTION_BUDGET)
    must_change_symbols = []
    payload.update(
        {
            "phase": "REPAIR_REQUIRED",
            "target_file": target_file,
            "must_change_symbols": must_change_symbols,
            "next_allowed_tools": ["read_file", "read_file_range", "read_symbol", "apply_patch", "edit_file", "write_file", "replace_in_file"],
            "forbidden_until_modified": ["run_command"],
            "rerun_after_modified": rerun_commands[0] if rerun_commands else None,
            "created_at_message_index": len(state.messages),
            "initial_inspection_budget": inspection_budget,
        }
    )
    return payload


def requires_non_empty_diff_repair(result: VerificationResult) -> bool:
    required = " ".join(result.required_fixes).lower()
    return "non-empty git diff" in required or (not result.git_diff.strip() and "diff" in required)


def task_specific_repair_hints(paths: list[str]) -> list[str]:
    file_names = {path.rsplit("/", 1)[-1] for path in paths}
    hints: list[str] = []
    if "calculator.py" in file_names:
        hints.append(
            "python-bugfix: inspect calculator.py and related tests, edit the implementation, "
            "then run the explicit verification command from the task contract before final_candidate."
        )
    if "cli.py" in file_names:
        hints.append(
            "python-cli: inspect cli.py and related tests, edit the implementation, "
            "then run the explicit verification command from the task contract before final_candidate."
        )
    return hints


def repair_action_start_index(state: AgentState, result: ToolResult | None = None) -> int:
    if result is None:
        return len(state.messages)
    for index in range(len(state.messages) - 1, -1, -1):
        message = state.messages[index]
        if message.get("role") != "tool":
            continue
        if str(message.get("tool") or "") != result.tool:
            continue
        if int(message.get("exit_code") or 0) != result.exit_code:
            continue
        if str(message.get("output") or "") != str(result.output or ""):
            continue
        message_metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        if dict(message_metadata) == dict(result.metadata or {}):
            return index + 1
    return len(state.messages)


def allowed_tool_definitions(definitions: list[Any], repair_mode: str | None) -> list[Any]:
    if repair_mode not in {"must_edit", "quality_repair"}:
        return definitions
    allowed = allowed_tools_for_repair_mode_name(repair_mode)
    return [definition for definition in definitions if getattr(definition, "name", None) in allowed]


def allowed_tool_definitions_for_state(definitions: list[Any], state: AgentState) -> list[Any]:
    refresh_targeted_repair_phase(state)
    source_continuation = continuation_allowed(state)
    if successful_source_inspection(state.messages, state.job.instruction) is not None and not source_continuation:
        definitions = [definition for definition in definitions if getattr(definition, "name", None) != "inspect_source"]
    if source_progress_forced(state):
        return [definition for definition in definitions if getattr(definition, "name", None) in (EDIT_TOOLS | LOCAL_INSPECTION_TOOLS)]
    if state.repair_mode == "targeted_repair" and state.active_repair_action:
        allowed = targeted_repair_allowed_tools_for_phase(state) - targeted_repair_hard_forbidden_tools(state)
        if source_continuation:
            allowed.add("inspect_source")
        return [definition for definition in definitions if getattr(definition, "name", None) in allowed]
    status_output = state.latest_git_status.output if state.latest_git_status is not None else ""
    workflow = workflow_snapshot(state, status_output)
    if crawler_source_research_priority_active(state, workflow):
        allowed = initial_crawler_source_tools(state)
        return [definition for definition in definitions if getattr(definition, "name", None) in allowed]
    if duplicate_inspection_edit_forced(state, workflow):
        allowed = EDIT_TOOLS | LOCAL_INSPECTION_TOOLS
        return [definition for definition in definitions if getattr(definition, "name", None) in allowed]
    if (
        workflow.phase == WorkflowPhase.EDIT_REQUIRED
        and not successful_edit_tool_called(state)
        and exploratory_tool_calls(state) >= INITIAL_NO_DIFF_EXPLORATION_BUDGET
    ):
        allowed = EDIT_TOOLS | LOCAL_INSPECTION_TOOLS
        return [definition for definition in definitions if getattr(definition, "name", None) in allowed]
    if workflow.phase == WorkflowPhase.TEST_REQUIRED and missing_must_modify_targets(state):
        allowed = {"write_file", "apply_patch", "git_status", "git_diff"}
        return [definition for definition in definitions if getattr(definition, "name", None) in allowed]
    return allowed_tool_definitions(definitions, state.repair_mode)


def repair_mode_tool_block(state: AgentState, tool_name: str) -> str:
    refresh_targeted_repair_phase(state)
    if tool_name == "inspect_source" and continuation_allowed(state):
        return ""
    if state.repair_mode == "targeted_repair" and state.active_repair_action:
        if tool_name in targeted_repair_hard_forbidden_tools(state):
            return repair_mode_forbidden_detail(state, tool_name)
        if tool_name in targeted_repair_allowed_tools_for_phase(state):
            return ""
        return repair_mode_forbidden_detail(state, tool_name)
    if state.repair_mode not in {"must_edit", "quality_repair", "targeted_repair"}:
        return ""
    allowed = allowed_tools_for_repair_mode(state)
    if tool_name in allowed:
        return ""
    return repair_mode_forbidden_detail(state, tool_name)


def allowed_tools_for_repair_mode(state: AgentState) -> set[str]:
    if state.repair_mode == "targeted_repair":
        return allowed_tools_for_repair_mode_name(state.repair_mode) - targeted_repair_hard_forbidden_tools(state)
    return allowed_tools_for_repair_mode_name(state.repair_mode)


def targeted_repair_hard_forbidden_tools(state: AgentState) -> set[str]:
    action = state.active_repair_action or {}
    explicitly_forbidden = {str(tool) for tool in action.get("forbidden_tools") or [] if str(tool)}
    unsafe_forbidden = {"web_search", "fetch_url", "preview", "logs"}
    return explicitly_forbidden & unsafe_forbidden


def targeted_repair_allowed_tools_for_phase(state: AgentState) -> set[str]:
    if targeted_repair_modified_target(state):
        return {"run_command", "git_status", "git_diff"}
    return FOCUSED_REPAIR_READ_TOOLS | EDIT_TOOLS | TARGETED_REPAIR_GIT_TOOLS


def allowed_tools_for_repair_mode_name(repair_mode: str | None) -> set[str]:
    if repair_mode == "must_edit":
        return EDIT_TOOLS | LOCAL_INSPECTION_TOOLS
    if repair_mode in {"quality_repair", "targeted_repair"}:
        return {"read_file", "edit_file", "write_file", "replace_in_file", "apply_patch", "run_command", "git_status", "git_diff"}
    return set()


def repair_mode_forbidden_detail(state: AgentState, tool_name: str) -> str:
    if state.repair_mode == "targeted_repair" and state.active_repair_action:
        instruction = str(state.active_repair_action.get("instruction") or "")
        if not targeted_repair_modified_target(state):
            return (
                f"{tool_name} is blocked until an active target file is modified. "
                "Use focused target reads, edit tools, git_status, or git_diff now.\n"
                f"{instruction}"
            )
        return (
            f"{tool_name} is blocked by the active targeted repair action.\n"
            f"Rerun the exact repair command or inspect git status/diff now.\n{instruction}"
        )
    if state.repair_mode == "must_edit":
        return (
            f"{tool_name} is blocked while repair_mode=must_edit. "
            "Call edit_file, write_file, replace_in_file, or apply_patch to change a target file."
        )
    return f"{tool_name} is blocked while repair_mode={state.repair_mode}."


def targeted_repair_exploration_block(state: AgentState, tool_name: str) -> str:
    _ = state, tool_name
    return ""


def targeted_repair_action_block(state: AgentState, tool_name: str, args: dict[str, object]) -> str:
    _ = args
    if state.repair_mode != "targeted_repair" or not state.active_repair_action:
        return ""
    if tool_name in targeted_repair_hard_forbidden_tools(state):
        return (
            f"{tool_name} is blocked by the active targeted repair action. "
            "Use local read/edit/run_command/git tools for this repair instead."
        )
    return ""


def targeted_repair_read_policy_result(
    state: AgentState,
    tool_name: str,
    args: dict[str, object],
) -> ToolResult | None:
    if (
        state.repair_mode != "targeted_repair"
        or not state.active_repair_action
        or tool_name not in FOCUSED_REPAIR_READ_TOOLS
    ):
        return None
    path = normalize_workspace_relative_path(str(args.get("path") or ""))
    targets = targeted_repair_targets(state)
    if targets and not any(status_change_covers_target(path, target) for target in targets):
        return repair_read_blocked_result(
            tool_name,
            path,
            "repair_read_not_targeted",
            targets,
            "Read one of the active target files or edit the most likely source target.",
        )
    if repeated_targeted_repair_read(state, tool_name, args):
        return repair_read_blocked_result(
            tool_name,
            path,
            "repair_read_repeated",
            targets,
            "Use a different range/symbol if the prior output was truncated; otherwise edit the target now.",
        )
    budget = targeted_repair_inspection_budget(state)
    observed = targeted_repair_read_count(state)
    if observed < budget or not usable_targeted_repair_read_exists(state):
        return None
    if truncated_read_followup_allowed(state, tool_name, path):
        return None
    return repair_read_blocked_result(
        tool_name,
        path,
        "repair_read_budget_exhausted",
        targets,
        "The focused read budget is exhausted. Edit the target file now, then rerun the required command.",
        observed=observed,
        budget=budget,
    )


def repair_read_blocked_result(
    tool_name: str,
    path: str,
    reason: str,
    targets: set[str],
    next_action: str,
    *,
    observed: int | None = None,
    budget: int | None = None,
) -> ToolResult:
    target_text = ", ".join(sorted(targets)) or "<unspecified>"
    output = f"{reason}: read not executed. Target files: {target_text}. {next_action}"
    metadata: dict[str, Any] = {
        "path": path,
        "blocked": True,
        "reason": reason,
        "target_files": sorted(targets),
        "next_action": next_action,
    }
    if observed is not None:
        metadata["observed_reads"] = observed
    if budget is not None:
        metadata["read_budget"] = budget
    return ToolResult(tool=tool_name, output=output, exit_code=0, metadata=metadata)


def note_targeted_repair_tool_result(state: AgentState, result: ToolResult) -> None:
    if state.repair_mode != "targeted_repair" or not state.active_repair_action:
        return
    if not result.ok:
        refresh_targeted_repair_phase(state)
        return
    if result.tool in FOCUSED_REPAIR_READ_TOOLS and useful_targeted_repair_read_result(state, result):
        state.targeted_repair_inspections += 1
    elif result.tool in {"edit_file", "write_file", "replace_in_file", "apply_patch"}:
        state.targeted_repair_edits += 1
    refresh_targeted_repair_phase(state)


def refresh_targeted_repair_phase(state: AgentState) -> None:
    if state.repair_mode != "targeted_repair" or not state.active_repair_action:
        state.targeted_repair_phase = None
        return
    if targeted_repair_modified_target(state):
        state.targeted_repair_phase = "rerun_required"
        return
    budget = targeted_repair_inspection_budget(state)
    observed = max(state.targeted_repair_inspections, targeted_repair_read_count(state))
    state.targeted_repair_inspections = observed
    state.targeted_repair_phase = "edit_forced" if observed >= budget else "inspect_allowed"


def targeted_repair_inspection_budget(state: AgentState) -> int:
    action = state.active_repair_action or {}
    try:
        raw_budget = action.get("initial_inspection_budget")
        budget = 2 if raw_budget is None else int(raw_budget)
    except (TypeError, ValueError):
        budget = 2
    return max(0, budget)


def targeted_repair_targets(state: AgentState) -> set[str]:
    action = state.active_repair_action or {}
    return {normalize_workspace_relative_path(str(path)) for path in action.get("target_files") or [] if str(path)}


def path_matches_any_target(path: str, targets: set[str]) -> bool:
    normalized = normalize_workspace_relative_path(path)
    return normalized in targets


def patch_touches_any_target(patch: str, targets: set[str]) -> bool:
    for line in patch.splitlines():
        stripped = line.strip()
        for prefix in ("*** Update File:", "*** Add File:", "*** Delete File:"):
            if stripped.startswith(prefix):
                path = normalize_workspace_relative_path(stripped[len(prefix) :].strip())
                if path in targets:
                    return True
        if stripped.startswith(("--- ", "+++ ")):
            parts = stripped.split(maxsplit=1)
            if len(parts) == 2:
                path = normalize_workspace_relative_path(parts[1].removeprefix("a/").removeprefix("b/"))
                if path in targets:
                    return True
            continue
        if stripped.startswith("diff --git "):
            for token in stripped.split():
                if token.startswith(("a/", "b/")):
                    path = normalize_workspace_relative_path(token[2:])
                    if path in targets:
                        return True
    return False


def targeted_repair_read_count(state: AgentState) -> int:
    if state.active_repair_action is None:
        return 0
    count = 0
    for message in state.messages[state.active_repair_started_at :]:
        if useful_targeted_repair_read_message(state, message):
            count += 1
    return count


def useful_targeted_repair_read_result(state: AgentState, result: ToolResult) -> bool:
    return useful_targeted_repair_read_message(
        state,
        {
            "role": "tool",
            "tool": result.tool,
            "exit_code": result.exit_code,
            "output": result.output,
            "truncated": result.truncated,
            "metadata": result.metadata or {},
        },
    )


def useful_targeted_repair_read_message(state: AgentState, message: dict[str, Any]) -> bool:
    if (
        message.get("role") != "tool"
        or message.get("tool") not in FOCUSED_REPAIR_READ_TOOLS
        or int(message.get("exit_code") or 0) != 0
        or not str(message.get("output") or "").strip()
    ):
        return False
    metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
    if metadata.get("blocked"):
        return False
    path = normalize_workspace_relative_path(str(metadata.get("path") or ""))
    targets = targeted_repair_targets(state)
    return bool(path) and (not targets or any(status_change_covers_target(path, target) for target in targets))


def usable_targeted_repair_read_exists(state: AgentState) -> bool:
    return any(
        useful_targeted_repair_read_message(state, message)
        for message in state.messages[state.active_repair_started_at :]
    )


def repeated_targeted_repair_read(state: AgentState, tool_name: str, args: dict[str, object]) -> bool:
    requested = targeted_repair_read_identity(tool_name, args)
    for message in state.messages[state.active_repair_started_at :]:
        if message.get("role") != "tool" or message.get("tool") != tool_name or int(message.get("exit_code") or 0) != 0:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        if metadata.get("blocked"):
            continue
        if targeted_repair_read_identity(tool_name, metadata) == requested:
            return True
    return False


def targeted_repair_read_identity(tool_name: str, values: dict[str, object]) -> tuple[object, ...]:
    path = normalize_workspace_relative_path(str(values.get("path") or ""))
    if tool_name == "read_file_range":
        return tool_name, path, int_or_default(values.get("start_line"), 1), int_or_default(values.get("end_line"), 120)
    if tool_name == "read_symbol":
        return tool_name, path, str(values.get("symbol") or ""), int_or_default(values.get("context_lines"), 5)
    return tool_name, path


def int_or_default(value: object, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def truncated_read_followup_allowed(state: AgentState, tool_name: str, path: str) -> bool:
    if tool_name not in {"read_file_range", "read_symbol"}:
        return False
    for message in reversed(state.messages[state.active_repair_started_at :]):
        if message.get("role") != "tool" or message.get("tool") not in FOCUSED_REPAIR_READ_TOOLS:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        seen_path = normalize_workspace_relative_path(str(metadata.get("path") or ""))
        if seen_path != path or metadata.get("blocked"):
            continue
        return bool(
            message.get("truncated")
            or metadata.get("source_truncated")
            or metadata.get("prompt_output_truncated")
        )
    return False


def targeted_repair_modified_target(state: AgentState) -> bool:
    action = state.active_repair_action or {}
    targets = targeted_repair_targets(state)
    rerun_commands = [str(command) for command in action.get("rerun_commands") or [] if str(command)]
    if not targets:
        return successful_edit_tool_called(state)
    for message in reversed(state.messages[state.active_repair_started_at :]):
        if message.get("role") != "tool":
            continue
        tool = str(message.get("tool") or "")
        if tool == "run_command" and int(message.get("exit_code") or 0) != 0 and rerun_commands:
            metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
            observed = str(metadata.get("command") or "")
            if any(commands_equivalent(observed, command) for command in rerun_commands):
                return False
        if tool not in {"write_file", "edit_file", "replace_in_file", "apply_patch"} or int(message.get("exit_code") or 0) != 0:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        path = normalize_workspace_relative_path(str(metadata.get("path") or ""))
        if tool == "apply_patch" or path in targets:
            return True
    return False


def targeted_repair_rerun_satisfied(state: AgentState) -> bool:
    action = state.active_repair_action or {}
    commands = [str(command) for command in action.get("rerun_commands") or [] if str(command)]
    if not commands:
        return bool(action) and (targeted_repair_modified_target(state) or targeted_repair_target_changed_in_status(state))
    if not targeted_repair_modified_target(state):
        return False
    for message in reversed(state.messages[state.active_repair_started_at :]):
        if message.get("role") != "tool":
            continue
        tool = str(message.get("tool") or "")
        if tool in {"write_file", "edit_file", "replace_in_file", "apply_patch"} and int(message.get("exit_code") or 0) == 0:
            return False
        if int(message.get("exit_code") or 0) != 0:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        observed = str(metadata.get("command") or "")
        for command in commands:
            if commands_equivalent(observed, command):
                return True
    return False


def targeted_repair_target_changed_in_status(state: AgentState) -> bool:
    targets = targeted_repair_targets(state)
    if not targets or state.latest_git_status is None:
        return False
    changed = changed_paths_from_status(state.latest_git_status.output)
    return any(status_change_covers_target(path, target) for path in changed for target in targets)


def status_change_covers_target(path: str, target: str) -> bool:
    changed = normalize_workspace_relative_path(path).rstrip("/")
    wanted = normalize_workspace_relative_path(target).rstrip("/")
    if not changed or not wanted:
        return False
    return changed == wanted or wanted.startswith(f"{changed}/") or changed.startswith(f"{wanted}/")


def targeted_rerun_missing_detail(state: AgentState) -> str:
    action = state.active_repair_action or {}
    commands = [str(command) for command in action.get("rerun_commands") or [] if str(command)]
    rerun = ", ".join(commands) or "the failing command"
    return f"You fixed a targeted repair issue but have not rerun the required command. Rerun: {rerun}"


def recent_tool_results(state: AgentState) -> list[ToolResult]:
    results: list[ToolResult] = []
    for message in state.messages:
        if message.get("role") != "tool":
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        results.append(
            ToolResult(
                tool=str(message.get("tool") or ""),
                output=str(message.get("output") or ""),
                exit_code=int(message.get("exit_code") or 0),
                metadata=metadata,
                truncated=bool(message.get("truncated")),
            )
        )
    return results


def edit_required_tool_block(state: AgentState, workflow: Any, tool_name: str, args: dict[str, object]) -> str:
    _ = args
    if workflow.phase != WorkflowPhase.EDIT_REQUIRED:
        return ""
    if tool_name in {"write_file", "edit_file", "replace_in_file"}:
        target_block = edit_required_target_file_block(state, tool_name, args)
        if target_block:
            return target_block
    if tool_name in EDIT_TOOLS or tool_name in LOCAL_INSPECTION_TOOLS:
        return ""
    if successful_edit_tool_called(state):
        return ""
    if exploratory_tool_calls(state) < INITIAL_NO_DIFF_EXPLORATION_BUDGET:
        return ""
    targets = state.task_contract.must_modify_files if state.task_contract is not None else []
    target_text = f" Target files: {', '.join(targets[:5])}." if targets else ""
    state.repair_mode = "must_edit"
    return f"{tool_name} is blocked while EDIT_REQUIRED after repeated inspection without a diff. Create or edit a target file now.{target_text}"


def exploratory_tool_calls(state: AgentState) -> int:
    return sum(
        1
        for message in state.messages
        if message.get("role") == "tool" and message.get("tool") not in {"write_file", "edit_file", "replace_in_file", "apply_patch"}
    )


def edit_tool_attempted(state: AgentState) -> bool:
    return any(
        message.get("role") == "tool" and message.get("tool") in {"write_file", "edit_file", "replace_in_file", "apply_patch"}
        for message in state.messages
    )


def latest_successful_read_index(state: AgentState, path: str) -> int | None:
    normalized = normalize_workspace_relative_path(path)
    for index in range(len(state.messages) - 1, -1, -1):
        message = state.messages[index]
        if message.get("role") != "tool" or message.get("tool") != "read_file" or int(message.get("exit_code") or 0) != 0:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        if metadata.get("cached_duplicate") is True:
            continue
        seen = normalize_workspace_relative_path(str(metadata.get("path") or ""))
        if seen == normalized:
            return index
    return None


def path_changed_after_message(state: AgentState, path: str, index: int) -> bool:
    normalized = normalize_workspace_relative_path(path)
    for message in state.messages[index + 1 :]:
        if message.get("role") != "tool" or message.get("tool") not in {"write_file", "edit_file", "replace_in_file", "apply_patch"}:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        changed = normalize_workspace_relative_path(str(metadata.get("path") or ""))
        if message.get("tool") == "apply_patch" or changed == normalized:
            return True
    return False


def crawler_source_research_priority_active(state: AgentState, workflow: Any) -> bool:
    if workflow.phase != WorkflowPhase.EDIT_REQUIRED:
        return False
    if not is_crawler_instruction(state.job.instruction):
        return False
    if not instruction_source_urls(state.job.instruction):
        return False
    return not source_research_succeeded(state)


def initial_crawler_source_tools(state: AgentState) -> set[str]:
    base = {"inspect_source", "git_status", "read_file", "read_file_range", "read_symbol", "list_files"}
    if attempted_source_urls(state.messages):
        return base | {"fetch_url", "web_search"}
    return base


def source_research_succeeded(state: AgentState) -> bool:
    return successful_source_inspection(state.messages, state.job.instruction) is not None


def source_inspection_required_tool_block(state: AgentState, tool_name: str) -> str:
    if not crawler_source_inspection_required(state.job.instruction):
        return ""
    if successful_source_inspection(state.messages, state.job.instruction) is not None:
        return ""
    allowed = initial_crawler_source_tools(state)
    if tool_name in allowed:
        return ""
    candidates = instruction_source_urls(state.job.instruction)
    candidate_text = ", ".join(candidates[:3]) or "the literal source URL"
    return (
        "Source inspection is required before editing or running verification commands. The current diff must remain empty. "
        f"Call inspect_source from the sandbox for one of these literal candidates: {candidate_text}. "
        "A local scaffold read, web_search, or host fetch_url does not satisfy this stage."
    )


def explicit_source_fetch_attempted(state: AgentState) -> bool:
    urls = set(instruction_source_urls(state.job.instruction))
    if not urls:
        return False
    for message in state.messages:
        if message.get("role") != "tool" or message.get("tool") != "fetch_url":
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        url = str(metadata.get("url") or "")
        if url in urls:
            return True
    return False


def edit_required_target_file_block(state: AgentState, tool_name: str, args: dict[str, object]) -> str:
    targets = [normalize_workspace_path(path) for path in (state.task_contract.must_modify_files if state.task_contract is not None else [])]
    if not targets:
        return ""
    raw_path = str(args.get("path") or "").strip()
    normalized = normalize_workspace_path(raw_path)
    if not normalized:
        return ""
    if normalized in targets:
        return ""
    target_text = ", ".join(targets[:5])
    return (
        f"{tool_name} must modify one of the required target files while EDIT_REQUIRED. "
        f"Use one of: {target_text}."
    )


def normalize_workspace_path(path: str) -> str:
    normalized = str(path or "").replace("\\", "/").strip()
    if normalized.startswith("/workspace/"):
        normalized = normalized[len("/workspace/") :]
    elif normalized.startswith("/workspace"):
        normalized = normalized[len("/workspace") :].lstrip("/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def required_test_tool_block(state: AgentState, workflow: Any, tool_name: str, args: dict[str, object]) -> str:
    if tool_name == "inspect_source" and continuation_allowed(state, getattr(workflow, "phase", None)):
        return ""
    if workflow.phase != WorkflowPhase.TEST_REQUIRED:
        return ""
    missing = getattr(workflow, "missing_commands", None) or []
    if not missing:
        return ""
    if state.repair_mode == "targeted_repair" and state.active_repair_action:
        if tool_name in EDIT_TOOLS or tool_name in LOCAL_INSPECTION_TOOLS or tool_name in {"run_command", "git_status", "git_diff"}:
            return ""
    if getattr(workflow, "required_tests_attempted", False) and not getattr(workflow, "required_tests_passed", False):
        if tool_name in EDIT_TOOLS or tool_name in LOCAL_INSPECTION_TOOLS or tool_name in {"run_command", "git_status", "git_diff"}:
            return ""
    missing_targets = missing_must_modify_targets(state)
    if target_edit_allowed_while_tests_missing(state, tool_name, args):
        return ""
    if test_file_edit_allowed_while_tests_missing(tool_name, args):
        return ""
    if missing_targets:
        return (
            f"{tool_name} is blocked while TEST_REQUIRED because required target files are still missing from edits: "
            f"{', '.join(missing_targets[:5])}. Create or edit the missing target file first."
        )
    next_command = str(missing[0])
    if required_command_failed_after_latest_edit(state, next_command):
        return ""
    if tool_name != "run_command":
        return f"{tool_name} is blocked while TEST_REQUIRED. Run this exact command first: {next_command}"
    observed = str(args.get("command") or "")
    expected = next_command
    if not commands_equivalent(observed, expected):
        return f"Wrong command for TEST_REQUIRED. Run this exact command first: {next_command}"
    return ""


def duplicate_read_file_block(state: AgentState, workflow: Any, tool_name: str, args: dict[str, object]) -> str:
    if tool_name != "read_file":
        return ""
    if workflow.phase != WorkflowPhase.EDIT_REQUIRED or getattr(workflow, "diff_exists", False):
        return ""
    if successful_edit_tool_called(state) or edit_tool_attempted(state):
        return ""
    if exploratory_tool_calls(state) < 3:
        return ""
    path = normalize_workspace_relative_path(str(args.get("path") or ""))
    if not path:
        return ""
    latest_read_index = latest_successful_read_index(state, path)
    if latest_read_index is None:
        return ""
    if path_changed_after_message(state, path, latest_read_index):
        return ""
    if duplicate_inspection_rejection_count(state) > 0:
        targets = target_candidates_for_edit_pressure(state)
        target_text = f"\nCandidate target files: {', '.join(targets[:5])}." if targets else ""
        return (
            f"You already read {path}, and its content is still available in context.\n"
            "The git diff is still empty.\n"
            "Another read_file call is not useful now. Available next tools: write_file, edit_file, replace_in_file, apply_patch, git_status, git_diff.\n"
            "Do not call read_file again. Edit the most likely source file now; use write_file if exact replacement text is uncertain."
            f"{target_text}"
        )
    return (
        f"You already read {path}, and its content is still available in context.\n"
        "The git diff is still empty.\n"
        "Do not reread the same file again. Edit the most likely target file now using edit_file/write_file/apply_patch."
    )


def duplicate_inspection_pressure_tool_block(state: AgentState, workflow: Any, tool_name: str) -> str:
    _ = state, workflow, tool_name
    return ""


def duplicate_inspection_required_command_retarget(state: AgentState, workflow: Any) -> str:
    if workflow.phase != WorkflowPhase.TEST_REQUIRED or not getattr(workflow, "diff_exists", False):
        return ""
    contract = state.task_contract
    commands = list(contract.must_run_commands if contract is not None else [])
    if not commands:
        return ""
    if any(required_command_attempted(state, command) for command in commands):
        return ""
    return str(commands[0])


def required_command_attempted(state: AgentState, command: str) -> bool:
    for message in state.messages:
        if message.get("role") != "tool" or message.get("tool") != "run_command":
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        observed = str(metadata.get("command") or "")
        if observed and commands_equivalent(observed, command):
            return True
    return False


def cached_duplicate_read_result(state: AgentState, args: dict[str, object], detail: str) -> ToolResult:
    path = normalize_workspace_relative_path(str(args.get("path") or ""))
    original_index = latest_successful_read_index(state, path) if path else None
    excerpt = cached_read_excerpt(state, original_index)
    targets = target_candidates_for_edit_pressure(state)
    target_text = f"\nCandidate target files: {', '.join(targets[:5])}." if targets else ""
    output = (
        f"You already read {path}, and the file has not changed.\n"
        "Git diff is still empty.\n"
        "Do not reread this file again. Edit the most likely target file now using write_file/edit_file/apply_patch."
        f"{target_text}"
    )
    if excerpt:
        output += f"\n\nCached excerpt from the previous read:\n{excerpt}"
    elif detail:
        output += f"\n\n{detail}"
    return ToolResult(
        tool="read_file",
        output=output,
        exit_code=0,
        metadata={
            "path": path,
            "cached_duplicate": True,
            "original_read_index": original_index,
        },
    )


def cached_read_excerpt(state: AgentState, index: int | None, *, max_lines: int = 80, max_chars: int = 4000) -> str:
    if index is None or index < 0 or index >= len(state.messages):
        return ""
    output = str(state.messages[index].get("output") or "")
    if not output:
        return ""
    lines = output.splitlines()
    excerpt = "\n".join(lines[:max_lines])
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rstrip() + "\n...[truncated]"
    return excerpt


def duplicate_inspection_edit_forced(state: AgentState, workflow: Any) -> bool:
    if source_progress_forced(state):
        return True
    if workflow.phase != WorkflowPhase.EDIT_REQUIRED or getattr(workflow, "diff_exists", False):
        return False
    return not successful_edit_tool_called(state) and duplicate_inspection_rejection_count(state) > 0


def duplicate_inspection_rejection_count(state: AgentState) -> int:
    return sum(
        1
        for message in state.messages
        if message.get("role") == "system"
        and message.get("kind") == "feedback"
        and "duplicate_inspection_after_edit_pressure" in str(message.get("content") or "")
    )


def target_candidates_for_edit_pressure(state: AgentState) -> list[str]:
    if state.task_contract is None:
        return []
    strict_targets, candidate_targets = target_file_guidance(state.job.instruction, state.task_contract.must_modify_files)
    return candidate_targets or strict_targets


def targeted_repair_rerun_command_block(state: AgentState, args: dict[str, object]) -> str:
    command = next_targeted_repair_rerun_command(state)
    if not command:
        return ""
    observed = str(args.get("command") or "")
    if commands_equivalent(observed, command):
        return ""
    return f"Active targeted repair was modified; rerun only the repair command now: {command}"


def targeted_repair_successful_commands_after_latest_edit(state: AgentState) -> list[str]:
    start = state.active_repair_started_at
    targets = targeted_repair_targets(state)
    for index, message in enumerate(state.messages[state.active_repair_started_at :], start=state.active_repair_started_at):
        if message.get("role") != "tool" or int(message.get("exit_code") or 0) != 0:
            continue
        tool = str(message.get("tool") or "")
        if tool not in {"write_file", "edit_file", "replace_in_file", "apply_patch"}:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        path = normalize_workspace_relative_path(str(metadata.get("path") or ""))
        if tool == "apply_patch" or not targets or path in targets:
            start = index + 1
    observed: list[str] = []
    for message in state.messages[start:]:
        if message.get("role") != "tool" or int(message.get("exit_code") or 0) != 0:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        command = str(metadata.get("command") or "")
        if command:
            observed.append(command)
    return observed


def next_targeted_repair_rerun_command(state: AgentState) -> str:
    action = state.active_repair_action or {}
    commands = [str(command) for command in action.get("rerun_commands") or [] if str(command)]
    if not commands:
        return ""
    observed = targeted_repair_successful_commands_after_latest_edit(state)
    for command in commands:
        if not any(commands_equivalent(seen, command) for seen in observed):
            return command
    return ""


def controller_owned_required_command(state: AgentState, workflow: Any) -> str:
    if state.verification_scheduler is None:
        commands = [str(command) for command in (state.task_contract.must_run_commands if state.task_contract is not None else []) if str(command)]
        state.verification_scheduler = VerificationScheduler.from_explicit_commands(commands)
    scheduler = state.verification_scheduler
    if not scheduler.commands:
        return ""
    for node in scheduler.commands:
        if command_was_run(state, node.command) and not scheduler.is_fresh_success(node.command):
            scheduler.record(node.command, True)
    if state.active_repair_action:
        if state.repair_mode != "targeted_repair" or not targeted_repair_modified_target(state):
            return ""
    else:
        if state.repair_mode is not None or workflow.phase != WorkflowPhase.TEST_REQUIRED:
            return ""
        if not getattr(workflow, "diff_exists", False) or not successful_edit_tool_called(state):
            return ""
        if missing_must_modify_targets(state):
            return ""
    command = scheduler.next_command() or ""
    if not command:
        return ""
    profile = state.profile or select_task_profile(state.job.instruction)
    if profile.name == "crawler":
        return command
    return command if "\n" in command or heredoc_delimiter_from_command(command) is not None else ""


def targeted_repair_forced_tool(
    state: AgentState,
    tool_name: str,
    args: dict[str, object] | None = None,
) -> tuple[str, dict[str, object], str] | None:
    if state.repair_mode == "targeted_repair" and state.active_repair_action:
        return None

    status_output = state.latest_git_status.output if state.latest_git_status is not None else ""
    workflow = workflow_snapshot(state, status_output)
    if workflow.phase != WorkflowPhase.TEST_REQUIRED:
        return None
    if missing_must_modify_targets(state):
        return None
    missing = list(getattr(workflow, "missing_commands", None) or [])
    if not missing:
        return None
    command = str(missing[0])
    observed = str((args or {}).get("command") or "") if tool_name == "run_command" else ""
    if tool_name == "run_command" and commands_equivalent(observed, command):
        return None
    return "run_command", {"command": command}, "test_required_requires_exact_command"
    return None


def preferred_targeted_repair_target(state: AgentState, targets: list[str]) -> str:
    _ = state
    for candidate in targets:
        normalized = normalize_workspace_relative_path(candidate)
        name = normalized.rsplit("/", 1)[-1]
        if normalized and not normalized.startswith("tests/") and not name.startswith("test_"):
            return candidate
    return targets[0]


def json_dumps_compact(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def test_file_edit_allowed_while_tests_missing(tool_name: str, args: dict[str, object]) -> bool:
    if tool_name == "apply_patch":
        patch = str(args.get("patch") or "")
        return any(path.startswith(("tests/", "test_")) for path in patch_paths(patch))
    if tool_name not in {"write_file", "edit_file", "replace_in_file"}:
        return False
    path = normalize_workspace_relative_path(str(args.get("path") or ""))
    return path.startswith("tests/") or path.rsplit("/", 1)[-1].startswith("test_")


def patch_paths(patch: str) -> list[str]:
    paths: list[str] = []
    for line in patch.splitlines():
        stripped = line.strip()
        for prefix in ("*** Update File:", "*** Add File:", "*** Delete File:"):
            if stripped.startswith(prefix):
                path = normalize_workspace_relative_path(stripped[len(prefix) :].strip())
                if path and path not in paths and path != "/dev/null":
                    paths.append(path)
        if not stripped.startswith(("--- ", "+++ ", "diff --git ")):
            continue
        for token in stripped.split():
            if token.startswith(("a/", "b/")):
                path = normalize_workspace_relative_path(token[2:])
                if path and path not in paths and path != "/dev/null":
                    paths.append(path)
    return paths


def crawler_external_source_tool_block(state: AgentState, tool_name: str, args: dict[str, object]) -> str:
    policy_block = source_tool_block(state, tool_name, args)
    if policy_block or tool_name == "inspect_source":
        return policy_block
    if tool_name in EDIT_TOOLS:
        return ""
    if not is_crawler_instruction(state.job.instruction):
        return ""
    allowed = instruction_source_domains(state.job.instruction)
    if not allowed:
        return ""
    if tool_name == "inspect_source":
        raw_url = str(args.get("url") or "")
        candidates = instruction_source_urls(state.job.instruction)
        if raw_url in candidates:
            return ""
        return (
            "inspect_source must use a literal source candidate from the task without changing its query string: "
            + ", ".join(candidates[:5])
        )
    if tool_name == "fetch_url":
        raw_url = str(args.get("url") or "")
        host = urlparse(raw_url).hostname or ""
        if host and source_domain_allowed(host, allowed):
            return ""
        return (
            f"fetch_url is blocked for {host or '<missing host>'}. "
            f"This crawler task may only inspect the planned source domain(s): {', '.join(sorted(allowed))}."
        )
    if tool_name == "web_search":
        query = str(args.get("query") or args.get("q") or "").lower()
        blocked_terms = {"cisa.gov", "cisa", "cis benchmark", "cis control", "security advisory"}
        if any(term in query for term in blocked_terms):
            return (
                "web_search is blocked because the query drifted to CISA/CIS security content. "
                f"Keep research anchored to the planned source domain(s): {', '.join(sorted(allowed))}."
            )
    if tool_name in {"write_file", "edit_file", "replace_in_file"}:
        content = str(args.get("content") or args.get("new_text") or args.get("replacement") or "")
        blocked = blocked_domains_in_text(content, allowed)
        if blocked:
            return (
                f"{tool_name} content references source domain(s) outside the plan: {', '.join(blocked)}. "
                f"Rewrite the file for the planned source domain(s): {', '.join(sorted(allowed))}."
            )
    if tool_name == "apply_patch":
        patch = str(args.get("patch") or "")
        blocked = blocked_domains_in_text(patch, allowed)
        if blocked:
            return (
                f"apply_patch content references source domain(s) outside the plan: {', '.join(blocked)}. "
                f"Patch only for the planned source domain(s): {', '.join(sorted(allowed))}."
            )
    return ""


def crawler_corrected_fetch_url_args(state: AgentState, tool_name: str, args: dict[str, object]) -> dict[str, object] | None:
    if tool_name != "fetch_url" or not is_crawler_instruction(state.job.instruction):
        return None
    source_urls = instruction_source_urls(state.job.instruction)
    allowed = instruction_source_domains(state.job.instruction)
    if not source_urls or not allowed:
        return None
    raw_url = str(args.get("url") or "")
    host = urlparse(raw_url).hostname or ""
    if host and source_domain_allowed(host, allowed):
        return None
    corrected = dict(args)
    corrected["url"] = source_urls[0]
    corrected["goal"] = (
        f"Inspect the planned crawler source {source_urls[0]} and derive selectors/fields for the requested target. "
        "Ignore unrelated CIS/CISA/security-control interpretations."
    )
    return corrected


def instruction_source_domains(instruction: str) -> set[str]:
    domains: set[str] = set()
    for url in instruction_source_urls(instruction):
        host = urlparse(url).hostname
        if host:
            normalized = normalize_detected_host(host)
            if normalized:
                domains.add(normalized)
    for match in re.finditer(r'"allowed_domains"\s*:\s*\[([^\]]+)\]', instruction or ""):
        for domain in re.findall(r'"([^"]+)"', match.group(1)):
            normalized = normalize_detected_host(domain)
            if normalized:
                domains.add(normalized)
    return domains


def normalize_detected_host(host: str) -> str:
    normalized = host.lower().strip(".")
    if "${" in normalized:
        normalized = normalized.split("${", 1)[0].rstrip(".")
    if "{" in normalized:
        normalized = normalized.split("{", 1)[0].rstrip(".")
    return normalized


def source_domain_allowed(host: str, allowed: set[str]) -> bool:
    normalized = normalize_detected_host(host)
    if not normalized:
        return False
    return any(normalized == domain or normalized.endswith(f".{domain}") for domain in allowed)


def blocked_domains_in_text(text: str, allowed: set[str]) -> list[str]:
    blocked: list[str] = []
    candidates: list[str] = []
    for match in re.finditer(r"https?://[^\s'\"`)>]+", text.lower()):
        host = urlparse(match.group(0).rstrip(".,;:")).hostname
        if host:
            candidates.append(host)
    for match in re.finditer(r"\b(?:source_url|url|domain|host)\s*=\s*[\"']([^\"']+)[\"']", text.lower()):
        value = match.group(1).strip()
        host = urlparse(value).hostname or value
        if "." in host:
            candidates.append(host)
    for host in candidates:
        host = normalize_detected_host(host)
        if not host:
            continue
        if source_domain_allowed(host, allowed):
            continue
        if host in {"example.com", "localhost"} or host.endswith(".example"):
            continue
        if host not in blocked:
            blocked.append(host)
    return blocked[:5]


def latest_failed_required_command(state: AgentState) -> ToolResult | None:
    required = set(normalize_command(command) for command in (state.task_contract.must_run_commands if state.task_contract else []) if command)
    if not required:
        return None
    resolved: set[str] = set()
    for message in reversed(state.messages):
        if message.get("role") != "tool":
            continue
        if message.get("tool") != "run_command":
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        command = normalize_command(str(metadata.get("command") or ""))
        if command not in required:
            continue
        if int(message.get("exit_code") or 0) == 0:
            resolved.add(command)
            continue
        if command in resolved:
            continue
        return ToolResult(
            tool="run_command",
            output=str(message.get("output") or ""),
            exit_code=int(message.get("exit_code") or 1),
            metadata=metadata,
            truncated=bool(message.get("truncated")),
        )
    return None


def target_edit_allowed_while_tests_missing(state: AgentState, tool_name: str, args: dict[str, object]) -> bool:
    if tool_name not in {"write_file", "edit_file", "replace_in_file", "apply_patch"}:
        return False
    missing_targets = missing_must_modify_targets(state)
    if not missing_targets:
        return False
    if tool_name == "apply_patch":
        return True
    path = normalize_workspace_relative_path(str(args.get("path") or ""))
    return bool(path and path in missing_targets)


def missing_must_modify_targets(state: AgentState) -> list[str]:
    task_contract = state.task_contract
    if task_contract is None or not task_contract.must_modify_files:
        return []
    strict_targets, candidate_targets = target_file_guidance(state.job.instruction, task_contract.must_modify_files)
    candidate_target_set = {normalize_workspace_relative_path(candidate) for candidate in candidate_targets}
    required_targets = strict_targets or [
        path
        for path in task_contract.must_modify_files
        if normalize_workspace_relative_path(path) not in candidate_target_set
    ]
    if not required_targets:
        return []
    status = state.latest_git_status.output if state.latest_git_status is not None else ""
    changed = {normalize_workspace_relative_path(path) for path in changed_paths_from_status(status)}
    edited_paths: set[str] = set()
    for message in state.messages:
        if message.get("role") != "tool":
            continue
        if message.get("tool") not in {"write_file", "edit_file", "replace_in_file", "apply_patch"}:
            continue
        if int(message.get("exit_code") or 0) != 0:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        path = normalize_workspace_relative_path(str(metadata.get("path") or ""))
        if path:
            edited_paths.add(path)
    return [
        path
        for raw_path in required_targets
        if (path := normalize_workspace_relative_path(raw_path)) and path not in changed and path not in edited_paths
        and not generated_artifact_target(path)
    ]


def generated_artifact_target(path: str) -> bool:
    normalized = normalize_workspace_relative_path(path)
    return normalized.startswith(("data/", "output/", "outputs/", "artifacts/"))


def normalize_workspace_relative_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    if normalized.startswith("/workspace/"):
        normalized = normalized[len("/workspace/") :]
    normalized = normalized.lstrip("./")
    normalized = posixpath.normpath(normalized)
    return "" if normalized == "." else normalized


def required_command_failed_after_latest_edit(state: AgentState, command: str) -> bool:
    expected = command
    seen_edit = False
    for message in reversed(state.messages):
        if message.get("role") != "tool":
            continue
        tool = str(message.get("tool") or "")
        if tool in {"edit_file", "write_file", "replace_in_file", "apply_patch"} and int(message.get("exit_code") or 0) == 0:
            seen_edit = True
            break
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        observed = str(metadata.get("command") or "")
        if commands_equivalent(observed, expected) and int(message.get("exit_code") or 0) != 0:
            return True
    return False if seen_edit else False


def decision_to_step(decision, usage_meter: LLMUsageMeter | None = None) -> dict[str, object]:
    payload: dict[str, object] = {"type": "llm_decision", "decision_type": decision.type}
    if decision.tool_name:
        payload["tool"] = decision.tool_name
        payload["args"] = sanitize_tool_args(decision.args or {})
    if decision.summary:
        payload["summary"] = truncate_text(decision.summary, 2000)
    if getattr(decision, "verification", None):
        payload["verification"] = truncate_text(decision.verification, 2000)
    if getattr(decision, "no_test_reason", None):
        payload["no_test_reason"] = truncate_text(decision.no_test_reason, 1000)
    if getattr(decision, "remaining_risks", None):
        payload["remaining_risks"] = [truncate_text(str(risk), 500) for risk in decision.remaining_risks or []]
    if getattr(decision, "reasoning", None):
        payload["reasoning"] = truncate_text(str(decision.reasoning), 4000)
    if getattr(decision, "reasoning_records", None):
        payload["reasoning_records"] = [
            {
                key: truncate_text(str(value), 4000) if isinstance(value, str) else value
                for key, value in dict(record).items()
                if value is not None and value != ""
            }
            for record in decision.reasoning_records or []
            if isinstance(record, dict)
        ][:10]
    if usage_meter is not None:
        payload["usage"] = usage_meter.snapshot()
    return payload


def tool_exception_result(tool_name: str, exc: Exception) -> ToolResult:
    detail = truncate_text(str(exc), 2000)
    return ToolResult(
        tool=tool_name,
        output=f"tool invocation failed: {detail}",
        exit_code=1,
        metadata={"exception_type": type(exc).__name__, "error": detail},
    )


def sanitize_tool_args(args: dict[str, Any]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, value in args.items():
        if key == "content" and isinstance(value, str):
            sanitized[key] = {"bytes": len(value.encode("utf-8")), "preview": truncate_text(value, 500)}
            continue
        sanitized[key] = sanitize_value(value)
    return sanitized


def sanitize_value(value: Any) -> object:
    if isinstance(value, str):
        return truncate_text(value, 1000)
    if isinstance(value, dict):
        return {str(key): sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return truncate_text(str(value), 1000)


def summarize_output(output: str, *, max_lines: int = 20, max_chars: int = 4000) -> str:
    lines = output.splitlines()
    summary = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        summary += f"\n... truncated {len(lines) - max_lines} lines"
    return truncate_text(summary, max_chars)


def truncate_text(text: str, limit: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="replace") + "\n<truncated>"


def non_retryable_llm_error(exc: Exception) -> bool:
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code in {401, 403}:
        return True
    text = str(exc).lower()
    return "401 unauthorized" in text or "403 forbidden" in text or "invalid api key" in text or "incorrect api key" in text


def provider_failure_reason(exc: Exception) -> str | None:
    if isinstance(exc, ProviderUnavailableError):
        if exc.category == "provider_auth_failed":
            return "llm_auth_failed"
        return f"llm_provider_unavailable:{exc.category}"
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    text = str(exc).lower()
    if status_code in {502, 503, 504} or any(fragment in text for fragment in ("502 bad gateway", "503 service unavailable", "504 gateway timeout", "no_upstream_capacity")):
        return "llm_provider_unavailable:provider_upstream_unavailable"
    if status_code == 429 or "rate limit" in text or "too many requests" in text:
        return "llm_provider_unavailable:provider_rate_limited"
    if "connection refused" in text or "connection reset" in text or "server disconnected" in text or "timeout" in text:
        return "llm_provider_unavailable:provider_network_error"
    return None


def retryable_provider_failure(reason: str | None, exc: Exception) -> bool:
    if isinstance(exc, ProviderUnavailableError):
        return bool(exc.retryable)
    return reason in {
        "llm_provider_unavailable:provider_upstream_unavailable",
        "llm_provider_unavailable:provider_rate_limited",
        "llm_provider_unavailable:provider_network_error",
    }
