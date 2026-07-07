from __future__ import annotations

import asyncio
import posixpath
import re
from typing import Any
from urllib.parse import urlparse

from docode.agent.context import ContextManager, ContextPack
from docode.agent.inspector import ProjectInspector
from docode.agent.prompts import DOCODE_SYSTEM_PROMPT
from docode.agent.quality_gate import QualityGate, QualityGateResult
from docode.agent.repair_planner import RepairAction, format_repair_action, plan_repair_from_tool_result
from docode.agent.reviewer import CodeReviewer, ReviewResult
from docode.agent.state import AgentState
from docode.agent.stuck import REPAIR_ALLOWED_TOOLS, StuckDetector, git_status_clean
from docode.agent.stop_policy import StopPolicy
from docode.agent.task_contract import TaskContract, is_crawler_instruction, task_contract_from_instruction
from docode.agent.verifier import CodingVerifier, VerificationResult, changed_files_from_diff, verification_evidence_from_steps
from docode.agent.workflow import (
    WorkflowPhase,
    changed_paths_from_status,
    commands_equivalent,
    final_candidate_gate,
    successful_edit_tool_called,
    workflow_snapshot,
)
from docode.artifacts.exporter import ArtifactExporter, terminal_artifact_id
from docode.dobox.tools import DoBoxTools
from docode.dobox.types import ToolResult
from docode.llm.runtime import AgentDecision, DecisionLLM, LLMUsageMeter, ProviderUnavailableError
from docode.storage.models import CodingJob, JobStatus
from docode.storage.repository import JobRepository

INITIAL_NO_DIFF_EXPLORATION_BUDGET = 3


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

    async def run(self, job: CodingJob) -> CodingJob:
        job = await self.repository.update_job(job.id, status=JobStatus.RUNNING)
        state = AgentState(job=job)
        await self.bootstrap(state)

        while True:
            cancelled = await self.cancelled_job(job.id)
            if cancelled is not None:
                return cancelled

            self.sync_llm_usage(state)
            stop = self.stop_policy.evaluate(state)
            if stop.should_stop:
                finalized = await self.maybe_auto_finalize_before_stop(state, stop.reason or "stopped")
                if finalized is not None:
                    return finalized
                return await self.fail(job.id, stop.reason or "stopped")

            refresh_targeted_repair_phase(state)
            context_pack = await self.collect_observation(state)
            observation = context_pack.render()
            await self.repository.add_step(job.id, "system", observation_step(context_pack))
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
                if current_workflow.phase == WorkflowPhase.FINAL_READY and state.repair_mode is None:
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
                forced_repair = targeted_repair_forced_tool(state, tool_name)
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
                review_retargeted_args = targeted_review_repair_retarget_args(state, tool_name)
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
                retargeted_args = crawler_missing_artifact_file_retarget_args(state, current_workflow, tool_name, tool_args)
                if retargeted_args is not None:
                    await self.repository.add_step(
                        job.id,
                        "system",
                        {
                            "type": "decision_retargeted",
                            "reason": "crawler_required_artifact_file_missing",
                            "from_tool": tool_name,
                            "to_tool": "write_file",
                            "target": retargeted_args.get("path"),
                        },
                    )
                    tool_name = "write_file"
                    tool_args = retargeted_args
                repair_tool_block = repair_mode_tool_block(state, tool_name)
                if repair_tool_block:
                    await self.record_rejected_decision(
                        state,
                        reason=f"{state.repair_mode}_tool_forbidden" if state.repair_mode else "repair_mode_tool_forbidden",
                        detail=repair_tool_block,
                    )
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
                if current_workflow.phase == WorkflowPhase.FINAL_READY and state.repair_mode is None:
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
        action = plan_repair_from_tool_result(tool=result.tool, output=result.output, metadata=result.metadata or {})
        if action is None:
            return
        await self.activate_targeted_repair(state, action, result=result)

    async def maybe_activate_required_command_repair(self, state: AgentState, workflow: Any) -> bool:
        if workflow.phase != WorkflowPhase.TEST_REQUIRED:
            return False
        failed = latest_failed_required_command(state)
        if failed is None:
            return False
        action = plan_repair_from_tool_result(tool=failed.tool, output=failed.output, metadata=failed.metadata or {})
        if action is None:
            return False
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
        if state.repair_mode is not None:
            return None
        status = await self.tools.git_status()
        state.latest_git_status = status
        current_workflow = workflow_snapshot(state, status.output)
        if current_workflow.phase != WorkflowPhase.FINAL_READY:
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
        if state.active_repair_action and not targeted_repair_rerun_satisfied(state):
            await self.record_rejected_decision(
                state,
                reason="targeted_repair_rerun_missing",
                detail=targeted_rerun_missing_detail(state),
            )
            return None
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
        quality = await self.quality_gate.run(tools=self.tools, task_contract=state.task_contract, instruction=job.instruction)
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
        evidence = verification_evidence_from_steps(await self.repository.list_steps(job.id)).with_no_test_reason(decision.no_test_reason)
        try:
            verification = await asyncio.wait_for(self.verifier.verify(job, self.tools, evidence=evidence), timeout=180)
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
        count = state.failure_signatures.get(action.signature, 0) + 1
        state.failure_signatures[action.signature] = count
        state.repair_action_attempts = count
        action_payload = repair_action_contract(action, state)
        state.active_repair_action = action_payload
        state.active_repair_started_at = len(state.messages)
        state.targeted_repair_phase = "inspect_allowed"
        state.targeted_repair_inspections = 0
        state.targeted_repair_edits = 0
        state.repair_mode = "targeted_repair"
        state.add_feedback(format_repair_action(action, repeated_count=count))
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
        inspection = await self.inspector.inspect(state.job.instruction, self.tools)
        state.inspection = inspection
        state.task_contract = task_contract_from_instruction(state.job.instruction)
        await self.repository.add_step(
            state.job.id,
            "system",
            {
                "type": "bootstrap",
                "listing": inspection.listing,
                "important_files": list(inspection.important_files),
                "detected_commands": inspection.detected_commands,
                "plan": inspection.plan,
                "acceptance_criteria": inspection.acceptance_criteria,
                "task_contract": {
                    "must_modify_files": state.task_contract.must_modify_files,
                    "must_run_commands": state.task_contract.must_run_commands,
                    "forbidden_finish_conditions": state.task_contract.forbidden_finish_conditions,
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
        return self.context_manager.build_pack(
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
        )

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
                f"- next required tool: edit_file/apply_patch/write_file {target}\n"
                f"- do not run tests again until {target} changes\n"
            )
            if commands:
                compact += f"- rerun after patch: {commands[0]}\n"
            state.add_feedback(f"{compact}\n{reason}: {truncate_text(detail, 600)}")
        else:
            state.add_feedback(f"{reason}: {truncate_text(detail, 1000)}{next_command}")
        state.iteration += 1

    def sync_llm_usage(self, state: AgentState) -> None:
        if self.usage_meter is not None:
            state.llm_tokens_used = self.usage_meter.total_tokens
            state.llm_cost_used = self.usage_meter.cost

    def _llm_messages(self, state: AgentState) -> list[dict[str, Any]]:
        if not state.messages:
            return []
        return [compact_llm_message(message) for message in state.messages[-4:]]


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
            "relevant_fetch_urls": result.evidence.relevant_fetch_urls or [],
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
        "latest_evidence": context_pack.latest_evidence,
        "recent_messages": context_pack.recent_messages,
    }


def enrich_tool_result_metadata(tool_name: str, args: dict[str, object], result: ToolResult) -> ToolResult:
    metadata = dict(result.metadata or {})
    if "path" not in metadata and tool_name in {"write_file", "edit_file", "replace_in_file", "read_file"}:
        path = args.get("path")
        if path:
            metadata["path"] = str(path)
    if "command" not in metadata and tool_name == "run_command":
        command = args.get("command")
        if command:
            metadata["command"] = str(command)
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
        compact["output"] = truncate_text(str(message["output"]), 500)
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        keep = {}
        for key in ("path", "command", "reason", "url", "status_code", "content_type", "prompt_output_truncated"):
            if key in metadata:
                keep[key] = metadata[key]
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


def repair_action_from_quality_gate(result: QualityGateResult) -> RepairAction | None:
    issue_text = "\n".join(f"{issue.code}: {issue.message}" for issue in result.blockers())
    if not issue_text:
        return None
    return plan_repair_from_tool_result(
        tool="run_command",
        output=issue_text,
        metadata={"command": "python3 crawler.py --dry-run"},
    )


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
    instruction += (
        "For crawler artifacts, make crawler.py write .araneae/sink/events.jsonl at runtime, "
        "align parsed records with the declared schema, strengthen fixtures, and update parser tests."
    )
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
        if "tests/test_parser.py" in lower or "test file" in lower or "parser tests" in lower:
            preferred.append("tests/test_parser.py")
        if "fixtures/sample.html" in lower or "fixture" in lower:
            preferred.append("fixtures/sample.html")
        if (
            "crawler.py" in lower
            or "dry-run" in lower
            or "preflight" in lower
            or "events.jsonl" in lower
            or "output format" in lower
        ):
            preferred.append("crawler.py")
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
        ordered = []
        for path in preferred:
            if path in available and path not in ordered:
                ordered.append(path)
        ordered.extend(path for path in task_contract.must_modify_files if path not in ordered)
        return ordered
    return []


def repair_action_contract(action: RepairAction, state: AgentState) -> dict[str, Any]:
    payload = action.to_dict()
    target_files = [str(path) for path in payload.get("target_files") or [] if str(path)]
    target_file = target_files[0] if target_files else None
    rerun_commands = [str(command) for command in payload.get("rerun_commands") or [] if str(command)]
    instruction = str(payload.get("instruction") or "")
    must_change_symbols = []
    for symbol in ("number_from_text", "parse_trending", "parse_repositories", "parse_repos"):
        if symbol in instruction:
            must_change_symbols.append(symbol)
    if payload.get("category") == "missing_required_field" and "crawler.py" in target_files:
        must_change_symbols.extend(symbol for symbol in ("parse_trending", "number_from_text") if symbol not in must_change_symbols)
    payload.update(
        {
            "phase": "REPAIR_REQUIRED",
            "target_file": target_file,
            "must_change_symbols": must_change_symbols,
            "next_allowed_tools": ["read_file", "apply_patch", "edit_file", "write_file"],
            "forbidden_until_modified": ["run_command"],
            "rerun_after_modified": rerun_commands[0] if rerun_commands else None,
            "created_at_message_index": len(state.messages),
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
            "python-bugfix: read calculator.py, edit retry_count so it returns attempts, run "
            "`python3 -m unittest discover -s tests`, run git_diff, then final_candidate."
        )
    if "cli.py" in file_names:
        hints.append(
            "python-cli: read cli.py, edit `print('TODO')` to print a greeting using args.name, run "
            "`python3 cli.py --name Ada`, run git_diff, then final_candidate."
        )
    return hints


def allowed_tool_definitions(definitions: list[Any], repair_mode: str | None) -> list[Any]:
    if repair_mode not in {"must_edit", "quality_repair", "targeted_repair"}:
        return definitions
    allowed = allowed_tools_for_repair_mode_name(repair_mode)
    return [definition for definition in definitions if getattr(definition, "name", None) in allowed]


def allowed_tool_definitions_for_state(definitions: list[Any], state: AgentState) -> list[Any]:
    refresh_targeted_repair_phase(state)
    if state.repair_mode == "targeted_repair" and state.active_repair_action:
        allowed = targeted_repair_allowed_tools_for_phase(state)
        return [definition for definition in definitions if getattr(definition, "name", None) in allowed]
    status_output = state.latest_git_status.output if state.latest_git_status is not None else ""
    workflow = workflow_snapshot(state, status_output)
    if crawler_source_research_priority_active(state, workflow):
        allowed = initial_crawler_source_tools(state)
        return [definition for definition in definitions if getattr(definition, "name", None) in allowed]
    if (
        workflow.phase == WorkflowPhase.EDIT_REQUIRED
        and not successful_edit_tool_called(state)
        and exploratory_tool_calls(state) >= INITIAL_NO_DIFF_EXPLORATION_BUDGET
    ):
        allowed = {"write_file", "edit_file", "replace_in_file", "apply_patch", "git_status", "git_diff"}
        return [definition for definition in definitions if getattr(definition, "name", None) in allowed]
    if workflow.phase == WorkflowPhase.TEST_REQUIRED and missing_must_modify_targets(state):
        allowed = {"write_file", "apply_patch", "git_status", "git_diff"}
        return [definition for definition in definitions if getattr(definition, "name", None) in allowed]
    return allowed_tool_definitions(definitions, state.repair_mode)


def repair_mode_tool_block(state: AgentState, tool_name: str) -> str:
    refresh_targeted_repair_phase(state)
    if state.repair_mode not in {"must_edit", "quality_repair", "targeted_repair"}:
        return ""
    allowed = allowed_tools_for_repair_mode(state)
    if tool_name in allowed:
        return ""
    return repair_mode_forbidden_detail(state, tool_name)


def allowed_tools_for_repair_mode(state: AgentState) -> set[str]:
    if state.repair_mode == "targeted_repair":
        return targeted_repair_allowed_tools_for_phase(state)
    return allowed_tools_for_repair_mode_name(state.repair_mode)


def targeted_repair_allowed_tools_for_phase(state: AgentState) -> set[str]:
    action = state.active_repair_action or {}

    # 一旦 target file 已经在本轮 repair 后被修改，才允许 rerun command。
    if targeted_repair_modified_target(state):
        return {"run_command", "git_status", "git_diff"}

    # target file 还没改之前，绝对不要暴露 run_command。
    # inspect_allowed 允许读一次，也允许模型直接改。
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

    # edit_forced 阶段只允许修改，不允许继续读/搜/跑测试。
    if state.targeted_repair_phase == "edit_forced":
        return {
            "edit_file",
            "write_file",
            "replace_in_file",
            "apply_patch",
        }

    # fallback：保守处理，不允许 run_command
    return {
        "read_file",
        "edit_file",
        "write_file",
        "replace_in_file",
        "apply_patch",
    }


def allowed_tools_for_repair_mode_name(repair_mode: str | None) -> set[str]:
    if repair_mode == "must_edit":
        return {"edit_file", "write_file", "replace_in_file", "apply_patch", "git_status", "git_diff"}
    if repair_mode in {"quality_repair", "targeted_repair"}:
        return {"read_file", "edit_file", "write_file", "replace_in_file", "apply_patch", "run_command", "git_status", "git_diff"}
    return set()


def repair_mode_forbidden_detail(state: AgentState, tool_name: str) -> str:
    if state.repair_mode == "targeted_repair" and state.active_repair_action:
        instruction = str(state.active_repair_action.get("instruction") or "")
        return (
            f"{tool_name} is blocked while repair_mode=targeted_repair.\n"
            f"You must follow the active repair action:\n{instruction}"
        )
    if state.repair_mode == "must_edit":
        return (
            f"{tool_name} is blocked while repair_mode=must_edit. "
            "Call edit_file, write_file, replace_in_file, or apply_patch to change a target file."
        )
    return f"{tool_name} is blocked while repair_mode={state.repair_mode}."


def targeted_repair_exploration_block(state: AgentState, tool_name: str) -> str:
    refresh_targeted_repair_phase(state)
    if state.repair_mode != "targeted_repair" or not state.active_repair_action:
        return ""
    if targeted_repair_modified_target(state):
        return ""
    if state.targeted_repair_phase != "edit_forced":
        return ""
    if tool_name in {"read_file", "search", "list_files", "run_command"}:
        targets = state.active_repair_action.get("target_files") or []
        target_text = ", ".join(str(target) for target in targets) or "the target file"
        return f"You have already inspected enough context for the active targeted repair. Modify {target_text} now."
    return ""


def targeted_repair_action_block(state: AgentState, tool_name: str, args: dict[str, object]) -> str:
    refresh_targeted_repair_phase(state)
    if state.repair_mode != "targeted_repair" or not state.active_repair_action:
        return ""
    targets = targeted_repair_targets(state)
    if not targets:
        return ""
    target_text = ", ".join(sorted(targets))
    if targeted_repair_modified_target(state):
        if tool_name == "run_command":
            return targeted_repair_rerun_command_block(state, args)
        return ""
    if tool_name in {"git_status", "git_diff"}:
        return ""
    if tool_name == "run_command":
        return f"Active targeted repair requires modifying {target_text} before running commands."
    if tool_name in {"write_file", "edit_file", "replace_in_file"}:
        path = normalize_workspace_relative_path(str(args.get("path") or ""))
        if path and path_matches_any_target(path, targets):
            return ""
        return f"Active targeted repair requires modifying {target_text}; attempted edit target was {path or '<missing>'}."
    if tool_name == "apply_patch":
        patch = str(args.get("patch") or "")
        if patch_touches_any_target(patch, targets):
            return ""
        return f"Active targeted repair patch must touch {target_text}."
    return ""


def note_targeted_repair_tool_result(state: AgentState, result: ToolResult) -> None:
    if state.repair_mode != "targeted_repair" or not state.active_repair_action:
        return
    if not result.ok:
        refresh_targeted_repair_phase(state)
        return
    if result.tool in {"read_file", "search", "list_files"}:
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
        if message.get("role") == "tool" and message.get("tool") in {"read_file", "search", "list_files"}:
            count += 1
    return count


def targeted_repair_modified_target(state: AgentState) -> bool:
    action = state.active_repair_action or {}
    targets = targeted_repair_targets(state)
    if not targets:
        return successful_edit_tool_called(state)
    for message in reversed(state.messages[state.active_repair_started_at :]):
        if message.get("role") != "tool":
            continue
        tool = str(message.get("tool") or "")
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
        return bool(action) and targeted_repair_modified_target(state)
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
        observed = " ".join(str(metadata.get("command") or "").split())
        for command in commands:
            expected = " ".join(command.split())
            if commands_equivalent(observed, expected):
                return True
    return False


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
    if state.repair_mode == "must_edit" and tool_name not in {"write_file", "edit_file", "replace_in_file", "apply_patch"}:
        return f"{tool_name} is blocked while EDIT_REQUIRED. Create or edit a target file now."
    if tool_name in {"write_file", "edit_file", "replace_in_file"}:
        target_block = edit_required_target_file_block(state, tool_name, args)
        if target_block:
            return target_block
    if tool_name in {"write_file", "edit_file", "replace_in_file", "apply_patch"}:
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


def crawler_source_research_priority_active(state: AgentState, workflow: Any) -> bool:
    if workflow.phase != WorkflowPhase.EDIT_REQUIRED:
        return False
    if successful_edit_tool_called(state):
        return False
    if exploratory_tool_calls(state) >= INITIAL_NO_DIFF_EXPLORATION_BUDGET:
        return False
    if not is_crawler_instruction(state.job.instruction):
        return False
    if not instruction_source_urls(state.job.instruction):
        return False
    return not source_research_succeeded(state)


def initial_crawler_source_tools(state: AgentState) -> set[str]:
    base = {"write_file", "edit_file", "replace_in_file", "apply_patch"}
    if explicit_source_fetch_attempted(state):
        return base | {"fetch_url", "web_search"}
    return base | {"fetch_url"}


def instruction_source_urls(instruction: str) -> list[str]:
    urls: list[str] = []
    for match in re.findall(r"https?://[^\s'\"`)>]+", instruction or ""):
        cleaned = match.rstrip(".,;:")
        if cleaned and cleaned not in urls:
            urls.append(cleaned)
    return urls


def source_research_succeeded(state: AgentState) -> bool:
    return any(
        message.get("role") == "tool"
        and message.get("tool") in {"fetch_url", "web_search"}
        and int(message.get("exit_code") or 0) == 0
        for message in state.messages
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
    if workflow.phase != WorkflowPhase.TEST_REQUIRED:
        return ""
    missing = getattr(workflow, "missing_commands", None) or []
    if not missing:
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
    observed = " ".join(str(args.get("command") or "").strip().split())
    expected = " ".join(next_command.strip().split())
    if not commands_equivalent(observed, expected):
        return f"Wrong command for TEST_REQUIRED. Run this exact command first: {next_command}"
    return ""


def targeted_repair_rerun_command_block(state: AgentState, args: dict[str, object]) -> str:
    action = state.active_repair_action or {}
    commands = [str(command) for command in action.get("rerun_commands") or [] if str(command)]
    if not commands:
        return ""
    observed = " ".join(str(args.get("command") or "").strip().split())
    if any(commands_equivalent(observed, command) for command in commands):
        return ""
    return f"Active targeted repair was modified; rerun only the repair command now: {commands[0]}"


def crawler_missing_artifact_file_retarget_args(
    state: AgentState,
    workflow: Any,
    tool_name: str,
    args: dict[str, object],
) -> dict[str, object] | None:
    if workflow.phase != WorkflowPhase.TEST_REQUIRED:
        return None
    if not is_crawler_instruction(state.job.instruction):
        return None
    missing = missing_must_modify_targets(state)
    if not missing:
        return None
    if tool_name == "run_command":
        return None
    requested = normalize_workspace_relative_path(str(args.get("path") or ""))
    if tool_name == "write_file" and requested in missing:
        return None
    target = missing[0]
    content = default_crawler_artifact_file_content(target, state)
    if content is None:
        return None
    return {"path": target, "content": content}


def targeted_repair_forced_tool(state: AgentState, tool_name: str) -> tuple[str, dict[str, object], str] | None:
    if state.repair_mode != "targeted_repair" or not state.active_repair_action:
        return None
    if targeted_repair_modified_target(state):
        return None
    targets = sorted(targeted_repair_targets(state))
    if not targets:
        return None
    target = targets[0]
    read_count = targeted_repair_read_count(state)
    if read_count <= 0 and tool_name in {"run_command", "git_status", "git_diff", "search", "list_files"}:
        return (
            "read_file",
            {"path": target},
            "active_repair_requires_inspection_or_patch",
        )
    if read_count > 0 and tool_name in {"run_command", "read_file", "search", "list_files"}:
        content = default_crawler_artifact_file_content(target, state)
        if content is not None:
            return (
                "write_file",
                {"path": target, "content": content},
                "active_repair_requires_target_patch",
            )
    return None


def targeted_review_repair_retarget_args(state: AgentState, tool_name: str) -> dict[str, object] | None:
    action = state.active_repair_action or {}
    if state.repair_mode != "targeted_repair" or action.get("category") not in {
        "review_repair",
        "json_semantic_failure",
        "missing_required_field",
    }:
        return None
    if targeted_repair_modified_target(state):
        return None
    if tool_name != "run_command":
        return None
    action_targets = [normalize_workspace_relative_path(str(path)) for path in action.get("target_files") or [] if str(path)]
    target = next((path for path in action_targets if default_crawler_artifact_file_content(path, state) is not None), "")
    if not target:
        return None
    content = default_crawler_artifact_file_content(target, state)
    if content is None:
        return None
    return {"path": target, "content": content}


def default_crawler_artifact_file_content(path: str, state: AgentState) -> str | None:
    source_urls = instruction_source_urls(state.job.instruction)
    source_url = source_urls[0] if source_urls else "https://github.com/trending"
    domains = sorted(instruction_source_domains(state.job.instruction)) or ["github.com"]
    objective_match = re.search(r"Objective id:\s*([A-Za-z0-9_.:-]+)", state.job.instruction)
    objective_id = objective_match.group(1) if objective_match else state.job.id
    if path == "crawler.py":
        return f'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html.parser
import json
import re
import urllib.request
from pathlib import Path

SOURCE_URL = "{source_url}"
OBJECTIVE_ID = "{objective_id}"
SCHEMA = "https_github_com_trending"
SINK_PATH = Path(".araneae/sink/events.jsonl")
FIXTURE_PATH = Path("fixtures/sample.html")


def number_from_text(value: str) -> int:
    text = value.replace(",", "").strip().lower()
    match = re.search(r"(\\d+(?:\\.\\d+)?)\\s*([km]?)", text)
    if not match:
        return 0
    number = float(match.group(1))
    suffix = match.group(2)
    if suffix == "k":
        number *= 1000
    elif suffix == "m":
        number *= 1000000
    return int(number)


class TrendingParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, object]] = []
        self.current: dict[str, object] | None = None
        self.capture: str | None = None
        self.link_seen = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        class_name = attrs_dict.get("class", "")
        if tag == "article" and "Box-row" in class_name:
            self.current = {{"description": "", "language": "", "stars_today": 0, "total_stars": 0, "forks": 0}}
            self.link_seen = False
        if self.current is None:
            return
        href = attrs_dict.get("href", "")
        if tag == "a" and href.startswith("/") and href.count("/") >= 2 and not self.link_seen:
            parts = [part.strip() for part in href.strip("/").split("/")[:2]]
            if len(parts) == 2:
                self.current["owner"] = parts[0]
                self.current["repository_name"] = parts[1]
                self.current["url"] = "https://github.com/" + "/".join(parts)
                self.link_seen = True
        if tag == "p":
            self.capture = "description"
        elif tag == "span":
            self.capture = "span"
        elif tag == "a" and "Link--muted" in class_name:
            self.capture = "stat"

    def handle_data(self, data: str) -> None:
        if self.current is None:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self.capture == "description" and not self.current.get("description"):
            self.current["description"] = text
        elif self.capture == "span":
            if "stars today" in text.lower():
                self.current["stars_today"] = number_from_text(text)
            elif not self.current.get("language") and not any(char.isdigit() for char in text):
                self.current["language"] = text
        elif self.capture == "stat":
            if not self.current.get("total_stars"):
                self.current["total_stars"] = number_from_text(text)
            elif not self.current.get("forks"):
                self.current["forks"] = number_from_text(text)

    def handle_endtag(self, tag: str) -> None:
        if tag in {{"p", "span", "a"}}:
            self.capture = None
        if tag == "article" and self.current is not None:
            if self.current.get("owner") and self.current.get("repository_name"):
                self.records.append(normalize_record(self.current))
            self.current = None


def normalize_record(raw: dict[str, object]) -> dict[str, object]:
    return {{
        "repository_name": str(raw.get("repository_name") or ""),
        "owner": str(raw.get("owner") or ""),
        "description": str(raw.get("description") or ""),
        "language": str(raw.get("language") or ""),
        "stars_today": int(raw.get("stars_today") or 0),
        "total_stars": int(raw.get("total_stars") or 0),
        "forks": int(raw.get("forks") or 0),
        "url": str(raw.get("url") or ""),
    }}


def parse_trending(html: str) -> list[dict[str, object]]:
    parser = TrendingParser()
    parser.feed(html)
    return parser.records


def fetch_source() -> str:
    request = urllib.request.Request(SOURCE_URL, headers={{"User-Agent": "cis-araneae-github-trends/1.0"}})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def write_sink(records: list[dict[str, object]], path: Path = SINK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            event = {{"type": "record", "schema": SCHEMA, "objective_id": OBJECTIVE_ID, "record": record}}
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\\n")


def write_sample_csv(records: list[dict[str, object]]) -> None:
    Path("fixtures").mkdir(exist_ok=True)
    with Path("fixtures/sample.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["repository_name", "owner", "description", "language", "stars_today", "total_stars", "forks", "url"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def load_fixture() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


def preflight() -> int:
    html = load_fixture()
    records = parse_trending(html)
    if not records:
        raise SystemExit("fixture produced no records")
    print(f"preflight ok: {{len(records)}} fixture record(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        return preflight()
    html = load_fixture() if args.dry_run else fetch_source()
    records = parse_trending(html)
    if not records:
        raise SystemExit("no GitHub trending records parsed")
    write_sink(records)
    if args.dry_run:
        write_sample_csv(records)
        print(f"dry-run complete: wrote {{len(records)}} record(s) to {{SINK_PATH}}")
    else:
        print(f"crawl complete: wrote {{len(records)}} record(s) to {{SINK_PATH}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    if path == "manifest.json":
        return json_dumps_compact(
            {
                "name": "github-trending-crawler",
                "version": "0.1.0",
                "runtime": "python3",
                "entry_command": "python crawler.py",
                "hashslip": {"enabled": True},
                "preflight": {"required": True},
                "safety": {
                    "allowed_domains": domains,
                    "blocked_private_networks": True,
                    "disable_shell": True,
                    "network_egress_policy": "manifest_only",
                    "secrets_access": "none",
                },
            }
        )
    if path == "requirements.txt":
        return ""
    if path == "tests/test_parser.py":
        return (
            "import importlib.util\n"
            "import pathlib\n"
            "import unittest\n\n"
            "ROOT = pathlib.Path(__file__).resolve().parents[1]\n"
            "SPEC = importlib.util.spec_from_file_location('crawler', ROOT / 'crawler.py')\n"
            "crawler = importlib.util.module_from_spec(SPEC)\n"
            "SPEC.loader.exec_module(crawler)\n\n\n"
            "class ParserTest(unittest.TestCase):\n"
            "    def test_parse_fixture_records(self):\n"
            "        html = (ROOT / 'fixtures' / 'sample.html').read_text(encoding='utf-8')\n"
            "        records = crawler.parse_trending(html)\n"
            "        self.assertGreaterEqual(len(records), 2)\n"
            "        first = records[0]\n"
            "        self.assertEqual(first['owner'], 'owner')\n"
            "        self.assertEqual(first['repository_name'], 'repo')\n"
            "        self.assertEqual(first['url'], 'https://github.com/owner/repo')\n"
            "        self.assertEqual(first['language'], 'Python')\n"
            "        self.assertEqual(first['stars_today'], 56)\n"
            "        self.assertEqual(first['total_stars'], 1234)\n"
            "        self.assertEqual(first['forks'], 78)\n\n"
            "    def test_number_parser(self):\n"
            "        self.assertEqual(crawler.number_from_text('1.2k'), 1200)\n"
            "        self.assertEqual(crawler.number_from_text('56 stars today'), 56)\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
    if path == "fixtures/sample.html":
        return (
            "<!doctype html><html><body><main>\n"
            "<article class=\"Box-row\">\n"
            "<h2 class=\"h3 lh-condensed\"><a href=\"/owner/repo\">owner / repo</a></h2>\n"
            "<p class=\"col-9 color-fg-muted my-1 pr-4\">A sample trending repository.</p>\n"
            "<span itemprop=\"programmingLanguage\">Python</span>\n"
            "<a class=\"Link--muted d-inline-block mr-3\" href=\"/owner/repo/stargazers\">1,234</a>\n"
            "<a class=\"Link--muted d-inline-block mr-3\" href=\"/owner/repo/forks\">78</a>\n"
            "<span class=\"d-inline-block float-sm-right\">56 stars today</span>\n"
            "</article>\n"
            "<article class=\"Box-row\">\n"
            "<h2 class=\"h3 lh-condensed\"><a href=\"/acme/tools\">acme / tools</a></h2>\n"
            "<p class=\"col-9 color-fg-muted my-1 pr-4\">Useful developer tools.</p>\n"
            "<span itemprop=\"programmingLanguage\">Go</span>\n"
            "<a class=\"Link--muted d-inline-block mr-3\" href=\"/acme/tools/stargazers\">2,345</a>\n"
            "<a class=\"Link--muted d-inline-block mr-3\" href=\"/acme/tools/forks\">120</a>\n"
            "<span class=\"d-inline-block float-sm-right\">42 stars today</span>\n"
            "</article>\n"
            "</main></body></html>\n"
        )
    if path == "fixtures/sample.csv":
        return "repository_name,owner,description,language,stars_today,total_stars,forks,url\nrepo,owner,A sample trending repository.,Python,56,1234,0,https://github.com/owner/repo\n"
    if path == "schemas/output.schema.json":
        return json_dumps_compact(
            {
                "type": "object",
                "properties": {
                    "repository_name": {"type": "string"},
                    "owner": {"type": "string"},
                    "description": {"type": "string"},
                    "language": {"type": "string"},
                    "stars_today": {"type": "integer"},
                    "total_stars": {"type": "integer"},
                    "forks": {"type": "integer"},
                    "url": {"type": "string"},
                },
            }
        )
    if path == "sources.json":
        return json_dumps_compact([{"name": "GitHub Trending Page", "url": source_url, "allowed_domains": domains}])
    if path == "CHANGELOG.md":
        return "# Changelog\n\n- Initial GitHub Trending crawler artifact.\n"
    if path == "README.md":
        return "# GitHub Trending Crawler\n\nRun `python crawler.py --preflight` or `python crawler.py --dry-run`.\n"
    return None


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
    if not is_crawler_instruction(state.job.instruction):
        return ""
    allowed = instruction_source_domains(state.job.instruction)
    if not allowed:
        return ""
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


def normalize_command(command: str) -> str:
    return " ".join(command.strip().split())


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
        for raw_path in task_contract.must_modify_files
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
    expected = " ".join(command.strip().split())
    seen_edit = False
    for message in reversed(state.messages):
        if message.get("role") != "tool":
            continue
        tool = str(message.get("tool") or "")
        if tool in {"edit_file", "write_file", "replace_in_file", "apply_patch"} and int(message.get("exit_code") or 0) == 0:
            seen_edit = True
            break
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        observed = " ".join(str(metadata.get("command") or "").strip().split())
        if observed == expected and int(message.get("exit_code") or 0) != 0:
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
