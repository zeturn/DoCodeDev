from __future__ import annotations

import asyncio
import posixpath
from typing import Any

from docode.agent.context import ContextManager, ContextPack
from docode.agent.inspector import ProjectInspector
from docode.agent.prompts import DOCODE_SYSTEM_PROMPT
from docode.agent.quality_gate import QualityGate, QualityGateResult
from docode.agent.repair_planner import RepairAction, format_repair_action, plan_repair_from_tool_result
from docode.agent.reviewer import CodeReviewer, ReviewResult
from docode.agent.state import AgentState
from docode.agent.stuck import REPAIR_ALLOWED_TOOLS, StuckDetector, git_status_clean
from docode.agent.stop_policy import StopPolicy
from docode.agent.task_contract import TaskContract, task_contract_from_instruction
from docode.agent.verifier import CodingVerifier, VerificationResult, changed_files_from_diff, verification_evidence_from_steps
from docode.agent.workflow import WorkflowPhase, changed_paths_from_status, final_candidate_gate, successful_edit_tool_called, workflow_snapshot
from docode.artifacts.exporter import ArtifactExporter, terminal_artifact_id
from docode.dobox.tools import DoBoxTools
from docode.dobox.types import ToolResult
from docode.llm.runtime import DecisionLLM, LLMUsageMeter, ProviderUnavailableError
from docode.storage.models import CodingJob, JobStatus
from docode.storage.repository import JobRepository


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
                repair_tool_block = repair_mode_tool_block(state, decision.tool_name)
                if repair_tool_block:
                    await self.record_rejected_decision(
                        state,
                        reason=f"{state.repair_mode}_tool_forbidden" if state.repair_mode else "repair_mode_tool_forbidden",
                        detail=repair_tool_block,
                    )
                    continue
                targeted_action_block = targeted_repair_action_block(state, decision.tool_name, decision.args or {})
                if targeted_action_block:
                    await self.record_rejected_decision(
                        state,
                        reason="targeted_repair_wrong_action",
                        detail=targeted_action_block,
                    )
                    continue
                targeted_exploration_block = targeted_repair_exploration_block(state, decision.tool_name)
                if targeted_exploration_block:
                    await self.record_rejected_decision(
                        state,
                        reason="targeted_repair_exploration_limit",
                        detail=targeted_exploration_block,
                    )
                    continue
                edit_command_block = edit_required_tool_block(state, current_workflow, decision.tool_name, decision.args or {})
                if edit_command_block:
                    await self.record_rejected_decision(
                        state,
                        reason="edit_required_tool_forbidden",
                        detail=edit_command_block,
                        workflow_state=current_workflow.to_dict(),
                    )
                    continue
                test_command_block = required_test_tool_block(state, current_workflow, decision.tool_name, decision.args or {})
                if test_command_block:
                    await self.record_rejected_decision(
                        state,
                        reason="test_required_tool_forbidden",
                        detail=test_command_block,
                        workflow_state=current_workflow.to_dict(),
                    )
                    continue
                if current_workflow.phase == WorkflowPhase.FINAL_READY and state.repair_mode not in {"must_edit", "quality_repair"}:
                    await self.record_rejected_decision(
                        state,
                        reason="final_ready_tool_forbidden",
                        detail=(
                            f"{decision.tool_name} is blocked because the workflow is already FINAL_READY. "
                            "Submit final_candidate now so the verifier can review the completed diff."
                        ),
                        workflow_state=current_workflow.to_dict(),
                    )
                    continue
                cancelled = await self.cancelled_job(job.id)
                if cancelled is not None:
                    return cancelled
                await self.repository.add_step(
                    job.id,
                    "tool",
                    {
                        "type": "tool_call",
                        "tool": decision.tool_name,
                        "args": sanitize_tool_args(decision.args or {}),
                    },
                )
                try:
                    result = await self.tools.call(decision.tool_name, decision.args or {})
                except Exception as exc:
                    result = tool_exception_result(decision.tool_name, exc)
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
                    continue
                if state.active_repair_action and not targeted_repair_rerun_satisfied(state):
                    await self.record_rejected_decision(
                        state,
                        reason="targeted_repair_rerun_missing",
                        detail=targeted_rerun_missing_detail(state),
                    )
                    continue
                gate = final_candidate_gate(state, status.output)
                if not gate.allowed:
                    if gate.repair_mode:
                        state.repair_mode = gate.repair_mode
                    await self.record_rejected_decision(
                        state,
                        reason=gate.reason,
                        detail=gate.detail,
                        workflow_state=gate.snapshot.to_dict(),
                    )
                    continue
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
                    continue
                state.quality_gate_passed = True
                if state.repair_mode == "quality_repair":
                    state.repair_mode = None
                review = await self.run_independent_review(state, quality, final_summary)
                if review is not None:
                    self.sync_llm_usage(state)
                    await self.repository.add_step(job.id, "system", review.to_dict())
                    if not review.passed:
                        state.repair_mode = "quality_repair"
                        state.add_feedback(review_repair_feedback(review))
                        state.iteration += 1
                        continue
                await self.repository.update_job(job.id, status=JobStatus.VERIFYING)
                evidence = verification_evidence_from_steps(await self.repository.list_steps(job.id)).with_no_test_reason(decision.no_test_reason)
                verification = await self.verifier.verify(job, self.tools, evidence=evidence)
                self.sync_llm_usage(state)
                await self.repository.add_step(job.id, "verifier", verification_to_dict(verification))
                stop = self.stop_policy.evaluate(state)
                if stop.should_stop:
                    return await self.fail(job.id, stop.reason or "stopped")
                if verification.passed:
                    artifacts = await self.exporter.export_success(job, verification, final_summary)
                    artifact_id = terminal_artifact_id(artifacts)
                    return await self.repository.update_job(
                        job.id,
                        status=JobStatus.SUCCEEDED,
                        result_summary=final_summary,
                        artifact_id=artifact_id,
                    )
                if requires_non_empty_diff_repair(verification):
                    state.repair_mode = "must_edit"
                elif verification.required_fixes:
                    state.repair_mode = "must_edit"
                state.add_feedback(verification_repair_feedback(verification, state.task_contract))
                state.iteration += 1
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

    async def activate_targeted_repair(self, state: AgentState, action: RepairAction, result: ToolResult | None = None) -> None:
        command = str((result.metadata or {}).get("command") or "") if result is not None else ""
        if command:
            state.last_failed_command = command
        count = state.failure_signatures.get(action.signature, 0) + 1
        state.failure_signatures[action.signature] = count
        state.repair_action_attempts = count
        state.active_repair_action = action.to_dict()
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
                "repair_action": action.to_dict(),
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
        state.add_observation(f"Task: {state.job.instruction}")
        inspection = await self.inspector.inspect(state.job.instruction, self.tools)
        state.inspection = inspection
        state.task_contract = task_contract_from_instruction(state.job.instruction)
        state.add_observation("Project inspection:\n" + inspection.summary())
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
                return await self.llm.decide(
                    system=DOCODE_SYSTEM_PROMPT,
                    messages=state.messages,
                    tools=tools,
                    context=observation,
                )
            except Exception as exc:
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
        if isinstance(missing_commands, list) and missing_commands:
            next_command = f"\nNext required command: {missing_commands[0]}"
        state.add_feedback(f"{reason}: {truncate_text(detail, 1000)}{next_command}")
        state.iteration += 1

    def sync_llm_usage(self, state: AgentState) -> None:
        if self.usage_meter is not None:
            state.llm_tokens_used = self.usage_meter.total_tokens
            state.llm_cost_used = self.usage_meter.cost


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
    allowed = action.get("allowed_tools")
    if isinstance(allowed, list) and allowed:
        base = {str(tool) for tool in allowed}
    else:
        base = allowed_tools_for_repair_mode_name("targeted_repair")
    base -= {"web_search", "fetch_url", "preview", "logs"}
    if state.targeted_repair_phase == "edit_forced":
        base -= {"read_file", "search", "list_files"}
        base |= {"edit_file", "write_file", "replace_in_file", "apply_patch", "run_command", "git_status", "git_diff"}
    return base


def allowed_tools_for_repair_mode_name(repair_mode: str | None) -> set[str]:
    if repair_mode == "must_edit":
        return set(REPAIR_ALLOWED_TOOLS)
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
            "Call read_file, edit_file, write_file, replace_in_file, apply_patch, git_status, or git_diff first."
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
    if targeted_repair_modified_target(state):
        return ""
    targets = targeted_repair_targets(state)
    if not targets:
        return ""
    target_text = ", ".join(sorted(targets))
    if tool_name in {"git_status", "git_diff"}:
        return ""
    if tool_name == "run_command" and state.targeted_repair_phase == "edit_forced":
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
        if not line.startswith(("--- ", "+++ ", "diff --git ")):
            continue
        for target in targets:
            if f" a/{target}" in line or f" b/{target}" in line or line.endswith(target):
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
            if observed == expected or expected in observed:
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
    if state.repair_mode == "must_edit":
        return ""
    if tool_name in {"write_file", "edit_file", "replace_in_file", "apply_patch"}:
        return ""
    if successful_edit_tool_called(state):
        return ""
    if exploratory_tool_calls(state) < 8:
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


def required_test_tool_block(state: AgentState, workflow: Any, tool_name: str, args: dict[str, object]) -> str:
    if workflow.phase != WorkflowPhase.TEST_REQUIRED:
        return ""
    missing = getattr(workflow, "missing_commands", None) or []
    if not missing:
        return ""
    missing_targets = missing_must_modify_targets(state)
    if target_edit_allowed_while_tests_missing(state, tool_name, args):
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
    if observed != expected:
        return f"Wrong command for TEST_REQUIRED. Run this exact command first: {next_command}"
    return ""


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
    ]


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
