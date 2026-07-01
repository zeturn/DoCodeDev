from __future__ import annotations

import asyncio
from typing import Any

from docode.agent.context import ContextManager, ContextPack
from docode.agent.inspector import ProjectInspector
from docode.agent.prompts import DOCODE_SYSTEM_PROMPT
from docode.agent.state import AgentState
from docode.agent.stuck import REPAIR_ALLOWED_TOOLS, StuckDetector, git_status_clean
from docode.agent.stop_policy import StopPolicy
from docode.agent.task_contract import TaskContract, task_contract_from_instruction
from docode.agent.verifier import CodingVerifier, VerificationResult, changed_files_from_diff, verification_evidence_from_steps
from docode.agent.workflow import WorkflowPhase, final_candidate_gate, workflow_snapshot
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
                if state.repair_mode == "must_edit" and decision.tool_name not in REPAIR_ALLOWED_TOOLS:
                    await self.record_rejected_decision(
                        state,
                        reason="repair_mode_tool_forbidden",
                        detail=(
                            f"{decision.tool_name} is blocked while repair_mode=must_edit. "
                            "Call read_file, edit_file, write_file, replace_in_file, apply_patch, git_status, or git_diff first."
                        ),
                    )
                    continue
                current_workflow = workflow_snapshot(state, state.latest_git_status.output if state.latest_git_status else "")
                if current_workflow.phase == WorkflowPhase.FINAL_READY and state.repair_mode != "must_edit":
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
                state.iteration += 1
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
        tools = allowed_tool_definitions(self.tools.definitions(), state.repair_mode)
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
        state.add_feedback(f"{reason}: {truncate_text(detail, 1000)}")
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
    if repair_mode != "must_edit":
        return definitions
    return [definition for definition in definitions if getattr(definition, "name", None) in REPAIR_ALLOWED_TOOLS]


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
