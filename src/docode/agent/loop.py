from __future__ import annotations

from typing import Any

from docode.agent.inspector import ProjectInspector
from docode.agent.prompts import DOCODE_SYSTEM_PROMPT
from docode.agent.state import AgentState
from docode.agent.stop_policy import StopPolicy
from docode.agent.verifier import CodingVerifier, VerificationResult
from docode.artifacts.exporter import ArtifactExporter, terminal_artifact_id
from docode.dobox.tools import DoBoxTools
from docode.dobox.types import ToolResult
from docode.llm.runtime import DecisionLLM, LLMUsageMeter
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
        usage_meter: LLMUsageMeter | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.verifier = verifier
        self.repository = repository
        self.exporter = exporter
        self.stop_policy = stop_policy
        self.inspector = inspector or ProjectInspector()
        self.usage_meter = usage_meter

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

            observation = await self.collect_observation(state)
            await self.repository.add_step(job.id, "system", {"type": "observation", "content": observation})
            try:
                decision = await self.llm.decide(
                    system=DOCODE_SYSTEM_PROMPT,
                    messages=state.messages,
                    tools=self.tools.definitions(),
                    context=observation,
                )
            except Exception as exc:
                await self.record_model_failure(state, "llm_decision_failed", str(exc))
                continue
            self.sync_llm_usage(state)
            await self.repository.add_step(job.id, "llm", decision_to_step(decision, self.usage_meter))
            stop = self.stop_policy.evaluate(state)
            if stop.should_stop:
                return await self.fail(job.id, stop.reason or "stopped")

            if decision.type == "tool_call" and decision.tool_name:
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
                final_summary = (decision.summary or "").strip()
                if not final_summary:
                    await self.record_model_failure(
                        state,
                        "final_summary_missing",
                        "final_candidate must include a non-empty summary before verification can complete",
                    )
                    continue
                await self.repository.update_job(job.id, status=JobStatus.VERIFYING)
                verification = await self.verifier.verify(job, self.tools)
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
                state.add_feedback(verification_repair_feedback(verification))
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
            },
        )

    async def collect_observation(self, state: AgentState) -> str:
        status = await self.tools.git_status()
        inspection = state.inspection.summary() if state.inspection else "Project inspection unavailable."
        return (
            f"Instruction: {state.job.instruction}\n"
            f"{inspection}\n"
            f"Git status:\n{status.output}\n"
            f"Recent messages: {len(state.messages)}"
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
    }


def verification_repair_feedback(result: VerificationResult) -> str:
    parts = [result.reason]
    if result.required_fixes:
        parts.append("Required fixes:\n" + "\n".join(f"- {fix}" for fix in result.required_fixes))
    if result.smoke_result is not None and result.smoke_result.exit_code != 0:
        command = result.smoke_result.metadata.get("command") if result.smoke_result.metadata else None
        if command:
            parts.append("Smoke command:\n" + command)
        parts.append("Smoke output:\n" + truncate_text(result.smoke_result.output, 4000))
    return "\n\n".join(part for part in parts if part)


def decision_to_step(decision, usage_meter: LLMUsageMeter | None = None) -> dict[str, object]:
    payload: dict[str, object] = {"type": "llm_decision", "decision_type": decision.type}
    if decision.tool_name:
        payload["tool"] = decision.tool_name
        payload["args"] = sanitize_tool_args(decision.args or {})
    if decision.summary:
        payload["summary"] = truncate_text(decision.summary, 2000)
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
