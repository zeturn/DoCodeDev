from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from docode.agent.inspector import ProjectInspection
from docode.dobox.types import ToolResult
from docode.storage.models import CodingJob


@dataclass(frozen=True, slots=True)
class ContextPack:
    task_contract: str
    repo_map: str
    working_memory: str
    file_memory: str
    latest_evidence: str
    recent_messages: list[dict[str, Any]] = field(default_factory=list)

    def render(self) -> str:
        sections = [
            ("Task Contract", self.task_contract),
            ("Repo Map", self.repo_map),
            ("Working Memory", self.working_memory),
            ("File Memory", self.file_memory),
            ("Latest Evidence", self.latest_evidence),
            ("Recent Messages", json.dumps(self.recent_messages, ensure_ascii=False, indent=2)),
        ]
        return "\n\n".join(f"## {title}\n{body}".rstrip() for title, body in sections if body)


class ContextManager:
    def __init__(self, *, recent_message_limit: int = 8, section_bytes: int = 12_000) -> None:
        self.recent_message_limit = recent_message_limit
        self.section_bytes = section_bytes

    def build_pack(
        self,
        *,
        job: CodingJob,
        inspection: ProjectInspection | None,
        messages: list[dict[str, Any]],
        git_status: ToolResult,
        iteration: int,
        tool_calls_count: int,
        llm_tokens_used: int,
        llm_cost_used: float,
    ) -> ContextPack:
        task_contract = self.task_contract(job)
        repo_map = self.repo_map(inspection)
        working_memory = self.working_memory(
            inspection=inspection,
            messages=messages,
            iteration=iteration,
            tool_calls_count=tool_calls_count,
            llm_tokens_used=llm_tokens_used,
            llm_cost_used=llm_cost_used,
        )
        file_memory = self.file_memory(inspection, messages)
        latest_evidence = self.latest_evidence(git_status, messages)
        recent_messages = [compact_message(message) for message in messages[-self.recent_message_limit :]]
        return ContextPack(
            task_contract=clip_text(task_contract, self.section_bytes),
            repo_map=clip_text(repo_map, self.section_bytes),
            working_memory=clip_text(working_memory, self.section_bytes),
            file_memory=clip_text(file_memory, self.section_bytes),
            latest_evidence=clip_text(latest_evidence, self.section_bytes),
            recent_messages=recent_messages,
        )

    def task_contract(self, job: CodingJob) -> str:
        return (
            f"Instruction:\n{job.instruction}\n\n"
            "Constraints:\n"
            f"- provider/model: {job.provider}/{job.model}\n"
            f"- quality: {getattr(job, 'quality', 'balanced')}\n"
            f"- max_iterations: {job.max_iterations}\n"
            f"- max_tool_calls: {job.max_tool_calls}\n"
            f"- artifact_mode: {job.artifact_mode}\n"
            f"- sandbox_network_mode: {job.sandbox_network_mode}"
        )

    def repo_map(self, inspection: ProjectInspection | None) -> str:
        if inspection is None:
            return "Project inspection unavailable."
        return inspection.summary()

    def working_memory(
        self,
        *,
        inspection: ProjectInspection | None,
        messages: list[dict[str, Any]],
        iteration: int,
        tool_calls_count: int,
        llm_tokens_used: int,
        llm_cost_used: float,
    ) -> str:
        completed, failed = summarize_tool_history(messages)
        feedback = summarize_feedback(messages)
        plan = "\n".join(f"- {item}" for item in (inspection.plan if inspection else [])) or "- No plan detected yet."
        acceptance = "\n".join(f"- {item}" for item in (inspection.acceptance_criteria if inspection else [])) or "- No acceptance criteria detected yet."
        return (
            f"Iteration: {iteration}\n"
            f"Tool calls: {tool_calls_count}\n"
            f"LLM usage: {llm_tokens_used} tokens, ${llm_cost_used:.4f}\n\n"
            f"Current Plan:\n{plan}\n\n"
            f"Acceptance Criteria:\n{acceptance}\n\n"
            f"Completed Steps:\n{completed or '- None yet.'}\n\n"
            f"Failed Steps / Repair Attempts:\n{failed or '- None yet.'}\n\n"
            f"Verifier / Model Feedback:\n{feedback or '- None yet.'}"
        )

    def file_memory(self, inspection: ProjectInspection | None, messages: list[dict[str, Any]]) -> str:
        important = sorted(inspection.important_files) if inspection else []
        touched = sorted(touched_paths(messages))
        parts: list[str] = []
        if important:
            parts.append("Important files from inspection:\n" + "\n".join(f"- {path}" for path in important))
        if touched:
            parts.append("Touched or inspected paths:\n" + "\n".join(f"- {path}" for path in touched))
        return "\n\n".join(parts) if parts else "No file memory yet."

    def latest_evidence(self, git_status: ToolResult, messages: list[dict[str, Any]]) -> str:
        latest_tools = [message for message in messages if message.get("role") == "tool"][-5:]
        tool_summaries = "\n".join(tool_evidence(message) for message in latest_tools)
        return (
            f"Git status:\n{git_status.output or '<clean>'}\n\n"
            f"Latest tool evidence:\n{tool_summaries or '- No tool calls yet.'}"
        )


def summarize_tool_history(messages: list[dict[str, Any]]) -> tuple[str, str]:
    completed: list[str] = []
    failed: list[str] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        if int(message.get("exit_code") or 0) == 0:
            line = tool_evidence(message, output_limit=180)
            completed.append("- " + line)
        else:
            line = tool_evidence(message, output_limit=1200)
            failed.append("- " + line)
    return "\n".join(completed[-6:]), "\n".join(failed[-12:])


def summarize_feedback(messages: list[dict[str, Any]]) -> str:
    feedback = [
        "- " + clip_text(str(message.get("content") or ""), 1200)
        for message in messages
        if message.get("kind") == "feedback"
    ]
    return "\n".join(feedback[-8:])


def compact_message(message: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("role", "kind", "tool", "exit_code", "truncated"):
        if key in message:
            compact[key] = message[key]
    if "content" in message:
        compact["content"] = clip_text(str(message["content"]), 1200)
    if "output" in message:
        compact["output"] = clip_text(str(message["output"]), 1200)
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        compact["metadata"] = compact_metadata(metadata)
    return compact


def tool_evidence(message: dict[str, Any], *, output_limit: int = 900) -> str:
    tool = str(message.get("tool") or "tool")
    exit_code = int(message.get("exit_code") or 0)
    metadata = compact_metadata(message.get("metadata") if isinstance(message.get("metadata"), dict) else {})
    if message.get("truncated") and output_limit <= 200:
        original = metadata.get("original_output_bytes")
        output = f"<truncated output: {original} bytes>" if original else "<truncated output>"
    else:
        output = clip_text(str(message.get("output") or ""), output_limit)
    suffix = f" metadata={json.dumps(metadata, ensure_ascii=False)}" if metadata else ""
    truncated = " truncated" if message.get("truncated") else ""
    return f"{tool} exit={exit_code}{truncated}: {output or '<no output>'}{suffix}"


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keep = {}
    for key in (
        "path",
        "command",
        "detected",
        "rejected",
        "reason",
        "url",
        "status_code",
        "content_type",
        "original_bytes",
        "returned_bytes",
        "prompt_output_truncated",
        "original_output_lines",
        "original_output_bytes",
    ):
        if key in metadata:
            keep[key] = metadata[key]
    return keep


def touched_paths(messages: list[dict[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for message in messages:
        metadata = message.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("path"), str):
            paths.add(metadata["path"])
    return paths


def clip_text(text: str, limit: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="replace") + "\n<truncated>"
