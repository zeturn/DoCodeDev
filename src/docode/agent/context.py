from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from docode.agent.inspector import ProjectInspection
from docode.agent.stuck import git_status_clean
from docode.agent.task_contract import TaskContract
from docode.agent.workflow import parse_status_line
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
        ]
        return "\n\n".join(f"## {title}\n{body}".rstrip() for title, body in sections if body)


class ContextManager:
    def __init__(self, *, recent_message_limit: int = 3, section_bytes: int = 2_000) -> None:
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
        task_contract: TaskContract | None = None,
        repair_mode: str | None = None,
        active_repair_action: dict[str, Any] | None = None,
        targeted_repair_phase: str | None = None,
        workflow_phase: str | None = None,
    ) -> ContextPack:
        compact_mode = workflow_phase in {"EDIT_REQUIRED", "TEST_REQUIRED", "FINAL_READY"}
        section_bytes = 1_200 if compact_mode else self.section_bytes
        recent_message_limit = 2 if compact_mode else self.recent_message_limit
        task_contract_text = self.task_contract(
            job,
            task_contract=task_contract,
            repair_mode=repair_mode,
            active_repair_action=active_repair_action,
            targeted_repair_phase=targeted_repair_phase,
        )
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
        latest_evidence = self.latest_evidence(
            git_status,
            messages,
            repair_mode=repair_mode,
            active_repair_action=active_repair_action,
            targeted_repair_phase=targeted_repair_phase,
        )
        recent_messages = [
            compact_message(message, content_limit=500 if compact_mode else 1200)
            for message in messages[-recent_message_limit:]
        ]
        return ContextPack(
            task_contract=clip_text(task_contract_text, 1_600 if compact_mode else section_bytes),
            repo_map=clip_text(repo_map, 700 if compact_mode else section_bytes),
            working_memory=clip_text(working_memory, 900 if compact_mode else section_bytes),
            file_memory=clip_text(file_memory, 700 if compact_mode else section_bytes),
            latest_evidence=clip_text(latest_evidence, 1_000 if compact_mode else section_bytes),
            recent_messages=recent_messages,
        )

    def task_contract(
        self,
        job: CodingJob,
        *,
        task_contract: TaskContract | None = None,
        repair_mode: str | None = None,
        active_repair_action: dict[str, Any] | None = None,
        targeted_repair_phase: str | None = None,
    ) -> str:
        parts = [
            f"Instruction:\n{job.instruction}\n\n"
            "Constraints:\n"
            f"- provider/model: {job.provider}/{job.model}\n"
            f"- quality: {getattr(job, 'quality', 'balanced')}\n"
            f"- max_iterations: {job.max_iterations}\n"
            f"- max_tool_calls: {job.max_tool_calls}\n"
            f"- artifact_mode: {job.artifact_mode}\n"
            f"- sandbox_network_mode: {job.sandbox_network_mode}"
        ]
        if task_contract is not None:
            mandatory: list[str] = []
            mandatory.extend(f"You must modify {path}" for path in task_contract.must_modify_files)
            mandatory.append("You must produce non-empty git diff before final_candidate")
            mandatory.extend(f"You must run suggested command: {command}" for command in task_contract.must_run_commands)
            mandatory.extend(task_contract.forbidden_finish_conditions)
            if is_crawler_instruction(job.instruction):
                mandatory.extend(crawler_contract_requirements())
            parts.append("Mandatory:\n" + "\n".join(f"- {item}" for item in mandatory))
        source_urls = instruction_source_urls(job.instruction)
        if source_urls:
            domains = [urlparse(url).netloc for url in source_urls if urlparse(url).netloc]
            parts.append(
                "Source Guidance:\n"
                + "\n".join(f"- preferred_source_url: {url}" for url in source_urls[:5])
                + ("\n" + "\n".join(f"- preferred_source_domain: {domain}" for domain in domains[:5]) if domains else "")
                + "\n- First inspect these sources with fetch_url before broadening to web_search."
                + "\n- If web_search becomes necessary, keep queries anchored to these URLs/domains and the requested target."
            )
        if repair_mode:
            parts.append(
                "Repair Mode:\n"
                "- The next action must be read_file, edit_file, write_file, replace_in_file, apply_patch, git_status, or git_diff.\n"
                "- final_candidate and run_command are blocked until git_status shows a modified file."
            )
        if active_repair_action:
            phase = targeted_repair_phase or "inspect_allowed"
            targets = ", ".join(str(path) for path in active_repair_action.get("target_files") or []) or "the target file"
            next_action = f"modify {targets} now" if phase == "edit_forced" else f"inspect {targets} briefly, then modify it"
            parts.append(
                "Active Targeted Repair:\n"
                + json.dumps(active_repair_action, ensure_ascii=False, indent=2)
                + "\n\n"
                + f"Targeted repair phase: {phase}\n"
                + f"Next required action: {next_action}"
            )
        return "\n\n".join(parts)

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

    def latest_evidence(
        self,
        git_status: ToolResult,
        messages: list[dict[str, Any]],
        *,
        repair_mode: str | None = None,
        active_repair_action: dict[str, Any] | None = None,
        targeted_repair_phase: str | None = None,
    ) -> str:
        latest_tools = [message for message in messages if message.get("role") == "tool"][-5:]
        tool_summaries = "\n".join(tool_evidence(message) for message in latest_tools)
        clean = git_status_clean(git_status.output)
        changed = changed_files_from_status(git_status.output)
        final_allowed = "no" if clean or repair_mode == "must_edit" else "yes"
        required_next = next_missing_command(messages)
        active = ""
        if active_repair_action:
            target_files = ", ".join(str(path) for path in active_repair_action.get("target_files") or []) or "<none>"
            rerun = ", ".join(str(command) for command in active_repair_action.get("rerun_commands") or []) or "<none>"
            active = (
                "\n\nActive Targeted Repair:\n"
                f"- category: {active_repair_action.get('category')}\n"
                f"- reason: {active_repair_action.get('reason')}\n"
                f"- phase: {targeted_repair_phase or 'inspect_allowed'}\n"
                f"- target_files: {target_files}\n"
                f"- rerun: {rerun}\n"
                f"- next_required_action: {'modify ' + target_files + ' now' if targeted_repair_phase == 'edit_forced' else 'inspect briefly, then modify target file'}"
            )
        return (
            f"Git status:\n{git_status.output or '<clean>'}\n\n"
            "Current Git Diff State:\n"
            f"- git_status_clean: {str(clean).lower()}\n"
            f"- changed_files: {json.dumps(changed, ensure_ascii=False)}\n"
            f"- final_candidate_allowed: {final_allowed}{final_candidate_reason(clean, repair_mode)}\n"
            f"- required_next_command: {required_next or '<none>'}\n\n"
            f"Latest tool evidence:\n{tool_summaries or '- No tool calls yet.'}"
            f"{active}"
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


def compact_message(message: dict[str, Any], *, content_limit: int = 1200) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("role", "kind", "tool", "exit_code", "truncated"):
        if key in message:
            compact[key] = message[key]
    if "content" in message:
        compact["content"] = clip_text(str(message["content"]), content_limit)
    if "output" in message:
        compact["output"] = clip_text(str(message["output"]), content_limit)
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        compact["metadata"] = compact_metadata(metadata)
    return compact


def instruction_source_urls(instruction: str) -> list[str]:
    urls: list[str] = []
    for match in re.findall(r"https?://[^\s'\"`)>]+", instruction or "", flags=re.IGNORECASE):
        cleaned = match.rstrip(".,;:")
        if cleaned and cleaned not in urls:
            urls.append(cleaned)
    return urls


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


def next_missing_command(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        content = message.get("content")
        if isinstance(content, str):
            for line in content.splitlines():
                if line.startswith("Next required command:"):
                    return line.split(":", 1)[1].strip()
        metadata = message.get("metadata")
        if isinstance(metadata, dict):
            commands = metadata.get("missing_commands")
            if isinstance(commands, list) and commands:
                return str(commands[0])
    return ""


def touched_paths(messages: list[dict[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for message in messages:
        metadata = message.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("path"), str):
            paths.add(metadata["path"])
    return paths


def changed_files_from_status(output: str) -> list[str]:
    files: list[str] = []
    for line in output.splitlines():
        _, path = parse_status_line(line)
        if not path:
            continue
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[-1].strip()
        normalized = path.replace("\\", "/")
        if normalized in {".docode_probe", ".docode_probe_api"} or normalized.startswith(".docode_probe") or normalized.startswith(".git/"):
            continue
        if path and path not in files:
            files.append(path)
    return files


def final_candidate_reason(clean: bool, repair_mode: str | None) -> str:
    if clean:
        return " because no file changes exist"
    if repair_mode == "must_edit":
        return " because repair_mode requires an edit confirmation first"
    return " after tests pass"


def is_crawler_instruction(instruction: str) -> bool:
    lowered = (instruction or "").lower()
    return any(keyword in lowered for keyword in ("crawler", "scraper", "scrape", "爬虫", "抓取", "采集", "数据源"))


def crawler_contract_requirements() -> list[str]:
    return [
        "Crawler dependency policy: prefer Python standard library; do not use undeclared third-party packages",
        "If a third-party Python package is required, declare it in requirements.txt or pyproject.toml and verify imports in a venv",
        "Do not retry system pip install after externally-managed-environment failures",
        "Crawler dry-run must write the requested output artifact and final verification must prove it parses",
        "Prefer an offline fixture mode so parser behavior is reproducible without live network access",
    ]


def clip_text(text: str, limit: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="replace") + "\n<truncated>"


def strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", value)
