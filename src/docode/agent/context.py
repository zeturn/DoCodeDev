from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from docode.agent.inspector import ProjectInspection
from docode.agent.stuck import git_status_clean
from docode.agent.task_contract import TaskContract
from docode.dobox.types import ToolResult
from docode.git_changes import changed_paths_from_status, strip_ansi
from docode.storage.models import CodingJob


@dataclass(frozen=True, slots=True)
class ContextPack:
    task_contract: str
    repo_map: str
    working_memory: str
    file_memory: str
    action_summary: str
    latest_evidence: str
    recent_messages: list[dict[str, Any]] = field(default_factory=list)

    def render(self) -> str:
        sections = [
            ("Task Contract", self.task_contract),
            ("Repo Map", self.repo_map),
            ("Working Memory", self.working_memory),
            ("File Memory", self.file_memory),
            ("Action Summary", self.action_summary),
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
        repair_active = repair_mode == "targeted_repair" and active_repair_action is not None
        compact_mode = (workflow_phase in {"EDIT_REQUIRED", "TEST_REQUIRED", "FINAL_READY"}) and not repair_active
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
        file_memory = self.file_memory(inspection, messages, repair_active=repair_active)
        action_summary = self.action_summary(
            job=job,
            messages=messages,
            git_status=git_status,
            task_contract=task_contract,
            repair_mode=repair_mode,
        )
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
            file_memory=clip_text(file_memory, 700 if compact_mode else (3_000 if repair_active else section_bytes)),
            action_summary=clip_text(action_summary, 900 if compact_mode else 1_600),
            latest_evidence=clip_text(latest_evidence, 1_000 if compact_mode else (8_000 if repair_active else section_bytes)),
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
            strict_targets, candidate_targets = target_file_guidance(job.instruction, task_contract.must_modify_files)
            mandatory.extend(f"You must modify {path}" for path in strict_targets)
            if candidate_targets:
                mandatory.append(f"Candidate target files: {', '.join(candidate_targets)}")
                mandatory.append("You must produce a non-empty diff in at least one relevant source file before finalization")
                mandatory.append("Prefer editing the file most directly responsible for the failing behavior")
            else:
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
                "- Use the active repair details as structured guidance for the next fix.\n"
                "- You may inspect, edit, and rerun as needed, but final_candidate still requires passing workflow evidence."
            )
        if active_repair_action:
            phase = targeted_repair_phase or "inspect_allowed"
            targets = ", ".join(str(path) for path in active_repair_action.get("target_files") or []) or "the target file"
            next_action = f"inspect or modify {targets}, then rerun the relevant command when useful"
            parts.append(
                "Active Targeted Repair:\n"
                + json.dumps(active_repair_action, ensure_ascii=False, indent=2)
                + "\n\n"
                + f"Targeted repair phase: {phase}\n"
                + f"Suggested next action: {next_action}"
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

    def file_memory(self, inspection: ProjectInspection | None, messages: list[dict[str, Any]], *, repair_active: bool = False) -> str:
        important = sorted(inspection.important_files) if inspection else []
        touched = sorted(touched_paths(messages))
        parts: list[str] = []
        if important:
            parts.append("Important files from inspection:\n" + "\n".join(f"- {path}" for path in important))
        if touched:
            parts.append("Touched or inspected paths:\n" + "\n".join(f"- {path}" for path in touched))
        if repair_active:
            snippets = repair_file_snippets(messages)
            if snippets:
                parts.append("Repair file snippets already available in context:\n" + snippets)
        return "\n\n".join(parts) if parts else "No file memory yet."

    def action_summary(
        self,
        *,
        job: CodingJob,
        messages: list[dict[str, Any]],
        git_status: ToolResult,
        task_contract: TaskContract | None = None,
        repair_mode: str | None = None,
    ) -> str:
        inspected = sorted(inspected_paths(messages))
        if not inspected or not git_status_clean(git_status.output) or edit_successful(messages):
            return ""

        source_paths = [path for path in inspected if not is_test_path(path)]
        test_paths = [path for path in inspected if is_test_path(path)]
        if not source_paths and not test_paths:
            return ""

        strict_targets, candidate_targets = target_file_guidance(job.instruction, (task_contract.must_modify_files if task_contract else []))
        likely_targets = candidate_targets or strict_targets or source_paths
        lines = ["Already inspected:"]
        lines.extend(f"- {path}" for path in inspected[:8])
        if len(inspected) > 8:
            lines.append(f"- ... {len(inspected) - 8} more")

        lines.extend(
            [
                "",
                "Current state:",
                "- Git diff is empty.",
                "- The task requires a code change before finalization.",
                "- You have already inspected source and/or test files.",
            ]
        )
        if likely_targets:
            lines.append(f"- Candidate target files: {', '.join(likely_targets[:6])}.")

        lines.extend(
            [
                "",
                "Next action:",
                "- Choose the most likely source file and edit it now.",
                "- Do not repeatedly reread the same files unless you need a specific missing line.",
                "- After editing, run the explicit verification command.",
            ]
        )
        if repeated_inspection_without_diff(messages):
            lines.extend(
                [
                    "",
                    "Repeated inspection warning:",
                    "You have repeatedly inspected files, but no source file has changed yet.",
                    "The job cannot finish with a clean git diff.",
                    "Unless you need a specific missing line, stop rereading the same files and edit the most likely target source file now.",
                ]
            )
        if repair_mode == "must_edit":
            lines.extend(
                [
                    "",
                    "Edit pressure:",
                    "repair_mode=must_edit is active; the next useful action should modify a relevant source file.",
                ]
            )
        return "\n".join(lines)

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
            snippets = repair_file_snippets(messages, output_limit=1600)
            active = (
                "\n\nActive Targeted Repair:\n"
                f"- category: {active_repair_action.get('category')}\n"
                f"- reason: {active_repair_action.get('reason')}\n"
                f"- phase: {targeted_repair_phase or 'inspect_allowed'}\n"
                f"- target_files: {target_files}\n"
                f"- rerun: {rerun}\n"
                "- suggested_next_action: inspect, edit, or rerun based on the latest failure output"
                + ("\n\nRepair file snippets:\n" + snippets if snippets else "")
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


def repair_file_snippets(messages: list[dict[str, Any]], *, output_limit: int = 1400) -> str:
    snippets: list[str] = []
    seen: set[tuple[str, str]] = set()
    for message in reversed(messages):
        if message.get("role") != "tool" or int(message.get("exit_code") or 0) != 0:
            continue
        tool = str(message.get("tool") or "")
        if tool not in {"read_file", "read_file_range", "read_symbol"}:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        path = str(metadata.get("path") or metadata.get("resolved_path") or "<unknown>")
        symbol = str(metadata.get("symbol") or "")
        key = (path, symbol or tool)
        if key in seen:
            continue
        seen.add(key)
        header = f"### {tool}: {path}" + (f" symbol={symbol}" if symbol else "")
        snippets.append(header + "\n" + clip_text(str(message.get("output") or ""), output_limit))
        if len(snippets) >= 4:
            break
    return "\n\n".join(reversed(snippets))


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keep = {}
    for key in (
        "path",
        "symbol",
        "start_line",
        "end_line",
        "definition_line",
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


INSPECTION_TOOLS = {"read_file", "read_file_range", "read_symbol", "list_files", "search"}
EDIT_TOOLS = {"write_file", "edit_file", "replace_in_file", "apply_patch"}
EDIT_TARGET_VERBS = {"change", "edit", "fix", "implement", "modify", "refactor", "repair", "update"}


def inspected_paths(messages: list[dict[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for message in messages:
        if message.get("role") != "tool" or int(message.get("exit_code") or 0) != 0:
            continue
        if str(message.get("tool") or "") not in INSPECTION_TOOLS:
            continue
        metadata = message.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("path"), str):
            path = metadata["path"].strip()
            if path and path not in {"."}:
                paths.add(path)
    return paths


def edit_successful(messages: list[dict[str, Any]]) -> bool:
    return any(
        message.get("role") == "tool"
        and str(message.get("tool") or "") in EDIT_TOOLS
        and int(message.get("exit_code") or 0) == 0
        for message in messages
    )


def repeated_inspection_without_diff(messages: list[dict[str, Any]]) -> bool:
    inspection_count = sum(
        1
        for message in messages
        if message.get("role") == "tool"
        and str(message.get("tool") or "") in INSPECTION_TOOLS
        and int(message.get("exit_code") or 0) == 0
    )
    return inspection_count >= 3 and not edit_successful(messages)


def is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    return normalized.startswith("tests/") or "/tests/" in normalized or name.startswith("test_") or name.endswith("_test.py")


def target_file_guidance(instruction: str, targets: list[str]) -> tuple[list[str], list[str]]:
    normalized_targets = [target for target in targets if target]
    if not normalized_targets:
        return [], []
    if target_hint_mentions_any(instruction, normalized_targets):
        return explicit_edit_targets(instruction, normalized_targets), [
            target for target in normalized_targets if target not in explicit_edit_targets(instruction, normalized_targets)
        ]
    strict = explicit_edit_targets(instruction, normalized_targets)
    missing_from_instruction = [target for target in normalized_targets if target not in instruction]
    strict = unique_list([*strict, *missing_from_instruction])
    candidates = [target for target in normalized_targets if target not in strict]
    return strict, candidates


def target_hint_mentions_any(instruction: str, targets: list[str]) -> bool:
    for raw_line in (instruction or "").splitlines():
        line = raw_line.lower()
        if not any(marker in line for marker in ("target file:", "target files:", "edit file:", "edit files:")):
            continue
        if any(target.lower() in line for target in targets):
            return True
    return False


def explicit_edit_targets(instruction: str, targets: list[str]) -> list[str]:
    strict: list[str] = []
    target_lookup = {target.lower(): target for target in targets}
    for raw_line in (instruction or "").splitlines():
        line = raw_line.strip()
        if not line or any(marker in line.lower() for marker in ("target file:", "target files:", "edit file:", "edit files:")):
            continue
        previous_strict = False
        previous_end = 0
        matches = sorted(
            (
                (match.start(), match.end(), target)
                for target in targets
                for match in re.finditer(re.escape(target), line, flags=re.IGNORECASE)
            ),
            key=lambda item: item[0],
        )
        for start, end, target in matches:
            between = line[previous_end:start]
            if edit_verb_immediately_before(between) or (previous_strict and conjunction_only(between)):
                strict.append(target)
                previous_strict = True
            else:
                previous_strict = False
            previous_end = end
    return unique_list(strict)


def edit_verb_immediately_before(text: str) -> bool:
    words = re.findall(r"[a-zA-Z_]+", text.lower())
    if not words:
        return False
    if words[-1] in EDIT_TARGET_VERBS:
        return True
    return len(words) >= 2 and words[-1] in {"file", "files"} and words[-2] in EDIT_TARGET_VERBS


def conjunction_only(text: str) -> bool:
    return bool(re.fullmatch(r"\s*(?:,|and|or|\+)\s*", text, flags=re.IGNORECASE))


def unique_list(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def changed_files_from_status(output: str) -> list[str]:
    return changed_paths_from_status(output)


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

