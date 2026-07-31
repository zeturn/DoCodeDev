from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from docode.agent.source_inspection import canonical_source_url, instruction_source_urls, source_origin, successful_source_inspection
from docode.agent.task_contract import is_crawler_instruction
from dataclasses import dataclass
from docode.agent.profiles import select_task_profile

SOURCE_TO_EDIT_TOOLS = {"read_file", "read_file_range", "read_symbol", "list_files", "write_file", "edit_file", "replace_in_file", "apply_patch", "git_status", "git_diff"}
EDIT_TOOLS = {"write_file", "edit_file", "replace_in_file", "apply_patch"}
MAX_UNIQUE_SOURCE_INSPECTIONS = 4
MAX_OPTIONAL_PRE_EDIT_SOURCE_URLS = 2
MAX_PRE_EDIT_MODEL_DECISIONS = 2


@dataclass(frozen=True, slots=True)
class SourcePolicyDecision:
    allowed: bool
    reason: str = ""


class SourcePolicy:
    def evaluate(self, state: Any, tool_name: str, args: dict[str, object]) -> SourcePolicyDecision:
        profile = getattr(state, "profile", None) or select_task_profile(state.job.instruction)
        if profile.name != "crawler" or tool_name not in {"inspect_source", "fetch_url"}:
            return SourcePolicyDecision(True)
        raw_url = str(args.get("url") or "").strip()
        parsed = urlparse(raw_url)
        if parsed.scheme not in set(profile.allowed_source_schemes) or not parsed.hostname:
            return SourcePolicyDecision(False, f"{tool_name} requires an absolute URL using an allowed source scheme.")
        if tool_name == "inspect_source":
            identity = canonical_source_url(raw_url)
            if identity and identity in successful_source_identities(state):
                return SourcePolicyDecision(False, "duplicate_source_inspection: This source is already available in Source Inspection memory. Do not inspect it again or switch modes to refetch it. Read or edit the implementation now.")
            if unique_network_source_count(state) >= profile.budget_policy.maximum_source_requests:
                return SourcePolicyDecision(False, "source_request_budget_exhausted: edit or verify using retained source evidence.")
        if source_origin(raw_url) not in allowed_source_origins(state):
            origins = ", ".join(f"{scheme}://{host}" + ("" if port == (443 if scheme == "https" else 80) else f":{port}") for scheme, host, port in sorted(allowed_source_origins(state)))
            return SourcePolicyDecision(False, f"{tool_name} is restricted to the same source origin(s): {origins or 'the task source origin'}.")
        return SourcePolicyDecision(True)


def successful_source_identities(state: Any) -> list[str]:
    identities: list[str] = []
    for message in state.messages:
        if message.get("role") != "tool" or message.get("tool") != "inspect_source" or int(message.get("exit_code") or 0) != 0:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        identity = canonical_source_url(str(metadata.get("requested_url") or metadata.get("url") or ""))
        if identity and identity not in identities:
            identities.append(identity)
    return identities


def unique_network_source_count(state: Any) -> int:
    return len({
        canonical_source_url(str(metadata.get("requested_url") or metadata.get("url") or ""))
        for message in state.messages
        if message.get("role") == "tool" and message.get("tool") == "inspect_source" and int(message.get("exit_code") or 0) == 0
        and isinstance((metadata := message.get("metadata")), dict) and not metadata.get("cached")
        and canonical_source_url(str(metadata.get("requested_url") or metadata.get("url") or ""))
    })


def source_progress_forced(state: Any) -> bool:
    profile = getattr(state, "profile", None) or select_task_profile(state.job.instruction)
    if profile.name != "crawler" or successful_source_inspection(state.messages, state.job.instruction) is None:
        return False
    latest_edit = max((i for i, m in enumerate(state.messages) if m.get("role") == "tool" and m.get("tool") in EDIT_TOOLS and int(m.get("exit_code") or 0) == 0), default=-1)
    if any(m.get("kind") == "feedback" and "duplicate_source_inspection" in str(m.get("content") or "") for m in state.messages[latest_edit + 1:]):
        return True
    if latest_edit >= 0:
        return False
    identities = successful_source_identities(state)
    first_source = next((i for i, m in enumerate(state.messages) if m.get("role") == "tool" and m.get("tool") == "inspect_source" and int(m.get("exit_code") or 0) == 0), len(state.messages))
    decisions = sum(1 for m in state.messages[first_source + 1:] if m.get("role") == "tool" and not (m.get("metadata") or {}).get("controller_owned"))
    return max(0, len(identities) - 1) >= min(MAX_OPTIONAL_PRE_EDIT_SOURCE_URLS, profile.budget_policy.maximum_source_requests) or decisions >= profile.budget_policy.maximum_pre_edit_decisions


def continuation_allowed(state: Any, phase: Any | None = None) -> bool:
    profile = getattr(state, "profile", None) or select_task_profile(state.job.instruction)
    if profile.name != "crawler" or successful_source_inspection(state.messages, state.job.instruction) is None:
        return False
    if source_progress_forced(state) or unique_network_source_count(state) >= profile.budget_policy.maximum_source_requests:
        return False
    if phase is not None and str(getattr(phase, "value", phase)) == "FINAL_READY":
        return False
    return True


def allowed_source_origins(state: Any) -> set[tuple[str, str, int]]:
    origins = {origin for url in instruction_source_urls(state.job.instruction) if (origin := source_origin(url)) is not None}
    for message in state.messages:
        if message.get("role") != "tool" or message.get("tool") != "inspect_source" or int(message.get("exit_code") or 0) != 0:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        for key in ("requested_url", "final_url", "url"):
            if (origin := source_origin(str(metadata.get(key) or ""))) is not None:
                origins.add(origin)
    return origins


def source_tool_block(state: Any, tool_name: str, args: dict[str, object]) -> str:
    decision = SourcePolicy().evaluate(state, tool_name, args)
    return "" if decision.allowed else decision.reason
