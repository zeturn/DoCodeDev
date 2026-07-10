from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

from docode.agent.task_contract import is_crawler_instruction, text_outside_verification_blocks


SOURCE_WORDS = ("source", "feed", "endpoint", "input", "url", "源", "数据源", "接口", "输入")
CONTROL_ENDPOINTS = {"/__reset", "/__metrics"}
EDIT_TOOLS = {"write_file", "edit_file", "replace_in_file", "apply_patch"}


@dataclass(frozen=True, slots=True)
class SourceInspectionEvidence:
    requested_url: str
    final_url: str
    status_code: int | None
    execution_scope: str
    mode: str
    body: str
    before_first_edit: bool
    successful: bool
    message_index: int
    controller_owned: bool = False
    error: str = ""

    def to_dict(self, *, include_body: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if not include_body:
            payload.pop("body", None)
        return payload


def crawler_source_inspection_required(instruction: str) -> bool:
    return is_crawler_instruction(instruction) and bool(instruction_source_urls(instruction))


def instruction_source_urls(instruction: str) -> list[str]:
    """Return likely source URLs, preferring prose before verification commands."""

    main_text = text_outside_verification_blocks(instruction)
    main_candidates = extracted_urls(main_text)
    all_candidates = extracted_urls(instruction)
    ranked_main = rank_source_urls(main_text, main_candidates)
    fallback = rank_source_urls(instruction, [url for url in all_candidates if url not in ranked_main])
    return [*ranked_main, *fallback]


def extracted_urls(text: str) -> list[str]:
    urls: list[str] = []
    for match in re.finditer(r"https?://[^\s'\"`)>]+", text or "", flags=re.IGNORECASE):
        cleaned = match.group(0).rstrip(".,;:")
        if not cleaned or source_control_url(cleaned) or cleaned in urls:
            continue
        urls.append(cleaned)
    return urls


def rank_source_urls(text: str, urls: list[str]) -> list[str]:
    positions = {url: text.find(url) for url in urls}

    def score(url: str) -> tuple[int, int]:
        position = positions[url]
        nearby = text[max(0, position - 100) : position + len(url) + 40].lower()
        explicit = any(word in nearby for word in SOURCE_WORDS)
        return (0 if explicit else 1, position)

    return sorted(urls, key=score)


def source_control_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return path in CONTROL_ENDPOINTS


def source_inspection_evidence(
    messages: list[dict[str, Any]],
    instruction: str,
) -> list[SourceInspectionEvidence]:
    candidates = set(instruction_source_urls(instruction))
    first_edit = next(
        (
            index
            for index, message in enumerate(messages)
            if message.get("role") == "tool"
            and message.get("tool") in EDIT_TOOLS
            and int(message.get("exit_code") or 0) == 0
        ),
        len(messages),
    )
    evidence: list[SourceInspectionEvidence] = []
    for index, message in enumerate(messages):
        if message.get("role") != "tool" or message.get("tool") != "inspect_source":
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        payload = source_result_payload(message)
        requested_url = str(payload.get("requested_url") or metadata.get("requested_url") or metadata.get("url") or "")
        final_url = str(payload.get("final_url") or metadata.get("final_url") or requested_url)
        status_code = optional_int(payload.get("status_code", metadata.get("status_code")))
        scope = str(payload.get("execution_scope") or metadata.get("execution_scope") or "")
        mode = str(payload.get("mode") or metadata.get("mode") or "raw")
        body = str(payload.get("body") or "")
        error = str(payload.get("error") or metadata.get("error") or "")
        source_identity_matches = not candidates or requested_url in candidates
        successful = (
            int(message.get("exit_code") or 0) == 0
            and scope == "sandbox"
            and status_code is not None
            and 200 <= status_code < 400
            and source_identity_matches
            and (bool(body) or mode == "headers")
        )
        evidence.append(
            SourceInspectionEvidence(
                requested_url=requested_url,
                final_url=final_url,
                status_code=status_code,
                execution_scope=scope,
                mode=mode,
                body=body,
                before_first_edit=index < first_edit,
                successful=successful,
                message_index=index,
                controller_owned=bool(metadata.get("controller_owned")),
                error=error,
            )
        )
    return evidence


def successful_source_inspection(messages: list[dict[str, Any]], instruction: str) -> SourceInspectionEvidence | None:
    return next(
        (item for item in source_inspection_evidence(messages, instruction) if item.successful and item.before_first_edit),
        None,
    )


def attempted_source_urls(messages: list[dict[str, Any]]) -> set[str]:
    attempted: set[str] = set()
    for message in messages:
        if message.get("role") != "tool" or message.get("tool") != "inspect_source":
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        payload = source_result_payload(message)
        url = str(payload.get("requested_url") or metadata.get("requested_url") or metadata.get("url") or "")
        if url:
            attempted.add(url)
    return attempted


def source_result_payload(message: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(str(message.get("output") or ""))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
