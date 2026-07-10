from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any
from urllib.parse import urlparse


MAX_UNIQUE_SOURCE_INSPECTIONS = 4


def apply_runtime_hotfix() -> None:
    """Install a narrow source/verification pipeline hotfix.

    This module is intentionally self-contained so it can be removed after the
    equivalent behavior is folded into the normal runtime modules.
    """

    import docode.agent.loop as loop
    import docode.agent.prompts as prompts
    import docode.agent.workflow as workflow
    from docode.dobox.tools import DoBoxTools
    from docode.dobox.types import ToolResult

    if getattr(loop, "_source_pipeline_hotfix_v1_applied", False):
        return

    _patch_prompt(loop, prompts)
    _patch_inspect_source_cache(DoBoxTools, ToolResult)
    _patch_source_tool_visibility(loop)
    _patch_source_domain_policy(loop)
    _patch_controller_verification_order(loop, workflow)
    setattr(loop, "_source_pipeline_hotfix_v1_applied", True)


def _patch_prompt(loop: Any, prompts: Any) -> None:
    original = str(prompts.DOCODE_SYSTEM_PROMPT)
    lines: list[str] = []
    inserted = False
    for line in original.splitlines():
        if line.startswith("For crawler parser tasks, preserve the public parser API"):
            if not inserted:
                lines.append(
                    "For crawler and data-source tasks, derive selectors, fields, pagination, namespaces, and URL handling from the actual inspect_source evidence. Do not assume a familiar site schema, fixed selectors, or historical crawler field names."
                )
                inserted = True
            continue
        if line.startswith("If the workspace contains a crawler scaffold, modify `crawler.py` early"):
            continue
        lines.append(line)
    patched = "\n".join(lines)
    prompts.DOCODE_SYSTEM_PROMPT = patched
    loop.DOCODE_SYSTEM_PROMPT = patched


def _patch_inspect_source_cache(tools_cls: Any, tool_result_cls: Any) -> None:
    if getattr(tools_cls, "_source_pipeline_hotfix_v1_applied", False):
        return
    setattr(tools_cls, "_source_pipeline_hotfix_v1_applied", True)
    original = tools_cls.inspect_source

    async def inspect_source(
        self: Any,
        url: str,
        mode: str = "raw",
        max_bytes: int = 50_000,
        timeout: int = 15,
    ) -> Any:
        normalized_url = str(url or "").strip()
        normalized_mode = str(mode or "raw").strip().lower()
        try:
            normalized_max_bytes = int(max_bytes)
        except (TypeError, ValueError):
            normalized_max_bytes = 50_000
        try:
            normalized_timeout = int(timeout)
        except (TypeError, ValueError):
            normalized_timeout = 15

        key = (normalized_url, normalized_mode)
        cache = getattr(self, "_docode_inspect_source_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_docode_inspect_source_cache", cache)
        cached = cache.get(key)
        if cached is not None and not _larger_refetch_needed(cached, normalized_max_bytes):
            return _cached_source_result(cached, tool_result_cls)

        result = await original(
            self,
            normalized_url,
            normalized_mode,
            normalized_max_bytes,
            normalized_timeout,
        )
        if not result.ok:
            return result

        enriched = _enrich_source_result(result, tool_result_cls)
        cache[key] = enriched
        return enriched

    tools_cls.inspect_source = inspect_source


def _larger_refetch_needed(cached: Any, requested_max_bytes: int) -> bool:
    metadata = cached.metadata if isinstance(cached.metadata, dict) else {}
    if not (cached.truncated or metadata.get("truncated")):
        return False
    try:
        returned = int(metadata.get("returned_bytes") or 0)
    except (TypeError, ValueError):
        returned = 0
    return requested_max_bytes > returned


def _cached_source_result(cached: Any, tool_result_cls: Any) -> Any:
    metadata = dict(cached.metadata or {})
    metadata.update({"cached": True, "network_request_performed": False})
    payload: dict[str, Any] = {
        "requested_url": metadata.get("requested_url") or metadata.get("url"),
        "final_url": metadata.get("final_url"),
        "status_code": metadata.get("status_code"),
        "content_type": metadata.get("content_type"),
        "mode": metadata.get("mode"),
        "cached": True,
        "network_request_performed": False,
        "truncated": bool(metadata.get("truncated") or cached.truncated),
        "structure_summary": metadata.get("structure_summary") or {},
        "instruction": "This exact source was already inspected. Reuse the existing raw source evidence and edit or inspect a distinct same-origin page; do not request this URL again.",
    }
    return tool_result_cls(
        tool="inspect_source",
        output=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        exit_code=cached.exit_code,
        metadata=metadata,
        truncated=cached.truncated,
    )


def _enrich_source_result(result: Any, tool_result_cls: Any) -> Any:
    try:
        payload = json.loads(str(result.output or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return result
    if not isinstance(payload, dict):
        return result
    body = payload.get("body")
    content_type = str(payload.get("content_type") or "")
    if isinstance(body, str) and body:
        summary = _source_structure_summary(body, content_type)
        if summary:
            payload["structure_summary"] = summary
    metadata = dict(result.metadata or {})
    if "structure_summary" in payload:
        metadata["structure_summary"] = payload["structure_summary"]
    metadata["cached"] = False
    metadata["network_request_performed"] = True
    return tool_result_cls(
        tool="inspect_source",
        output=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        exit_code=result.exit_code,
        metadata=metadata,
        truncated=result.truncated,
    )


def _source_structure_summary(body: str, content_type: str) -> dict[str, Any]:
    lowered = content_type.lower()
    if "json" in lowered:
        try:
            value = json.loads(body)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        if isinstance(value, dict):
            list_fields = {str(key): len(item) for key, item in value.items() if isinstance(item, list)}
            cursor_fields = {
                str(key): item
                for key, item in value.items()
                if isinstance(key, str) and any(token in key.lower() for token in ("cursor", "next", "page"))
            }
            return {
                "kind": "json_object",
                "top_level_keys": [str(key) for key in value.keys()][:20],
                "list_fields": list_fields,
                "pagination_fields": cursor_fields,
            }
        if isinstance(value, list):
            sample_keys = list(value[0].keys())[:20] if value and isinstance(value[0], dict) else []
            return {"kind": "json_array", "length": len(value), "sample_keys": sample_keys}
        return {"kind": type(value).__name__}

    tag_counts = Counter(match.group(1).lower() for match in re.finditer(r"<\s*([A-Za-z][A-Za-z0-9:._-]*)\b", body))
    class_counts: Counter[str] = Counter()
    for match in re.finditer(r"\bclass\s*=\s*(['\"])(.*?)\1", body, flags=re.IGNORECASE | re.DOTALL):
        for name in match.group(2).split():
            if name:
                class_counts[name] += 1
    data_attributes = sorted(
        set(match.group(1).lower() for match in re.finditer(r"\b(data-[A-Za-z0-9_.:-]+)\s*=", body, flags=re.IGNORECASE))
    )[:20]
    namespaces = sorted(
        set((match.group(1) or "default") for match in re.finditer(r"\bxmlns(?::([A-Za-z0-9_.-]+))?\s*=", body, flags=re.IGNORECASE))
    )[:20]
    repeated_tags = [{"tag": tag, "count": count} for tag, count in tag_counts.most_common(12) if count > 1]
    kind = "xml" if "xml" in lowered or body.lstrip().startswith("<?xml") else "html"
    return {
        "kind": kind,
        "root_or_first_tag": next(iter(tag_counts), ""),
        "repeated_tags": repeated_tags,
        "class_names": [name for name, _ in class_counts.most_common(20)],
        "data_attributes": data_attributes,
        "namespace_prefixes": namespaces,
        "relative_links_present": bool(re.search(r"\b(?:href|src)\s*=\s*['\"]/(?!/)", body, flags=re.IGNORECASE)),
    }


def _patch_source_tool_visibility(loop: Any) -> None:
    original_definitions = loop.allowed_tool_definitions_for_state
    original_repair_block = loop.repair_mode_tool_block

    def allowed_tool_definitions_for_state(definitions: list[Any], state: Any) -> list[Any]:
        selected = list(original_definitions(definitions, state))
        if not loop.is_crawler_instruction(state.job.instruction):
            return selected
        if loop.successful_source_inspection(state.messages, state.job.instruction) is None:
            return selected
        status_output = state.latest_git_status.output if state.latest_git_status is not None else ""
        snapshot = loop.workflow_snapshot(state, status_output)
        if snapshot.phase == loop.WorkflowPhase.FINAL_READY:
            return selected
        if _unique_source_urls(state) >= MAX_UNIQUE_SOURCE_INSPECTIONS:
            return selected
        if any(getattr(definition, "name", None) == "inspect_source" for definition in selected):
            return selected
        inspect_definition = next(
            (definition for definition in definitions if getattr(definition, "name", None) == "inspect_source"),
            None,
        )
        if inspect_definition is not None:
            selected.append(inspect_definition)
        return selected

    def repair_mode_tool_block(state: Any, tool_name: str) -> str:
        if (
            tool_name == "inspect_source"
            and loop.is_crawler_instruction(state.job.instruction)
            and loop.successful_source_inspection(state.messages, state.job.instruction) is not None
            and _unique_source_urls(state) < MAX_UNIQUE_SOURCE_INSPECTIONS
        ):
            return ""
        return original_repair_block(state, tool_name)

    loop.allowed_tool_definitions_for_state = allowed_tool_definitions_for_state
    loop.repair_mode_tool_block = repair_mode_tool_block


def _unique_source_urls(state: Any) -> int:
    return len(
        {
            str(metadata.get("requested_url") or metadata.get("url") or "")
            for message in state.messages
            if message.get("role") == "tool"
            and message.get("tool") == "inspect_source"
            and int(message.get("exit_code") or 0) == 0
            and isinstance((metadata := message.get("metadata")), dict)
            and not metadata.get("cached")
            and str(metadata.get("requested_url") or metadata.get("url") or "")
        }
    )


def _patch_source_domain_policy(loop: Any) -> None:
    def crawler_external_source_tool_block(state: Any, tool_name: str, args: dict[str, object]) -> str:
        if not loop.is_crawler_instruction(state.job.instruction):
            return ""
        allowed_domains = loop.instruction_source_domains(state.job.instruction)
        if tool_name == "inspect_source":
            raw_url = str(args.get("url") or "").strip()
            parsed = urlparse(raw_url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                return "inspect_source requires an absolute HTTP or HTTPS URL."
            allowed_origins = _allowed_source_origins(loop, state)
            if _url_origin(raw_url) in allowed_origins:
                return ""
            allowed_text = ", ".join(sorted(_format_origin(origin) for origin in allowed_origins)) or "the task source origin"
            return f"inspect_source is restricted to the same source origin(s): {allowed_text}."
        if tool_name == "fetch_url":
            raw_url = str(args.get("url") or "")
            host = urlparse(raw_url).hostname or ""
            if host and loop.source_domain_allowed(host, allowed_domains):
                return ""
            return (
                f"fetch_url is blocked for {host or '<missing host>'}. "
                f"This crawler task may only inspect the planned source domain(s): {', '.join(sorted(allowed_domains))}."
            )
        if tool_name == "web_search":
            query = str(args.get("query") or args.get("q") or "").lower()
            blocked_terms = {"cisa.gov", "cisa", "cis benchmark", "cis control", "security advisory"}
            if any(term in query for term in blocked_terms):
                return (
                    "web_search is blocked because the query drifted to unrelated security-control content. "
                    f"Keep research anchored to the planned source domain(s): {', '.join(sorted(allowed_domains))}."
                )
        # Network policy applies to network tools, not XML namespaces, schema
        # identifiers, documentation URLs, or output links written into code.
        return ""

    loop.crawler_external_source_tool_block = crawler_external_source_tool_block


def _allowed_source_origins(loop: Any, state: Any) -> set[tuple[str, str, int]]:
    origins = {origin for url in loop.instruction_source_urls(state.job.instruction) if (origin := _url_origin(url)) is not None}
    for message in state.messages:
        if message.get("role") != "tool" or message.get("tool") != "inspect_source" or int(message.get("exit_code") or 0) != 0:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        for key in ("requested_url", "final_url", "url"):
            origin = _url_origin(str(metadata.get(key) or ""))
            if origin is not None:
                origins.add(origin)
    return origins


def _url_origin(url: str) -> tuple[str, str, int] | None:
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return parsed.scheme.lower(), parsed.hostname.lower().rstrip("."), port


def _format_origin(origin: tuple[str, str, int]) -> str:
    scheme, host, port = origin
    default = 443 if scheme == "https" else 80
    suffix = "" if port == default else f":{port}"
    return f"{scheme}://{host}{suffix}"


def _patch_controller_verification_order(loop: Any, workflow: Any) -> None:
    def controller_owned_required_command(state: Any, snapshot: Any) -> str:
        commands = [
            str(command)
            for command in (state.task_contract.must_run_commands if state.task_contract is not None else [])
            if str(command).strip()
        ]
        if not commands:
            return ""

        if state.active_repair_action:
            if state.repair_mode != "targeted_repair" or not loop.targeted_repair_modified_target(state):
                return ""
        else:
            if state.repair_mode is not None:
                return ""
            if snapshot.phase != loop.WorkflowPhase.TEST_REQUIRED:
                return ""
            if not getattr(snapshot, "diff_exists", False) or not loop.successful_edit_tool_called(state):
                return ""
            if loop.missing_must_modify_targets(state):
                return ""

        # Every successful source edit starts a new verification epoch. Run the
        # explicit plan from the beginning so producer commands refresh their
        # artifacts before validator commands consume them.
        for command in commands:
            if not workflow.command_was_run(state, command):
                return command
        return ""

    loop.controller_owned_required_command = controller_owned_required_command
