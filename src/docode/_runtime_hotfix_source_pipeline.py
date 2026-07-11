from __future__ import annotations

import json
import re
from collections import Counter
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse, urlunparse


MAX_UNIQUE_SOURCE_INSPECTIONS = 4
MAX_OPTIONAL_PRE_EDIT_SOURCE_URLS = 2
MAX_PRE_EDIT_MODEL_DECISIONS = 2
SOURCE_TO_EDIT_TOOLS = {
    "read_file",
    "read_file_range",
    "read_symbol",
    "list_files",
    "write_file",
    "edit_file",
    "replace_in_file",
    "apply_patch",
    "git_status",
    "git_diff",
}


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
        if (
            not isinstance(url, str)
            or normalized_url != url
            or normalized_mode not in {"raw", "text", "json", "headers"}
            or isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or not 1_024 <= max_bytes <= 200_000
            or isinstance(timeout, bool)
            or not isinstance(timeout, int)
            or not 1 <= timeout <= 30
        ):
            return await original(self, url, mode, max_bytes, timeout)
        try:
            normalized_max_bytes = int(max_bytes)
        except (TypeError, ValueError):
            normalized_max_bytes = 50_000
        try:
            normalized_timeout = int(timeout)
        except (TypeError, ValueError):
            normalized_timeout = 15

        key = _canonical_source_url(normalized_url) or normalized_url
        cache = getattr(self, "_docode_inspect_source_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_docode_inspect_source_cache", cache)
        cached = cache.get(key)
        if cached is not None and not _larger_refetch_needed(cached, normalized_max_bytes):
            return _cached_source_result(cached, tool_result_cls, normalized_mode)

        # Fetch a reusable representation once. Crawler controller inspection
        # uses raw mode, and later text/json/header views are derived locally.
        result = await original(
            self,
            normalized_url,
            "raw",
            normalized_max_bytes,
            normalized_timeout,
        )
        if not result.ok:
            return result

        enriched = _enrich_source_result(result, tool_result_cls)
        cache[key] = enriched
        if normalized_mode == "raw":
            return enriched
        return _derived_source_result(enriched, tool_result_cls, normalized_mode, from_cache=False)

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


def _cached_source_result(cached: Any, tool_result_cls: Any, requested_mode: str) -> Any:
    if requested_mode != "raw":
        return _derived_source_result(cached, tool_result_cls, requested_mode, from_cache=True)
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
        "instruction": "This source is already available in Source Inspection memory. Do not inspect it again. Read or edit the implementation now.",
    }
    return tool_result_cls(
        tool="inspect_source",
        output=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        exit_code=cached.exit_code,
        metadata=metadata,
        truncated=cached.truncated,
    )


def _derived_source_result(source: Any, tool_result_cls: Any, requested_mode: str, *, from_cache: bool) -> Any:
    try:
        payload = json.loads(str(source.output or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return source
    if not isinstance(payload, dict):
        return source
    body = str(payload.get("body") or "")
    content_type = str(payload.get("content_type") or "").lower()
    try:
        if requested_mode == "json":
            body = json.dumps(json.loads(body), ensure_ascii=False, separators=(",", ":"))
        elif requested_mode == "text" and "html" in content_type:
            parser = _CachedTextExtractor()
            parser.feed(body)
            body = unescape(parser.text())
        elif requested_mode == "headers":
            body = json.dumps({"content-type": payload.get("content_type") or ""}, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        payload["error"] = f"cached {requested_mode} conversion failed: {exc}"
        body = ""
    payload.update(
        {
            "mode": requested_mode,
            "body": body,
            "cached": from_cache,
            "network_request_performed": not from_cache,
        }
    )
    metadata = dict(source.metadata or {})
    metadata.update({"mode": requested_mode, "cached": from_cache, "network_request_performed": not from_cache})
    return tool_result_cls(
        tool="inspect_source",
        output=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        exit_code=source.exit_code if not payload.get("error") else 1,
        metadata=metadata,
        truncated=source.truncated,
    )


class _CachedTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        _ = attrs
        if tag in {"script", "style", "noscript", "template"}:
            self.skip += 1
        elif not self.skip and tag in {"p", "br", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "template"} and self.skip:
            self.skip -= 1
        elif not self.skip and tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)

    def text(self) -> str:
        return "\n".join(compact for line in "".join(self.parts).splitlines() if (compact := " ".join(line.split())))


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
            list_fields = {
                str(key): len(item)
                for key, item in list(value.items())[:50]
                if isinstance(item, list)
            }
            cursor_fields = {
                str(key): _safe_summary_value(str(key), item)
                for key, item in list(value.items())[:50]
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


def _safe_summary_value(key: str, value: Any) -> Any:
    lowered_key = key.lower()
    sensitive_markers = ("password", "secret", "api_key", "apikey", "access_token", "refresh_token", "authorization", "cookie")
    if any(marker in lowered_key for marker in sensitive_markers):
        return "[redacted]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 200 else value[:200] + "...[truncated]"
    if isinstance(value, list):
        return {"type": "array", "length": len(value)}
    if isinstance(value, dict):
        return {"type": "object", "keys": [str(item) for item in list(value)[:20]]}
    return type(value).__name__


def _patch_source_tool_visibility(loop: Any) -> None:
    original_definitions = loop.allowed_tool_definitions_for_state
    original_repair_block = loop.repair_mode_tool_block
    original_required_test_block = loop.required_test_tool_block
    original_duplicate_edit_forced = loop.duplicate_inspection_edit_forced

    def allowed_tool_definitions_for_state(definitions: list[Any], state: Any) -> list[Any]:
        selected = list(original_definitions(definitions, state))
        if not loop.is_crawler_instruction(state.job.instruction):
            return selected
        if _source_progress_forced(loop, state):
            return [definition for definition in selected if getattr(definition, "name", None) in SOURCE_TO_EDIT_TOOLS]
        status_output = state.latest_git_status.output if state.latest_git_status is not None else ""
        snapshot = loop.workflow_snapshot(state, status_output)
        if not _source_inspection_continuation_allowed(loop, state, snapshot):
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
            and _source_inspection_continuation_allowed(loop, state)
        ):
            return ""
        return original_repair_block(state, tool_name)

    def required_test_tool_block(state: Any, snapshot: Any, tool_name: str, args: dict[str, object]) -> str:
        if (
            tool_name == "inspect_source"
            and _source_inspection_continuation_allowed(loop, state, snapshot)
        ):
            return ""
        return original_required_test_block(state, snapshot, tool_name, args)

    def duplicate_inspection_edit_forced(state: Any, snapshot: Any) -> bool:
        return _source_progress_forced(loop, state) or original_duplicate_edit_forced(state, snapshot)

    loop.allowed_tool_definitions_for_state = allowed_tool_definitions_for_state
    loop.repair_mode_tool_block = repair_mode_tool_block
    loop.required_test_tool_block = required_test_tool_block
    loop.duplicate_inspection_edit_forced = duplicate_inspection_edit_forced


def _source_inspection_continuation_allowed(loop: Any, state: Any, snapshot: Any | None = None) -> bool:
    if not loop.is_crawler_instruction(state.job.instruction):
        return False
    if loop.successful_source_inspection(state.messages, state.job.instruction) is None:
        return False
    if _source_progress_forced(loop, state):
        return False
    if not loop.successful_edit_tool_called(state):
        if _optional_pre_edit_source_urls(state) >= MAX_OPTIONAL_PRE_EDIT_SOURCE_URLS:
            return False
        if _pre_edit_model_decisions(state) >= MAX_PRE_EDIT_MODEL_DECISIONS:
            return False
    if _unique_source_urls(state) >= MAX_UNIQUE_SOURCE_INSPECTIONS:
        return False
    if snapshot is None:
        status_output = state.latest_git_status.output if state.latest_git_status is not None else ""
        snapshot = loop.workflow_snapshot(state, status_output)
    return snapshot.phase != loop.WorkflowPhase.FINAL_READY


def _unique_source_urls(state: Any) -> int:
    return len(
        {
            identity
            for message in state.messages
            if message.get("role") == "tool"
            and message.get("tool") == "inspect_source"
            and int(message.get("exit_code") or 0) == 0
            and isinstance((metadata := message.get("metadata")), dict)
            and not metadata.get("cached")
            and (identity := _canonical_source_url(str(metadata.get("requested_url") or metadata.get("url") or "")))
        }
    )


def _successful_source_identities(state: Any) -> list[str]:
    identities: list[str] = []
    for message in state.messages:
        if message.get("role") != "tool" or message.get("tool") != "inspect_source" or int(message.get("exit_code") or 0) != 0:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        identity = _canonical_source_url(str(metadata.get("requested_url") or metadata.get("url") or ""))
        if identity and identity not in identities:
            identities.append(identity)
    return identities


def _optional_pre_edit_source_urls(state: Any) -> int:
    identities = _successful_source_identities(state)
    return max(0, len(identities) - 1)


def _pre_edit_model_decisions(state: Any) -> int:
    first_source_index = next(
        (
            index
            for index, message in enumerate(state.messages)
            if message.get("role") == "tool"
            and message.get("tool") == "inspect_source"
            and int(message.get("exit_code") or 0) == 0
        ),
        None,
    )
    if first_source_index is None:
        return 0
    decisions = 0
    for message in state.messages[first_source_index + 1 :]:
        if (
            message.get("role") == "tool"
            and message.get("tool") in {"write_file", "edit_file", "replace_in_file", "apply_patch"}
            and int(message.get("exit_code") or 0) == 0
        ):
            return 0
        if message.get("role") == "tool":
            metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
            if not metadata.get("controller_owned"):
                decisions += 1
        elif message.get("kind") == "feedback":
            decisions += 1
    return decisions


def _duplicate_source_inspection_attempted(state: Any) -> bool:
    latest_edit = -1
    for index, message in enumerate(state.messages):
        if (
            message.get("role") == "tool"
            and message.get("tool") in {"write_file", "edit_file", "replace_in_file", "apply_patch"}
            and int(message.get("exit_code") or 0) == 0
        ):
            latest_edit = index
    return any(
        message.get("kind") == "feedback" and "duplicate_source_inspection" in str(message.get("content") or "")
        for message in state.messages[latest_edit + 1 :]
    )


def _source_progress_forced(loop: Any, state: Any) -> bool:
    if not loop.is_crawler_instruction(state.job.instruction):
        return False
    if loop.successful_source_inspection(state.messages, state.job.instruction) is None:
        return False
    if _duplicate_source_inspection_attempted(state):
        return True
    if loop.successful_edit_tool_called(state):
        return False
    return (
        _optional_pre_edit_source_urls(state) >= MAX_OPTIONAL_PRE_EDIT_SOURCE_URLS
        or _pre_edit_model_decisions(state) >= MAX_PRE_EDIT_MODEL_DECISIONS
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
            identity = _canonical_source_url(raw_url)
            if identity and identity in _successful_source_identities(state):
                return (
                    "duplicate_source_inspection: This source is already available in Source Inspection memory. "
                    "Do not inspect it again or switch modes to refetch it. Read or edit the implementation now."
                )
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
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return None
    return parsed.scheme.lower(), parsed.hostname.lower().rstrip("."), port


def _canonical_source_url(url: str) -> str | None:
    parsed = urlparse(str(url or "").strip())
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    host = parsed.hostname.lower().rstrip(".")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    default_port = 443 if scheme == "https" else 80
    netloc = host if port is None or port == default_port else f"{host}:{port}"
    return urlunparse((scheme, netloc, parsed.path, parsed.params, parsed.query, ""))


def _format_origin(origin: tuple[str, str, int]) -> str:
    scheme, host, port = origin
    default = 443 if scheme == "https" else 80
    suffix = "" if port == default else f":{port}"
    return f"{scheme}://{host}{suffix}"


def _patch_controller_verification_order(loop: Any, workflow: Any) -> None:
    original = loop.controller_owned_required_command

    def controller_owned_required_command(state: Any, snapshot: Any) -> str:
        if not loop.is_crawler_instruction(state.job.instruction):
            return original(state, snapshot)

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
