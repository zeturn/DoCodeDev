from __future__ import annotations

import json
import re
from collections import Counter
from html import unescape
from html.parser import HTMLParser
from typing import Any

from docode.agent.source_inspection import canonical_source_url
from docode.dobox.types import ToolResult


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
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


def source_structure_summary(body: str, content_type: str) -> dict[str, Any]:
    lowered = content_type.lower()
    if "json" in lowered:
        try:
            value = json.loads(body)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        if isinstance(value, dict):
            return {
                "kind": "json_object",
                "top_level_keys": [str(key) for key in value][:20],
                "list_fields": {str(key): len(item) for key, item in list(value.items())[:50] if isinstance(item, list)},
                "pagination_fields": {
                    str(key): safe_summary_value(str(key), item)
                    for key, item in list(value.items())[:50]
                    if isinstance(key, str) and any(token in key.lower() for token in ("cursor", "next", "page"))
                },
            }
        if isinstance(value, list):
            keys = list(value[0])[:20] if value and isinstance(value[0], dict) else []
            return {"kind": "json_array", "length": len(value), "sample_keys": keys}
        return {"kind": type(value).__name__}

    tags = Counter(match.group(1).lower() for match in re.finditer(r"<\s*([A-Za-z][A-Za-z0-9:._-]*)\b", body))
    classes: Counter[str] = Counter()
    for match in re.finditer(r"\bclass\s*=\s*(['\"])(.*?)\1", body, flags=re.IGNORECASE | re.DOTALL):
        classes.update(name for name in match.group(2).split() if name)
    namespaces = sorted(
        set((match.group(1) or "default") for match in re.finditer(r"\bxmlns(?::([A-Za-z0-9_.-]+))?\s*=", body, flags=re.IGNORECASE))
    )[:20]
    return {
        "kind": "xml" if "xml" in lowered or body.lstrip().startswith("<?xml") else "html",
        "root_or_first_tag": next(iter(tags), ""),
        "repeated_tags": [{"tag": tag, "count": count} for tag, count in tags.most_common(12) if count > 1],
        "class_names": [name for name, _ in classes.most_common(20)],
        "data_attributes": sorted(set(match.group(1).lower() for match in re.finditer(r"\b(data-[A-Za-z0-9_.:-]+)\s*=", body, flags=re.IGNORECASE)))[:20],
        "namespace_prefixes": namespaces,
        "relative_links_present": bool(re.search(r"\b(?:href|src)\s*=\s*['\"]/(?!/)", body, flags=re.IGNORECASE)),
    }


def safe_summary_value(key: str, value: Any) -> Any:
    if any(marker in key.lower() for marker in ("password", "secret", "api_key", "apikey", "access_token", "refresh_token", "authorization", "cookie")):
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


def enrich_source_result(result: ToolResult) -> ToolResult:
    try:
        payload = json.loads(result.output or "")
    except (TypeError, ValueError, json.JSONDecodeError):
        return result
    if not isinstance(payload, dict):
        return result
    body = payload.get("body")
    if isinstance(body, str) and body:
        payload["structure_summary"] = source_structure_summary(body, str(payload.get("content_type") or ""))
    metadata = dict(result.metadata or {})
    metadata.update({"cached": False, "network_request_performed": True})
    if payload.get("structure_summary"):
        metadata["structure_summary"] = payload["structure_summary"]
    return ToolResult(result.tool, json.dumps(payload, ensure_ascii=False, separators=(",", ":")), result.exit_code, metadata, result.truncated)


def cached_source_result(source: ToolResult, mode: str, *, cached: bool = True) -> ToolResult:
    try:
        payload = json.loads(source.output or "")
    except (TypeError, ValueError, json.JSONDecodeError):
        return source
    metadata = dict(source.metadata or {})
    if mode == "raw":
        compact = {key: payload.get(key) for key in ("requested_url", "final_url", "status_code", "content_type", "mode", "truncated", "structure_summary")}
        compact.update({"cached": True, "network_request_performed": False, "instruction": "Source evidence is already in memory; inspect implementation or edit now."})
        metadata.update({"cached": True, "network_request_performed": False})
        return ToolResult("inspect_source", json.dumps(compact, ensure_ascii=False, separators=(",", ":")), source.exit_code, metadata, source.truncated)
    body = str(payload.get("body") or "")
    error = ""
    try:
        if mode == "json":
            body = json.dumps(json.loads(body), ensure_ascii=False, separators=(",", ":"))
        elif mode == "text" and "html" in str(payload.get("content_type") or "").lower():
            parser = _TextExtractor()
            parser.feed(body)
            body = unescape(parser.text())
        elif mode == "headers":
            body = json.dumps({"content-type": payload.get("content_type") or ""}, separators=(",", ":"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        error = f"cached {mode} conversion failed: {exc}"
        body = ""
    payload.update({"mode": mode, "body": body, "cached": cached, "network_request_performed": not cached})
    if error:
        payload["error"] = error
    metadata.update({"mode": mode, "cached": cached, "network_request_performed": not cached})
    return ToolResult("inspect_source", json.dumps(payload, ensure_ascii=False, separators=(",", ":")), source.exit_code if not error else 1, metadata, source.truncated)


class SourceResponseCache:
    def __init__(self) -> None:
        self._responses: dict[str, ToolResult] = {}

    def get(self, url: str, mode: str, max_bytes: int) -> ToolResult | None:
        source = self._responses.get(canonical_source_url(url))
        if source is None:
            return None
        returned = int((source.metadata or {}).get("returned_bytes") or 0)
        if source.truncated and max_bytes > returned:
            return None
        return cached_source_result(source, mode)

    def put(self, url: str, result: ToolResult) -> ToolResult:
        enriched = enrich_source_result(result)
        self._responses[canonical_source_url(url)] = enriched
        return enriched
