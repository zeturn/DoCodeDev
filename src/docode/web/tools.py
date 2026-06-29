from __future__ import annotations

import html
import ipaddress
import json
import re
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

from docode.dobox.tools import ToolDefinition, filter_handler_args
from docode.dobox.types import ToolResult


DEFAULT_USER_AGENT = "DoCode/0.1 (+https://docode.local)"


@dataclass(frozen=True, slots=True)
class WebToolsConfig:
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_search_model: str = "gpt-4o-mini"
    openai_search_tool_type: str = "web_search"
    search_context_size: str = "low"
    fetch_timeout_seconds: float = 20.0
    output_limit_bytes: int = 200_000
    allow_private_hosts: bool = False


class WebTools:
    def __init__(self, config: WebToolsConfig) -> None:
        self.config = config
        self.search_client = OpenAIWebSearchClient(config)

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                "web_search",
                "Search the public web for candidate data sources and current documentation. Returns concise results with URLs.",
                {"query": "string"},
                self.web_search,
            ),
            ToolDefinition(
                "fetch_url",
                "Fetch a public HTTP/HTTPS webpage and return goal-focused extracted sections for source inspection.",
                {"url": "string", "goal": "string", "max_sections": "integer"},
                self.fetch_url,
            ),
        ]

    async def call(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        for definition in self.definitions():
            if definition.name == tool_name:
                return await definition.handler(**filter_handler_args(definition.handler, args))
        return ToolResult(tool=tool_name, output=f"unknown tool: {tool_name}", exit_code=127)

    async def web_search(self, query: str) -> ToolResult:
        if not isinstance(query, str) or not query.strip():
            return ToolResult(tool="web_search", output="query must be a non-empty string", exit_code=2)
        if not self.config.openai_api_key:
            return ToolResult(
                tool="web_search",
                output="OpenAI web search is not configured. Set DOCODE_OPENAI_API_KEY to enable web_search.",
                exit_code=2,
                metadata={"configured": False},
            )
        try:
            output, raw = await self.search_client.search(query.strip())
        except Exception as exc:
            return ToolResult(
                tool="web_search",
                output=f"web_search failed: {exc}",
                exit_code=1,
                metadata={"exception_type": type(exc).__name__},
            )
        return clipped_tool_result(
            "web_search",
            output,
            self.config.output_limit_bytes,
            metadata={"query": query.strip(), "response_id": raw.get("id"), "model": self.config.openai_search_model},
        )

    async def fetch_url(self, url: str, goal: str = "", max_sections: int = 8) -> ToolResult:
        if not isinstance(url, str) or not url.strip():
            return ToolResult(tool="fetch_url", output="url must be a non-empty string", exit_code=2)
        normalized = url.strip()
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ToolResult(tool="fetch_url", output="url must be an absolute http or https URL", exit_code=2, metadata={"url": normalized})
        if not self.config.allow_private_hosts and is_private_or_local_host(parsed.hostname or ""):
            return ToolResult(tool="fetch_url", output="rejected: URL host resolves to a private or local address", exit_code=2, metadata={"url": normalized})

        try:
            content, content_type, status_code = await fetch_public_url(normalized, self.config.fetch_timeout_seconds)
        except Exception as exc:
            return ToolResult(
                tool="fetch_url",
                output=f"fetch_url failed: {exc}",
                exit_code=1,
                metadata={"url": normalized, "exception_type": type(exc).__name__},
            )
        extraction = extract_url_content(
            url=normalized,
            content=content,
            content_type=content_type,
            goal=goal,
            max_sections=max_sections,
            output_limit_bytes=self.config.output_limit_bytes,
        )
        return ToolResult(
            tool="fetch_url",
            output=json.dumps(extraction.payload, ensure_ascii=False, indent=2),
            metadata={
                "url": normalized,
                "status_code": status_code,
                "content_type": content_type,
                "original_bytes": extraction.original_bytes,
                "returned_bytes": extraction.returned_bytes,
                "truncated": extraction.truncated,
                "goal": goal.strip() if isinstance(goal, str) else "",
            },
            truncated=extraction.truncated,
        )


class OpenAIWebSearchClient:
    def __init__(self, config: WebToolsConfig) -> None:
        self.config = config

    async def search(self, query: str) -> tuple[str, dict[str, Any]]:
        import httpx

        tool: dict[str, Any] = {"type": self.config.openai_search_tool_type, "search_context_size": self.config.search_context_size}
        payload = {
            "model": self.config.openai_search_model,
            "tools": [tool],
            "tool_choice": "required",
            "input": (
                "Search the web for this data-source discovery task. "
                "Return candidate source names, URLs, what data each source provides, access method, and caveats.\n\n"
                f"Query: {query}"
            ),
        }
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                self.config.openai_base_url.rstrip("/") + "/responses",
                headers={"Authorization": f"Bearer {self.config.openai_api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            if response.is_error:
                raise RuntimeError(f"OpenAI Responses API returned HTTP {response.status_code}: {response.text[:1000]}")
            data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("OpenAI Responses API returned a non-object payload")
        return extract_response_text(data), data


async def fetch_public_url(url: str, timeout_seconds: float) -> tuple[str, str, int]:
    import httpx

    async with httpx.AsyncClient(
        timeout=timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "text/html,text/plain,application/xhtml+xml,application/json;q=0.8,*/*;q=0.5"},
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        return response.text, content_type, response.status_code


def extract_response_text(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    parts: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            parts.extend(extract_content_text(item.get("content")))
    if parts:
        return "\n".join(part for part in parts if part)
    return json.dumps(data, ensure_ascii=False)[:4000]


def extract_content_text(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text") or item.get("summary")
            if isinstance(text, str):
                parts.append(text)
    return parts


def readable_text(content: str, content_type: str) -> str:
    if "html" not in content_type.lower():
        return content
    parser = ReadableHTMLParser()
    parser.feed(content)
    return parser.text()


class ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"head", "title", "script", "style", "noscript"}:
            self.skip_depth += 1
        if tag in {"p", "br", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"head", "title", "script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1
        if tag in {"p", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        text = html.unescape(" ".join(self.parts))
        return re.sub(r"[ \t\r\f\v]+", " ", re.sub(r"\n\s*\n+", "\n\n", text)).strip()


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    payload: dict[str, Any]
    original_bytes: int
    returned_bytes: int
    truncated: bool


def extract_url_content(
    *,
    url: str,
    content: str,
    content_type: str,
    goal: str = "",
    max_sections: int = 8,
    output_limit_bytes: int = 200_000,
) -> ExtractedPage:
    text = readable_text(content, content_type)
    title = extract_title(content, content_type)
    section_limit = max(1, min(int_or_default(max_sections, 8), 20))
    sections = select_relevant_sections(text, goal, section_limit)
    confidence = extraction_confidence(goal, sections)
    payload: dict[str, Any] = {
        "url": url,
        "title": title,
        "summary": summarize_sections(sections),
        "relevant_sections": sections,
        "confidence": confidence,
        "original_bytes": len(content.encode("utf-8")),
        "returned_bytes": 0,
        "truncated": False,
    }
    if confidence == "low":
        payload["warning"] = "No section strongly matched the goal; inspect another source or refine goal."
    payload, returned_bytes, truncated = fit_extraction_payload(payload, output_limit_bytes)
    payload["truncated"] = truncated
    payload["returned_bytes"] = len(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
    return ExtractedPage(
        payload=payload,
        original_bytes=len(content.encode("utf-8")),
        returned_bytes=int(payload["returned_bytes"]),
        truncated=truncated,
    )


def select_relevant_sections(text: str, goal: str, max_sections: int) -> list[dict[str, str]]:
    chunks = page_chunks(text)
    goal_terms = relevant_terms(goal)
    ranked: list[tuple[int, int, str]] = []
    for index, chunk in enumerate(chunks):
        lowered = chunk.lower()
        score = sum(1 for term in goal_terms if term in lowered)
        ranked.append((-score, index, chunk))
    selected = sorted(ranked)[:max_sections] if goal_terms else ranked[:max_sections]
    return [section_payload(chunk, goal_terms, matched_score=-score) for score, _, chunk in selected]


def page_chunks(text: str) -> list[str]:
    raw_chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n+", text) if chunk.strip()]
    chunks: list[str] = []
    index = 0
    while index < len(raw_chunks):
        chunk = raw_chunks[index]
        if index + 1 < len(raw_chunks) and looks_like_heading(chunk):
            chunks.append(chunk + "\n" + raw_chunks[index + 1])
            index += 2
            continue
        chunks.append(chunk)
        index += 1
    return chunks


def looks_like_heading(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and len(stripped) <= 120 and "\n" not in stripped and not stripped.endswith((".", ":", ";"))


def section_payload(chunk: str, goal_terms: set[str], *, matched_score: int) -> dict[str, str]:
    lines = [line.strip() for line in chunk.splitlines() if line.strip()]
    heading = lines[0][:120] if lines else "Section"
    if len(heading) > 80 and "." in heading:
        heading = heading.split(".", 1)[0][:80]
    text = chunk[:4000]
    if len(chunk) > len(text):
        text += "\n<truncated>"
    matched_terms = sorted(term for term in goal_terms if term in chunk.lower())
    if matched_terms:
        why = "matched goal terms: " + ", ".join(matched_terms[:8])
    elif matched_score > 0:
        why = "matched the requested goal"
    else:
        why = "early page content"
    return {"heading": heading or "Section", "text": text, "why_relevant": why}


def summarize_sections(sections: list[dict[str, str]]) -> str:
    text = "\n\n".join(section["text"] for section in sections)
    return text[:1200] + ("\n<truncated>" if len(text) > 1200 else "")


def extraction_confidence(goal: str, sections: list[dict[str, str]]) -> str:
    if not relevant_terms(goal):
        return "medium"
    if any(str(section.get("why_relevant", "")).startswith("matched") for section in sections):
        return "medium"
    return "low"


def fit_extraction_payload(payload: dict[str, Any], output_limit_bytes: int) -> tuple[dict[str, Any], int, bool]:
    limit = max(1000, output_limit_bytes)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    if len(encoded) <= limit:
        return payload, len(encoded), False

    clipped = dict(payload)
    sections = [dict(section) for section in payload.get("relevant_sections", []) if isinstance(section, dict)]
    per_section_limit = max(400, limit // max(1, len(sections) * 2))
    for section in sections:
        text = str(section.get("text", ""))
        if len(text.encode("utf-8")) > per_section_limit:
            section["text"] = text.encode("utf-8")[:per_section_limit].decode("utf-8", errors="replace") + "\n<truncated>"
    clipped["relevant_sections"] = sections
    clipped["summary"] = summarize_sections(sections)
    clipped["truncated"] = True

    while len(json.dumps(clipped, ensure_ascii=False, indent=2).encode("utf-8")) > limit and len(sections) > 1:
        sections.pop()
        clipped["relevant_sections"] = sections
        clipped["summary"] = summarize_sections(sections)

    while len(json.dumps(clipped, ensure_ascii=False, indent=2).encode("utf-8")) > limit and sections:
        largest = max(sections, key=lambda section: len(str(section.get("text", "")).encode("utf-8")))
        text = str(largest.get("text", ""))
        encoded_text = text.encode("utf-8")
        if len(encoded_text) <= 120:
            break
        largest["text"] = encoded_text[: max(120, len(encoded_text) // 2)].decode("utf-8", errors="replace") + "\n<truncated>"
        clipped["summary"] = summarize_sections(sections)

    encoded = json.dumps(clipped, ensure_ascii=False, indent=2).encode("utf-8")
    return clipped, len(encoded), True


def extract_title(content: str, content_type: str) -> str:
    if "html" not in content_type.lower():
        return ""
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", content)
    if not match:
        return ""
    return re.sub(r"\s+", " ", html.unescape(match.group(1))).strip()


def relevant_terms(goal: str) -> set[str]:
    if not isinstance(goal, str):
        return set()
    stopwords = {"the", "and", "for", "with", "from", "this", "that", "api", "url", "http", "https", "extract"}
    return {token for token in re.findall(r"[a-zA-Z0-9_./-]{3,}", goal.lower()) if token not in stopwords}


def int_or_default(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_private_or_local_host(hostname: str) -> bool:
    host = hostname.strip().lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        addresses = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return True
    return False


def clipped_tool_result(tool: str, output: str, limit: int, metadata: dict[str, Any] | None = None) -> ToolResult:
    encoded = output.encode("utf-8")
    if len(encoded) <= limit:
        return ToolResult(tool=tool, output=output, metadata=metadata)
    return ToolResult(tool=tool, output=encoded[:limit].decode("utf-8", errors="replace"), metadata=metadata, truncated=True)
