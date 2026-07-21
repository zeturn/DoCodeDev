from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse


INSPECT_SOURCE_MODES = {"raw", "text", "json", "headers"}
INSPECT_SOURCE_MIN_BYTES = 1_024
INSPECT_SOURCE_MAX_BYTES = 200_000
INSPECT_SOURCE_DEFAULT_BYTES = 50_000
INSPECT_SOURCE_MIN_TIMEOUT = 1
INSPECT_SOURCE_MAX_TIMEOUT = 30
INSPECT_SOURCE_DEFAULT_TIMEOUT = 15
INSPECT_SOURCE_COMMAND = [
    "bash",
    "-lc",
    'python_bin="$(command -v python3 || command -v python)"; '
    'test -n "$python_bin" || { echo "python interpreter unavailable" >&2; exit 127; }; '
    'exec "$python_bin" -c "$1" "$2"',
    "inspect_source",
]


# This program is passed as an argv value to the sandbox interpreter. The URL is
# transported in a JSON argv value and never interpolated into a shell command.
INSPECT_SOURCE_PROGRAM = r'''
import base64
import html
import json
import sys
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class TextExtractor(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self, convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "template"}:
            self.skip += 1
        elif not self.skip and tag in {"p", "br", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "template"} and self.skip:
            self.skip -= 1
        elif not self.skip and tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def text(self):
        lines = []
        for line in "".join(self.parts).splitlines():
            compact = " ".join(line.split())
            if compact:
                lines.append(compact)
        return "\n".join(lines)


def charset_for(headers):
    try:
        value = headers.get_content_charset()
    except Exception:
        value = None
    return value or "utf-8"


def content_length(headers):
    try:
        value = int(headers.get("content-length", ""))
        return value if value >= 0 else None
    except (TypeError, ValueError):
        return None


def base_payload(config):
    return {
        "requested_url": config["url"],
        "final_url": config["url"],
        "status_code": None,
        "content_type": "",
        "mode": config["mode"],
        "body": "",
        "original_bytes": 0,
        "returned_bytes": 0,
        "truncated": False,
        "execution_scope": "sandbox",
    }


def main():
    config = json.loads(sys.argv[1])
    payload = base_payload(config)
    request = Request(
        config["url"],
        headers={
            "User-Agent": "DoCode-Sandbox-Source-Inspector/1.0",
            "Accept": "text/html,application/xhtml+xml,application/json,application/xml,text/xml,text/plain,*/*;q=0.5",
        },
    )
    response = None
    exit_code = 0
    try:
        try:
            response = urlopen(request, timeout=config["timeout"])
        except HTTPError as exc:
            response = exc
            exit_code = 1

        headers = response.headers
        payload["final_url"] = response.geturl()
        payload["status_code"] = int(response.getcode())
        payload["content_type"] = headers.get("content-type", "")
        if config["mode"] == "headers":
            normalized_headers = {str(key).lower(): str(value) for key, value in headers.items()}
            payload["body"] = json.dumps(normalized_headers, ensure_ascii=False, sort_keys=True)
            payload["returned_bytes"] = len(payload["body"].encode("utf-8"))
        else:
            data = response.read(config["max_bytes"] + 1)
            captured = data[: config["max_bytes"]]
            declared = content_length(headers)
            payload["original_bytes"] = max(len(data), declared or 0)
            payload["returned_bytes"] = len(captured)
            payload["truncated"] = len(data) > config["max_bytes"] or bool(declared and declared > len(captured))
            content_type = payload["content_type"].lower()
            textual = any(token in content_type for token in ("text/", "json", "xml", "html", "javascript", "x-www-form-urlencoded"))
            if config["mode"] == "raw" and not textual:
                payload["body"] = base64.b64encode(captured).decode("ascii")
                payload["body_encoding"] = "base64"
            else:
                decoded = captured.decode(charset_for(headers), errors="replace")
                payload["body_encoding"] = charset_for(headers)
                if config["mode"] == "json":
                    payload["body"] = json.dumps(json.loads(decoded), ensure_ascii=False, separators=(",", ":"))
                elif config["mode"] == "text" and "html" in content_type:
                    parser = TextExtractor()
                    parser.feed(decoded)
                    payload["body"] = html.unescape(parser.text())
                else:
                    payload["body"] = decoded
    except Exception as exc:
        exit_code = 1
        payload["error"] = "%s: %s" % (type(exc).__name__, exc)
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return exit_code


sys.exit(main())
'''


def inspect_source_validation_error(url: object, mode: object, max_bytes: object, timeout: object) -> str | None:
    if not isinstance(url, str) or not url.strip():
        return "url must be a non-empty string"
    normalized = url.strip()
    if normalized != url or any(ord(character) < 32 for character in normalized):
        return "url must not contain surrounding whitespace or control characters"
    if re.search(r"[`|;<>]|\$\(|&&|\|\|", normalized):
        return "url must not contain shell control fragments"
    parsed = urlparse(normalized)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        return "url must be an absolute http or https URL"
    if parsed.username is not None or parsed.password is not None:
        return "url must not contain embedded credentials"
    try:
        parsed.port
    except ValueError:
        return "url contains an invalid port"
    if mode not in INSPECT_SOURCE_MODES:
        return "mode must be one of: raw, text, json, headers"
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not INSPECT_SOURCE_MIN_BYTES <= max_bytes <= INSPECT_SOURCE_MAX_BYTES:
        return f"max_bytes must be an integer between {INSPECT_SOURCE_MIN_BYTES} and {INSPECT_SOURCE_MAX_BYTES}"
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not INSPECT_SOURCE_MIN_TIMEOUT <= timeout <= INSPECT_SOURCE_MAX_TIMEOUT:
        return f"timeout must be an integer between {INSPECT_SOURCE_MIN_TIMEOUT} and {INSPECT_SOURCE_MAX_TIMEOUT}"
    return None


def inspect_source_error_payload(
    *,
    url: str,
    mode: str,
    error: str,
    status_code: int | None = None,
) -> dict[str, Any]:
    return {
        "requested_url": url,
        "final_url": url,
        "status_code": status_code,
        "content_type": "",
        "mode": mode,
        "body": "",
        "original_bytes": 0,
        "returned_bytes": 0,
        "truncated": False,
        "execution_scope": "sandbox",
        "error": error,
    }


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
