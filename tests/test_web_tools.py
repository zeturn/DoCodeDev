from __future__ import annotations

import json
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from docode.web.tools import WebTools, WebToolsConfig, extract_response_text, extract_url_content, readable_text


class WebToolsTests(IsolatedAsyncioTestCase):
    async def test_web_search_requires_openai_key(self) -> None:
        tools = WebTools(WebToolsConfig(openai_api_key=""))

        result = await tools.web_search("public transit GTFS data")

        self.assertEqual(result.exit_code, 2)
        self.assertIn("DOCODE_OPENAI_API_KEY", result.output)

    async def test_web_search_uses_openai_search_client(self) -> None:
        tools = WebTools(WebToolsConfig(openai_api_key="key-1"))
        tools.search_client.search = AsyncMock(return_value=("result with https://example.com", {"id": "resp_1"}))

        result = await tools.web_search("example source")

        self.assertEqual(result.exit_code, 0)
        self.assertIn("https://example.com", result.output)
        self.assertEqual(result.metadata["response_id"], "resp_1")
        tools.search_client.search.assert_awaited_once_with("example source")

    async def test_fetch_url_rejects_localhost_by_default(self) -> None:
        tools = WebTools(WebToolsConfig())

        result = await tools.fetch_url("http://localhost:3000/private")

        self.assertEqual(result.exit_code, 2)
        self.assertIn("private or local", result.output)

    async def test_fetch_url_returns_structured_extraction(self) -> None:
        tools = WebTools(WebToolsConfig(output_limit_bytes=3000))
        with patch("docode.web.tools.is_private_or_local_host", return_value=False), patch(
            "docode.web.tools.fetch_public_url",
            new=AsyncMock(
                return_value=(
                    "<html><head><title>X</title><style>.x{}</style></head><body>"
                    "<h1>Authentication</h1><script>x()</script><p>Use bearer tokens.</p>"
                    "<h2>Rate limits</h2><p>100 requests per minute.</p></body></html>",
                    "text/html",
                    200,
                )
            ),
        ):
            result = await tools.fetch_url("https://example.com/page", goal="authentication rate limits", max_sections=2)

        self.assertEqual(result.exit_code, 0)
        payload = json.loads(result.output)
        self.assertEqual(payload["title"], "X")
        self.assertEqual(payload["url"], "https://example.com/page")
        self.assertEqual(len(payload["relevant_sections"]), 2)
        self.assertIn("Use bearer tokens", result.output)
        self.assertIn("100 requests per minute", result.output)
        self.assertNotIn("x()", result.output)
        self.assertEqual(payload["confidence"], "medium")
        self.assertEqual(result.metadata["original_bytes"], payload["original_bytes"])

    async def test_fetch_url_low_confidence_when_goal_does_not_match(self) -> None:
        tools = WebTools(WebToolsConfig(output_limit_bytes=3000))
        with patch("docode.web.tools.is_private_or_local_host", return_value=False), patch(
            "docode.web.tools.fetch_public_url",
            new=AsyncMock(return_value=("<html><title>Docs</title><body><p>Welcome to the docs.</p></body></html>", "text/html", 200)),
        ):
            result = await tools.fetch_url("https://example.com/page", goal="authentication rate limits", max_sections=2)

        self.assertEqual(result.exit_code, 0)
        payload = json.loads(result.output)
        self.assertEqual(payload["confidence"], "low")
        self.assertIn("No section strongly matched", payload["warning"])

    def test_extract_url_content_returns_valid_json_sized_payload(self) -> None:
        extraction = extract_url_content(
            url="https://example.com/docs",
            content="<html><title>Docs</title><body><h1>API Authentication</h1><p>" + ("token " * 1000) + "</p></body></html>",
            content_type="text/html",
            goal="authentication token",
            max_sections=4,
            output_limit_bytes=1400,
        )

        encoded = json.dumps(extraction.payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.assertLessEqual(len(encoded), 1400)
        self.assertTrue(extraction.truncated)
        self.assertEqual(extraction.payload["title"], "Docs")

    def test_extract_response_text_from_responses_payload(self) -> None:
        text = extract_response_text(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "Candidate: https://example.com"},
                        ],
                    }
                ]
            }
        )

        self.assertEqual(text, "Candidate: https://example.com")

    def test_readable_text_strips_script_and_style(self) -> None:
        text = readable_text("<html><style>.hidden{}</style><body><h1>A</h1><script>b()</script><p>C</p></body></html>", "text/html")

        self.assertIn("A", text)
        self.assertIn("C", text)
        self.assertNotIn("hidden", text)
        self.assertNotIn("b()", text)
