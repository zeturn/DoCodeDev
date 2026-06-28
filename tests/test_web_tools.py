from __future__ import annotations

from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from docode.web.tools import WebTools, WebToolsConfig, extract_response_text, readable_text


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

    async def test_fetch_url_returns_readable_html(self) -> None:
        tools = WebTools(WebToolsConfig())
        with patch("docode.web.tools.is_private_or_local_host", return_value=False), patch(
            "docode.web.tools.fetch_public_url",
            new=AsyncMock(return_value=("<html><head><title>X</title><style>.x{}</style></head><body><h1>Hello</h1><script>x()</script><p>World</p></body></html>", "text/html", 200)),
        ):
            result = await tools.fetch_url("https://example.com/page")

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Hello", result.output)
        self.assertIn("World", result.output)
        self.assertNotIn("x()", result.output)

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
