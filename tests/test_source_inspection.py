from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase

from docode.agent.loop import CodingAgentLoop
from docode.agent.source_inspection import (
    instruction_source_urls,
    source_inspection_evidence,
    successful_source_inspection,
)
from docode.agent.state import AgentState
from docode.agent.stop_policy import StopPolicy
from docode.agent.verifier import CodingVerifier
from docode.artifacts.exporter import ArtifactExporter
from docode.dobox.types import ToolResult
from docode.storage.models import CodingJob, new_id
from docode.storage.repository import InMemoryJobRepository
from tests.crawler_benchmark_v1.definitions import CASES, LOCAL_BASE_URL


class SourceInspectionEvidenceTests(TestCase):
    def test_existing_crawler_benchmark_cases_select_their_real_source_not_control_urls(self) -> None:
        for case in CASES:
            candidates = instruction_source_urls(case.instruction)
            expected = LOCAL_BASE_URL + case.source_path if case.controlled else case.source_path
            self.assertEqual(candidates[0], expected, case.name)
            self.assertTrue(all("/__reset" not in url and "/__metrics" not in url for url in candidates), case.name)

    def test_source_url_selection_prefers_main_description_and_excludes_control_endpoints(self) -> None:
        instruction = """Build a crawler.

Source endpoint: https://example.test/items?cursor=
Backup feed: https://backup.test/feed.xml

Verification commands:
1. python -c \"open('https://example.test/__reset')\"
2. python crawler.py https://fallback.test/data.json output.json
3. python -c \"open('https://example.test/__metrics')\"
"""

        self.assertEqual(
            instruction_source_urls(instruction),
            [
                "https://example.test/items?cursor=",
                "https://backup.test/feed.xml",
                "https://fallback.test/data.json",
            ],
        )

    def test_only_successful_sandbox_inspection_satisfies_source_evidence(self) -> None:
        instruction = "crawl https://example.test/feed.xml"
        messages = [
            {"role": "tool", "tool": "web_search", "exit_code": 0, "metadata": {"query": "example feed"}},
            {"role": "tool", "tool": "fetch_url", "exit_code": 0, "metadata": {"url": "https://example.test/feed.xml"}},
            {
                "role": "tool",
                "tool": "inspect_source",
                "exit_code": 0,
                "output": json.dumps(
                    {
                        "requested_url": "https://example.test/feed.xml",
                        "final_url": "https://example.test/current.xml",
                        "status_code": 200,
                        "execution_scope": "sandbox",
                        "mode": "raw",
                        "body": "<rss><item /></rss>",
                    }
                ),
                "metadata": {"controller_owned": True},
            },
            {"role": "tool", "tool": "write_file", "exit_code": 0, "metadata": {"path": "crawler.py"}},
        ]

        evidence = source_inspection_evidence(messages, instruction)

        self.assertEqual(len(evidence), 1)
        self.assertTrue(evidence[0].successful)
        self.assertTrue(evidence[0].before_first_edit)
        self.assertTrue(evidence[0].controller_owned)
        self.assertEqual(successful_source_inspection(messages, instruction), evidence[0])

    def test_inspection_after_first_edit_records_ordering_failure(self) -> None:
        instruction = "crawl https://example.test/feed.xml"
        messages = [
            {"role": "tool", "tool": "write_file", "exit_code": 0, "metadata": {"path": "crawler.py"}},
            {
                "role": "tool",
                "tool": "inspect_source",
                "exit_code": 0,
                "output": json.dumps(
                    {
                        "requested_url": "https://example.test/feed.xml",
                        "final_url": "https://example.test/feed.xml",
                        "status_code": 200,
                        "execution_scope": "sandbox",
                        "mode": "raw",
                        "body": "data",
                    }
                ),
            },
        ]

        self.assertFalse(source_inspection_evidence(messages, instruction)[0].before_first_edit)


class ControllerSourceInspectionTests(IsolatedAsyncioTestCase):
    async def test_controller_inspects_primary_source_once_and_records_evidence(self) -> None:
        class SourceTools:
            def __init__(self) -> None:
                self.calls: list[tuple[str, dict[str, object]]] = []

            async def call(self, name, args):
                self.calls.append((name, dict(args)))
                payload = {
                    "requested_url": args["url"],
                    "final_url": args["url"],
                    "status_code": 200,
                    "content_type": "application/json",
                    "mode": args["mode"],
                    "body": '{"items":[1]}',
                    "original_bytes": 13,
                    "returned_bytes": 13,
                    "truncated": False,
                    "execution_scope": "sandbox",
                }
                return ToolResult(tool=name, output=json.dumps(payload), metadata={key: value for key, value in payload.items() if key != "body"})

        with TemporaryDirectory() as tmp:
            repository = InMemoryJobRepository()
            instruction = "crawl source https://example.test/items?cursor= into JSON"
            job = await repository.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction=instruction))
            tools = SourceTools()
            loop = CodingAgentLoop(
                llm=object(),
                tools=tools,
                verifier=CodingVerifier(),
                repository=repository,
                exporter=ArtifactExporter(Path(tmp), repository),
                stop_policy=StopPolicy(),
            )
            state = AgentState(job=job)

            first = await loop.maybe_execute_controller_source_inspection(state)
            second = await loop.maybe_execute_controller_source_inspection(state)

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(tools.calls[0][1]["url"], "https://example.test/items?cursor=")
        evidence = source_inspection_evidence(state.messages, instruction)[0]
        self.assertTrue(evidence.successful)
        self.assertTrue(evidence.before_first_edit)
        steps = await repository.list_steps(job.id)
        self.assertTrue(any(step.content.get("type") == "source_inspection_auto_execution" for step in steps))
        self.assertTrue(any(step.content.get("type") == "source_inspection_evidence" for step in steps))
