from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from docode.agent.loop import CodingAgentLoop
from docode.agent.quality_gate import QualityGate
from docode.agent.stop_policy import StopPolicy
from docode.agent.task_contract import task_contract_from_instruction
from docode.artifacts.exporter import ArtifactExporter
from docode.llm.runtime import AgentDecision
from docode.storage.models import CodingJob, JobStatus, new_id

from tests.test_smoke_product_parser_job import (
    FORBIDDEN_SMOKE_STRINGS,
    PARSER_IMPLEMENTATION,
    PRODUCTS_HTML_WITH_TRAILING_SPACE,
    REQUIRED_COMMAND,
    FixtureProductParserTools,
    RecordingRepository,
    RequiredCommandVerifier,
)


BROKEN_PARSER_IMPLEMENTATION = """from html.parser import HTMLParser


class _ProductCardParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.products = []
        self.current = None
        self.field = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set(attrs.get("class", "").split())
        if tag == "div" and "product-card" in classes:
            self.current = {
                "id": attrs.get("data-id", ""),
                "name": "",
                "url": "",
                "price": 0.0,
                "rating": 0.0,
                "in_stock": False,
            }
            return
        if self.current is None:
            return
        if tag == "a" and "name" in classes:
            self.field = "name"
            self.current["url"] = attrs.get("href", "")
        elif tag == "span" and "price" in classes:
            self.field = "price"
        elif tag == "span" and "rating" in classes:
            self.field = "rating"
        elif tag == "span" and "stock" in classes:
            self.field = "stock"

    def handle_data(self, data):
        if self.current is None or self.field is None:
            return
        text = data.strip()
        if not text:
            return
        if self.field == "name":
            self.current["name"] += text
        elif self.field == "price":
            self.current["price"] = float(text.replace("$", "").strip())
        elif self.field == "rating":
            self.current["rating"] = float(text)
        elif self.field == "stock":
            self.current["in_stock"] = False

    def handle_endtag(self, tag):
        if self.current is not None and tag in {"a", "span"}:
            self.field = None
        elif self.current is not None and tag == "div":
            self.products.append(self.current)
            self.current = None
            self.field = None


def parse_products(html_text: str):
    parser = _ProductCardParser()
    parser.feed(html_text)
    return parser.products
"""


class ProductParserRepairSmokeLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.saw_repair_feedback = False

    async def decide(self, *, system, messages, tools, context):
        _ = system, tools, context
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(type="tool_call", tool_name="read_file", args={"path": "tests/test_parser.py"})
        if self.calls == 2:
            return AgentDecision(type="tool_call", tool_name="read_file", args={"path": "fixtures/products.html"})
        if self.calls == 3:
            return AgentDecision(type="tool_call", tool_name="read_file", args={"path": "parser.py"})
        if self.calls == 4:
            return AgentDecision(
                type="tool_call",
                tool_name="write_file",
                args={"path": "parser.py", "content": BROKEN_PARSER_IMPLEMENTATION},
            )
        if self.calls == 5:
            return AgentDecision(type="tool_call", tool_name="run_command", args={"command": REQUIRED_COMMAND})
        if self.calls == 6:
            return AgentDecision(
                type="tool_call",
                tool_name="write_file",
                args={"path": "fixtures/products.html", "content": PRODUCTS_HTML_WITH_TRAILING_SPACE},
            )
        if self.calls == 7:
            return AgentDecision(type="tool_call", tool_name="run_command", args={"command": REQUIRED_COMMAND})
        if self.calls == 8:
            self.saw_repair_feedback = any(
                message.get("kind") == "feedback"
                and ("parsed_value_mismatch" in str(message.get("content")) or "repair_mode=targeted_repair" in str(message.get("content")))
                for message in messages
            )
            return AgentDecision(type="tool_call", tool_name="read_file", args={"path": "parser.py"})
        if self.calls == 9:
            return AgentDecision(
                type="tool_call",
                tool_name="write_file",
                args={"path": "parser.py", "content": PARSER_IMPLEMENTATION},
            )
        if self.calls == 10:
            return AgentDecision(type="tool_call", tool_name="run_command", args={"command": REQUIRED_COMMAND})
        return AgentDecision(type="final_candidate", summary="Repaired product parser and verified tests.")


class ProductParserRepairSmokeJobTests(IsolatedAsyncioTestCase):
    async def test_product_parser_repair_after_failing_required_command(self) -> None:
        fixture_root = Path(__file__).resolve().parent / "fixtures" / "repos" / "product_parser"
        instruction = (
            "Implement parser.py so parse_products parses fixtures/products.html and the tests pass.\n\n"
            "Verification commands:\n"
            f"1. {REQUIRED_COMMAND}"
        )
        self.assertIn(REQUIRED_COMMAND, task_contract_from_instruction(instruction).must_run_commands)

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            shutil.copytree(fixture_root, workspace)
            repo = RecordingRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="smoke", instruction=instruction))
            tools = FixtureProductParserTools(workspace)
            llm = ProductParserRepairSmokeLLM()

            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=RequiredCommandVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp) / "artifacts", repo, workspace_file_reader=tools.read_file),
                stop_policy=StopPolicy(max_iterations=14, max_runtime_seconds=60),
                quality_gate=QualityGate(),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertIn(JobStatus.RUNNING, repo.status_updates)
            self.assertIn(JobStatus.SUCCEEDED, repo.status_updates)
            self.assertTrue(llm.saw_repair_feedback)
            self.assertGreaterEqual(tools.commands.count(REQUIRED_COMMAND), 2)

            steps = await repo.list_steps(job.id)
            command_results = [
                step
                for step in steps
                if step.content.get("type") == "tool_result"
                and step.content.get("tool") == "run_command"
                and step.content.get("metadata", {}).get("command") == REQUIRED_COMMAND
            ]
            failing_results = [step for step in command_results if step.content.get("exit_code")]
            passing_results = [step for step in command_results if step.content.get("exit_code") == 0]
            self.assertTrue(failing_results)
            self.assertTrue(passing_results)
            self.assertLess(steps.index(failing_results[0]), steps.index(passing_results[-1]))

            repair_steps = [step for step in steps if step.content.get("type") == "repair_action"]
            self.assertTrue(repair_steps)
            repair_payload = repair_steps[0].content.get("repair_action", {})
            self.assertEqual(repair_payload.get("category"), "parsed_value_mismatch")
            self.assertIn("parser.py", repair_payload.get("target_files", []))
            self.assertIn(REQUIRED_COMMAND, repair_payload.get("rerun_commands", []))

            parser_repair_reads = [
                step
                for step in steps
                if step.content.get("type") == "tool_result"
                and step.content.get("tool") == "read_file"
                and step.content.get("metadata", {}).get("path") == "parser.py"
                and steps.index(step) > steps.index(repair_steps[0])
            ]
            parser_repair_writes = [
                step
                for step in steps
                if step.content.get("type") == "tool_result"
                and step.content.get("tool") == "write_file"
                and step.content.get("metadata", {}).get("path") == "parser.py"
                and steps.index(step) > steps.index(repair_steps[0])
            ]
            self.assertTrue(parser_repair_reads or parser_repair_writes)

            parser_source = (workspace / "parser.py").read_text(encoding="utf-8")
            self.assertIn("HTMLParser", parser_source)
            self.assertIn('text.lower() == "in stock"', parser_source)
            self.assertNotIn("Desk Lamp", parser_source)
            self.assertNotIn("Notebook", parser_source)
            self.assertFalse("sku-001" in parser_source and "sku-002" in parser_source)
            self.assertTrue(any(step.kind == "verifier" for step in steps))

            artifacts = await repo.list_artifacts(job.id)
            artifact_kinds = {artifact.kind for artifact in artifacts}
            self.assertIn("report", artifact_kinds)
            self.assertIn("result", artifact_kinds)
            self.assertIsNotNone(result.artifact_id)

            combined_steps = "\n".join(str(step.content) for step in steps)
            combined_files = "\n".join(tools.snapshot_files().values())
            for forbidden in FORBIDDEN_SMOKE_STRINGS:
                self.assertNotIn(forbidden, combined_steps)
                self.assertNotIn(forbidden, combined_files)
            self.assertFalse(any(step.content.get("reason") == "active_repair_controller_forced_target_edit" for step in steps))
