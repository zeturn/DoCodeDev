from __future__ import annotations

import difflib
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from docode.agent.loop import CodingAgentLoop
from docode.agent.quality_gate import QualityGate
from docode.agent.stop_policy import StopPolicy
from docode.agent.task_contract import task_contract_from_instruction
from docode.agent.verifier import VerificationResult
from docode.artifacts.exporter import ArtifactExporter
from docode.dobox.types import ToolResult
from docode.llm.runtime import AgentDecision
from docode.storage.models import CodingJob, JobStatus, new_id
from docode.storage.repository import InMemoryJobRepository


REQUIRED_COMMAND = "python -m unittest discover -s tests"

FORBIDDEN_SMOKE_STRINGS = (
    "Git" + "Hub",
    "Trend" + "ing",
    "repo" + "sitories",
    "owner" + "/repo",
    "sta" + "rs",
    "for" + "ks",
    "Box" + "-row",
    "crawler" + ".py",
    "https://git" + "hub.com",
    "http://git" + "hub.com",
)

PARSER_IMPLEMENTATION = """from html.parser import HTMLParser


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
            self.current["in_stock"] = text.lower() == "in stock"

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

PRODUCTS_HTML_WITH_TRAILING_SPACE = """<div class="product-card" data-id="sku-001">
  <a class="name" href="/products/desk-lamp">Desk Lamp</a>
  <span class="price">$24.99</span>
  <span class="rating">4.7</span>
  <span class="stock">In stock</span>
</div>
<div class="product-card" data-id="sku-002">
  <a class="name" href="/products/notebook">Notebook</a>
  <span class="price">$5.50</span>
  <span class="rating">4.2</span>
  <span class="stock">Out of stock</span>
</div>

"""


class RecordingRepository(InMemoryJobRepository):
    def __init__(self) -> None:
        super().__init__()
        self.status_updates: list[JobStatus] = []

    async def update_job(self, job_id: str, **changes: object) -> CodingJob:
        updated = await super().update_job(job_id, **changes)
        if "status" in changes:
            self.status_updates.append(updated.status)
        return updated


class ProductParserSmokeLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.saw_required_command_feedback = False

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
                args={"path": "parser.py", "content": PARSER_IMPLEMENTATION},
            )
        if self.calls == 5:
            return AgentDecision(type="final_candidate", summary="Implemented product parser before verification.")
        if self.calls == 6:
            self.saw_required_command_feedback = any(
                message.get("kind") == "feedback" and REQUIRED_COMMAND in str(message.get("content"))
                for message in messages
            )
            return AgentDecision(type="tool_call", tool_name="run_command", args={"command": REQUIRED_COMMAND})
        if self.calls == 7:
            return AgentDecision(
                type="tool_call",
                tool_name="write_file",
                args={"path": "fixtures/products.html", "content": PRODUCTS_HTML_WITH_TRAILING_SPACE},
            )
        if self.calls == 8:
            return AgentDecision(type="tool_call", tool_name="run_command", args={"command": REQUIRED_COMMAND})
        return AgentDecision(type="final_candidate", summary="Implemented product parser and verified tests.")


class FixtureProductParserTools:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.initial_files = self.snapshot_files()
        self.commands: list[str] = []

    def definitions(self):
        return []

    def set_detected_command(self, name: str, command: str | None) -> None:
        _ = name, command

    async def call(self, tool_name: str, args: dict[str, object]) -> ToolResult:
        if tool_name == "read_file":
            return await self.read_file(str(args["path"]))
        if tool_name == "write_file":
            return await self.write_file(str(args["path"]), str(args["content"]))
        if tool_name == "edit_file":
            return await self.edit_file(str(args["path"]), str(args["old_text"]), str(args["new_text"]))
        if tool_name == "run_command":
            return await self.run_command(str(args["command"]))
        if tool_name == "git_status":
            return await self.git_status()
        if tool_name == "git_diff":
            return await self.git_diff()
        if tool_name == "list_files":
            return await self.list_files(str(args.get("path") or "."))
        raise AssertionError(tool_name)

    async def list_files(self, path: str = ".") -> ToolResult:
        _ = path
        paths = sorted(
            file.relative_to(self.workspace).as_posix()
            for file in self.workspace.rglob("*")
            if file.is_file()
        )
        return ToolResult(tool="list_files", output="\n".join(paths) + "\n")

    async def read_file(self, path: str) -> ToolResult:
        normalized = normalize_path(path)
        target = safe_workspace_path(self.workspace, normalized)
        if not target.exists():
            return ToolResult(tool="read_file", output=f"{normalized} not found", exit_code=1, metadata={"path": normalized})
        return ToolResult(tool="read_file", output=target.read_text(encoding="utf-8"), metadata={"path": normalized})

    async def write_file(self, path: str, content: str) -> ToolResult:
        normalized = normalize_path(path)
        target = safe_workspace_path(self.workspace, normalized)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(tool="write_file", output=f"wrote {normalized}", metadata={"path": normalized})

    async def edit_file(self, path: str, old_text: str, new_text: str) -> ToolResult:
        current = (await self.read_file(path)).output
        if old_text not in current:
            return ToolResult(tool="edit_file", output="old_text not found", exit_code=1, metadata={"path": normalize_path(path)})
        return await self.write_file(path, current.replace(old_text, new_text, 1))

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        _ = cwd
        self.commands.append(command)
        if command.startswith("git add -N"):
            return ToolResult(tool="run_command", output="", metadata={"command": command})
        completed = subprocess.run(
            executable_python_command(command),
            cwd=self.workspace,
            shell=True,
            text=True,
            capture_output=True,
            check=False,
        )
        return ToolResult(
            tool="run_command",
            output=completed.stdout + completed.stderr,
            exit_code=completed.returncode,
            metadata={"command": command},
        )

    async def git_status(self) -> ToolResult:
        output = "".join(f" M {path}\n" for path in self.changed_files())
        return ToolResult(tool="git_status", output=output)

    async def git_diff(self) -> ToolResult:
        parts: list[str] = []
        current = self.snapshot_files()
        for path in sorted(set(self.initial_files) | set(current)):
            before = self.initial_files.get(path, "").splitlines(keepends=True)
            after = current.get(path, "").splitlines(keepends=True)
            if before == after:
                continue
            parts.append(f"diff --git a/{path} b/{path}\n")
            parts.extend(difflib.unified_diff(before, after, fromfile=f"a/{path}", tofile=f"b/{path}"))
        return ToolResult(tool="git_diff", output="".join(parts))

    async def run_tests(self) -> ToolResult:
        return ToolResult(tool="run_tests", output="no test command auto-detected", metadata={"detected": False})

    async def run_build(self) -> ToolResult:
        return ToolResult(tool="run_build", output="no build command detected", metadata={"detected": False})

    async def run_lint(self) -> ToolResult:
        return ToolResult(tool="run_lint", output="no lint command detected", metadata={"detected": False})

    async def detect_test_command(self):
        return None

    async def detect_build_command(self):
        return None

    async def detect_lint_command(self):
        return None

    def changed_files(self) -> list[str]:
        current = self.snapshot_files()
        return [path for path in sorted(set(self.initial_files) | set(current)) if self.initial_files.get(path) != current.get(path)]

    def snapshot_files(self) -> dict[str, str]:
        return {
            file.relative_to(self.workspace).as_posix(): file.read_text(encoding="utf-8")
            for file in self.workspace.rglob("*")
            if file.is_file() and "__pycache__" not in file.parts and not file.name.endswith(".pyc")
        }


class RequiredCommandVerifier:
    async def verify(self, job, tools, evidence=None):
        _ = job, evidence
        status = await tools.git_status()
        diff = await tools.git_diff()
        command_ok = REQUIRED_COMMAND in tools.commands
        return VerificationResult(
            passed=bool(diff.output.strip()) and command_ok,
            confidence=0.95,
            reason="Product parser smoke command verified.",
            required_fixes=[] if command_ok else [f"run required command: {REQUIRED_COMMAND}"],
            git_status=status.output,
            git_diff=diff.output,
            status_result=status,
            test_result=ToolResult(tool="run_command", output="OK\n", metadata={"command": REQUIRED_COMMAND}) if command_ok else None,
        )


class ProductParserSmokeJobTests(IsolatedAsyncioTestCase):
    async def test_product_parser_reads_fixture_runs_required_command_before_success(self) -> None:
        fixture_root = Path(__file__).resolve().parent / "fixtures" / "repos" / "product_parser"
        instruction = (
            "Implement parser.py so parse_products parses fixtures/products.html and the tests pass.\n\n"
            "Verification commands:\n"
            f"1. {REQUIRED_COMMAND}"
        )
        task_contract = task_contract_from_instruction(instruction)
        self.assertIn(REQUIRED_COMMAND, task_contract.must_run_commands)

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            shutil.copytree(fixture_root, workspace)
            repo = RecordingRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="smoke", instruction=instruction))
            tools = FixtureProductParserTools(workspace)
            llm = ProductParserSmokeLLM()

            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=RequiredCommandVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp) / "artifacts", repo, workspace_file_reader=tools.read_file),
                stop_policy=StopPolicy(max_iterations=10, max_runtime_seconds=60),
                quality_gate=QualityGate(),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertIn(JobStatus.RUNNING, repo.status_updates)
            self.assertIn(JobStatus.SUCCEEDED, repo.status_updates)
            self.assertTrue(llm.saw_required_command_feedback)
            self.assertEqual(tools.commands.count(REQUIRED_COMMAND), 1)

            parser_source = (workspace / "parser.py").read_text(encoding="utf-8")
            self.assertIn("HTMLParser", parser_source)
            self.assertNotIn("Desk Lamp", parser_source)
            self.assertNotIn("Notebook", parser_source)
            self.assertFalse("sku-001" in parser_source and "sku-002" in parser_source)

            steps = await repo.list_steps(job.id)
            read_paths = {
                step.content.get("metadata", {}).get("path")
                for step in steps
                if step.content.get("type") == "tool_result" and step.content.get("tool") == "read_file"
            }
            self.assertIn("tests/test_parser.py", read_paths)
            self.assertIn("fixtures/products.html", read_paths)
            self.assertIn("parser.py", read_paths)
            self.assertTrue(any(step.content.get("type") == "tool_result" and step.content.get("tool") == "write_file" for step in steps))
            command_steps = [
                step
                for step in steps
                if step.content.get("type") == "tool_call"
                and step.content.get("tool") == "run_command"
                and step.content.get("args", {}).get("command") == REQUIRED_COMMAND
            ]
            self.assertEqual(len(command_steps), 1)
            rejected_steps = [step for step in steps if step.content.get("type") == "decision_rejected"]
            self.assertTrue(any(step.content.get("reason") == "final_candidate_tests_missing" for step in rejected_steps))
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


def normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    if normalized.startswith("/workspace/"):
        normalized = normalized[len("/workspace/") :]
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def executable_python_command(command: str) -> str:
    if command.startswith("python -m "):
        return f"{shlex.quote(sys.executable)} {command[len('python ') :]}"
    return command


def safe_workspace_path(workspace: Path, path: str) -> Path:
    target = (workspace / path).resolve()
    root = workspace.resolve()
    if target != root and root not in target.parents:
        raise ValueError(path)
    return target
