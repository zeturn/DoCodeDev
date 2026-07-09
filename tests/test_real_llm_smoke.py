from __future__ import annotations

import difflib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, skipUnless

from docode.agent.loop import CodingAgentLoop
from docode.agent.quality_gate import QualityGate
from docode.agent.stop_policy import StopPolicy
from docode.agent.verifier import VerificationResult
from docode.artifacts.exporter import ArtifactExporter
from docode.config import load_config
from docode.dobox.tools import ToolDefinition
from docode.dobox.types import ToolResult
from docode.integrations.basaltpass import BasaltPassTokenExchangeClient
from docode.llm.credentials import APICredCredentialResolver, ProviderCredential
from docode.llm.runtime import build_docode_llm
from docode.storage.models import CodingJob, JobStatus, new_id

from tests.test_smoke_readme_job import DiffAcceptingVerifier, RecordingRepository, normalize_path


REAL_LLM_SMOKE_ENABLED = os.getenv("DOCODE_REAL_LLM_SMOKE", "").lower() in {"1", "true", "yes", "on"}
REQUIRED_CALCULATOR_COMMAND = "python -m unittest discover -s tests"


class RealLLMReadmeFixtureTools:
    def __init__(self, fixture_root: Path) -> None:
        self.files = {
            path.relative_to(fixture_root).as_posix(): path.read_text(encoding="utf-8")
            for path in fixture_root.rglob("*")
            if path.is_file()
        }
        self.initial_files = dict(self.files)
        self.commands: list[str] = []

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition("read_file", "Read a file from the workspace.", {"path": "string"}, self.read_file),
            ToolDefinition("write_file", "Write a file in the workspace.", {"path": "string", "content": "string"}, self.write_file),
            ToolDefinition(
                "edit_file",
                "Replace exact text in an existing workspace file.",
                {"path": "string", "old_text": "string", "new_text": "string"},
                self.edit_file,
            ),
            ToolDefinition("list_files", "List files in the workspace.", {"path": "string"}, self.list_files),
            ToolDefinition("git_status", "Return git porcelain status.", {}, self.git_status),
            ToolDefinition("git_diff", "Return git diff.", {}, self.git_diff),
            ToolDefinition("run_command", "Run a simple local smoke command.", {"command": "string", "cwd": "string"}, self.run_command),
            ToolDefinition("run_tests", "Run detected tests if available.", {}, self.run_tests),
            ToolDefinition("run_build", "Run detected build if available.", {}, self.run_build),
            ToolDefinition("run_lint", "Run detected lint if available.", {}, self.run_lint),
        ]

    def set_detected_command(self, name: str, command: str | None) -> None:
        _ = name, command

    async def call(self, tool_name: str, args: dict[str, object]) -> ToolResult:
        for definition in self.definitions():
            if definition.name == tool_name:
                if tool_name in {"git_status", "git_diff", "run_tests", "run_build", "run_lint"}:
                    return await definition.handler()
                return await definition.handler(**{key: value for key, value in args.items() if key in definition.parameters})
        return ToolResult(tool=tool_name, output=f"unknown tool: {tool_name}", exit_code=127)

    async def list_files(self, path: str = ".") -> ToolResult:
        _ = path
        return ToolResult(tool="list_files", output="\n".join(sorted(self.files)) + "\n")

    async def read_file(self, path: str) -> ToolResult:
        normalized = normalize_path(path)
        if normalized not in self.files:
            return ToolResult(tool="read_file", output=f"{normalized} not found", exit_code=1, metadata={"path": normalized})
        return ToolResult(tool="read_file", output=self.files[normalized], metadata={"path": normalized})

    async def write_file(self, path: str, content: str) -> ToolResult:
        normalized = normalize_path(path)
        self.files[normalized] = content
        return ToolResult(tool="write_file", output=f"wrote {normalized}", metadata={"path": normalized})

    async def edit_file(self, path: str, old_text: str, new_text: str) -> ToolResult:
        normalized = normalize_path(path)
        current = self.files.get(normalized)
        if current is None:
            return ToolResult(tool="edit_file", output=f"{normalized} not found", exit_code=1, metadata={"path": normalized})
        if old_text not in current:
            return ToolResult(tool="edit_file", output="old_text not found", exit_code=1, metadata={"path": normalized})
        self.files[normalized] = current.replace(old_text, new_text, 1)
        return ToolResult(tool="edit_file", output=f"edited {normalized}", metadata={"path": normalized})

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        _ = cwd
        self.commands.append(command)
        return ToolResult(tool="run_command", output="ok\n", metadata={"command": command})

    async def git_status(self) -> ToolResult:
        changed = [path for path in sorted(set(self.initial_files) | set(self.files)) if self.initial_files.get(path) != self.files.get(path)]
        return ToolResult(tool="git_status", output="".join(f" M {path}\n" for path in changed))

    async def git_diff(self) -> ToolResult:
        parts: list[str] = []
        for path in sorted(set(self.initial_files) | set(self.files)):
            before = self.initial_files.get(path, "").splitlines(keepends=True)
            after = self.files.get(path, "").splitlines(keepends=True)
            if before == after:
                continue
            parts.append(f"diff --git a/{path} b/{path}\n")
            parts.extend(difflib.unified_diff(before, after, fromfile=f"a/{path}", tofile=f"b/{path}"))
        return ToolResult(tool="git_diff", output="".join(parts))

    async def run_tests(self) -> ToolResult:
        return ToolResult(tool="run_tests", output="no test command detected", metadata={"detected": False})

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


class RealLLMCalculatorFixtureTools:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.initial_files = self.snapshot_files()
        self.commands: list[str] = []

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition("read_file", "Read a file from the workspace.", {"path": "string"}, self.read_file),
            ToolDefinition("write_file", "Write a file in the workspace.", {"path": "string", "content": "string"}, self.write_file),
            ToolDefinition(
                "edit_file",
                "Replace exact text in an existing workspace file.",
                {"path": "string", "old_text": "string", "new_text": "string"},
                self.edit_file,
            ),
            ToolDefinition("list_files", "List files in the workspace.", {"path": "string"}, self.list_files),
            ToolDefinition("git_status", "Return git porcelain status.", {}, self.git_status),
            ToolDefinition("git_diff", "Return git diff.", {}, self.git_diff),
            ToolDefinition("run_command", "Run a command in the workspace.", {"command": "string", "cwd": "string"}, self.run_command),
            ToolDefinition("run_tests", "Run detected tests if available.", {}, self.run_tests),
            ToolDefinition("run_build", "Run detected build if available.", {}, self.run_build),
            ToolDefinition("run_lint", "Run detected lint if available.", {}, self.run_lint),
        ]

    def set_detected_command(self, name: str, command: str | None) -> None:
        _ = name, command

    async def call(self, tool_name: str, args: dict[str, object]) -> ToolResult:
        for definition in self.definitions():
            if definition.name == tool_name:
                if tool_name in {"git_status", "git_diff", "run_tests", "run_build", "run_lint"}:
                    return await definition.handler()
                return await definition.handler(**{key: value for key, value in args.items() if key in definition.parameters})
        return ToolResult(tool=tool_name, output=f"unknown tool: {tool_name}", exit_code=127)

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
        executable_command = command
        if command == "python3" or command.startswith("python3 "):
            executable_command = f'"{sys.executable}"{command[len("python3") :]}'
        elif command == "python" or command.startswith("python "):
            executable_command = f'"{sys.executable}"{command[len("python") :]}'
        completed = subprocess.run(
            executable_command,
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
        return ToolResult(tool="git_status", output="".join(f" M {path}\n" for path in self.changed_files()))

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
        return ToolResult(tool="run_tests", output="no test command detected", metadata={"detected": False})

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


class ExactCommandDiffVerifier:
    async def verify(self, job, tools, evidence=None):
        _ = job, evidence
        status = await tools.git_status()
        diff = await tools.git_diff()
        command_ok = any(command == REQUIRED_CALCULATOR_COMMAND for command in tools.commands)
        return VerificationResult(
            passed=bool(diff.output.strip()) and command_ok,
            confidence=0.95,
            reason="Required command and diff verified.",
            required_fixes=[] if command_ok else [f"run required command: {REQUIRED_CALCULATOR_COMMAND}"],
            git_status=status.output,
            git_diff=diff.output,
            status_result=status,
            test_result=ToolResult(tool="run_command", output="OK\n", metadata={"command": REQUIRED_CALCULATOR_COMMAND}) if command_ok else None,
        )


async def build_real_llm_or_skip(testcase: IsolatedAsyncioTestCase, job: CodingJob):
    config = load_config()
    requested_provider = os.getenv("DOCODE_REAL_LLM_PROVIDER") or "deepseek"
    local_credentials: dict[str, ProviderCredential] = {}
    if requested_provider == "openai" and config.direct_openai_enabled and config.openai_api_key:
        local_credentials["openai"] = ProviderCredential(
            provider="openai",
            model=os.getenv("DOCODE_REAL_LLM_MODEL") or config.default_model,
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
        )
    deepseek_api_key = os.getenv("DOCODE_DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if requested_provider == "deepseek" and deepseek_api_key:
        local_credentials["deepseek"] = ProviderCredential(
            provider="deepseek",
            model=os.getenv("DOCODE_REAL_LLM_MODEL") or os.getenv("DOCODE_DEEPSEEK_MODEL") or "deepseek-chat",
            api_key=deepseek_api_key,
            base_url=os.getenv("DOCODE_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
    apicred_token = await apicred_token_from_basaltpass_or_skip(testcase, config)
    if not apicred_token:
        apicred_token = config.apicred_token
    if not apicred_token and not local_credentials:
        testcase.skipTest(
            f"DOCODE_REAL_LLM_SMOKE=1 is set for provider {requested_provider!r}, but no usable LLM credentials were configured. "
            "Checked DOCODE_BASALTPASS_SUBJECT_TOKEN, BASALTPASS_ACCESS_TOKEN, DOCODE_APICRED_TOKEN, "
            "DOCODE_DIRECT_OPENAI_ENABLED with DOCODE_OPENAI_API_KEY/OPENAI_API_KEY, and "
            "DOCODE_DEEPSEEK_API_KEY/DEEPSEEK_API_KEY."
        )
    if not apicred_token and local_credentials:
        requested_provider = os.getenv("DOCODE_REAL_LLM_PROVIDER") or next(iter(local_credentials))
        credential = local_credentials.get(requested_provider)
        if credential is None:
            testcase.skipTest(
                f"direct local credentials are configured for {sorted(local_credentials)}, "
                f"but DOCODE_REAL_LLM_PROVIDER requested {requested_provider!r}."
            )
        job.provider = credential.provider
        job.model = os.getenv("DOCODE_REAL_LLM_MODEL") or config.default_model or credential.model
    resolver = APICredCredentialResolver(
        config.apicred_base_url,
        apicred_token,
        config.apicred_mode,
        local_credentials=local_credentials,
        retry_attempts=1,
        retry_delays=(),
    )
    if apicred_token:
        provider, model = await resolve_deepseek_model_or_skip(testcase, resolver, config)
        job.provider = provider
        job.model = model
    elif local_credentials:
        credential = local_credentials[requested_provider]
        job.provider = credential.provider
        job.model = os.getenv("DOCODE_REAL_LLM_MODEL") or credential.model
    try:
        return await build_docode_llm(job, resolver)
    except Exception as exc:
        testcase.skipTest(f"real LLM runtime is not available from current provider config: {exc}")


async def apicred_token_from_basaltpass_or_skip(testcase: IsolatedAsyncioTestCase, config) -> str | None:
    subject_token = os.getenv("DOCODE_BASALTPASS_SUBJECT_TOKEN") or os.getenv("BASALTPASS_ACCESS_TOKEN")
    if not subject_token:
        return None
    if not config.basaltpass_enabled:
        testcase.skipTest("BasaltPass subject token is set, but DOCODE_BASALTPASS_ENABLED is not enabled.")
    exchanger = BasaltPassTokenExchangeClient(config.basaltpass_base_url, config.basaltpass_client_id, config.basaltpass_client_secret)
    if not exchanger.configured:
        testcase.skipTest("BasaltPass token exchange requires BASALTPASS_BASE_URL, BASALTPASS_OAUTH_CLIENT_ID, and BASALTPASS_OAUTH_CLIENT_SECRET.")
    try:
        token = await exchanger.exchange(
            subject_token=subject_token,
            resource=config.basaltpass_apicred_resource,
            scope=config.basaltpass_apicred_scope,
        )
    except Exception as exc:
        testcase.skipTest(f"BasaltPass token exchange for APICred failed: {exc}")
    if not token:
        testcase.skipTest("BasaltPass token exchange returned no APICred token.")
    return token


async def resolve_deepseek_model_or_skip(testcase: IsolatedAsyncioTestCase, resolver: APICredCredentialResolver, config) -> tuple[str, str]:
    requested_provider = os.getenv("DOCODE_REAL_LLM_PROVIDER") or "deepseek"
    requested_model = os.getenv("DOCODE_REAL_LLM_MODEL")
    try:
        catalog = await resolver.list_providers(user_id="real-llm-smoke")
    except Exception as exc:
        testcase.skipTest(f"APICred model catalog is unavailable: {exc}")
    models = catalog.get(requested_provider) or []
    if requested_provider not in catalog:
        testcase.skipTest(f"APICred catalog does not expose provider {requested_provider!r}. Available providers: {sorted(catalog)}")
    if requested_model:
        if models and requested_model not in models:
            testcase.skipTest(f"APICred provider {requested_provider!r} does not expose requested model {requested_model!r}.")
        return requested_provider, requested_model
    if config.default_provider == requested_provider and config.default_model and (not models or config.default_model in models):
        return requested_provider, config.default_model
    if models:
        return requested_provider, models[0]
    testcase.skipTest(f"APICred provider {requested_provider!r} did not list any models; set DOCODE_REAL_LLM_MODEL explicitly.")


def summarize_job_steps(steps, *, limit: int = 40) -> str:
    lines: list[str] = []
    for index, step in enumerate(steps[-limit:]):
        content = dict(step.content)
        if "output" in content:
            content["output"] = str(content["output"])[:600]
        if "summary" in content:
            content["summary"] = str(content["summary"])[:600]
        lines.append(f"{index}: {step.kind} {content}")
    return "\n".join(lines)


def final_candidate_step_indices(steps) -> list[int]:
    return [
        index
        for index, step in enumerate(steps)
        if (
            step.content.get("type") == "llm_decision"
            and step.content.get("decision_type") == "final_candidate"
        )
        or step.content.get("type") == "auto_final_candidate"
    ]


def safe_workspace_path(workspace: Path, path: str) -> Path:
    target = (workspace / path).resolve()
    root = workspace.resolve()
    if target != root and root not in target.parents:
        raise ValueError(path)
    return target


@skipUnless(REAL_LLM_SMOKE_ENABLED, "set DOCODE_REAL_LLM_SMOKE=1 to run optional real LLM smoke tests")
class RealLLMSmokeTests(IsolatedAsyncioTestCase):
    async def test_real_llm_readme_edit_with_fake_tools(self) -> None:
        fixture_root = Path(__file__).resolve().parent / "fixtures" / "repos" / "readme_edit"
        with TemporaryDirectory() as tmp:
            repo = RecordingRepository()
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="real-llm-smoke",
                    instruction="Update README.md by adding one sentence that says this project has a smoke test.",
                    max_iterations=12,
                )
            )
            tools = RealLLMReadmeFixtureTools(fixture_root)
            llm = await build_real_llm_or_skip(self, job)
            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=DiffAcceptingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo, workspace_file_reader=tools.read_file),
                stop_policy=StopPolicy(max_iterations=12, max_runtime_seconds=120, max_consecutive_failures=8),
                quality_gate=QualityGate(),
            )

            result = await loop.run(job)
            steps = await repo.list_steps(job.id)
            if result.status != JobStatus.SUCCEEDED:
                self.fail(f"real LLM README smoke failed with status={result.status}\n\nRecent steps:\n{summarize_job_steps(steps)}")

            self.assertIn(JobStatus.RUNNING, repo.status_updates)
            self.assertTrue(
                any(step.content.get("type") == "tool_result" and step.content.get("tool") in {"write_file", "edit_file"} for step in steps),
                summarize_job_steps(steps),
            )
            self.assertTrue(
                any(step.content.get("type") == "llm_decision" and step.content.get("decision_type") == "final_candidate" for step in steps)
                or any(step.content.get("type") == "auto_final_candidate" for step in steps),
                summarize_job_steps(steps),
            )
            self.assertIn("smoke test", tools.files["README.md"].lower())

            artifacts = await repo.list_artifacts(job.id)
            artifact_kinds = {artifact.kind for artifact in artifacts}
            self.assertIn("report", artifact_kinds)
            self.assertIn("result", artifact_kinds)

    async def test_real_llm_calculator_bugfix_with_fake_tools(self) -> None:
        fixture_root = Path(__file__).resolve().parent / "fixtures" / "repos" / "calculator_bug"
        instruction = (
            "Fix calculator.py so the tests pass.\n\n"
            "Verification commands:\n"
            f"1. {REQUIRED_CALCULATOR_COMMAND}"
        )
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            shutil.copytree(fixture_root, workspace)
            repo = RecordingRepository()
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="real-llm-smoke",
                    instruction=instruction,
                    max_iterations=12,
                )
            )
            tools = RealLLMCalculatorFixtureTools(workspace)
            llm = await build_real_llm_or_skip(self, job)
            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=ExactCommandDiffVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp) / "artifacts", repo, workspace_file_reader=tools.read_file),
                stop_policy=StopPolicy(max_iterations=12, max_runtime_seconds=120, max_consecutive_failures=8),
                quality_gate=QualityGate(),
            )

            result = await loop.run(job)
            steps = await repo.list_steps(job.id)
            if result.status != JobStatus.SUCCEEDED:
                self.fail(f"real LLM calculator smoke failed with status={result.status}\n\nRecent steps:\n{summarize_job_steps(steps)}")

            read_paths = {
                step.content.get("metadata", {}).get("path")
                for step in steps
                if step.content.get("type") == "tool_result" and step.content.get("tool") == "read_file"
            }
            self.assertTrue({"calculator.py", "tests/test_calculator.py"} & read_paths, summarize_job_steps(steps))
            self.assertTrue(
                any(
                    step.content.get("type") == "tool_result"
                    and step.content.get("tool") in {"write_file", "edit_file"}
                    and step.content.get("metadata", {}).get("path") == "calculator.py"
                    for step in steps
                ),
                summarize_job_steps(steps),
            )
            command_results = [
                (index, step)
                for index, step in enumerate(steps)
                if step.content.get("type") == "tool_result"
                and step.content.get("tool") == "run_command"
                and step.content.get("metadata", {}).get("command") == REQUIRED_CALCULATOR_COMMAND
            ]
            passing_command_indices = [index for index, step in command_results if step.content.get("exit_code") == 0]
            self.assertTrue(passing_command_indices, summarize_job_steps(steps))
            final_indices = final_candidate_step_indices(steps)
            self.assertTrue(final_indices, summarize_job_steps(steps))
            self.assertLess(passing_command_indices[0], final_indices[-1], summarize_job_steps(steps))
            self.assertIn("return a + b", (workspace / "calculator.py").read_text(encoding="utf-8"))

            artifacts = await repo.list_artifacts(job.id)
            artifact_kinds = {artifact.kind for artifact in artifacts}
            self.assertIn("report", artifact_kinds)
            self.assertIn("result", artifact_kinds)
