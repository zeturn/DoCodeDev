from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse


@dataclass(frozen=True, slots=True)
class EvalCaseResult:
    name: str
    status: str
    success: bool
    iterations: int = 0
    tool_calls: int = 0
    tokens: int = 0
    cost: float = 0.0
    failure_reason: str | None = None
    verification_plan_failures: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EvalScenario:
    name: str
    category: str
    instruction: str
    files: dict[str, str]
    expected_checks: list[str]
    artifact_mode: str = "patch"

    def to_manifest_entry(self, repo_dir: Path, git_initialized: bool) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "instruction": self.instruction,
            "repo_path": str(repo_dir),
            "repo_url": repo_dir.resolve().as_uri(),
            "artifact_mode": self.artifact_mode,
            "expected_checks": self.expected_checks,
            "git_initialized": git_initialized,
        }


@dataclass(frozen=True, slots=True)
class EvalReport:
    total: int
    succeeded: int
    failed: int
    success_rate: float
    iterations: int
    tool_calls: int
    tokens: int
    cost: float
    failure_reasons: dict[str, int]
    verification_plan_failures: dict[str, int]
    cases: list[EvalCaseResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "success_rate": self.success_rate,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "tokens": self.tokens,
            "cost": self.cost,
            "failure_reasons": self.failure_reasons,
            "verification_plan_failures": self.verification_plan_failures,
            "cases": [asdict(case) for case in self.cases],
        }


def load_eval_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError(f"eval manifest must contain a cases list: {path}")
    return data


def eval_case_result_from_job(case: dict[str, Any], job: Any, steps: list[Any]) -> dict[str, Any]:
    status = status_value(getattr(job, "status", "missing"))
    usage = latest_usage_snapshot(steps)
    verification = latest_verification_step(steps)
    result = {
        "name": str(case.get("name") or getattr(job, "id", "unknown")),
        "category": case.get("category"),
        "instruction": case.get("instruction"),
        "job_id": getattr(job, "id", None),
        "status": status,
        "success": status == "succeeded",
        "iterations": count_steps(steps, "llm_decision"),
        "tool_calls": count_steps(steps, "tool_call"),
        "tokens": int_or_zero(usage.get("total_tokens") or usage.get("tokens")),
        "cost": float_or_zero(usage.get("cost")),
        "failure_reason": getattr(job, "failure_reason", None),
        "artifact_id": getattr(job, "artifact_id", None),
        "verification": verification,
    }
    return result


def write_eval_case_result(result: dict[str, Any], results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    name = str(result.get("name") or "case").replace("/", "-")
    path = results_dir / f"{name}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def run_eval(fixtures_dir: Path) -> EvalReport:
    cases = [load_eval_case(path) for path in sorted(fixtures_dir.glob("*.json"))]
    total = len(cases)
    succeeded = sum(1 for case in cases if case.success)
    failed = total - succeeded
    return EvalReport(
        total=total,
        succeeded=succeeded,
        failed=failed,
        success_rate=(succeeded / total) if total else 0.0,
        iterations=sum(case.iterations for case in cases),
        tool_calls=sum(case.tool_calls for case in cases),
        tokens=sum(case.tokens for case in cases),
        cost=sum(case.cost for case in cases),
        failure_reasons=count_values(case.failure_reason for case in cases if case.failure_reason),
        verification_plan_failures=count_values(failure for case in cases for failure in case.verification_plan_failures),
        cases=cases,
    )


def scaffold_eval_suite(output_dir: Path, *, force: bool = False) -> dict[str, Any]:
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repos_dir = output_dir / "repos"
    repos_dir.mkdir(exist_ok=True)

    cases: list[dict[str, Any]] = []
    for scenario in default_eval_scenarios():
        repo_dir = repos_dir / scenario.name
        if repo_dir.exists():
            if not force:
                raise FileExistsError(f"eval scenario repo already exists: {repo_dir}")
            shutil.rmtree(repo_dir)
        repo_dir.mkdir(parents=True)
        for relative_path, content in scenario.files.items():
            path = repo_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        git_initialized = initialize_git_repo(repo_dir)
        cases.append(scenario.to_manifest_entry(repo_dir, git_initialized))

    manifest = {
        "version": 1,
        "description": "DoCode small-repository eval suite covering common coding-agent task shapes.",
        "cases": cases,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


@contextmanager
def managed_local_repo_server(manifest: dict[str, Any], *, enabled: bool = True):
    if not enabled:
        yield manifest
        return

    repo_paths = local_repo_paths(manifest)
    if not repo_paths:
        yield manifest
        return
    git = shutil.which("git")
    if git is None:
        yield manifest
        return

    base_path = common_repo_base_path(repo_paths)
    port = free_tcp_port()
    process = subprocess.Popen(
        [
            git,
            "daemon",
            "--verbose",
            "--export-all",
            "--reuseaddr",
            f"--base-path={base_path}",
            f"--port={port}",
            "--listen=0.0.0.0",
            str(base_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        probe_url = git_url_for_repo(repo_paths[0], base_path, "127.0.0.1", port)
        wait_for_git_repo_server(git, probe_url, process)
        yield manifest_with_served_local_repos(manifest, base_path=base_path, host="host.docker.internal", port=port)
    finally:
        stop_process(process)


def manifest_with_served_local_repos(manifest: dict[str, Any], *, base_path: Path, host: str, port: int) -> dict[str, Any]:
    served = dict(manifest)
    served_cases: list[dict[str, Any]] = []
    for case in manifest.get("cases", []):
        served_case = dict(case)
        repo_path = local_repo_path_for_case(served_case)
        if repo_path is not None:
            served_case["repo_url"] = git_url_for_repo(repo_path, base_path, host, port)
            served_case["local_repo_url"] = case.get("repo_url")
        served_cases.append(served_case)
    served["cases"] = served_cases
    return served


def local_repo_paths(manifest: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for case in manifest.get("cases", []):
        if isinstance(case, dict):
            path = local_repo_path_for_case(case)
            if path is not None:
                paths.append(path)
    return paths


def local_repo_path_for_case(case: dict[str, Any]) -> Path | None:
    repo_path = case.get("repo_path")
    if repo_path:
        path = Path(str(repo_path)).expanduser().resolve()
        return path if path.exists() else None
    repo_url = str(case.get("repo_url") or "")
    parsed = urlparse(repo_url)
    if parsed.scheme == "file":
        path = Path(parsed.path).expanduser().resolve()
        return path if path.exists() else None
    return None


def common_repo_base_path(repo_paths: list[Path]) -> Path:
    parents = [str(path.parent) for path in repo_paths]
    return Path(os.path.commonpath(parents)).resolve()


def git_url_for_repo(repo_path: Path, base_path: Path, host: str, port: int) -> str:
    relative = repo_path.resolve().relative_to(base_path.resolve()).as_posix()
    return f"git://{host}:{port}/{quote(relative)}"


def free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_git_repo_server(git: str, url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 5.0
    last_error = "not checked"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"git daemon exited with code {process.returncode}")
        result = subprocess.run([git, "ls-remote", url, "HEAD"], capture_output=True, text=True, check=False, timeout=2)
        if result.returncode == 0:
            return
        last_error = (result.stderr or result.stdout or "").strip()
        time.sleep(0.1)
    raise RuntimeError(f"git daemon did not become ready for {url}: {last_error}")


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def default_eval_scenarios() -> list[EvalScenario]:
    return [
        EvalScenario(
            name="python-bugfix",
            category="Python bugfix",
            instruction="Fix the retry_count bug in calculator.py and keep the existing tests passing.",
            files={
                "calculator.py": "def retry_count(attempts):\n    return attempts - 1\n",
                "tests/test_calculator.py": "import unittest\nfrom calculator import retry_count\n\nclass CalculatorTests(unittest.TestCase):\n    def test_retry_count(self):\n        self.assertEqual(retry_count(3), 3)\n\nif __name__ == '__main__':\n    unittest.main()\n",
                "README.md": "# Python Bugfix Fixture\n",
            },
            expected_checks=["python3 -m unittest discover -s tests"],
        ),
        EvalScenario(
            name="python-cli",
            category="Python CLI",
            instruction="Turn cli.py into a working command line tool that prints a greeting for --name.",
            files={
                "cli.py": "import argparse\n\ndef main():\n    parser = argparse.ArgumentParser()\n    parser.add_argument('--name', default='world')\n    args = parser.parse_args()\n    print('TODO')\n\nif __name__ == '__main__':\n    main()\n",
                "README.md": "# Python CLI Fixture\nRun `python3 cli.py --name Ada`.\n",
            },
            expected_checks=["python3 cli.py --name Ada"],
        ),
        EvalScenario(
            name="crawler",
            category="crawler",
            instruction="Build a small crawler that parses fixtures/source.html and writes data/output.json with at least 2 records.",
            files={
                "crawler.py": "from pathlib import Path\n\nSOURCE = Path('fixtures/source.html')\nOUTPUT = Path('data/output.json')\n\ndef main():\n    OUTPUT.parent.mkdir(exist_ok=True)\n    OUTPUT.write_text('[]\\n', encoding='utf-8')\n\nif __name__ == '__main__':\n    main()\n",
                "fixtures/source.html": "<html><body><ul><li data-name='alpha'>Alpha</li><li data-name='beta'>Beta</li></ul></body></html>\n",
                "README.md": "# Crawler Fixture\n",
            },
            expected_checks=["python3 crawler.py", "python3 -c \"import json; assert len(json.load(open('data/output.json'))) >= 2\""],
        ),
        EvalScenario(
            name="api-adapter",
            category="API adapter",
            instruction="Implement client.parse_items_response so it extracts item names from a JSON API response.",
            files={
                "client.py": "import json\n\ndef parse_items_response(text):\n    data = json.loads(text)\n    return []\n",
                "tests/test_client.py": "import unittest\nfrom client import parse_items_response\n\nclass ClientTests(unittest.TestCase):\n    def test_parse_items(self):\n        payload = '{\"items\":[{\"name\":\"north\"},{\"name\":\"south\"}]}'\n        self.assertEqual(parse_items_response(payload), ['north', 'south'])\n\nif __name__ == '__main__':\n    unittest.main()\n",
            },
            expected_checks=["python3 -m unittest discover -s tests"],
        ),
        EvalScenario(
            name="readme-only",
            category="README-only",
            instruction="Update README.md with installation and usage sections. Do not change code.",
            files={
                "README.md": "# Tiny Tool\n\nTODO: document this project.\n",
                "tool.py": "def run():\n    return 'ok'\n",
            },
            expected_checks=["README contains installation and usage sections"],
        ),
        EvalScenario(
            name="js-bugfix",
            category="JS bugfix",
            instruction="Fix the JavaScript sum bug and keep node tests passing.",
            files={
                "package.json": "{\"scripts\":{\"test\":\"node test.js\"},\"type\":\"commonjs\"}\n",
                "sum.js": "function sum(values) { return values.length; }\nmodule.exports = { sum };\n",
                "test.js": "const { sum } = require('./sum');\nif (sum([1,2,3]) !== 6) throw new Error('bad sum');\nconsole.log('ok');\n",
            },
            expected_checks=["npm test"],
        ),
        EvalScenario(
            name="no-test-project",
            category="no test project",
            instruction="Fix the config parser typo. There are no automated tests; explain the manual verification.",
            files={
                "config_parser.py": "def parse_enabled(value):\n    return str(value).lower() in {'true', 'yes', 'onn'}\n",
                "README.md": "# No Test Fixture\n",
            },
            expected_checks=["python3 -m py_compile config_parser.py"],
        ),
        EvalScenario(
            name="bad-web-source-repair",
            category="bad web source repair",
            instruction="Replace the broken data source URL in source_config.py with a documented working source and record the verification evidence.",
            files={
                "source_config.py": "SOURCE_URL = 'https://api.example.invalid/missing'\n",
                "README.md": "# Source Repair Fixture\n",
            },
            expected_checks=["fetch_url evidence required", "python3 -m py_compile source_config.py"],
        ),
        EvalScenario(
            name="large-command-output",
            category="large command output",
            instruction="Fix noisy.py so tests pass even when command output is very large.",
            files={
                "noisy.py": "def status_line(index):\n    return f'line {index}'\n",
                "tests/test_noisy.py": "import unittest\nfrom noisy import status_line\n\nclass NoisyTests(unittest.TestCase):\n    def test_final_line(self):\n        self.assertEqual(status_line(999), 'done 999')\n\nif __name__ == '__main__':\n    unittest.main()\n",
                "generate_output.py": "for i in range(5000):\n    print('line', i)\n",
            },
            expected_checks=["python3 generate_output.py", "python3 -m unittest discover -s tests"],
        ),
        EvalScenario(
            name="github-pr-artifact-export",
            category="GitHub PR artifact export",
            instruction="Make a minimal code change and prepare the job for PR artifact export mode.",
            files={
                "README.md": "# PR Export Fixture\n",
                "module.py": "VALUE = 'old'\n",
            },
            expected_checks=["git diff is non-empty", "artifact_mode=pr"],
            artifact_mode="pr",
        ),
    ]


def write_eval_report(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_eval_case(path: Path) -> EvalCaseResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"eval case must be an object: {path}")
    status = str(data.get("status") or "")
    success = bool(data.get("success", status.lower() in {"succeeded", "success", "passed"}))
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    verification = data.get("verification") if isinstance(data.get("verification"), dict) else {}
    return EvalCaseResult(
        name=str(data.get("name") or path.stem),
        status=status or ("succeeded" if success else "failed"),
        success=success,
        iterations=int_or_zero(data.get("iterations")),
        tool_calls=int_or_zero(data.get("tool_calls")),
        tokens=int_or_zero(data.get("tokens") or usage.get("total_tokens") or usage.get("tokens")),
        cost=float_or_zero(data.get("cost") or usage.get("cost")),
        failure_reason=str(data.get("failure_reason") or "") or None,
        verification_plan_failures=verification_failures(verification),
    )


def verification_failures(verification: dict[str, Any]) -> list[str]:
    fixes = verification.get("required_fixes") or verification.get("verification_plan_failures") or []
    if isinstance(fixes, str):
        return [fixes]
    if isinstance(fixes, list):
        return [str(fix) for fix in fixes if str(fix)]
    return []


def latest_usage_snapshot(steps: list[Any]) -> dict[str, Any]:
    for step in reversed(steps):
        content = step_content(step)
        usage = content.get("usage")
        if isinstance(usage, dict):
            return usage
    return {}


def latest_verification_step(steps: list[Any]) -> dict[str, Any]:
    for step in reversed(steps):
        content = step_content(step)
        if content.get("type") == "verification" or content.get("verification_plan") is not None:
            return {
                "passed": content.get("passed"),
                "reason": content.get("reason"),
                "required_fixes": content.get("required_fixes") or [],
                "verification_plan": content.get("verification_plan"),
                "evidence": content.get("evidence"),
            }
    return {}


def count_steps(steps: list[Any], step_type: str) -> int:
    return sum(1 for step in steps if step_content(step).get("type") == step_type)


def step_content(step: Any) -> dict[str, Any]:
    content = getattr(step, "content", step)
    return content if isinstance(content, dict) else {}


def status_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def float_or_zero(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def initialize_git_repo(path: Path) -> bool:
    git = shutil.which("git")
    if git is None:
        return False
    commands = [
        [git, "init"],
        [git, "config", "user.email", "docode-eval@example.test"],
        [git, "config", "user.name", "DoCode Eval"],
        [git, "add", "."],
        [git, "commit", "-m", "Initial eval fixture"],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if result.returncode != 0:
            return False
    return True
