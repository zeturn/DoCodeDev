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
from typing import Any, Literal
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
    failure_class: str | None = None
    failure_category: str | None = None
    infra_diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class EvalThresholds:
    min_success_rate: float | None = None
    max_average_tool_calls: float | None = None
    max_total_cost: float | None = None

    def to_dict(self) -> dict[str, float]:
        values: dict[str, float] = {}
        if self.min_success_rate is not None:
            values["min_success_rate"] = self.min_success_rate
        if self.max_average_tool_calls is not None:
            values["max_average_tool_calls"] = self.max_average_tool_calls
        if self.max_total_cost is not None:
            values["max_total_cost"] = self.max_total_cost
        return values

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "EvalThresholds":
        data = data or {}
        return cls(
            min_success_rate=optional_float(data.get("min_success_rate")),
            max_average_tool_calls=optional_float(data.get("max_average_tool_calls")),
            max_total_cost=optional_float(data.get("max_total_cost")),
        )


@dataclass(frozen=True, slots=True)
class EvalAssertion:
    passed: bool
    failures: list[str]
    thresholds: EvalThresholds

    @property
    def regression(self) -> bool:
        return not self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "regression": self.regression,
            "thresholds": self.thresholds.to_dict(),
            "threshold_failures": self.failures,
        }


@dataclass(frozen=True, slots=True)
class EvalComparison:
    previous_total: int
    previous_succeeded: int
    previous_success_rate: float
    success_rate_delta: float
    succeeded_delta: int
    avg_iterations_delta: float
    avg_tool_calls_delta: float
    avg_tokens_delta: float
    cost_delta: float
    newly_succeeded: list[str]
    newly_failed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvalModelSummary:
    model: str
    total: int
    succeeded: int
    success_rate: float
    avg_iterations: float
    avg_tool_calls: float
    avg_tokens: float
    cost: float
    main_failure: str | None
    comparison: EvalComparison | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "model": self.model,
            "total": self.total,
            "succeeded": self.succeeded,
            "success_rate": self.success_rate,
            "avg_iterations": self.avg_iterations,
            "avg_tool_calls": self.avg_tool_calls,
            "avg_tokens": self.avg_tokens,
            "cost": self.cost,
            "main_failure": self.main_failure,
        }
        if self.comparison is not None:
            data["comparison"] = self.comparison.to_dict()
        return data


@dataclass(frozen=True, slots=True)
class EvalMatrixReport:
    models: list[EvalModelSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": [model.to_dict() for model in self.models],
            "best_success_rate": max((model.success_rate for model in self.models), default=0.0),
            "total_models": len(self.models),
        }


@dataclass(frozen=True, slots=True)
class EvalCheck:
    type: Literal["command", "file_contains", "file_exists", "json_len_at_least", "evidence", "artifact"]
    command: str | None = None
    path: str | None = None
    contains: list[str] = field(default_factory=list)
    min_len: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.type}
        if self.command:
            data["command"] = self.command
        if self.path:
            data["path"] = self.path
        if self.contains:
            data["contains"] = self.contains
        if self.min_len is not None:
            data["min_len"] = self.min_len
        return data


@dataclass(frozen=True, slots=True)
class EvalScenario:
    name: str
    category: str
    instruction: str
    files: dict[str, str]
    expected_checks: list[EvalCheck]
    hints: dict[str, Any] | None = None
    artifact_mode: str = "patch"

    def to_manifest_entry(self, repo_dir: Path, git_initialized: bool) -> dict[str, Any]:
        entry = {
            "name": self.name,
            "category": self.category,
            "instruction": self.instruction,
            "repo_path": str(repo_dir),
            "repo_url": repo_dir.resolve().as_uri(),
            "artifact_mode": self.artifact_mode,
            "expected_checks": [check.to_dict() for check in self.expected_checks],
            "git_initialized": git_initialized,
        }
        if self.hints:
            entry["hints"] = self.hints
        return entry


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
    failure_classes: dict[str, int]
    failure_categories: dict[str, int]
    verification_plan_failures: dict[str, int]
    cases: list[EvalCaseResult]
    assertion: EvalAssertion | None = None
    comparison: EvalComparison | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "success_rate": self.success_rate,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "tokens": self.tokens,
            "cost": self.cost,
            "failure_reasons": self.failure_reasons,
            "failure_classes": self.failure_classes,
            "failure_categories": self.failure_categories,
            "verification_plan_failures": self.verification_plan_failures,
            "cases": [asdict(case) for case in self.cases],
        }
        if self.assertion is not None:
            data.update(self.assertion.to_dict())
        if self.comparison is not None:
            data["comparison"] = self.comparison.to_dict()
        return data


def load_eval_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError(f"eval manifest must contain a cases list: {path}")
    return data


def eval_case_result_from_job(case: dict[str, Any], job: Any, steps: list[Any]) -> dict[str, Any]:
    status = status_value(getattr(job, "status", "missing"))
    usage = latest_usage_snapshot(steps)
    verification = latest_verification_step(steps)
    failure_reason = getattr(job, "failure_reason", None)
    infra_diagnostics = collect_infra_diagnostics(steps, verification)
    failure_class, failure_category = classify_failure(status, failure_reason, steps, verification, infra_diagnostics)
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
        "failure_reason": failure_reason,
        "failure_class": failure_class,
        "infra_diagnostics": infra_diagnostics or None,
        "artifact_id": getattr(job, "artifact_id", None),
        "verification": verification,
    }
    if failure_category:
        result["failure_category"] = failure_category
    return result


def write_eval_case_result(result: dict[str, Any], results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    name = str(result.get("name") or "case").replace("/", "-")
    path = results_dir / f"{name}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def run_eval(fixtures_dir: Path, thresholds: EvalThresholds | None = None) -> EvalReport:
    cases = [load_eval_case(path) for path in sorted(fixtures_dir.glob("*.json"))]
    total = len(cases)
    succeeded = sum(1 for case in cases if case.success)
    failed = total - succeeded
    report = EvalReport(
        total=total,
        succeeded=succeeded,
        failed=failed,
        success_rate=(succeeded / total) if total else 0.0,
        iterations=sum(case.iterations for case in cases),
        tool_calls=sum(case.tool_calls for case in cases),
        tokens=sum(case.tokens for case in cases),
        cost=sum(case.cost for case in cases),
        failure_reasons=count_values(case.failure_reason for case in cases if case.failure_reason),
        failure_classes=count_values(case.failure_class for case in cases if case.failure_class),
        failure_categories=count_values(case.failure_category for case in cases if case.failure_category),
        verification_plan_failures=count_values(failure for case in cases for failure in case.verification_plan_failures),
        cases=cases,
    )
    if thresholds is None or not thresholds.to_dict():
        return report
    return with_eval_assertion(report, thresholds)


def with_eval_assertion(report: EvalReport, thresholds: EvalThresholds) -> EvalReport:
    return EvalReport(
        total=report.total,
        succeeded=report.succeeded,
        failed=report.failed,
        success_rate=report.success_rate,
        iterations=report.iterations,
        tool_calls=report.tool_calls,
        tokens=report.tokens,
        cost=report.cost,
        failure_reasons=report.failure_reasons,
        failure_classes=report.failure_classes,
        failure_categories=report.failure_categories,
        verification_plan_failures=report.verification_plan_failures,
        cases=report.cases,
        assertion=assert_eval_report(report, thresholds),
        comparison=report.comparison,
    )


def assert_eval_report(report: EvalReport, thresholds: EvalThresholds) -> EvalAssertion:
    failures: list[str] = []
    if thresholds.min_success_rate is not None and report.success_rate < thresholds.min_success_rate:
        failures.append(f"success_rate {report.success_rate:.3f} < min_success_rate {thresholds.min_success_rate:.3f}")
    average_tool_calls = (report.tool_calls / report.total) if report.total else 0.0
    if thresholds.max_average_tool_calls is not None and average_tool_calls > thresholds.max_average_tool_calls:
        failures.append(
            f"average_tool_calls {average_tool_calls:.3f} > max_average_tool_calls {thresholds.max_average_tool_calls:.3f}"
        )
    if thresholds.max_total_cost is not None and report.cost > thresholds.max_total_cost:
        failures.append(f"total_cost {report.cost:.6f} > max_total_cost {thresholds.max_total_cost:.6f}")
    return EvalAssertion(passed=not failures, failures=failures, thresholds=thresholds)


def load_eval_report(path: Path) -> EvalReport:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"eval report must be an object: {path}")
    cases = [eval_case_from_mapping(item) for item in data.get("cases", []) if isinstance(item, dict)]
    assertion = None
    if isinstance(data.get("thresholds"), dict):
        assertion = EvalAssertion(
            passed=not bool(data.get("regression")),
            failures=[str(failure) for failure in data.get("threshold_failures", [])],
            thresholds=EvalThresholds.from_mapping(data.get("thresholds")),
        )
    comparison = None
    if isinstance(data.get("comparison"), dict):
        comparison = eval_comparison_from_mapping(data["comparison"])
    return EvalReport(
        total=int_or_zero(data.get("total")),
        succeeded=int_or_zero(data.get("succeeded")),
        failed=int_or_zero(data.get("failed")),
        success_rate=float_or_zero(data.get("success_rate")),
        iterations=int_or_zero(data.get("iterations")),
        tool_calls=int_or_zero(data.get("tool_calls")),
        tokens=int_or_zero(data.get("tokens")),
        cost=float_or_zero(data.get("cost")),
        failure_reasons=dict(data.get("failure_reasons") or {}),
        failure_classes=dict(data.get("failure_classes") or {}),
        failure_categories=dict(data.get("failure_categories") or {}),
        verification_plan_failures=dict(data.get("verification_plan_failures") or {}),
        cases=cases,
        assertion=assertion,
        comparison=comparison,
    )


def with_eval_comparison(report: EvalReport, previous: EvalReport) -> EvalReport:
    return EvalReport(
        total=report.total,
        succeeded=report.succeeded,
        failed=report.failed,
        success_rate=report.success_rate,
        iterations=report.iterations,
        tool_calls=report.tool_calls,
        tokens=report.tokens,
        cost=report.cost,
        failure_reasons=report.failure_reasons,
        failure_classes=report.failure_classes,
        failure_categories=report.failure_categories,
        verification_plan_failures=report.verification_plan_failures,
        cases=report.cases,
        assertion=report.assertion,
        comparison=compare_eval_reports(report, previous),
    )


def compare_eval_reports(current: EvalReport, previous: EvalReport) -> EvalComparison:
    current_cases = {case.name: case.success for case in current.cases}
    previous_cases = {case.name: case.success for case in previous.cases}
    shared_names = sorted(set(current_cases) & set(previous_cases))
    newly_succeeded = [name for name in shared_names if current_cases[name] and not previous_cases[name]]
    newly_failed = [name for name in shared_names if previous_cases[name] and not current_cases[name]]
    return EvalComparison(
        previous_total=previous.total,
        previous_succeeded=previous.succeeded,
        previous_success_rate=previous.success_rate,
        success_rate_delta=current.success_rate - previous.success_rate,
        succeeded_delta=current.succeeded - previous.succeeded,
        avg_iterations_delta=average(current.iterations, current.total) - average(previous.iterations, previous.total),
        avg_tool_calls_delta=average(current.tool_calls, current.total) - average(previous.tool_calls, previous.total),
        avg_tokens_delta=average(current.tokens, current.total) - average(previous.tokens, previous.total),
        cost_delta=current.cost - previous.cost,
        newly_succeeded=newly_succeeded,
        newly_failed=newly_failed,
    )


def eval_comparison_from_mapping(data: dict[str, Any]) -> EvalComparison:
    return EvalComparison(
        previous_total=int_or_zero(data.get("previous_total")),
        previous_succeeded=int_or_zero(data.get("previous_succeeded")),
        previous_success_rate=float_or_zero(data.get("previous_success_rate")),
        success_rate_delta=float_or_zero(data.get("success_rate_delta")),
        succeeded_delta=int_or_zero(data.get("succeeded_delta")),
        avg_iterations_delta=float_or_zero(data.get("avg_iterations_delta")),
        avg_tool_calls_delta=float_or_zero(data.get("avg_tool_calls_delta")),
        avg_tokens_delta=float_or_zero(data.get("avg_tokens_delta")),
        cost_delta=float_or_zero(data.get("cost_delta")),
        newly_succeeded=[str(name) for name in data.get("newly_succeeded", [])],
        newly_failed=[str(name) for name in data.get("newly_failed", [])],
    )


def summarize_eval_matrix(reports: dict[str, EvalReport], previous_reports: dict[str, EvalReport] | None = None) -> EvalMatrixReport:
    previous_reports = previous_reports or {}
    summaries = [
        EvalModelSummary(
            model=model,
            total=report.total,
            succeeded=report.succeeded,
            success_rate=report.success_rate,
            avg_iterations=average(report.iterations, report.total),
            avg_tool_calls=average(report.tool_calls, report.total),
            avg_tokens=average(report.tokens, report.total),
            cost=report.cost,
            main_failure=main_failure(report),
            comparison=compare_eval_reports(report, previous_reports[model]) if model in previous_reports else None,
        )
        for model, report in sorted(reports.items())
    ]
    return EvalMatrixReport(models=summaries)


def average(value: int | float, total: int) -> float:
    return (float(value) / total) if total else 0.0


def main_failure(report: EvalReport) -> str | None:
    for counts in (report.failure_categories, report.failure_classes, report.failure_reasons, report.verification_plan_failures):
        winner = most_common_key(counts)
        if winner:
            return winner
    return None


def most_common_key(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


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
    deadline = time.monotonic() + 20.0
    last_error = "not checked"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"git daemon exited with code {process.returncode}")
        try:
            result = subprocess.run([git, "ls-remote", url, "HEAD"], capture_output=True, text=True, check=False, timeout=5)
        except subprocess.TimeoutExpired:
            last_error = "git ls-remote timed out"
            time.sleep(0.25)
            continue
        if result.returncode == 0:
            return
        last_error = (result.stderr or result.stdout or "").strip()
        time.sleep(0.25)
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
            expected_checks=[EvalCheck(type="command", command="python3 -m unittest discover -s tests")],
            hints={
                "target_files": ["calculator.py"],
                "expected_behavior": "retry_count(3) should return 3 by returning attempts unchanged.",
                "suggested_commands": ["python3 -m unittest discover -s tests"],
            },
        ),
        EvalScenario(
            name="python-cli",
            category="Python CLI",
            instruction="Turn cli.py into a working command line tool that prints a greeting for --name.",
            files={
                "cli.py": "import argparse\n\ndef main():\n    parser = argparse.ArgumentParser()\n    parser.add_argument('--name', default='world')\n    args = parser.parse_args()\n    print('TODO')\n\nif __name__ == '__main__':\n    main()\n",
                "README.md": "# Python CLI Fixture\nRun `python3 cli.py --name Ada`.\n",
            },
            expected_checks=[EvalCheck(type="command", command="python3 cli.py --name Ada")],
            hints={
                "target_files": ["cli.py"],
                "expected_behavior": "python3 cli.py --name Ada should print a greeting that includes Ada.",
                "suggested_commands": ["python3 cli.py --name Ada"],
            },
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
            expected_checks=[
                EvalCheck(type="command", command="python3 crawler.py"),
                EvalCheck(type="json_len_at_least", path="data/output.json", min_len=2),
            ],
            hints={
                "target_files": ["crawler.py"],
                "expected_behavior": "data/output.json should contain at least two records parsed from fixtures/source.html.",
                "suggested_commands": ["python3 crawler.py", "python3 -c \"import json; assert len(json.load(open('data/output.json'))) >= 2\""],
            },
        ),
        EvalScenario(
            name="api-adapter",
            category="API adapter",
            instruction="Implement client.parse_items_response so it extracts item names from a JSON API response.",
            files={
                "client.py": "import json\n\ndef parse_items_response(text):\n    data = json.loads(text)\n    return []\n",
                "tests/test_client.py": "import unittest\nfrom client import parse_items_response\n\nclass ClientTests(unittest.TestCase):\n    def test_parse_items(self):\n        payload = '{\"items\":[{\"name\":\"north\"},{\"name\":\"south\"}]}'\n        self.assertEqual(parse_items_response(payload), ['north', 'south'])\n\nif __name__ == '__main__':\n    unittest.main()\n",
            },
            expected_checks=[EvalCheck(type="command", command="python3 -m unittest discover -s tests")],
            hints={
                "target_files": ["client.py"],
                "expected_behavior": "parse_items_response should return item names from the API response in order.",
                "suggested_commands": ["python3 -m unittest discover -s tests"],
            },
        ),
        EvalScenario(
            name="readme-only",
            category="README-only",
            instruction="Update README.md with installation and usage sections. Do not change code.",
            files={
                "README.md": "# Tiny Tool\n\nTODO: document this project.\n",
                "tool.py": "def run():\n    return 'ok'\n",
            },
            expected_checks=[EvalCheck(type="file_contains", path="README.md", contains=["Installation", "Usage"])],
            hints={
                "target_files": ["README.md"],
                "expected_behavior": "README should include installation and usage sections.",
                "suggested_commands": ["README contains installation and usage sections"],
            },
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
            expected_checks=[EvalCheck(type="command", command="npm test")],
            hints={
                "target_files": ["sum.js"],
                "expected_behavior": "sum([1,2,3]) should return 6.",
                "suggested_commands": ["npm test"],
            },
        ),
        EvalScenario(
            name="no-test-project",
            category="no test project",
            instruction="Fix the config parser typo. There are no automated tests; explain the manual verification.",
            files={
                "config_parser.py": "def parse_enabled(value):\n    return str(value).lower() in {'true', 'yes', 'onn'}\n",
                "README.md": "# No Test Fixture\n",
            },
            expected_checks=[EvalCheck(type="command", command="python3 -m py_compile config_parser.py")],
            hints={
                "target_files": ["config_parser.py"],
                "expected_behavior": "parse_enabled should recognize 'on' as enabled, not the typo 'onn'.",
                "suggested_commands": ["python3 -m py_compile config_parser.py"],
            },
        ),
        EvalScenario(
            name="bad-web-source-repair",
            category="bad web source repair",
            instruction="Replace the broken data source URL in source_config.py with a documented working source and record the verification evidence.",
            files={
                "source_config.py": "SOURCE_URL = 'https://api.example.invalid/missing'\n",
                "README.md": "# Source Repair Fixture\n",
            },
            expected_checks=[
                EvalCheck(type="evidence"),
                EvalCheck(type="command", command="python3 -m py_compile source_config.py"),
            ],
            hints={
                "target_files": ["source_config.py"],
                "expected_behavior": "SOURCE_URL should point to a documented working source and include verification evidence.",
                "suggested_commands": ["fetch_url evidence required", "python3 -m py_compile source_config.py"],
            },
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
            expected_checks=[
                EvalCheck(type="command", command="python3 generate_output.py"),
                EvalCheck(type="command", command="python3 -m unittest discover -s tests"),
            ],
            hints={
                "target_files": ["noisy.py"],
                "expected_behavior": "status_line(999) should return 'done 999'.",
                "suggested_commands": ["python3 generate_output.py", "python3 -m unittest discover -s tests"],
            },
        ),
        EvalScenario(
            name="github-pr-artifact-export",
            category="GitHub PR artifact export",
            instruction="Make a minimal code change and prepare the job for PR artifact export mode.",
            files={
                "README.md": "# PR Export Fixture\n",
                "module.py": "VALUE = 'old'\n",
            },
            expected_checks=[EvalCheck(type="artifact", path="pr")],
            hints={
                "target_files": ["module.py"],
                "expected_behavior": "Make a minimal code change suitable for PR artifact export.",
                "suggested_commands": ["git diff is non-empty", "artifact_mode=pr"],
            },
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
    return eval_case_from_mapping(data, default_name=path.stem)


def eval_case_from_mapping(data: dict[str, Any], *, default_name: str = "case") -> EvalCaseResult:
    status = str(data.get("status") or "")
    success = bool(data.get("success", status.lower() in {"succeeded", "success", "passed"}))
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    verification = data.get("verification") if isinstance(data.get("verification"), dict) else {}
    failure_class = str(data.get("failure_class") or "") or None
    failure_category = str(data.get("failure_category") or "") or None
    infra_diagnostics = data.get("infra_diagnostics") if isinstance(data.get("infra_diagnostics"), dict) else None
    failure_class, failure_category = normalize_loaded_failure_class(
        failure_class,
        failure_category,
        failure_reason=str(data.get("failure_reason") or "") or None,
        verification_failures=verification_failures(verification),
        infra_diagnostics=infra_diagnostics,
    )
    return EvalCaseResult(
        name=str(data.get("name") or default_name),
        status=status or ("succeeded" if success else "failed"),
        success=success,
        iterations=int_or_zero(data.get("iterations")),
        tool_calls=int_or_zero(data.get("tool_calls")),
        tokens=int_or_zero(data.get("tokens") or usage.get("total_tokens") or usage.get("tokens")),
        cost=float_or_zero(data.get("cost") or usage.get("cost")),
        failure_reason=str(data.get("failure_reason") or "") or None,
        verification_plan_failures=verification_failures(verification),
        failure_class=failure_class,
        failure_category=failure_category,
        infra_diagnostics=infra_diagnostics,
    )


def normalize_loaded_failure_class(
    failure_class: str | None,
    failure_category: str | None,
    *,
    failure_reason: str | None,
    verification_failures: list[str],
    infra_diagnostics: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    probe = (infra_diagnostics or {}).get("workspace_probe")
    if (
        failure_class == "infra_failed"
        and failure_category == "workspace_inconsistent"
        and isinstance(probe, dict)
        and probe.get("passed") is True
    ):
        if verification_failures:
            return "verifier_failed", "verifier_plan_failed"
        if failure_reason == "max_consecutive_failures_exceeded":
            return "agent_failed", "max_consecutive_failures_exceeded"
    return failure_class, failure_category


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
                "smoke_result": content.get("smoke_result"),
            }
    return {}


def classify_failure(
    status: str,
    failure_reason: str | None,
    steps: list[Any],
    verification: dict[str, Any],
    infra_diagnostics: dict[str, Any],
) -> tuple[str | None, str | None]:
    if status == "succeeded":
        return None, None
    reason = (failure_reason or "").lower()
    step_contents = [step_content(step) for step in steps]
    combined = " ".join(str(content) for content in step_contents).lower()
    provider_combined = provider_failure_text(reason, step_contents)

    probe = infra_diagnostics.get("workspace_probe") if isinstance(infra_diagnostics.get("workspace_probe"), dict) else {}
    if reason.startswith("infrastructure_failed:") or probe.get("passed") is False:
        return "infra_failed", infra_category_from_reason(reason) or "workspace_inconsistent"
    if "post /api/projects failed" in reason or "failed to create project" in reason:
        return "infra_failed", "provider_call_failed"
    if "server disconnected" in reason or "connection refused" in reason or "connection reset" in reason:
        return "infra_failed", "provider_call_failed"
    if "model_not_found" in provider_combined or "model_not_available" in provider_combined or "model unavailable" in provider_combined:
        return "model_unavailable", "model_catalog_mismatch"
    if "llm_auth_failed" in reason or "401 unauthorized" in provider_combined or "403 forbidden" in provider_combined:
        return "model_unavailable", "provider_auth_failed"
    if "llm_provider_unavailable" in reason or "llm_provider_unavailable" in provider_combined:
        return "model_unavailable", provider_unavailable_category(reason, provider_combined)
    provider_category = provider_unavailable_category(reason, provider_combined)
    if provider_category != "provider_call_failed":
        return "model_unavailable", provider_category
    if reason.startswith("apicred_authorize_failed"):
        return "model_unavailable", "provider_call_failed"
    if "provider_call_failed" in combined:
        return "model_unavailable", "provider_call_failed"
    if "max_llm_tokens_exceeded" in reason:
        return "budget_exceeded", "max_llm_tokens_exceeded"
    if "max_iterations_exceeded" in reason:
        return "budget_exceeded", "max_iterations_exceeded"
    if "max_tool_calls_exceeded" in reason:
        return "budget_exceeded", "max_tool_calls_exceeded"
    if "max_consecutive_failures_exceeded" in reason:
        if "unsupported decision type" in combined:
            return "agent_failed", "parser_failed"
        if verification.get("passed") is False:
            return "verifier_failed", "verifier_plan_failed"
        return "agent_failed", "max_consecutive_failures_exceeded"
    if "unsupported decision type" in combined:
        return "agent_failed", "parser_failed"
    if verification.get("passed") is False or verification.get("required_fixes"):
        return "verifier_failed", "verifier_plan_failed"
    if "workspace_diagnostic" in infra_diagnostics:
        return "verifier_failed", "workspace_inconsistent"
    return "agent_failed", None


def infra_category_from_reason(reason: str) -> str | None:
    if ":" not in reason:
        return None
    category = reason.split(":", 1)[1].strip().replace(" ", "_")
    return category or None


def provider_unavailable_category(reason: str, combined: str) -> str:
    text = f"{reason}\n{combined}".lower()
    for category in (
        "provider_upstream_unavailable",
        "provider_rate_limited",
        "provider_timeout",
        "provider_network_error",
        "model_catalog_mismatch",
        "provider_auth_failed",
    ):
        if category in text:
            return category
    if "no_upstream_capacity" in text or "502 bad gateway" in text or "503 service unavailable" in text or "504 gateway timeout" in text:
        return "provider_upstream_unavailable"
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return "provider_rate_limited"
    if "timeout" in text or "timed out" in text:
        return "provider_timeout"
    if "connection refused" in text or "connection reset" in text or "server disconnected" in text or "all connection attempts failed" in text:
        return "provider_network_error"
    return "provider_call_failed"


def provider_failure_text(reason: str, step_contents: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    if provider_failure_reason_like(reason):
        parts.append(reason)
    for content in step_contents:
        if provider_failure_step_like(content):
            parts.append(str(content))
    return " ".join(parts).lower()


def provider_failure_reason_like(reason: str) -> bool:
    return any(
        marker in reason
        for marker in (
            "llm_",
            "apicred_authorize_failed",
            "provider_",
            "model_not_found",
            "model_not_available",
            "model unavailable",
            "no_upstream_capacity",
        )
    )


def provider_failure_step_like(content: dict[str, Any]) -> bool:
    content_type = str(content.get("type") or "").lower()
    if content_type in {"llm_error", "apicred_authorize_failed"}:
        return True
    reason = str(content.get("reason") or "").lower()
    return provider_failure_reason_like(reason)


def collect_infra_diagnostics(steps: list[Any], verification: dict[str, Any]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for step in steps:
        content = step_content(step)
        if content.get("type") == "workspace_probe":
            diagnostics["workspace_probe"] = {
                "passed": content.get("passed"),
                "category": content.get("category"),
                "diagnostics": content.get("diagnostics"),
            }
        metadata = content.get("metadata") if isinstance(content.get("metadata"), dict) else {}
        if "workspace_diagnostic" in metadata:
            diagnostics["workspace_diagnostic"] = metadata["workspace_diagnostic"]
    smoke = verification.get("smoke_result") if isinstance(verification.get("smoke_result"), dict) else {}
    smoke_metadata = smoke.get("metadata") if isinstance(smoke.get("metadata"), dict) else {}
    if "workspace_diagnostic" in smoke_metadata:
        diagnostics["workspace_diagnostic"] = smoke_metadata["workspace_diagnostic"]
    return diagnostics


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


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


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
