from __future__ import annotations

import hashlib
import json
import os
import statistics
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import IsolatedAsyncioTestCase, skipUnless

from docode.agent.loop import CodingAgentLoop
from docode.agent.quality_gate import QualityGate
from docode.agent.stop_policy import StopPolicy
from docode.agent.verifier import CodingVerifier
from docode.artifacts.exporter import ArtifactExporter
from docode.config import load_config
from docode.dobox.client import DoBoxClient
from docode.dobox.tools import DoBoxTools
from docode.runtime.smoke import check_http_health, ensure_dobox_smoke_token
from docode.storage.models import CodingJob, new_id

from tests.crawler_benchmark_v1.definitions import (
    BENCHMARK_BRANCH,
    CASES,
    LOCAL_BASE_URL,
    REAL_SOURCE_URL,
    RUNTIME_COMMIT,
    RUNTIME_TAG,
    CrawlerCase,
)
from tests.crawler_benchmark_v1.harness import (
    EDIT_TOOLS,
    command_results,
    materialize_workspace,
    sanitize,
    summarize_steps,
    validate_controlled_payload,
    validate_live_payload,
    variant_source,
)
from tests.test_real_llm_smoke import build_real_llm_or_skip
from tests.test_smoke_readme_job import RecordingRepository


REAL_ENABLED = os.getenv("DOCODE_CRAWLER_BENCHMARK_V1", "").lower() in {"1", "true", "yes", "on"}
RUNS_PER_CASE = 3
RESULT_ROOT = Path(os.getenv("DOCODE_CRAWLER_BENCHMARK_RESULT_ROOT", ".docode/evals/crawler-benchmark-v1-d9579ed"))
RESULT_PATH = RESULT_ROOT / "results.json"
RUN_ROOT = RESULT_ROOT / "runs"
TRACE_ROOT = RESULT_ROOT / "traces"
DEFINITION_FILES = ("definitions.py", "fixture_service.py", "harness.py", "reference_solutions.py", "test_deterministic.py", "test_real_benchmark.py")


def definition_digest() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in DEFINITION_FILES:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def latest_step_value(steps: list[Any], *, kind: str | None = None, content_type: str | None = None, key: str = "passed") -> Any:
    matches = []
    for step in steps:
        if kind is not None and step.kind != kind:
            continue
        if content_type is not None and step.content.get("type") != content_type:
            continue
        matches.append(step.content)
    return matches[-1].get(key) if matches else None


def exact_required_command_evidence(steps: list[Any], required_commands: tuple[str, str]) -> list[dict[str, Any]]:
    observed: dict[str, list[int | None]] = {command: [] for command in required_commands}
    for step in steps:
        content = step.content
        if content.get("type") != "tool_result" or content.get("tool") != "run_command":
            continue
        command = str((content.get("metadata") or {}).get("command") or "")
        if command in observed:
            observed[command].append(content.get("exit_code"))
    return [
        {"command": command, "exit_codes": observed[command], "passed": any(code == 0 for code in observed[command])}
        for command in required_commands
    ]


def source_inspected_before_first_edit(steps: list[Any], case: CrawlerCase) -> bool:
    first_edit = min(
        (
            step.step_index
            for step in steps
            if step.content.get("type") == "tool_result"
            and step.content.get("tool") in EDIT_TOOLS
            and step.content.get("exit_code") == 0
        ),
        default=None,
    )
    marker = case.source_path if not case.controlled else case.source_path.split("?", 1)[0]
    for step in steps:
        if first_edit is not None and step.step_index >= first_edit:
            break
        content = step.content
        if content.get("type") != "tool_result" or content.get("exit_code") != 0:
            continue
        metadata_text = json.dumps(content.get("metadata") or {}, default=str)
        output = str(content.get("output") or content.get("summary") or "")
        if marker in metadata_text or (content.get("tool") == "fetch_url" and marker in output):
            return True
    return False


def classify_run(
    *,
    status: str,
    failure_reason: str | None,
    functionally_correct: bool,
    required_commands_passed: bool,
    source_unavailable: bool,
    summary: dict[str, Any],
    verifier_result: Any,
    quality_result: Any,
    independent_failures: list[str],
) -> str:
    if source_unavailable:
        return "source_unavailable"
    if status == "succeeded" and functionally_correct and required_commands_passed:
        return "success"
    if status == "succeeded" and functionally_correct:
        return "verification_protocol_failure"
    if status == "succeeded" and independent_failures:
        return "code_generation_failure"
    lowered = (failure_reason or "").lower()
    if any(marker in lowered for marker in ("provider", "transport", "llm_auth", "rate_limit", "connection")):
        return "provider_or_transport_failure"
    if functionally_correct:
        if quality_result is False:
            return "quality_gate_false_positive"
        if verifier_result is False:
            return "verifier_false_negative"
        return "runtime_failed_after_functional_verification"
    if "timeout" in lowered or "max_runtime" in lowered or "max_tool_calls" in lowered:
        return "timeout_or_budget"
    if summary.get("successful_edits", 0) == 0:
        return "workspace_comprehension_failure" if summary.get("successful_reads", 0) else "failed_before_edit"
    if independent_failures and summary.get("repair_actions", 0):
        return "repair_loop"
    if independent_failures:
        return "code_generation_failure"
    return "unknown"


def aggregate_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    strict = lambda run: bool(run["strict_success"])
    successful = [run for run in runs if strict(run)]
    cases = [case.name for case in CASES]
    by_case = {name: sorted((run for run in runs if run["case"] == name), key=lambda item: item["run"]) for name in cases}
    first_runs = [items[0] for items in by_case.values() if items]
    complete_cases = [items for items in by_case.values() if len(items) == RUNS_PER_CASE]
    repaired_successes = [run for run in successful if run["repair_actions"] > 0]
    return {
        "sample_size": len(runs),
        "complete_cases": len(complete_cases),
        "overall_run_success_rate": len(successful) / len(runs),
        "pass_at_1": sum(strict(run) for run in first_runs) / len(first_runs) if first_runs else 0.0,
        "pass_at_3": sum(any(strict(run) for run in items) for items in complete_cases) / len(complete_cases) if complete_cases else None,
        "functional_correctness_rate": sum(run["functionally_correct"] for run in runs) / len(runs),
        "median_iterations": statistics.median(run["iterations"] for run in runs),
        "median_tool_calls": statistics.median(run["tool_calls"] for run in runs),
        "successful_runs_requiring_repair": len(repaired_successes) / len(successful) if successful else 0.0,
        "failed_before_any_edit": sum(run["status"] != "succeeded" and run["successful_edits"] == 0 for run in runs) / len(runs),
        "functional_but_runtime_failed": sum(run["status"] != "succeeded" and run["functionally_correct"] for run in runs) / len(runs),
        "source_inspected_before_edit": sum(run["source_inspected_before_edit"] for run in runs) / len(runs),
        "source_unavailable_runs": sum(run["category"] == "source_unavailable" for run in runs),
    }


@skipUnless(REAL_ENABLED, "set DOCODE_CRAWLER_BENCHMARK_V1=1 to run the frozen crawler benchmark")
class RealCrawlerBenchmarkV1Tests(IsolatedAsyncioTestCase):
    async def test_three_real_runs_per_case(self) -> None:
        config = await self._real_dobox_config()
        client = DoBoxClient(config.dobox_base_url, config.dobox_token)
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        TRACE_ROOT.mkdir(parents=True, exist_ok=True)
        frozen_digest = definition_digest()
        runs, invalid_attempts = self._load_existing(frozen_digest)

        completed_keys = {(run["case"], run["run"]) for run in runs}
        for run_number in range(1, RUNS_PER_CASE + 1):
            for case in CASES:
                if (case.name, run_number) in completed_keys:
                    continue
                try:
                    result = await self._run_case(client, case, run_number)
                except Exception as exc:
                    invalid_attempts.append(
                        sanitize(
                            {
                                "case": case.name,
                                "run": run_number,
                                "recorded_at": datetime.now(timezone.utc).isoformat(),
                                "reason": f"harness exception: {type(exc).__name__}: {exc}",
                            }
                        )
                    )
                    self._write_results(runs, invalid_attempts, frozen_digest)
                    raise
                runs.append(sanitize(result))
                completed_keys.add((case.name, run_number))
                self._write_run_summary(result)
                self._write_results(runs, invalid_attempts, frozen_digest)

        self.assertEqual(len(runs), len(CASES) * RUNS_PER_CASE)
        self.assertEqual({(run["case"], run["run"]) for run in runs}, {(case.name, number) for case in CASES for number in range(1, 4)})

    async def _run_case(self, client: DoBoxClient, case: CrawlerCase, run_number: int) -> dict[str, Any]:
        with TemporaryDirectory() as tmp:
            local_root = Path(tmp)
            fixture = materialize_workspace(case, local_root / "fixture")
            initial_files = {path.relative_to(fixture).as_posix() for path in fixture.rglob("*") if path.is_file()}
            repo = RecordingRepository()
            network_mode = "no_internet" if case.controlled else "project"
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="crawler-benchmark-v1",
                    instruction=case.instruction,
                    provider="deepseek",
                    model="deepseek-chat",
                    quality="balanced",
                    max_iterations=36,
                    max_runtime_seconds=900,
                    max_consecutive_failures=10,
                    max_tool_calls=80,
                    artifact_mode="patch",
                    sandbox_network_mode=network_mode,
                )
            )
            llm = await build_real_llm_or_skip(self, job)
            project = await client.create_project(
                name=f"crawler-v1-{case.name}-r{run_number}-{new_id('sample')}",
                network_mode=network_mode,
            )
            session = await client.create_agent_session(project.project_id, name=f"crawler-v1-{case.name}-r{run_number}")
            try:
                await self._seed_fixture(client, project.project_id, session.session_id, fixture)
                await self._ensure_python(client, project.project_id, session.session_id)
                if case.controlled:
                    await self._start_fixture_service(client, project.project_id, session.session_id, case)
                tools = DoBoxTools(
                    client,
                    project.project_id,
                    agent_session_id=session.session_id,
                    command_timeout_seconds=180,
                    output_limit_bytes=250_000,
                    command_overrides={"test": case.required_commands[0]},
                )
                job = await repo.update_job(
                    job.id,
                    dobox_project_id=project.project_id,
                    dobox_sandbox_id=project.sandbox_id,
                    dobox_agent_session_id=session.session_id,
                )
                loop = CodingAgentLoop(
                    llm=llm,
                    tools=tools,
                    verifier=CodingVerifier(),
                    repository=repo,
                    exporter=ArtifactExporter(
                        local_root / "artifacts",
                        repo,
                        workspace_archive_provider=lambda: client.archive_workspace(project.project_id, agent_session_id=session.session_id),
                        workspace_file_reader=lambda path: client.read_file(project.project_id, path, agent_session_id=session.session_id),
                    ),
                    stop_policy=StopPolicy(max_iterations=36, max_runtime_seconds=900, max_consecutive_failures=10, max_tool_calls=80),
                    quality_gate=QualityGate(),
                )
                completed = await loop.run(job)
                steps = await repo.list_steps(job.id)
                summary = summarize_steps(steps, initial_files)
                independent_failures, independent_results, source_unavailable = await self._inspect_final_workspace(
                    client, project.project_id, session.session_id, case
                )
                functionally_correct = not independent_failures and not source_unavailable
                status_result = await tools.git_status()
                artifacts = await repo.list_artifacts(job.id)
                verifier_result = latest_step_value(steps, kind="verifier")
                quality_result = latest_step_value(steps, content_type="quality_gate")
                required_evidence = exact_required_command_evidence(steps, case.required_commands)
                required_commands_passed = all(item["passed"] for item in required_evidence)
                source_inspected = source_inspected_before_first_edit(steps, case)
                category = classify_run(
                    status=completed.status.value,
                    failure_reason=completed.failure_reason,
                    functionally_correct=functionally_correct,
                    required_commands_passed=required_commands_passed,
                    source_unavailable=source_unavailable,
                    summary=summary,
                    verifier_result=verifier_result,
                    quality_result=quality_result,
                    independent_failures=independent_failures,
                )
                trace_path = TRACE_ROOT / f"{case.name}-run{run_number}-{completed.id}.json"
                trace = {
                    "case": case.name,
                    "run": run_number,
                    "provider": completed.provider or job.provider,
                    "model": completed.model or job.model,
                    "job_id": completed.id,
                    "dobox_project_id": project.project_id,
                    "dobox_sandbox_id": project.sandbox_id,
                    "status": completed.status.value,
                    "failure_reason": completed.failure_reason,
                    "metrics": summary,
                    "independent_results": independent_results,
                    "independent_failures": independent_failures,
                    "git_status": status_result.output,
                    "steps": [{"index": step.step_index, "kind": step.kind, "content": step.content} for step in steps],
                }
                trace_path.write_text(json.dumps(sanitize(trace), indent=2, default=str), encoding="utf-8")
                return {
                    "case": case.name,
                    "title": case.title,
                    "language": "Python/HTTP",
                    "mode": "real_deepseek_real_dobox",
                    "run": run_number,
                    "network_mode": network_mode,
                    "status": completed.status.value,
                    "failure_reason": completed.failure_reason,
                    **summary,
                    "verifier_result": verifier_result,
                    "quality_gate_result": quality_result,
                    "artifact_export_result": bool(artifacts and completed.artifact_id),
                    "changed_files": [line[3:].strip().replace("\\", "/") for line in status_result.output.splitlines() if len(line) >= 4],
                    "functionally_correct": functionally_correct,
                    "required_commands_passed": required_commands_passed,
                    "required_command_evidence": required_evidence,
                    "source_inspected_before_edit": source_inspected,
                    "strict_success": completed.status.value == "succeeded" and functionally_correct and required_commands_passed,
                    "independent_failures": independent_failures,
                    "independent_results": independent_results,
                    "category": category,
                    "short_reason": completed.failure_reason or completed.result_summary or category,
                    "trace_path": trace_path.as_posix(),
                    "commands": command_results(steps),
                }
            finally:
                await client.delete_project(project.project_id)

    async def _real_dobox_config(self):
        config = load_config()
        ok, detail = await check_http_health(config.dobox_base_url.rstrip("/") + "/health")
        if not ok:
            self.skipTest(f"DoBox unavailable at {config.dobox_base_url}: {detail}")
        token, token_check = await ensure_dobox_smoke_token(config)
        if token_check.status != "passed" or not token:
            self.skipTest(f"DoBox authentication failed: {token_check.detail}")
        config.dobox_token = token
        return config

    async def _seed_fixture(self, client: DoBoxClient, project_id: str, session_id: str, fixture: Path) -> None:
        for path in sorted(fixture.rglob("*")):
            if path.is_file():
                await client.write_file(
                    project_id,
                    path.relative_to(fixture).as_posix(),
                    path.read_text(encoding="utf-8"),
                    agent_session_id=session_id,
                )
        result = await client.run_command(
            project_id,
            ["sh", "-lc", "git init -b main && git config user.email crawler-eval@example.test && git config user.name 'Crawler Eval' && git add . && git commit -m 'Seed crawler benchmark'"],
            timeout_sec=60,
            agent_session_id=session_id,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"fixture git init failed: {result.output[-1200:]}")

    async def _ensure_python(self, client: DoBoxClient, project_id: str, session_id: str) -> None:
        result = await client.run_command(
            project_id,
            [
                "sh",
                "-lc",
                "command -v python >/dev/null 2>&1 || (mkdir -p /tmp/docode-bin && ln -sf \"$(command -v python3)\" /tmp/docode-bin/python && printf 'export PATH=/tmp/docode-bin:$PATH\\n' > \"$HOME/.bash_profile\" && printf 'export PATH=/tmp/docode-bin:$PATH\\n' > \"$HOME/.profile\"); export PATH=/tmp/docode-bin:$PATH; python --version",
            ],
            timeout_sec=60,
            agent_session_id=session_id,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Python preflight failed: {result.output[-1200:]}")

    async def _start_fixture_service(self, client: DoBoxClient, project_id: str, session_id: str, case: CrawlerCase) -> None:
        service_source = (Path(__file__).resolve().parent / "fixture_service.py").read_text(encoding="utf-8")
        await client.write_file(project_id, ".benchmark_source.py", service_source, agent_session_id=session_id)
        command = (
            f"export PATH=/tmp/docode-bin:$PATH; nohup python .benchmark_source.py --case {case.name} --port 8765 "
            ">/tmp/crawler-benchmark-source.log 2>&1 & echo $! >/tmp/crawler-benchmark-source.pid; "
            "python - <<'PY'\n"
            "import time, urllib.request\n"
            "for attempt in range(40):\n"
            "    try:\n"
            f"        urllib.request.urlopen('{LOCAL_BASE_URL}/__metrics', timeout=1).read(); break\n"
            "    except Exception:\n"
            "        if attempt == 39: raise\n"
            "        time.sleep(0.25)\n"
            "PY\n"
            "rm -f .benchmark_source.py"
        )
        result = await client.run_command(
            project_id,
            ["bash", "-lc", command],
            timeout_sec=30,
            output_limit=20_000,
            agent_session_id=session_id,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"fixture server failed to start: {result.output[-2000:]}")

    async def _inspect_final_workspace(
        self,
        client: DoBoxClient,
        project_id: str,
        session_id: str,
        case: CrawlerCase,
    ) -> tuple[list[str], list[dict[str, Any]], bool]:
        try:
            await client.read_file(project_id, case.target, agent_session_id=session_id)
        except Exception:
            return [f"missing target file: {case.target}"], [], False
        if case.controlled:
            source_url = variant_source(case, LOCAL_BASE_URL, True)
            reset_command = (
                f"python -c \"import urllib.request; urllib.request.urlopen('{LOCAL_BASE_URL}/__reset', timeout=5).read()\" "
                f"&& python {case.target} '{source_url}' {case.output}"
            )
            result = await client.run_command(
                project_id,
                ["bash", "-lc", "export PATH=/tmp/docode-bin:$PATH; " + reset_command],
                timeout_sec=180,
                output_limit=100_000,
                agent_session_id=session_id,
            )
            independent = [{"command": reset_command, "exit_code": result.exit_code, "output": result.output[-2000:]}]
            if result.exit_code != 0:
                return [f"hidden-variant execution failed: {result.output[-800:]}"], independent, False
            try:
                artifact = await client.read_file(project_id, case.output, agent_session_id=session_id)
                payload = json.loads(artifact.content)
            except Exception as exc:
                return [f"hidden-variant artifact is invalid: {type(exc).__name__}: {exc}"], independent, False
            metric_result = await client.run_command(
                project_id,
                [
                    "python",
                    "-c",
                    f"import urllib.request; print(urllib.request.urlopen('{LOCAL_BASE_URL}/__metrics', timeout=5).read().decode())",
                ],
                timeout_sec=15,
                agent_session_id=session_id,
            )
            try:
                observed = json.loads(metric_result.output.strip())
            except json.JSONDecodeError:
                observed = {"metrics_error": metric_result.output[-1000:]}
            failures = validate_controlled_payload(
                case,
                payload,
                base_url=LOCAL_BASE_URL,
                variant=True,
                observed_metrics=observed,
            )
            independent.append({"check": "hidden_variant_payload_and_requests", "failures": failures, "metrics": observed})
            return failures, independent, False

        command = f"python {case.target} {REAL_SOURCE_URL} {case.output}"
        result = await client.run_command(
            project_id,
            ["bash", "-lc", "export PATH=/tmp/docode-bin:$PATH; " + command],
            timeout_sec=180,
            output_limit=100_000,
            agent_session_id=session_id,
        )
        independent = [{"command": command, "exit_code": result.exit_code, "output": result.output[-2000:]}]
        if result.exit_code != 0:
            unavailable = not self._host_source_available()
            return (["live source unavailable during independent check"] if unavailable else [f"independent live execution failed: {result.output[-800:]}"]), independent, unavailable
        try:
            artifact = await client.read_file(project_id, case.output, agent_session_id=session_id)
            payload = json.loads(artifact.content)
        except Exception as exc:
            return [f"live artifact is invalid: {type(exc).__name__}: {exc}"], independent, False
        failures = validate_live_payload(payload)
        independent.append({"check": "live_structural_validation", "record_count": len(payload) if isinstance(payload, list) else None, "failures": failures})
        return failures, independent, False

    def _host_source_available(self) -> bool:
        try:
            with urllib.request.urlopen(REAL_SOURCE_URL, timeout=20) as response:
                return response.status == 200 and bool(response.read(512))
        except Exception:
            return False

    def _load_existing(self, frozen_digest: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not RESULT_PATH.is_file():
            return [], []
        payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if payload.get("definition_digest") != frozen_digest:
            raise RuntimeError("benchmark definition digest changed after formal collection began")
        return list(payload.get("runs", [])), list(payload.get("invalid_attempts", []))

    def _write_run_summary(self, result: dict[str, Any]) -> None:
        summary_path = RUN_ROOT / f"{result['case']}-run{result['run']}.json"
        summary_path.write_text(json.dumps(sanitize(result), indent=2, default=str), encoding="utf-8")

    def _write_results(self, runs: list[dict[str, Any]], invalid_attempts: list[dict[str, Any]], frozen_digest: str) -> None:
        ordered = sorted(runs, key=lambda item: (item["run"], item["case"]))
        payload = {
            "baseline": RUNTIME_COMMIT,
            "tag": RUNTIME_TAG,
            "branch": BENCHMARK_BRANCH,
            "provider": "deepseek",
            "model": "deepseek-chat",
            "quality": "balanced",
            "artifact_mode": "patch",
            "max_iterations": 36,
            "max_tool_calls": 80,
            "requested_runs_per_case": RUNS_PER_CASE,
            "definition_digest": frozen_digest,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metrics": aggregate_metrics(ordered),
            "invalid_attempts": invalid_attempts,
            "runs": ordered,
        }
        RESULT_PATH.write_text(json.dumps(sanitize(payload), indent=2, default=str), encoding="utf-8")
