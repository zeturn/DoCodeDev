from __future__ import annotations

import json
import os
import statistics
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
from docode.storage.models import CodingJob, JobStatus, new_id

from tests.holdout.definitions import CASES, HoldoutCase
from tests.holdout.harness import EDIT_TOOLS, READ_TOOLS, materialize_fixture, sanitize, summarize_steps
from tests.test_real_llm_smoke import build_real_llm_or_skip
from tests.test_smoke_readme_job import RecordingRepository


REAL_HOLDOUT_ENABLED = os.getenv("DOCODE_REAL_HOLDOUT", "").lower() in {"1", "true", "yes", "on"}
RESULT_ROOT = Path(".docode/evals/eff27a7-unseen-holdout")
RESULT_PATH = RESULT_ROOT / "results.json"
TRACE_ROOT = RESULT_ROOT / "traces"


def latest_step_value(steps: list[Any], *, kind: str | None = None, content_type: str | None = None, key: str = "passed") -> Any:
    matches = []
    for step in steps:
        if kind is not None and step.kind != kind:
            continue
        if content_type is not None and step.content.get("type") != content_type:
            continue
        matches.append(step.content)
    return matches[-1].get(key) if matches else None


def required_command_results(steps: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "command": str((step.content.get("metadata") or {}).get("command") or ""),
            "exit_code": step.content.get("exit_code"),
            "summary": str(step.content.get("summary") or "")[:500],
        }
        for step in steps
        if step.content.get("type") == "tool_result" and step.content.get("tool") == "run_command"
    ]


def read_and_rewrite_metrics(steps: list[Any], initial_files: set[str]) -> tuple[bool, bool]:
    first_edit = None
    read_before_edit = False
    whole_file_rewrite = False
    for step in steps:
        content = step.content
        if content.get("type") != "tool_result" or content.get("exit_code") != 0:
            continue
        tool = content.get("tool")
        if tool in EDIT_TOOLS:
            if first_edit is None:
                first_edit = step.step_index
            path = str((content.get("metadata") or {}).get("path") or "").replace("\\", "/")
            if tool == "write_file" and path in initial_files:
                whole_file_rewrite = True
        elif tool in READ_TOOLS and first_edit is None:
            read_before_edit = True
    return read_before_edit, whole_file_rewrite


def classify_run(
    *,
    status: str,
    failure_reason: str | None,
    functional_correct: bool,
    summary: dict[str, Any],
    verifier_result: Any,
    quality_result: Any,
    independent_failures: list[str],
) -> str:
    if status == "succeeded" and functional_correct:
        return "success"
    if status == "succeeded" and independent_failures:
        return "code_generation_failure"
    lowered = (failure_reason or "").lower()
    if any(marker in lowered for marker in ("provider", "transport", "llm_auth", "rate_limit", "connection")):
        return "provider_or_transport_failure"
    if functional_correct:
        if quality_result is False:
            return "quality_gate_false_positive"
        if verifier_result is False:
            return "verifier_false_negative"
        return "finalization_failure"
    if "timeout" in lowered or "max_runtime" in lowered or "max_tool_calls" in lowered:
        return "timeout_or_budget"
    if summary["tool_calls"] == 0:
        return "repository_understanding_failure"
    if summary.get("successful_edits", 0) == 0:
        return "repository_understanding_failure" if summary.get("successful_reads", 0) else "wrong_edit_target"
    if independent_failures and summary["repair_actions"]:
        return "repair_loop"
    if independent_failures:
        return "code_generation_failure"
    if summary["commands_run"] and summary["successful_commands"] < summary["commands_run"]:
        return "required_command_failure"
    return "unknown"


def aggregate_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    successes = [run for run in runs if run["status"] == "succeeded" and run["functionally_correct"]]
    cases = sorted({run["case"] for run in runs})
    first_runs = [run for run in runs if run["run"] == 1]
    runs_by_case = {name: [run for run in runs if run["case"] == name] for name in cases}
    three_sample_cases = [items for items in runs_by_case.values() if len(items) >= 3]
    pass_at_3 = (
        sum(any(item["status"] == "succeeded" and item["functionally_correct"] for item in items[:3]) for items in three_sample_cases) / len(three_sample_cases)
        if three_sample_cases
        else None
    )
    successful_with_repair = [run for run in successes if run["repair_actions"] > 0]
    return {
        "sample_size": len(runs),
        "task_success_rate": len(successes) / len(runs),
        "pass_at_1": sum(run["status"] == "succeeded" and run["functionally_correct"] for run in first_runs) / len(first_runs) if first_runs else 0.0,
        "pass_at_3": pass_at_3,
        "median_iterations": statistics.median(run["iterations"] for run in runs),
        "median_tool_calls": statistics.median(run["tool_calls"] for run in runs),
        "successful_tasks_requiring_repair": len(successful_with_repair) / len(successes) if successes else 0.0,
        "runs_failed_before_any_edit": sum(run["status"] != "succeeded" and run["successful_edits"] == 0 for run in runs) / len(runs),
        "runs_failed_after_functional_verification_before_finalization": sum(
            run["status"] != "succeeded" and run["functionally_correct"] for run in runs
        ) / len(runs),
    }


@skipUnless(REAL_HOLDOUT_ENABLED, "set DOCODE_REAL_HOLDOUT=1 to run the frozen real unseen holdout")
class RealUnseenHoldoutTests(IsolatedAsyncioTestCase):
    async def test_real_deepseek_dobox_holdout(self) -> None:
        run_count = max(1, min(3, int(os.getenv("DOCODE_HOLDOUT_RUNS", "1"))))
        requested_cases = {name.strip() for name in os.getenv("DOCODE_HOLDOUT_CASES", "").split(",") if name.strip()}
        cases = tuple(case for case in CASES if not requested_cases or case.name in requested_cases)
        if requested_cases and {case.name for case in cases} != requested_cases:
            unknown = sorted(requested_cases - {case.name for case in cases})
            self.fail(f"unknown holdout case(s): {unknown}")
        config = await self._real_dobox_config()
        client = DoBoxClient(config.dobox_base_url, config.dobox_token)
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        TRACE_ROOT.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        if requested_cases and RESULT_PATH.is_file():
            previous = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
            results = [run for run in previous.get("runs", []) if run.get("case") not in requested_cases]

        for run_number in range(1, run_count + 1):
            for case in cases:
                try:
                    result = await self._run_case(client, config, case, run_number)
                except Exception as exc:
                    result = {
                        "case": case.name,
                        "language": case.language,
                        "mode": "real_dobox",
                        "run": run_number,
                        "status": "failed",
                        "failure_reason": f"harness exception: {type(exc).__name__}: {exc}",
                        "iterations": 0,
                        "llm_decisions": 0,
                        "tool_calls": 0,
                        "commands_run": 0,
                        "successful_commands": 0,
                        "repair_actions": 0,
                        "successful_edits": 0,
                        "successful_reads": 0,
                        "final_candidate_attempted": False,
                        "verifier_result": None,
                        "quality_gate_result": None,
                        "artifact_export_result": False,
                        "changed_files": [],
                        "read_before_edit": False,
                        "whole_file_rewrite": False,
                        "functionally_correct": False,
                        "independent_failures": [str(exc)],
                        "category": "harness_failure",
                        "trace_path": None,
                    }
                results.append(sanitize(result))
                self._write_results(results, run_count)

        harness_failures = [run for run in results if run["category"] == "harness_failure"]
        if harness_failures:
            self.fail(f"holdout harness failed for {len(harness_failures)} run(s); see {RESULT_PATH}")

    async def _run_case(self, client: DoBoxClient, config: Any, case: HoldoutCase, run_number: int) -> dict[str, Any]:
        with TemporaryDirectory() as tmp:
            local_root = Path(tmp)
            fixture = materialize_fixture(case, local_root / "fixture")
            initial_files = {path.relative_to(fixture).as_posix() for path in fixture.rglob("*") if path.is_file()}
            repo = RecordingRepository()
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="holdout-real",
                    instruction=case.instruction,
                    max_iterations=36,
                    max_runtime_seconds=900,
                    max_consecutive_failures=10,
                    max_tool_calls=80,
                    sandbox_network_mode="no_internet",
                )
            )
            llm = await build_real_llm_or_skip(self, job)
            project = await client.create_project(
                name=f"docode-holdout-{case.name}-r{run_number}-{new_id('eval')}",
                network_mode="no_internet",
            )
            session = await client.create_agent_session(project.project_id, name=f"holdout-{case.name}-r{run_number}")
            try:
                await self._seed_fixture(client, project.project_id, session.session_id, fixture)
                await self._runtime_preflight(client, project.project_id, session.session_id, case)
                tools = DoBoxTools(
                    client,
                    project.project_id,
                    agent_session_id=session.session_id,
                    command_timeout_seconds=120,
                    output_limit_bytes=200_000,
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
                summary = summarize_steps(steps)
                contents = [step.content for step in steps]
                summary["successful_edits"] = len([
                    content for content in contents
                    if content.get("type") == "tool_result" and content.get("tool") in EDIT_TOOLS and content.get("exit_code") == 0
                ])
                summary["successful_reads"] = len([
                    content for content in contents
                    if content.get("type") == "tool_result" and content.get("tool") in READ_TOOLS and content.get("exit_code") == 0
                ])
                read_before_edit, whole_file_rewrite = read_and_rewrite_metrics(steps, initial_files)
                independent_failures, independent_results = await self._inspect_final_workspace(
                    client, project.project_id, session.session_id, case, fixture
                )
                functional_correct = not independent_failures
                status_result = await tools.git_status()
                artifacts = await repo.list_artifacts(job.id)
                verifier_result = latest_step_value(steps, kind="verifier")
                quality_result = latest_step_value(steps, content_type="quality_gate")
                category = classify_run(
                    status=completed.status.value,
                    failure_reason=completed.failure_reason,
                    functional_correct=functional_correct,
                    summary=summary,
                    verifier_result=verifier_result,
                    quality_result=quality_result,
                    independent_failures=independent_failures,
                )
                trace_path = TRACE_ROOT / f"{case.name}-run{run_number}-{completed.id}.json"
                trace = {
                    "case": case.name,
                    "language": case.language,
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
                    "language": case.language,
                    "mode": "real_dobox",
                    "run": run_number,
                    "status": completed.status.value,
                    "failure_reason": completed.failure_reason,
                    **summary,
                    "verifier_result": verifier_result,
                    "quality_gate_result": quality_result,
                    "artifact_export_result": bool(artifacts and completed.artifact_id),
                    "changed_files": [line[3:].strip().replace("\\", "/") for line in status_result.output.splitlines() if len(line) >= 4],
                    "read_before_edit": read_before_edit,
                    "whole_file_rewrite": whole_file_rewrite,
                    "functionally_correct": functional_correct,
                    "independent_failures": independent_failures,
                    "category": category,
                    "short_reason": completed.failure_reason or completed.result_summary or "",
                    "trace_path": str(trace_path),
                    "commands": required_command_results(steps),
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
            ["sh", "-lc", "git init -b main && git config user.email holdout@example.test && git config user.name 'DoCode Holdout' && git add . && git commit -m 'Seed unseen holdout'"],
            cwd="/workspace",
            timeout_sec=60,
            agent_session_id=session_id,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"fixture git init failed: {result.output[-1000:]}")

    async def _runtime_preflight(self, client: DoBoxClient, project_id: str, session_id: str, case: HoldoutCase) -> None:
        commands = [
            "command -v python >/dev/null 2>&1 || "
            "(mkdir -p /tmp/docode-bin && ln -sf \"$(command -v python3)\" /tmp/docode-bin/python && "
            "printf 'export PATH=/tmp/docode-bin:$PATH\\n' > \"$HOME/.bash_profile\" && "
            "printf 'export PATH=/tmp/docode-bin:$PATH\\n' > \"$HOME/.profile\")"
        ]
        if case.language == "TypeScript" or case.language == "Node.js":
            commands.append("command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1")
        if case.language == "Go":
            commands.append("command -v go >/dev/null 2>&1")
        result = await client.run_command(
            project_id,
            ["sh", "-lc", " && ".join(commands)],
            cwd="/workspace",
            timeout_sec=60,
            agent_session_id=session_id,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"runtime preflight failed for {case.language}: {result.output[-1000:]}")

    async def _inspect_final_workspace(
        self,
        client: DoBoxClient,
        project_id: str,
        session_id: str,
        case: HoldoutCase,
        fixture: Path,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        failures: list[str] = []
        results: list[dict[str, Any]] = []
        for path in case.expected_files:
            try:
                await client.read_file(project_id, path, agent_session_id=session_id)
            except Exception:
                failures.append(f"missing expected file: {path}")
        for command in case.required_commands:
            portable_command = f"export PATH=/tmp/docode-bin:$PATH; {command}"
            result = await client.run_command(
                project_id,
                ["sh", "-lc", portable_command],
                cwd="/workspace",
                timeout_sec=180,
                agent_session_id=session_id,
            )
            results.append({"command": command, "exit_code": result.exit_code, "output": result.output[-2000:]})
            if result.exit_code != 0:
                failures.append(f"independent command failed: {command}: {result.output[-500:]}")
        if case.name == "sable_manual":
            original = (fixture / "engine/quiet_core.py").read_text(encoding="utf-8")
            try:
                current = await client.read_file(project_id, "engine/quiet_core.py", agent_session_id=session_id)
            except Exception:
                current = None
            if current is None or current.content != original:
                failures.append("docs-only task changed engine/quiet_core.py")
        if case.name == "silver_source":
            try:
                artifact = await client.read_file(project_id, "mosaic-result.json", agent_session_id=session_id)
                payload = json.loads(artifact.content)
            except (TypeError, json.JSONDecodeError, RuntimeError):
                failures.append("mosaic-result.json is not valid JSON")
            else:
                expected = [
                    {"ember_code": "E-17", "caption": "Aster Vale", "drift_index": 9},
                    {"ember_code": "E-42", "caption": "Brass Willow", "drift_index": 14},
                ]
                if payload != expected:
                    failures.append(f"mosaic-result.json schema/value mismatch: {payload!r}")
        return failures, results

    def _write_results(self, results: list[dict[str, Any]], requested_runs: int) -> None:
        payload = {
            "baseline": "eff27a7cbe70408097591369787105ffc5aea777",
            "tag": "agent-baseline-eff27a7",
            "branch": "eval/unseen-holdout-eff27a7",
            "provider": os.getenv("DOCODE_REAL_LLM_PROVIDER", "deepseek"),
            "model": os.getenv("DOCODE_REAL_LLM_MODEL", "deepseek-chat"),
            "requested_runs_per_case": requested_runs,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metrics": aggregate_metrics(results),
            "runs": results,
        }
        RESULT_PATH.write_text(json.dumps(sanitize(payload), indent=2, default=str), encoding="utf-8")
