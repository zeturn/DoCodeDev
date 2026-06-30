from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from pathlib import Path

from docode.api.job_actions import CreateJobInput, create_coding_job
from docode.config import load_config
from docode.llm.credentials import APICredCredentialResolver
from docode.llm.model_policy import DocodeModelPolicy
from docode.eval import (
    EvalThresholds,
    eval_case_result_from_job,
    load_eval_manifest,
    load_eval_report,
    managed_local_repo_server,
    run_eval,
    scaffold_eval_suite,
    summarize_eval_matrix,
    with_eval_assertion,
    with_eval_comparison,
    write_eval_case_result,
    write_eval_report,
)
from docode.storage.db import build_repository
from docode.storage.models import public_job_dict
from docode.worker.queue import AsyncJobQueue

JobRunnerService = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DoCode development utilities.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    scripted = subcommands.add_parser("scripted-job", help="Run a deterministic scripted job against the configured DoBox API.")
    scripted.add_argument("instruction")
    scripted.add_argument("--repo-url")
    scripted.add_argument("--branch")
    scripted.add_argument("--github-repo")
    scripted.add_argument("--base-branch")
    scripted.add_argument("--max-iterations", type=int, default=5)
    scripted.add_argument("--artifact-mode", choices=["patch", "zip", "commit", "pr"], default="patch")

    smoke_check = subcommands.add_parser("smoke-check", help="Check configured runtime dependencies and write an evidence report.")
    smoke_check.add_argument("--report", default=".docode/smoke-check.json")
    smoke_check.add_argument("--start-dobox", action="store_true", help="Temporarily start the local DoBox backend if it is not reachable.")

    smoke_run = subcommands.add_parser("smoke-run", help="Run a scripted end-to-end smoke job against the configured DoBox API.")
    smoke_run.add_argument("--instruction", default="create a result file")
    smoke_run.add_argument("--report", default=".docode/smoke-run.json")
    smoke_run.add_argument("--start-dobox", action="store_true", help="Temporarily start the local DoBox backend for the smoke job if needed.")

    eval_parser = subcommands.add_parser("eval", help="Run DoCode eval utilities.")
    eval_subcommands = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_subcommands.add_parser("run", help="Aggregate eval fixture results into a report.")
    eval_run.add_argument("fixtures_dir")
    eval_run.add_argument("--report", default=".docode/eval-report.json")
    eval_run.add_argument("--baseline-report", help="Previous eval report to compare against.")
    add_eval_threshold_arguments(eval_run)
    eval_assert = eval_subcommands.add_parser("assert", help="Fail when an eval report does not meet configured thresholds.")
    eval_assert.add_argument("report")
    add_eval_threshold_arguments(eval_assert)
    eval_matrix = eval_subcommands.add_parser("matrix", help="Build a model comparison matrix from eval reports or result directories.")
    eval_matrix.add_argument(
        "runs",
        nargs="+",
        help="Model run in the form model=path, where path is an eval report JSON file or a per-case results directory.",
    )
    eval_matrix.add_argument("--baseline", action="append", default=[], help="Previous run in the form model=path.")
    eval_matrix.add_argument("--report", default=".docode/eval-matrix.json")
    eval_scaffold = eval_subcommands.add_parser("scaffold", help="Create the standard small-repository eval suite.")
    eval_scaffold.add_argument("output_dir")
    eval_scaffold.add_argument("--force", action="store_true", help="Replace an existing suite directory.")
    eval_jobs = eval_subcommands.add_parser("jobs", help="Run eval manifest cases through DoCode jobs and write per-case results.")
    eval_jobs.add_argument("manifest")
    eval_jobs.add_argument("--results-dir", default=".docode/eval-results")
    eval_jobs.add_argument("--provider")
    eval_jobs.add_argument("--model")
    eval_jobs.add_argument("--quality")
    eval_jobs.add_argument("--limit", type=int)
    eval_jobs.add_argument("--max-iterations", type=int)
    eval_jobs.add_argument("--max-runtime-seconds", type=int)
    eval_jobs.add_argument("--max-consecutive-failures", type=int)
    eval_jobs.add_argument("--max-tool-calls", type=int)
    eval_jobs.add_argument("--max-llm-tokens", type=int)
    eval_jobs.add_argument("--max-llm-cost", type=float)
    eval_jobs.add_argument("--user-id", default="eval")
    eval_jobs.add_argument("--start-dobox", action="store_true", help="Temporarily start the local DoBox backend for eval jobs if needed.")
    eval_jobs.add_argument("--include-hints", action="store_true", help="Append eval-only target file and command hints to each job instruction.")
    eval_jobs.add_argument("--no-serve-local-repos", action="store_true", help="Do not expose local eval repos through a temporary git server.")
    eval_jobs.add_argument(
        "--sandbox-retention",
        choices=["keep", "delete_on_success", "delete_always"],
        default="delete_always",
        help="Sandbox retention policy for eval jobs.",
    )

    args = parser.parse_args()
    if args.command == "scripted-job":
        asyncio.run(run_scripted_job(args))
    if args.command == "smoke-check":
        asyncio.run(run_smoke_check_command(args))
    if args.command == "smoke-run":
        asyncio.run(run_smoke_run_command(args))
    if args.command == "eval" and args.eval_command == "run":
        run_eval_command(args)
    if args.command == "eval" and args.eval_command == "assert":
        run_eval_assert_command(args)
    if args.command == "eval" and args.eval_command == "matrix":
        run_eval_matrix_command(args)
    if args.command == "eval" and args.eval_command == "scaffold":
        run_eval_scaffold_command(args)
    if args.command == "eval" and args.eval_command == "jobs":
        asyncio.run(run_eval_jobs_command(args))


async def run_scripted_job(args: argparse.Namespace) -> None:
    runner_cls = JobRunnerService
    if runner_cls is None:
        from docode.worker.runner import JobRunnerService as runner_cls

    config = load_config()
    repository = build_repository(config)
    queue = AsyncJobQueue()
    model_policy = DocodeModelPolicy(config, APICredCredentialResolver(config.apicred_base_url, config.apicred_token, config.apicred_mode))
    job = await create_coding_job(
        repository=repository,
        queue=queue,
        config=config,
        model_policy=model_policy,
        user_id="cli",
        request=CreateJobInput(
            instruction=args.instruction,
            repo_url=args.repo_url,
            branch=args.branch,
            github_repo=args.github_repo,
            base_branch=args.base_branch,
            provider="scripted",
            model="scripted",
            max_iterations=args.max_iterations,
            artifact_mode=args.artifact_mode,
            sandbox_network_mode=config.sandbox_network_mode,
        ),
    )
    runner = runner_cls(config=config, repository=repository)
    await runner.run_job(job.id)
    completed = await repository.get_job(job.id)
    artifacts = await repository.list_artifacts(job.id)
    print(public_job_dict(completed) if completed is not None else {"job_id": job.id, "status": "missing"})
    print({"artifacts": [asdict(artifact) for artifact in artifacts]})


async def run_smoke_check_command(args: argparse.Namespace) -> None:
    from docode.runtime.smoke import run_smoke_check, write_smoke_report

    report = await run_smoke_check(load_config(), start_dobox=args.start_dobox)
    write_smoke_report(report, Path(args.report))
    print(asdict(report))


async def run_smoke_run_command(args: argparse.Namespace) -> None:
    from docode.runtime.smoke import run_scripted_smoke_job, write_smoke_report

    report = await run_scripted_smoke_job(load_config(), instruction=args.instruction, start_dobox=args.start_dobox)
    write_smoke_report(report, Path(args.report))
    print(asdict(report))


def run_eval_command(args: argparse.Namespace) -> None:
    thresholds = eval_thresholds_from_args(args)
    report = run_eval(Path(args.fixtures_dir), thresholds=thresholds)
    baseline_report = getattr(args, "baseline_report", None)
    if baseline_report:
        report = with_eval_comparison(report, load_eval_report(Path(baseline_report)))
    write_eval_report(report, Path(args.report))
    print(report.to_dict())
    if report.assertion is not None and report.assertion.regression:
        raise SystemExit(1)


def run_eval_assert_command(args: argparse.Namespace) -> None:
    thresholds = eval_thresholds_from_args(args)
    if thresholds is None:
        raise SystemExit("at least one eval threshold is required")
    report = with_eval_assertion(load_eval_report(Path(args.report)), thresholds)
    write_eval_report(report, Path(args.report))
    print(report.assertion.to_dict() if report.assertion is not None else {"regression": False})
    if report.assertion is not None and report.assertion.regression:
        raise SystemExit(1)


def run_eval_matrix_command(args: argparse.Namespace) -> None:
    reports = load_named_eval_reports(args.runs)
    baselines = load_named_eval_reports(args.baseline)
    matrix = summarize_eval_matrix(reports, previous_reports=baselines)
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(matrix.to_dict()), encoding="utf-8")
    print(matrix.to_dict())


def load_named_eval_reports(entries: list[str]) -> dict[str, object]:
    reports = {}
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"eval matrix entries must use model=path: {entry}")
        model, raw_path = entry.split("=", 1)
        model = model.strip()
        if not model:
            raise SystemExit(f"eval matrix model name is empty: {entry}")
        reports[model] = load_eval_report_input(Path(raw_path))
    return reports


def load_eval_report_input(path: Path):
    if path.is_dir():
        return run_eval(path)
    return load_eval_report(path)


def json_dumps(data: object) -> str:
    import json

    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def add_eval_threshold_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-success-rate", type=float)
    parser.add_argument("--max-avg-tool-calls", type=float, dest="max_average_tool_calls")
    parser.add_argument("--max-cost", type=float, dest="max_total_cost")


def eval_thresholds_from_args(args: argparse.Namespace) -> EvalThresholds | None:
    thresholds = EvalThresholds(
        min_success_rate=args.min_success_rate,
        max_average_tool_calls=args.max_average_tool_calls,
        max_total_cost=args.max_total_cost,
    )
    return thresholds if thresholds.to_dict() else None


def run_eval_scaffold_command(args: argparse.Namespace) -> None:
    manifest = scaffold_eval_suite(Path(args.output_dir), force=args.force)
    print({"manifest": str(Path(args.output_dir) / "manifest.json"), "cases": len(manifest["cases"])})


async def run_eval_jobs_command(args: argparse.Namespace) -> None:
    if args.start_dobox:
        from docode.runtime.smoke import (
            SmokeReport,
            check_http_health,
            dependency_checks,
            ensure_dobox_smoke_token,
            is_fatal_smoke_failure,
            local_dobox_checks,
            managed_local_dobox,
            run_command_probe,
            write_smoke_report,
        )

        config = load_config()
        checks = await local_dobox_checks(config, run_command_probe)
        async with managed_local_dobox(config, check_http_health, True, checks) as start_checks:
            checks.extend(start_checks)
            checks.extend(await dependency_checks(config, check_http_health))
            token, token_check = await ensure_dobox_smoke_token(config)
            checks.append(token_check)
            status = "passed" if all(not is_fatal_smoke_failure(check) for check in checks) else "failed"
            preflight = SmokeReport(status=status, checks=checks)
            write_smoke_report(preflight, Path(args.results_dir) / "_meta" / "preflight.json")
            if preflight.status != "passed":
                failed = write_eval_preflight_failures(args, "eval_preflight_failed")
                print({"results_dir": args.results_dir, "status": "failed", "failure_reason": "eval_preflight_failed", "cases": failed})
                return
            config.dobox_token = token or config.dobox_token
            await run_eval_jobs_with_config(args, config)
            return

    await run_eval_jobs_with_config(args, load_config())


async def run_eval_jobs_with_config(args: argparse.Namespace, config) -> None:
    config.sandbox_retention = args.sandbox_retention
    runner_cls = JobRunnerService
    if runner_cls is None:
        from docode.worker.runner import JobRunnerService as runner_cls

    repository = build_repository(config)
    queue = AsyncJobQueue()
    model_policy = DocodeModelPolicy(config, APICredCredentialResolver(config.apicred_base_url, config.apicred_token, config.apicred_mode))
    runner = runner_cls(config=config, repository=repository)
    manifest = load_eval_manifest(Path(args.manifest))
    serve_local_repos = args.start_dobox and not args.no_serve_local_repos
    results: list[dict[str, object]] = []
    with managed_local_repo_server(manifest, enabled=serve_local_repos) as served_manifest:
        cases = served_manifest["cases"][: args.limit] if args.limit else served_manifest["cases"]
        for case in cases:
            repo_url = case.get("repo_url") or case.get("repo_path")
            instruction = eval_instruction_with_hints(case) if getattr(args, "include_hints", False) else str(case["instruction"])
            job = await create_coding_job(
                repository=repository,
                queue=queue,
                config=config,
                model_policy=model_policy,
                user_id=args.user_id,
                request=CreateJobInput(
                    instruction=instruction,
                    repo_url=str(repo_url) if repo_url else None,
                    provider=args.provider,
                    model=args.model,
                    quality=args.quality,
                    max_iterations=args.max_iterations,
                    max_runtime_seconds=args.max_runtime_seconds,
                    max_consecutive_failures=args.max_consecutive_failures,
                    max_tool_calls=args.max_tool_calls,
                    max_llm_tokens=args.max_llm_tokens,
                    max_llm_cost=args.max_llm_cost,
                    artifact_mode=str(case.get("artifact_mode") or "patch"),
                    sandbox_network_mode=config.sandbox_network_mode,
                ),
            )
            await runner.run_job(job.id)
            completed = await repository.get_job(job.id)
            steps = await repository.list_steps(job.id)
            result = eval_case_result_from_job(case, completed or job, steps)
            write_eval_case_result(result, Path(args.results_dir))
            results.append(result)
    print({"results_dir": args.results_dir, "cases": len(results), "succeeded": sum(1 for result in results if result.get("success"))})


def eval_instruction_with_hints(case: dict[str, object]) -> str:
    instruction = str(case["instruction"])
    hints: list[str] = []
    configured_hints = case.get("hints") if isinstance(case.get("hints"), dict) else {}
    target_files = list_hint_values(configured_hints.get("target_files")) if configured_hints else []
    expected_behavior = str(configured_hints.get("expected_behavior") or "").strip() if configured_hints else ""
    suggested_commands = list_hint_values(configured_hints.get("suggested_commands")) if configured_hints else []
    if target_files:
        hints.append("target file: " + ", ".join(target_files[:5]))
    if expected_behavior:
        hints.append("expected behavior: " + expected_behavior)
    if suggested_commands:
        hints.append("verify with: " + "; ".join(suggested_commands[:5]))
    files = case.get("files")
    if not target_files and isinstance(files, dict):
        target_files = [str(path) for path in files if not str(path).startswith("tests/")]
        if target_files:
            hints.append("Likely target files: " + ", ".join(target_files[:5]))
    checks = case.get("expected_checks")
    if not suggested_commands and isinstance(checks, list) and checks:
        hints.append("Suggested verification commands: " + "; ".join(str(check) for check in checks[:5]))
    if not hints:
        return instruction
    return instruction + "\n\nEvaluation hints:\n" + "\n".join(f"- {hint}" for hint in hints)


def list_hint_values(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def write_eval_preflight_failures(args: argparse.Namespace, failure_reason: str) -> int:
    manifest = load_eval_manifest(Path(args.manifest))
    cases = manifest["cases"][: args.limit] if args.limit else manifest["cases"]
    for case in cases:
        write_eval_case_result(
            {
                "name": case.get("name"),
                "category": case.get("category"),
                "instruction": case.get("instruction"),
                "status": "failed",
                "success": False,
                "failure_reason": failure_reason,
                "verification": {"required_fixes": [failure_reason]},
            },
            Path(args.results_dir),
        )
    return len(cases)


if __name__ == "__main__":
    main()
