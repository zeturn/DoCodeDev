from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from pathlib import Path

from docode.api.job_actions import CreateJobInput, create_coding_job
from docode.config import load_config
from docode.llm.credentials import APICredCredentialResolver
from docode.llm.model_policy import DocodeModelPolicy
from docode.eval import eval_case_result_from_job, load_eval_manifest, run_eval, scaffold_eval_suite, write_eval_case_result, write_eval_report
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
    eval_jobs.add_argument("--user-id", default="eval")

    args = parser.parse_args()
    if args.command == "scripted-job":
        asyncio.run(run_scripted_job(args))
    if args.command == "smoke-check":
        asyncio.run(run_smoke_check_command(args))
    if args.command == "smoke-run":
        asyncio.run(run_smoke_run_command(args))
    if args.command == "eval" and args.eval_command == "run":
        run_eval_command(args)
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
    report = run_eval(Path(args.fixtures_dir))
    write_eval_report(report, Path(args.report))
    print(report.to_dict())


def run_eval_scaffold_command(args: argparse.Namespace) -> None:
    manifest = scaffold_eval_suite(Path(args.output_dir), force=args.force)
    print({"manifest": str(Path(args.output_dir) / "manifest.json"), "cases": len(manifest["cases"])})


async def run_eval_jobs_command(args: argparse.Namespace) -> None:
    runner_cls = JobRunnerService
    if runner_cls is None:
        from docode.worker.runner import JobRunnerService as runner_cls

    config = load_config()
    repository = build_repository(config)
    queue = AsyncJobQueue()
    model_policy = DocodeModelPolicy(config, APICredCredentialResolver(config.apicred_base_url, config.apicred_token, config.apicred_mode))
    runner = runner_cls(config=config, repository=repository)
    manifest = load_eval_manifest(Path(args.manifest))
    cases = manifest["cases"][: args.limit] if args.limit else manifest["cases"]
    results: list[dict[str, object]] = []
    for case in cases:
        repo_url = case.get("repo_url") or case.get("repo_path")
        job = await create_coding_job(
            repository=repository,
            queue=queue,
            config=config,
            model_policy=model_policy,
            user_id=args.user_id,
            request=CreateJobInput(
                instruction=str(case["instruction"]),
                repo_url=str(repo_url) if repo_url else None,
                provider=args.provider,
                model=args.model,
                quality=args.quality,
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


if __name__ == "__main__":
    main()
