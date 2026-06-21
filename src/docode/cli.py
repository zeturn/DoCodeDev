from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from pathlib import Path

from docode.api.job_actions import CreateJobInput, create_coding_job
from docode.config import load_config
from docode.llm.credentials import APICredCredentialResolver
from docode.llm.model_policy import DocodeModelPolicy
from docode.runtime.smoke import run_scripted_smoke_job, run_smoke_check, write_smoke_report
from docode.storage.db import build_repository
from docode.worker.queue import AsyncJobQueue
from docode.worker.runner import JobRunnerService


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

    args = parser.parse_args()
    if args.command == "scripted-job":
        asyncio.run(run_scripted_job(args))
    if args.command == "smoke-check":
        asyncio.run(run_smoke_check_command(args))
    if args.command == "smoke-run":
        asyncio.run(run_smoke_run_command(args))


async def run_scripted_job(args: argparse.Namespace) -> None:
    config = load_config()
    repository = build_repository(config)
    queue = AsyncJobQueue()
    model_policy = DocodeModelPolicy(config, APICredCredentialResolver(config.apicred_base_url, config.apicred_token))
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
    runner = JobRunnerService(config=config, repository=repository)
    await runner.run_job(job.id)
    completed = await repository.get_job(job.id)
    artifacts = await repository.list_artifacts(job.id)
    print(asdict(completed) if completed is not None else {"job_id": job.id, "status": "missing"})
    print({"artifacts": [asdict(artifact) for artifact in artifacts]})


async def run_smoke_check_command(args: argparse.Namespace) -> None:
    report = await run_smoke_check(load_config(), start_dobox=args.start_dobox)
    write_smoke_report(report, Path(args.report))
    print(asdict(report))


async def run_smoke_run_command(args: argparse.Namespace) -> None:
    report = await run_scripted_smoke_job(load_config(), instruction=args.instruction, start_dobox=args.start_dobox)
    write_smoke_report(report, Path(args.report))
    print(asdict(report))


if __name__ == "__main__":
    main()
