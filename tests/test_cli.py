from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from docode.api.job_actions import JobActionError
from docode.cli import run_eval_jobs_command, run_scripted_job
from docode.config import DocodeConfig
from docode.storage.models import JobStatus
from docode.storage.repository import InMemoryJobRepository


class FakeRunner:
    ran_job_ids: list[str] = []

    def __init__(self, *, config: DocodeConfig, repository: InMemoryJobRepository) -> None:
        self.config = config
        self.repository = repository

    async def run_job(self, job_id: str) -> None:
        self.ran_job_ids.append(job_id)


class CompletingFakeRunner(FakeRunner):
    async def run_job(self, job_id: str) -> None:
        self.ran_job_ids.append(job_id)
        await self.repository.add_step(job_id, "llm", {"type": "llm_decision", "usage": {"total_tokens": 25, "cost": 0.02}})
        await self.repository.add_step(job_id, "tool", {"type": "tool_call", "tool": "run_tests"})
        await self.repository.add_step(
            job_id,
            "verifier",
            {"passed": True, "reason": "ok", "required_fixes": [], "verification_plan": {"required_commands": []}},
        )
        await self.repository.update_job(job_id, status=JobStatus.SUCCEEDED, artifact_id="artifact-1")


class CliTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        FakeRunner.ran_job_ids = []

    async def test_scripted_job_uses_shared_job_creation_policy(self) -> None:
        repo = InMemoryJobRepository()
        config = DocodeConfig(max_iterations=20, github_base_branch="develop", sandbox_network_mode="bridge")
        args = argparse.Namespace(
            instruction="fix build",
            repo_url="https://github.com/acme/app",
            branch="feature/cli",
            github_repo="acme/app",
            base_branch=None,
            max_iterations=3,
            artifact_mode=" PR ",
        )

        with (
            patch("docode.cli.load_config", return_value=config),
            patch("docode.cli.build_repository", return_value=repo),
            patch("docode.cli.JobRunnerService", FakeRunner),
            patch("builtins.print"),
        ):
            await run_scripted_job(args)

        jobs = await repo.list_jobs()
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(FakeRunner.ran_job_ids, [job.id])
        self.assertEqual(job.user_id, "cli")
        self.assertEqual(job.provider, "scripted")
        self.assertEqual(job.model, "scripted")
        self.assertEqual(job.max_iterations, 3)
        self.assertEqual(job.artifact_mode, "pr")
        self.assertEqual(job.base_branch, "develop")
        self.assertEqual(job.sandbox_network_mode, "project")

    async def test_scripted_job_rejects_invalid_runtime_policy_before_running(self) -> None:
        repo = InMemoryJobRepository()
        args = argparse.Namespace(
            instruction="fix build",
            repo_url=None,
            branch=None,
            github_repo=None,
            base_branch=None,
            max_iterations=0,
            artifact_mode="patch",
        )

        with (
            patch("docode.cli.load_config", return_value=DocodeConfig()),
            patch("docode.cli.build_repository", return_value=repo),
            patch("docode.cli.JobRunnerService", FakeRunner),
            patch("builtins.print"),
        ):
            with self.assertRaises(JobActionError) as raised:
                await run_scripted_job(args)

        self.assertEqual(raised.exception.detail, "max_iterations must be between 1 and 200")
        self.assertEqual(await repo.list_jobs(), [])
        self.assertEqual(FakeRunner.ran_job_ids, [])

    async def test_eval_jobs_command_runs_manifest_cases_and_writes_results(self) -> None:
        repo = InMemoryJobRepository()
        CompletingFakeRunner.ran_job_ids = []
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            results_dir = root / "results"
            manifest.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "name": "readme-only",
                                "instruction": "update README",
                                "repo_url": "file:///tmp/readme-only",
                                "artifact_mode": "patch",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                manifest=str(manifest),
                results_dir=str(results_dir),
                provider="dev",
                model=None,
                quality=None,
                limit=None,
                user_id="eval",
            )

            with (
                patch("docode.cli.load_config", return_value=DocodeConfig()),
                patch("docode.cli.build_repository", return_value=repo),
                patch("docode.cli.JobRunnerService", CompletingFakeRunner),
                patch("builtins.print"),
            ):
                await run_eval_jobs_command(args)

            jobs = await repo.list_jobs()
            self.assertEqual(len(jobs), 1)
            self.assertEqual(CompletingFakeRunner.ran_job_ids, [jobs[0].id])
            result = json.loads((results_dir / "readme-only.json").read_text(encoding="utf-8"))
            self.assertTrue(result["success"])
            self.assertEqual(result["tokens"], 25)
            self.assertEqual(result["tool_calls"], 1)


if __name__ == "__main__":
    import unittest

    unittest.main()
