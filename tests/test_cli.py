from __future__ import annotations

import argparse
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from docode.api.job_actions import JobActionError
from docode.cli import run_scripted_job
from docode.config import DocodeConfig
from docode.storage.repository import InMemoryJobRepository


class FakeRunner:
    ran_job_ids: list[str] = []

    def __init__(self, *, config: DocodeConfig, repository: InMemoryJobRepository) -> None:
        self.config = config
        self.repository = repository

    async def run_job(self, job_id: str) -> None:
        self.ran_job_ids.append(job_id)


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


if __name__ == "__main__":
    import unittest

    unittest.main()
