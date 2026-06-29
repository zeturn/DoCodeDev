from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from docode.api.job_actions import JobActionError
from docode.cli import run_eval_jobs_command, run_scripted_job
from docode.config import DocodeConfig
from docode.runtime.smoke import SmokeCheck
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
                start_dobox=False,
                no_serve_local_repos=True,
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

    async def test_eval_jobs_start_dobox_writes_preflight_and_uses_token(self) -> None:
        repo = InMemoryJobRepository()

        class TokenRecordingRunner(CompletingFakeRunner):
            seen_tokens: list[str] = []

            def __init__(self, *, config: DocodeConfig, repository: InMemoryJobRepository) -> None:
                super().__init__(config=config, repository=repository)
                self.seen_tokens.append(config.dobox_token)

        async def fake_local_checks(config, command_runner):
            return []

        @asynccontextmanager
        async def fake_managed(config, checker, start_dobox, existing_checks):
            self.assertTrue(start_dobox)
            yield [SmokeCheck("dobox_autostart", "passed", "test server")]

        async def fake_dependency_checks(config, checker):
            return [SmokeCheck("dobox_health", "passed", "HTTP 200")]

        async def fake_token(config):
            return "token-1", SmokeCheck("dobox_auth", "passed", "test token")

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
                start_dobox=True,
                no_serve_local_repos=True,
            )

            with (
                patch("docode.cli.load_config", return_value=DocodeConfig()),
                patch("docode.cli.build_repository", return_value=repo),
                patch("docode.cli.JobRunnerService", TokenRecordingRunner),
                patch("docode.runtime.smoke.local_dobox_checks", fake_local_checks),
                patch("docode.runtime.smoke.managed_local_dobox", fake_managed),
                patch("docode.runtime.smoke.dependency_checks", fake_dependency_checks),
                patch("docode.runtime.smoke.ensure_dobox_smoke_token", fake_token),
                patch("builtins.print"),
            ):
                await run_eval_jobs_command(args)

            self.assertEqual(TokenRecordingRunner.seen_tokens, ["token-1"])
            preflight = json.loads((results_dir / "_meta" / "preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(preflight["status"], "passed")
            result = json.loads((results_dir / "readme-only.json").read_text(encoding="utf-8"))
            self.assertTrue(result["success"])

    async def test_eval_jobs_start_dobox_preflight_failure_writes_case_failures(self) -> None:
        repo = InMemoryJobRepository()

        async def fake_local_checks(config, command_runner):
            return []

        @asynccontextmanager
        async def fake_managed(config, checker, start_dobox, existing_checks):
            yield [SmokeCheck("dobox_autostart", "failed", "docker unavailable")]

        async def fake_dependency_checks(config, checker):
            return []

        async def fake_token(config):
            return "token-1", SmokeCheck("dobox_auth", "passed", "test token")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            results_dir = root / "results"
            manifest.write_text(
                json.dumps(
                    {
                        "cases": [
                            {"name": "python-bugfix", "instruction": "fix", "repo_url": "file:///tmp/one"},
                            {"name": "readme-only", "instruction": "docs", "repo_url": "file:///tmp/two"},
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
                limit=1,
                user_id="eval",
                start_dobox=True,
                no_serve_local_repos=True,
            )

            with (
                patch("docode.cli.load_config", return_value=DocodeConfig()),
                patch("docode.cli.build_repository", return_value=repo),
                patch("docode.cli.JobRunnerService", CompletingFakeRunner),
                patch("docode.runtime.smoke.local_dobox_checks", fake_local_checks),
                patch("docode.runtime.smoke.managed_local_dobox", fake_managed),
                patch("docode.runtime.smoke.dependency_checks", fake_dependency_checks),
                patch("docode.runtime.smoke.ensure_dobox_smoke_token", fake_token),
                patch("builtins.print"),
            ):
                await run_eval_jobs_command(args)

            self.assertEqual(await repo.list_jobs(), [])
            preflight = json.loads((results_dir / "_meta" / "preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(preflight["status"], "failed")
            result = json.loads((results_dir / "python-bugfix.json").read_text(encoding="utf-8"))
            self.assertFalse(result["success"])
            self.assertEqual(result["failure_reason"], "eval_preflight_failed")
            self.assertFalse((results_dir / "readme-only.json").exists())


if __name__ == "__main__":
    import unittest

    unittest.main()
