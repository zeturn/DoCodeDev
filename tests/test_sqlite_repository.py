from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from docode.storage.models import CodingJob, JobStatus, new_id
from docode.storage.sqlite import SQLiteJobRepository


class SQLiteRepositoryTests(IsolatedAsyncioTestCase):
    async def test_persists_jobs_steps_and_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "docode.db"
            repo = SQLiteJobRepository(db_path)
            reopened: SQLiteJobRepository | None = None
            try:
                job = await repo.create_job(
                    CodingJob(
                        id=new_id("job"),
                        user_id="user-1",
                        instruction="ship it",
                        quality="strong",
                        apicred_access_token="bp_xat_sqlite",
                        github_repo="zeturn/example",
                        base_branch="develop",
                        dobox_agent_session_id="7",
                        max_consecutive_failures=9,
                        max_tool_calls=7,
                        max_llm_tokens=12345,
                        max_llm_cost=0.75,
                        artifact_mode="commit",
                        sandbox_network_mode="no_internet",
                    )
                )
                await repo.update_job(job.id, status=JobStatus.RUNNING, dobox_project_id="42")
                await repo.add_step(job.id, "tool", {"tool": "run_command", "exit_code": 0})
                await repo.add_artifact(job.id, "patch", "/tmp/patch.diff", 12)

                reopened = SQLiteJobRepository(db_path)
                loaded = await reopened.get_job(job.id)
                self.assertIsNotNone(loaded)
                assert loaded is not None
                self.assertEqual(loaded.status, JobStatus.RUNNING)
                self.assertEqual(loaded.apicred_access_token, "bp_xat_sqlite")
                self.assertEqual(loaded.dobox_project_id, "42")
                self.assertEqual(loaded.dobox_agent_session_id, "7")
                self.assertEqual(loaded.quality, "strong")
                self.assertEqual(loaded.max_consecutive_failures, 9)
                self.assertEqual(loaded.max_tool_calls, 7)
                self.assertEqual(loaded.max_llm_tokens, 12345)
                self.assertEqual(loaded.max_llm_cost, 0.75)
                self.assertEqual(loaded.artifact_mode, "commit")
                self.assertEqual(loaded.sandbox_network_mode, "no_internet")
                self.assertEqual(loaded.github_repo, "zeturn/example")
                self.assertEqual(loaded.base_branch, "develop")
                self.assertEqual(len(await reopened.list_steps(job.id)), 1)
                artifacts = await reopened.list_artifacts(job.id)
                self.assertEqual(artifacts[0].kind, "patch")
                self.assertEqual((await reopened.get_artifact(artifacts[0].id)).size_bytes, 12)
            finally:
                if reopened is not None:
                    reopened.close()
                repo.close()

    async def test_claim_job_only_moves_queued_job_to_preparing_once(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = SQLiteJobRepository(Path(tmp) / "docode.db")
            try:
                job = await repo.create_job(CodingJob(id=new_id("job"), user_id="user-1", instruction="claim me"))

                claimed = await repo.claim_job(job.id)
                second_claim = await repo.claim_job(job.id)

                self.assertIsNotNone(claimed)
                assert claimed is not None
                self.assertEqual(claimed.status, JobStatus.PREPARING)
                self.assertIsNone(second_claim)
            finally:
                repo.close()

    async def test_step_and_artifact_updates_refresh_job_timestamp(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = SQLiteJobRepository(Path(tmp) / "docode.db")
            try:
                job = await repo.create_job(CodingJob(id=new_id("job"), user_id="user-1", instruction="heartbeat"))
                first = await repo.get_job(job.id)
                assert first is not None
                await repo.add_step(job.id, "system", {"type": "heartbeat"})
                after_step = await repo.get_job(job.id)
                assert after_step is not None
                self.assertGreater(after_step.updated_at, first.updated_at)
                await repo.add_artifact(job.id, "zip", "/tmp/out.zip", 5)
                after_artifact = await repo.get_job(job.id)
                assert after_artifact is not None
                self.assertGreaterEqual(after_artifact.updated_at, after_step.updated_at)
                self.assertGreater(after_artifact.updated_at, first.updated_at)
            finally:
                repo.close()
