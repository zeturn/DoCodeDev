from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest import TestCase

from fastapi import FastAPI
from fastapi.testclient import TestClient

from docode.api.auth import UserContext
from docode.api.routes_jobs import make_jobs_router
from docode.config import DocodeConfig
from docode.storage.models import CodingJob, JobStatus
from docode.storage.repository import InMemoryJobRepository
from docode.worker.queue import AsyncJobQueue


async def alice_user() -> UserContext:
    return UserContext(user_id="alice")


def build_app(repository: InMemoryJobRepository) -> FastAPI:
    app = FastAPI()
    app.include_router(make_jobs_router(repository, AsyncJobQueue(), DocodeConfig(database_path=":memory:"), alice_user))
    return app


def seed_job(repository: InMemoryJobRepository, job: CodingJob) -> None:
    asyncio.run(repository.create_job(job))


class JobsApiTests(TestCase):
    def test_list_jobs_returns_only_current_users_jobs_newest_first(self) -> None:
        repository = InMemoryJobRepository()
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        seed_job(
            repository,
            CodingJob(
                id="job_alice_old",
                user_id="alice",
                instruction="older alice job",
                status=JobStatus.SUCCEEDED,
                created_at=base_time,
                updated_at=base_time,
            ),
        )
        seed_job(
            repository,
            CodingJob(
                id="job_bob_new",
                user_id="bob",
                instruction="bob job",
                status=JobStatus.RUNNING,
                created_at=base_time + timedelta(minutes=10),
                updated_at=base_time + timedelta(minutes=10),
            ),
        )
        seed_job(
            repository,
            CodingJob(
                id="job_alice_new",
                user_id="alice",
                instruction="newer alice job",
                status=JobStatus.RUNNING,
                created_at=base_time + timedelta(minutes=20),
                updated_at=base_time + timedelta(minutes=20),
            ),
        )

        response = TestClient(build_app(repository)).get("/v1/jobs")

        self.assertEqual(response.status_code, 200)
        self.assertEqual([job["id"] for job in response.json()], ["job_alice_new", "job_alice_old"])

    def test_list_jobs_status_filter_applies_after_ownership_filter(self) -> None:
        repository = InMemoryJobRepository()
        seed_job(repository, CodingJob(id="job_alice_running", user_id="alice", instruction="run", status=JobStatus.RUNNING))
        seed_job(repository, CodingJob(id="job_alice_failed", user_id="alice", instruction="fail", status=JobStatus.FAILED))
        seed_job(repository, CodingJob(id="job_bob_running", user_id="bob", instruction="hidden", status=JobStatus.RUNNING))

        response = TestClient(build_app(repository)).get("/v1/jobs?status=running")

        self.assertEqual(response.status_code, 200)
        self.assertEqual([job["id"] for job in response.json()], ["job_alice_running"])

    def test_list_jobs_rejects_unknown_status(self) -> None:
        response = TestClient(build_app(InMemoryJobRepository())).get("/v1/jobs?status=missing")

        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported job status", response.json()["detail"])
