from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from docode.api.auth import make_user_context_dependency
from docode.api.frontend import mount_frontend
from docode.api.routes_artifacts import make_artifacts_router
from docode.api.routes_health import router as health_router
from docode.api.routes_jobs import make_jobs_router
from docode.api.routes_runtime import make_runtime_router
from docode.config import load_config
from docode.storage.db import build_repository
from docode.storage.models import JobStatus
from docode.worker.queue import AsyncJobQueue
from docode.worker.recovery import recover_interrupted_jobs, recover_stale_active_jobs
from docode.worker.runner import JobRunnerService


config = load_config()
repository = build_repository(config)
queue = AsyncJobQueue()
runner = JobRunnerService(config=config, repository=repository)
user_context_dependency = make_user_context_dependency(config)


async def _queue_watchdog() -> None:
    while True:
        queue.start(runner.run_job)
        stale_job_ids = await recover_stale_active_jobs(repository, config.stale_job_requeue_seconds)
        for job_id in stale_job_ids:
            await queue.enqueue(job_id)
        for job in await repository.list_jobs({JobStatus.QUEUED}):
            await queue.enqueue(job.id)
        await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ = app
    queue.start(runner.run_job)
    recovered_job_ids = set(await recover_interrupted_jobs(repository))
    for job_id in recovered_job_ids:
        await queue.enqueue(job_id)
    for job in await repository.list_jobs({JobStatus.QUEUED}):
        if job.id in recovered_job_ids:
            continue
        await queue.enqueue(job.id)
    watchdog = asyncio.create_task(_queue_watchdog())
    try:
        yield
    finally:
        watchdog.cancel()
        try:
            await watchdog
        except asyncio.CancelledError:
            pass
        await queue.stop()


app = FastAPI(title="DoCode API", version="0.1.0", lifespan=lifespan)
app.include_router(health_router)
app.include_router(make_jobs_router(repository, queue, config, user_context_dependency))
app.include_router(make_artifacts_router(repository, user_context_dependency))
app.include_router(make_runtime_router(config, user_context_dependency))
mount_frontend(app)
