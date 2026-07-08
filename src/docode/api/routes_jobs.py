from __future__ import annotations
from dataclasses import asdict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from docode.api.artifact_actions import artifact_descriptor
from docode.api.auth import UserContext, get_user_context, require_owned_job
from docode.api.events import event_stream, load_result_payload, step_event_payload
from docode.api.job_actions import CreateJobInput, JobActionError, cancel_existing_job, create_coding_job
from docode.config import DocodeConfig
from docode.llm.credentials import APICredCredentialResolver
from docode.llm.model_policy import DocodeModelPolicy
from docode.storage.models import JobStatus, public_job_dict
from docode.storage.repository import JobRepository
from docode.worker.queue import AsyncJobQueue
from docode.worker.runner import JobRunnerService


class CreateJobRequest(BaseModel):
    instruction: str = Field(min_length=1)
    repo_url: str | None = None
    branch: str | None = None
    github_repo: str | None = None
    base_branch: str | None = None
    provider: str | None = None
    model: str | None = None
    quality: str | None = Field(default=None, pattern="^(fast|balanced|strong)$")
    max_iterations: int | None = Field(default=None, ge=1, le=200)
    max_runtime_seconds: int | None = Field(default=None, ge=30, le=24 * 60 * 60)
    max_consecutive_failures: int | None = Field(default=None, ge=1, le=200)
    max_tool_calls: int | None = Field(default=None, ge=1, le=1000)
    max_llm_tokens: int | None = Field(default=None, ge=1, le=10_000_000)
    max_llm_cost: float | None = Field(default=None, gt=0, le=10_000)
    artifact_mode: str | None = Field(default=None, pattern="^(patch|zip|commit|pr)$")
    sandbox_network_mode: str | None = None


def make_jobs_router(repository: JobRepository, queue: AsyncJobQueue, config: DocodeConfig, user_dependency=get_user_context) -> APIRouter:
    router = APIRouter(prefix="/v1/jobs", tags=["jobs"])
    runner = JobRunnerService(config=config, repository=repository)

    @router.get("")
    async def list_jobs(status: str | None = None, user: UserContext = Depends(user_dependency)) -> list[dict[str, object]]:
        status_filter: set[JobStatus] | None = None
        if status:
            try:
                status_filter = {JobStatus(status)}
            except ValueError as exc:
                allowed = ", ".join(status.value for status in JobStatus)
                raise HTTPException(status_code=400, detail=f"unsupported job status '{status}', expected one of: {allowed}") from exc

        jobs = [job for job in await repository.list_jobs(status_filter) if job.user_id == user.user_id]
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        return jsonable_encoder([public_job_dict(job) for job in jobs])

    @router.post("")
    async def create_job(req: CreateJobRequest, background_tasks: BackgroundTasks, user: UserContext = Depends(user_dependency)) -> dict[str, str]:
        resolver = APICredCredentialResolver(config.apicred_base_url, config.apicred_token, config.apicred_mode)
        resolver.use_access_token(user.apicred_access_token)
        model_policy = DocodeModelPolicy(config, resolver)
        try:
            job = await create_coding_job(
                repository=repository,
                queue=queue,
                config=config,
                model_policy=model_policy,
                user_id=user.user_id,
                apicred_access_token=user.apicred_access_token,
                request=CreateJobInput(
                    instruction=req.instruction,
                    repo_url=req.repo_url,
                    branch=req.branch,
                    github_repo=req.github_repo,
                    base_branch=req.base_branch,
                    provider=req.provider,
                    model=req.model,
                    quality=req.quality,
                    max_iterations=req.max_iterations,
                    max_runtime_seconds=req.max_runtime_seconds,
                    max_consecutive_failures=req.max_consecutive_failures,
                    max_tool_calls=req.max_tool_calls,
                    max_llm_tokens=req.max_llm_tokens,
                    max_llm_cost=req.max_llm_cost,
                    artifact_mode=req.artifact_mode,
                    sandbox_network_mode=req.sandbox_network_mode,
                ),
            )
        except JobActionError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        background_tasks.add_task(runner.run_job, job.id)
        return {"job_id": job.id, "status": job.status.value}

    @router.get("/{job_id}")
    async def get_job(job_id: str, user: UserContext = Depends(user_dependency)) -> dict[str, object]:
        job = await require_owned_job(repository, job_id, user)
        payload = public_job_dict(job)
        result = await load_result_payload(repository, job_id)
        if result is not None:
            payload["result"] = result
        return jsonable_encoder(payload)

    @router.get("/{job_id}/events")
    async def stream_events(job_id: str, user: UserContext = Depends(user_dependency)) -> StreamingResponse:
        await require_owned_job(repository, job_id, user)
        return StreamingResponse(event_stream(repository, job_id), media_type="text/event-stream")

    @router.get("/{job_id}/steps")
    async def get_steps(job_id: str, user: UserContext = Depends(user_dependency)) -> list[dict[str, object]]:
        await require_owned_job(repository, job_id, user)
        return jsonable_encoder([step_event_payload(step) for step in await repository.list_steps(job_id)])

    @router.get("/{job_id}/artifacts")
    async def get_artifacts(job_id: str, user: UserContext = Depends(user_dependency)) -> list[dict[str, object]]:
        await require_owned_job(repository, job_id, user)
        return jsonable_encoder([artifact_descriptor(artifact) for artifact in await repository.list_artifacts(job_id)])

    @router.post("/{job_id}/cancel")
    async def cancel_job(job_id: str, user: UserContext = Depends(user_dependency)) -> dict[str, str]:
        job = await require_owned_job(repository, job_id, user)
        return await cancel_existing_job(repository, queue, config, job)

    return router
