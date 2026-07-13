from __future__ import annotations

import asyncio
from dataclasses import replace

from .models import CodingJob, DocodeArtifact, DocodeStep, JobStatus, MissionJob, MissionSpec, new_id, utcnow


class JobRepository:
    async def create_job(self, job: CodingJob) -> CodingJob:
        raise NotImplementedError

    async def get_job(self, job_id: str) -> CodingJob | None:
        raise NotImplementedError

    async def list_jobs(self, statuses: set[JobStatus] | None = None) -> list[CodingJob]:
        raise NotImplementedError

    async def claim_job(self, job_id: str) -> CodingJob | None:
        raise NotImplementedError

    async def update_job(self, job_id: str, **changes: object) -> CodingJob:
        raise NotImplementedError

    async def add_step(self, job_id: str, kind: str, content: dict[str, object]) -> DocodeStep:
        raise NotImplementedError

    async def list_steps(self, job_id: str) -> list[DocodeStep]:
        raise NotImplementedError

    async def list_steps_after(self, job_id: str, step_index: int) -> list[DocodeStep]:
        return [step for step in await self.list_steps(job_id) if step.step_index > step_index]

    async def add_artifact(self, job_id: str, kind: str, path: str, size_bytes: int) -> DocodeArtifact:
        raise NotImplementedError

    async def get_artifact(self, artifact_id: str) -> DocodeArtifact | None:
        raise NotImplementedError

    async def list_artifacts(self, job_id: str) -> list[DocodeArtifact]:
        raise NotImplementedError

    async def create_mission_job(self, mission_job: MissionJob) -> MissionJob:
        raise NotImplementedError

    async def list_mission_jobs(self, mission_id: str | None = None) -> list[MissionJob]:
        raise NotImplementedError

    async def create_mission_spec(self, mission_spec: MissionSpec) -> MissionSpec:
        raise NotImplementedError

    async def get_mission_spec(self, mission_id: str) -> MissionSpec | None:
        raise NotImplementedError


class InMemoryJobRepository(JobRepository):
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._jobs: dict[str, CodingJob] = {}
        self._steps: dict[str, list[DocodeStep]] = {}
        self._artifacts: dict[str, list[DocodeArtifact]] = {}
        self._mission_jobs: dict[str, MissionJob] = {}
        self._mission_specs: dict[str, MissionSpec] = {}

    async def create_job(self, job: CodingJob) -> CodingJob:
        async with self._lock:
            self._jobs[job.id] = job
            self._steps[job.id] = []
            self._artifacts[job.id] = []
            return job

    async def get_job(self, job_id: str) -> CodingJob | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def list_jobs(self, statuses: set[JobStatus] | None = None) -> list[CodingJob]:
        async with self._lock:
            jobs = list(self._jobs.values())
        if statuses is None:
            return jobs
        return [job for job in jobs if job.status in statuses]

    async def claim_job(self, job_id: str) -> CodingJob | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != JobStatus.QUEUED:
                return None
            updated = replace(job, status=JobStatus.PREPARING, updated_at=utcnow())
            self._jobs[job_id] = updated
            return updated

    async def update_job(self, job_id: str, **changes: object) -> CodingJob:
        async with self._lock:
            job = self._jobs[job_id]
            if "status" in changes and not isinstance(changes["status"], JobStatus):
                changes["status"] = JobStatus(str(changes["status"]))
            changes["updated_at"] = utcnow()
            if changes.get("status") in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}:
                changes.setdefault("completed_at", utcnow())
            updated = replace(job, **changes)
            self._jobs[job_id] = updated
            return updated

    async def add_step(self, job_id: str, kind: str, content: dict[str, object]) -> DocodeStep:
        async with self._lock:
            step = DocodeStep(
                id=new_id("step"),
                job_id=job_id,
                step_index=len(self._steps.setdefault(job_id, [])),
                kind=kind,
                content=dict(content),
            )
            self._steps[job_id].append(step)
            job = self._jobs.get(job_id)
            if job is not None:
                self._jobs[job_id] = replace(job, updated_at=utcnow())
            return step

    async def list_steps(self, job_id: str) -> list[DocodeStep]:
        async with self._lock:
            return list(self._steps.get(job_id, []))

    async def list_steps_after(self, job_id: str, step_index: int) -> list[DocodeStep]:
        async with self._lock:
            return [step for step in self._steps.get(job_id, []) if step.step_index > step_index]

    async def add_artifact(self, job_id: str, kind: str, path: str, size_bytes: int) -> DocodeArtifact:
        async with self._lock:
            artifact = DocodeArtifact(
                id=new_id("art"),
                job_id=job_id,
                kind=kind,
                path=path,
                size_bytes=size_bytes,
            )
            self._artifacts.setdefault(job_id, []).append(artifact)
            job = self._jobs.get(job_id)
            if job is not None:
                self._jobs[job_id] = replace(job, updated_at=utcnow())
            return artifact

    async def get_artifact(self, artifact_id: str) -> DocodeArtifact | None:
        async with self._lock:
            for artifacts in self._artifacts.values():
                for artifact in artifacts:
                    if artifact.id == artifact_id:
                        return artifact
            return None

    async def list_artifacts(self, job_id: str) -> list[DocodeArtifact]:
        async with self._lock:
            return list(self._artifacts.get(job_id, []))

    async def create_mission_job(self, mission_job: MissionJob) -> MissionJob:
        async with self._lock:
            self._mission_jobs[mission_job.id] = mission_job
            return mission_job

    async def list_mission_jobs(self, mission_id: str | None = None) -> list[MissionJob]:
        async with self._lock:
            jobs = list(self._mission_jobs.values())
        if mission_id is None:
            return jobs
        return [job for job in jobs if job.mission_id == mission_id]

    async def create_mission_spec(self, mission_spec: MissionSpec) -> MissionSpec:
        async with self._lock:
            self._mission_specs[mission_spec.id] = mission_spec
            return mission_spec

    async def get_mission_spec(self, mission_id: str) -> MissionSpec | None:
        async with self._lock:
            return self._mission_specs.get(mission_id)


def terminal_status(status: JobStatus) -> bool:
    return status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}
