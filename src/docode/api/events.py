from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from docode.storage.models import JobStatus
from docode.storage.repository import JobRepository, terminal_status
from docode.storage.step_redaction import redacted_step_content


async def event_stream(repository: JobRepository, job_id: str):
    last_step_index = -1
    last_status: JobStatus | None = None
    while True:
        job = await repository.get_job(job_id)
        if job is None:
            yield sse("error", {"error": "job not found"})
            return

        if job.status != last_status:
            last_status = job.status
            yield sse("status", {"job_id": job.id, "status": job.status.value})

        for step in await repository.list_steps_after(job_id, last_step_index):
            last_step_index = step.step_index
            yield sse("step", step_event_payload(step))

        if terminal_status(job.status) and not awaiting_stopped_sandbox_finalization(job):
            yield sse(
                "done",
                {
                    "job_id": job.id,
                    "status": job.status.value,
                    "artifact_id": job.artifact_id,
                    "failure_reason": job.failure_reason,
                },
            )
            return

        await asyncio.sleep(1)


def awaiting_stopped_sandbox_finalization(job: Any) -> bool:
    return job.status == JobStatus.STOPPED and bool(job.dobox_project_id) and not job.artifact_id


def sse(event: str, data: object) -> str:
    payload = json.dumps(to_jsonable(data), ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def step_event_payload(step: Any) -> dict[str, object]:
    return {
        "step_id": step.id,
        "job_id": step.job_id,
        "step_index": step.step_index,
        "kind": step.kind,
        **stream_safe_step_content(step.content),
        "created_at": step.created_at,
    }


def stream_safe_step_content(content: dict[str, Any]) -> dict[str, object]:
    return redacted_step_content(content)


async def load_result_payload(repository: JobRepository, job_id: str) -> dict[str, Any] | None:
    for artifact in await repository.list_artifacts(job_id):
        if artifact.kind != "result":
            continue
        path = Path(artifact.path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None
    return None


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value
