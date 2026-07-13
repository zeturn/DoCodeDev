from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from docode.defaults import DEFAULT_MODEL, DEFAULT_PROVIDER, DEFAULT_QUALITY


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    VERIFYING = "verifying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(slots=True)
class CodingJob:
    id: str
    user_id: str
    instruction: str
    repo_url: str | None = None
    branch: str | None = None
    github_repo: str | None = None
    base_branch: str = "main"
    dobox_project_id: str | None = None
    dobox_sandbox_id: str | None = None
    dobox_agent_session_id: str | None = None
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    quality: str = DEFAULT_QUALITY
    apicred_access_token: str | None = field(default=None, repr=False)
    status: JobStatus = JobStatus.QUEUED
    max_iterations: int = 50
    max_runtime_seconds: int = 1800
    max_consecutive_failures: int = 5
    max_tool_calls: int = 100
    max_llm_tokens: int = 100_000
    max_llm_cost: float | None = None
    artifact_mode: str = "patch"
    sandbox_network_mode: str = "project"
    result_summary: str | None = None
    failure_reason: str | None = None
    artifact_id: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    completed_at: datetime | None = None


SENSITIVE_JOB_FIELDS = frozenset({"apicred_access_token"})


def public_job_dict(job: CodingJob) -> dict[str, Any]:
    return {field_name: getattr(job, field_name) for field_name in job.__dataclass_fields__ if field_name not in SENSITIVE_JOB_FIELDS}


@dataclass(slots=True)
class DocodeStep:
    id: str
    job_id: str
    step_index: int
    kind: str
    content: dict[str, Any]
    created_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class DocodeArtifact:
    id: str
    job_id: str
    kind: str
    path: str
    size_bytes: int
    created_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class MissionJob:
    id: str
    mission_id: str
    name: str
    kind: str = "moiip_rss_tdt"
    status: str = "created"
    araneae_task_id: str | None = None
    araneae_schedule_id: str | None = None
    hashslip_input_collection: str | None = None
    hashslip_output_collection: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
