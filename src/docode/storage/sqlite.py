from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import CodingJob, DocodeArtifact, DocodeStep, JobStatus, new_id, parse_datetime, utcnow
from .repository import JobRepository


class SQLiteJobRepository(JobRepository):
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = asyncio.Lock()
        self._migrate()

    def _migrate(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS docode_jobs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                instruction TEXT NOT NULL,
                repo_url TEXT,
                branch TEXT,
                github_repo TEXT,
                base_branch TEXT NOT NULL DEFAULT 'main',
                dobox_project_id TEXT,
                dobox_sandbox_id TEXT,
                dobox_agent_session_id TEXT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                apicred_access_token TEXT,
                status TEXT NOT NULL,
                max_iterations INTEGER NOT NULL,
                max_runtime_seconds INTEGER NOT NULL,
                max_consecutive_failures INTEGER NOT NULL DEFAULT 5,
                max_tool_calls INTEGER NOT NULL DEFAULT 100,
                max_llm_tokens INTEGER NOT NULL DEFAULT 100000,
                max_llm_cost REAL,
                artifact_mode TEXT NOT NULL DEFAULT 'patch',
                sandbox_network_mode TEXT NOT NULL DEFAULT 'project',
                result_summary TEXT,
                failure_reason TEXT,
                artifact_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_docode_jobs_status ON docode_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_docode_jobs_user_id ON docode_jobs(user_id);

            CREATE TABLE IF NOT EXISTS docode_steps (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_docode_steps_job_id ON docode_steps(job_id, step_index);

            CREATE TABLE IF NOT EXISTS docode_artifacts (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_docode_artifacts_job_id ON docode_artifacts(job_id);
            """
        )
        self._ensure_column("docode_jobs", "max_tool_calls", "INTEGER NOT NULL DEFAULT 100")
        self._ensure_column("docode_jobs", "max_consecutive_failures", "INTEGER NOT NULL DEFAULT 5")
        self._ensure_column("docode_jobs", "max_llm_tokens", "INTEGER NOT NULL DEFAULT 100000")
        self._ensure_column("docode_jobs", "max_llm_cost", "REAL")
        self._ensure_column("docode_jobs", "artifact_mode", "TEXT NOT NULL DEFAULT 'patch'")
        self._ensure_column("docode_jobs", "sandbox_network_mode", "TEXT NOT NULL DEFAULT 'project'")
        self._ensure_column("docode_jobs", "dobox_agent_session_id", "TEXT")
        self._ensure_column("docode_jobs", "github_repo", "TEXT")
        self._ensure_column("docode_jobs", "base_branch", "TEXT NOT NULL DEFAULT 'main'")
        self._ensure_column("docode_jobs", "apicred_access_token", "TEXT")
        self._conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        if column in {str(row["name"]) for row in rows}:
            return
        self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def close(self) -> None:
        self._conn.close()

    async def create_job(self, job: CodingJob) -> CodingJob:
        async with self._lock:
            self._conn.execute(
                """
                INSERT INTO docode_jobs (
                    id, user_id, instruction, repo_url, branch, github_repo, base_branch,
                    dobox_project_id, dobox_sandbox_id, dobox_agent_session_id, provider, model, apicred_access_token, status, max_iterations,
                    max_runtime_seconds, max_consecutive_failures, max_tool_calls, max_llm_tokens, max_llm_cost, artifact_mode, sandbox_network_mode, result_summary, failure_reason,
                    artifact_id, created_at,
                    updated_at, completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                job_to_row(job),
            )
            self._conn.commit()
            return job

    async def get_job(self, job_id: str) -> CodingJob | None:
        async with self._lock:
            row = self._conn.execute("SELECT * FROM docode_jobs WHERE id = ?", (job_id,)).fetchone()
            return job_from_row(row) if row else None

    async def list_jobs(self, statuses: set[JobStatus] | None = None) -> list[CodingJob]:
        async with self._lock:
            if statuses:
                values = [status.value for status in statuses]
                placeholders = ",".join("?" for _ in values)
                rows = self._conn.execute(
                    f"SELECT * FROM docode_jobs WHERE status IN ({placeholders}) ORDER BY created_at ASC",
                    values,
                ).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM docode_jobs ORDER BY created_at ASC").fetchall()
            return [job_from_row(row) for row in rows]

    async def claim_job(self, job_id: str) -> CodingJob | None:
        async with self._lock:
            now = iso(utcnow())
            cursor = self._conn.execute(
                "UPDATE docode_jobs SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
                (JobStatus.PREPARING.value, now, job_id, JobStatus.QUEUED.value),
            )
            if cursor.rowcount != 1:
                self._conn.commit()
                return None
            row = self._conn.execute("SELECT * FROM docode_jobs WHERE id = ?", (job_id,)).fetchone()
            self._conn.commit()
            return job_from_row(row) if row else None

    async def update_job(self, job_id: str, **changes: object) -> CodingJob:
        async with self._lock:
            row = self._conn.execute("SELECT * FROM docode_jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            job = job_from_row(row)
            if "status" in changes and not isinstance(changes["status"], JobStatus):
                changes["status"] = JobStatus(str(changes["status"]))
            changes["updated_at"] = utcnow()
            if changes.get("status") in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}:
                changes.setdefault("completed_at", utcnow())
            updated = job_replace(job, changes)
            self._conn.execute(
                """
                UPDATE docode_jobs SET
                    user_id = ?, instruction = ?, repo_url = ?, branch = ?, github_repo = ?,
                    base_branch = ?, dobox_project_id = ?, dobox_sandbox_id = ?, dobox_agent_session_id = ?, provider = ?, model = ?,
                    apicred_access_token = ?, status = ?, max_iterations = ?, max_runtime_seconds = ?, max_consecutive_failures = ?, max_tool_calls = ?,
                    max_llm_tokens = ?, max_llm_cost = ?, artifact_mode = ?, sandbox_network_mode = ?, result_summary = ?,
                    failure_reason = ?, artifact_id = ?, created_at = ?, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                job_update_row(updated),
            )
            self._conn.commit()
            return updated

    async def add_step(self, job_id: str, kind: str, content: dict[str, object]) -> DocodeStep:
        async with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(step_index), -1) + 1 AS next_index FROM docode_steps WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            step = DocodeStep(
                id=new_id("step"),
                job_id=job_id,
                step_index=int(row["next_index"]),
                kind=kind,
                content=dict(content),
            )
            self._conn.execute(
                "INSERT INTO docode_steps (id, job_id, step_index, kind, content, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                step_to_row(step),
            )
            self._conn.commit()
            return step

    async def list_steps(self, job_id: str) -> list[DocodeStep]:
        async with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM docode_steps WHERE job_id = ? ORDER BY step_index ASC",
                (job_id,),
            ).fetchall()
            return [step_from_row(row) for row in rows]

    async def list_steps_after(self, job_id: str, step_index: int) -> list[DocodeStep]:
        async with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM docode_steps WHERE job_id = ? AND step_index > ? ORDER BY step_index ASC",
                (job_id, step_index),
            ).fetchall()
            return [step_from_row(row) for row in rows]

    async def add_artifact(self, job_id: str, kind: str, path: str, size_bytes: int) -> DocodeArtifact:
        async with self._lock:
            artifact = DocodeArtifact(id=new_id("art"), job_id=job_id, kind=kind, path=path, size_bytes=size_bytes)
            self._conn.execute(
                "INSERT INTO docode_artifacts (id, job_id, kind, path, size_bytes, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                artifact_to_row(artifact),
            )
            self._conn.commit()
            return artifact

    async def get_artifact(self, artifact_id: str) -> DocodeArtifact | None:
        async with self._lock:
            row = self._conn.execute("SELECT * FROM docode_artifacts WHERE id = ?", (artifact_id,)).fetchone()
            return artifact_from_row(row) if row else None

    async def list_artifacts(self, job_id: str) -> list[DocodeArtifact]:
        async with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM docode_artifacts WHERE job_id = ? ORDER BY created_at ASC",
                (job_id,),
            ).fetchall()
            return [artifact_from_row(row) for row in rows]


def iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def job_to_row(job: CodingJob) -> tuple[object, ...]:
    return (
        job.id,
        job.user_id,
        job.instruction,
        job.repo_url,
        job.branch,
        job.github_repo,
        job.base_branch,
        job.dobox_project_id,
        job.dobox_sandbox_id,
        job.dobox_agent_session_id,
        job.provider,
        job.model,
        job.apicred_access_token,
        job.status.value,
        job.max_iterations,
        job.max_runtime_seconds,
        job.max_consecutive_failures,
        job.max_tool_calls,
        job.max_llm_tokens,
        job.max_llm_cost,
        job.artifact_mode,
        job.sandbox_network_mode,
        job.result_summary,
        job.failure_reason,
        job.artifact_id,
        iso(job.created_at),
        iso(job.updated_at),
        iso(job.completed_at),
    )


def job_update_row(job: CodingJob) -> tuple[object, ...]:
    return job_to_row(job)[1:] + (job.id,)


def job_from_row(row: sqlite3.Row) -> CodingJob:
    return CodingJob(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        instruction=str(row["instruction"]),
        repo_url=row["repo_url"],
        branch=row["branch"],
        github_repo=row["github_repo"] if "github_repo" in row.keys() else None,
        base_branch=str(row["base_branch"]) if "base_branch" in row.keys() and row["base_branch"] else "main",
        dobox_project_id=row["dobox_project_id"],
        dobox_sandbox_id=row["dobox_sandbox_id"],
        dobox_agent_session_id=row["dobox_agent_session_id"] if "dobox_agent_session_id" in row.keys() else None,
        provider=str(row["provider"]),
        model=str(row["model"]),
        apicred_access_token=row["apicred_access_token"] if "apicred_access_token" in row.keys() else None,
        status=JobStatus(str(row["status"])),
        max_iterations=int(row["max_iterations"]),
        max_runtime_seconds=int(row["max_runtime_seconds"]),
        max_consecutive_failures=int(row["max_consecutive_failures"]) if "max_consecutive_failures" in row.keys() else 5,
        max_tool_calls=int(row["max_tool_calls"]) if "max_tool_calls" in row.keys() else 100,
        max_llm_tokens=int(row["max_llm_tokens"]) if "max_llm_tokens" in row.keys() else 100_000,
        max_llm_cost=float(row["max_llm_cost"]) if "max_llm_cost" in row.keys() and row["max_llm_cost"] is not None else None,
        artifact_mode=str(row["artifact_mode"]) if "artifact_mode" in row.keys() else "patch",
        sandbox_network_mode=str(row["sandbox_network_mode"]) if "sandbox_network_mode" in row.keys() else "project",
        result_summary=row["result_summary"],
        failure_reason=row["failure_reason"],
        artifact_id=row["artifact_id"],
        created_at=parse_datetime(row["created_at"]) or utcnow(),
        updated_at=parse_datetime(row["updated_at"]) or utcnow(),
        completed_at=parse_datetime(row["completed_at"]),
    )


def job_replace(job: CodingJob, changes: dict[str, object]) -> CodingJob:
    values = {field: getattr(job, field) for field in job.__dataclass_fields__}
    values.update(changes)
    return CodingJob(**values)


def step_to_row(step: DocodeStep) -> tuple[object, ...]:
    return (
        step.id,
        step.job_id,
        step.step_index,
        step.kind,
        json.dumps(step.content, ensure_ascii=False),
        iso(step.created_at),
    )


def step_from_row(row: sqlite3.Row) -> DocodeStep:
    return DocodeStep(
        id=str(row["id"]),
        job_id=str(row["job_id"]),
        step_index=int(row["step_index"]),
        kind=str(row["kind"]),
        content=json.loads(str(row["content"])),
        created_at=parse_datetime(row["created_at"]) or utcnow(),
    )


def artifact_to_row(artifact: DocodeArtifact) -> tuple[object, ...]:
    return (artifact.id, artifact.job_id, artifact.kind, artifact.path, artifact.size_bytes, iso(artifact.created_at))


def artifact_from_row(row: sqlite3.Row) -> DocodeArtifact:
    return DocodeArtifact(
        id=str(row["id"]),
        job_id=str(row["job_id"]),
        kind=str(row["kind"]),
        path=str(row["path"]),
        size_bytes=int(row["size_bytes"]),
        created_at=parse_datetime(row["created_at"]) or utcnow(),
    )
