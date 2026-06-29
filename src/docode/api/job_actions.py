from __future__ import annotations

from dataclasses import dataclass

from docode.artifacts.terminal import export_stopped_artifacts
from docode.config import DocodeConfig
from docode.llm.model_policy import DocodeModelPolicy
from docode.sandbox import normalize_sandbox_network_mode
from docode.storage.models import CodingJob, JobStatus
from docode.storage.models import new_id
from docode.storage.repository import JobRepository, terminal_status
from docode.worker.queue import AsyncJobQueue


ALLOWED_ARTIFACT_MODES = frozenset({"patch", "zip", "commit", "pr"})
MAX_ITERATIONS_LIMIT = 200
MAX_RUNTIME_SECONDS_LIMIT = 24 * 60 * 60
MAX_CONSECUTIVE_FAILURES_LIMIT = 200
MAX_TOOL_CALLS_LIMIT = 1000
MAX_LLM_TOKENS_LIMIT = 10_000_000
MAX_LLM_COST_LIMIT = 10_000.0
HIGH_VALIDATION_DEFAULT_MAX_ITERATIONS = 100
HIGH_VALIDATION_DEFAULT_MAX_TOOL_CALLS = 250
HIGH_VALIDATION_DEFAULT_MAX_CONSECUTIVE_FAILURES = 12


@dataclass(frozen=True, slots=True)
class CreateJobInput:
    instruction: str
    repo_url: str | None = None
    branch: str | None = None
    github_repo: str | None = None
    base_branch: str | None = None
    provider: str | None = None
    model: str | None = None
    quality: str | None = None
    max_iterations: int | None = None
    max_runtime_seconds: int | None = None
    max_consecutive_failures: int | None = None
    max_tool_calls: int | None = None
    max_llm_tokens: int | None = None
    max_llm_cost: float | None = None
    artifact_mode: str | None = None
    sandbox_network_mode: str | None = None


class JobActionError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


async def create_coding_job(
    *,
    repository: JobRepository,
    queue: AsyncJobQueue,
    config: DocodeConfig,
    model_policy: DocodeModelPolicy,
    user_id: str,
    request: CreateJobInput,
    apicred_access_token: str | None = None,
) -> CodingJob:
    try:
        sandbox_network_mode = normalize_sandbox_network_mode(request.sandbox_network_mode or config.sandbox_network_mode)
    except ValueError as exc:
        raise JobActionError(400, str(exc)) from exc

    try:
        default_max_iterations = default_iterations_for_request(request, config)
        default_max_tool_calls = default_tool_calls_for_request(request, config)
        max_iterations = bounded_int("max_iterations", request.max_iterations, default_max_iterations, minimum=1, maximum=MAX_ITERATIONS_LIMIT)
        max_runtime_seconds = bounded_int(
            "max_runtime_seconds",
            request.max_runtime_seconds,
            config.max_runtime_seconds,
            minimum=30,
            maximum=MAX_RUNTIME_SECONDS_LIMIT,
        )
        max_consecutive_failures = bounded_int(
            "max_consecutive_failures",
            request.max_consecutive_failures,
            default_consecutive_failures_for_request(request),
            minimum=1,
            maximum=MAX_CONSECUTIVE_FAILURES_LIMIT,
        )
        max_tool_calls = bounded_int("max_tool_calls", request.max_tool_calls, default_max_tool_calls, minimum=1, maximum=MAX_TOOL_CALLS_LIMIT)
        max_llm_tokens = bounded_int("max_llm_tokens", request.max_llm_tokens, config.max_llm_tokens, minimum=1, maximum=MAX_LLM_TOKENS_LIMIT)
        max_llm_cost = runtime_cost_budget(request.max_llm_cost, config.max_llm_cost)
        artifact_mode = normalize_artifact_mode(request.artifact_mode)
    except ValueError as exc:
        raise JobActionError(400, str(exc)) from exc

    try:
        resolved_model = await model_policy.resolve(provider=request.provider, model=request.model, quality=request.quality, user_id=user_id)
    except ValueError as exc:
        raise JobActionError(400, str(exc)) from exc
    if not resolved_model.allowed:
        raise JobActionError(400, resolved_model.reason)

    job = CodingJob(
        id=new_id("job"),
        user_id=user_id,
        instruction=request.instruction,
        repo_url=request.repo_url,
        branch=request.branch,
        github_repo=request.github_repo,
        base_branch=request.base_branch or config.github_base_branch,
        provider=resolved_model.provider,
        model=resolved_model.model,
        quality=resolved_model.quality,
        apicred_access_token=apicred_access_token,
        max_iterations=max_iterations,
        max_runtime_seconds=max_runtime_seconds,
        max_consecutive_failures=max_consecutive_failures,
        max_tool_calls=max_tool_calls,
        max_llm_tokens=max_llm_tokens,
        max_llm_cost=max_llm_cost,
        artifact_mode=artifact_mode,
        sandbox_network_mode=sandbox_network_mode,
    )
    created = await repository.create_job(job)
    await queue.enqueue(created.id)
    return created


def positive_min(*values: float | None) -> float | None:
    budgets = [value for value in values if value is not None and value > 0]
    return min(budgets) if budgets else None


def bounded_int(name: str, requested: int | None, configured: int, *, minimum: int, maximum: int) -> int:
    value = requested if requested is not None else configured
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def default_iterations_for_request(request: CreateJobInput, config: DocodeConfig) -> int:
    if request.max_iterations is not None or not needs_expanded_repair_budget(request.instruction):
        return config.max_iterations
    return min(MAX_ITERATIONS_LIMIT, max(config.max_iterations, HIGH_VALIDATION_DEFAULT_MAX_ITERATIONS))


def default_tool_calls_for_request(request: CreateJobInput, config: DocodeConfig) -> int:
    if request.max_tool_calls is not None or not needs_expanded_repair_budget(request.instruction):
        return config.max_tool_calls
    return min(MAX_TOOL_CALLS_LIMIT, max(config.max_tool_calls, HIGH_VALIDATION_DEFAULT_MAX_TOOL_CALLS))


def default_consecutive_failures_for_request(request: CreateJobInput) -> int:
    if request.max_consecutive_failures is not None or not needs_expanded_repair_budget(request.instruction):
        return 5
    return HIGH_VALIDATION_DEFAULT_MAX_CONSECUTIVE_FAILURES


def needs_expanded_repair_budget(instruction: str) -> bool:
    lowered = (instruction or "").lower()
    keywords = (
        "crawler",
        "scraper",
        "scrape",
        "spider",
        "爬虫",
        "抓取",
        "采集",
        "下载",
        "每日",
        "长期",
        "数据源",
        "联网",
        "web_search",
        "fetch_url",
        "etl",
        "pipeline",
        "script",
        "脚本",
        "cli",
        "命令行",
    )
    return any(keyword in lowered for keyword in keywords)


def runtime_cost_budget(requested: float | None, configured: float | None) -> float | None:
    for name, value in (("max_llm_cost", requested), ("configured max_llm_cost", configured)):
        if value is not None and (value <= 0 or value > MAX_LLM_COST_LIMIT):
            raise ValueError(f"{name} must be greater than 0 and at most {MAX_LLM_COST_LIMIT:g}")
    return positive_min(requested, configured)


def normalize_artifact_mode(value: str | None) -> str:
    mode = (value or "patch").strip().lower()
    if mode not in ALLOWED_ARTIFACT_MODES:
        raise ValueError("artifact_mode must be patch, zip, commit, or pr")
    return mode


async def cancel_existing_job(repository: JobRepository, queue: AsyncJobQueue, config: DocodeConfig, job: CodingJob) -> dict[str, str]:
    if terminal_status(job.status):
        return {"job_id": job.id, "status": job.status.value}
    await repository.add_step(job.id, "system", {"type": "cancelled", "reason": "user_requested_cancel"})
    if job.dobox_project_id:
        await repository.update_job(job.id, status=JobStatus.STOPPED, failure_reason="cancelled")
        await queue.enqueue(job.id)
        return {"job_id": job.id, "status": "stopped"}
    artifact_id = await export_stopped_artifacts(repository, config.artifact_dir, job, "cancelled")
    await repository.update_job(job.id, status=JobStatus.STOPPED, failure_reason="cancelled", artifact_id=artifact_id)
    return {"job_id": job.id, "status": "stopped"}
