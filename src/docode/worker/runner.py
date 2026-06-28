from __future__ import annotations

from collections.abc import Awaitable, Callable

from docode.agent.loop import CodingAgentLoop
from docode.agent.stop_policy import StopPolicy
from docode.agent.tools import CompositeAgentTools
from docode.agent.verifier import CodingVerifier
from docode.artifacts.exporter import ArtifactExporter, terminal_artifact_id
from docode.artifacts.github import GitHubExporter
from docode.config import DocodeConfig
from docode.dobox.client import DoBoxClient
from docode.dobox.tools import DoBoxTools
from docode.llm.credentials import APICredCredentialResolver
from docode.llm.runtime import WeavVerifierJudge, build_docode_runtime
from docode.storage.models import JobStatus
from docode.storage.repository import JobRepository, terminal_status
from docode.web.tools import WebTools, WebToolsConfig


DoBoxClientFactory = Callable[[], DoBoxClient]
LLMFactory = Callable[[object], Awaitable[object]]
CredentialResolverFactory = Callable[[], APICredCredentialResolver]


class JobRunnerService:
    def __init__(
        self,
        *,
        config: DocodeConfig,
        repository: JobRepository,
        dobox_client_factory: DoBoxClientFactory | None = None,
        llm_factory: LLMFactory | None = None,
        credential_resolver_factory: CredentialResolverFactory | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.dobox_client_factory = dobox_client_factory or (lambda: DoBoxClient(self.config.dobox_base_url, self.config.dobox_token))
        self.llm_factory = llm_factory
        self.credential_resolver_factory = credential_resolver_factory or (
            lambda: APICredCredentialResolver(self.config.apicred_base_url, self.config.apicred_token, self.config.apicred_mode)
        )

    async def run_job(self, job_id: str) -> None:
        existing = await self.repository.get_job(job_id)
        if existing is None:
            return
        if existing.status == JobStatus.STOPPED:
            dobox = self.dobox_client_factory() if existing.dobox_project_id else None
            stopped = await self.finalize_stopped_job(job_id, dobox)
            if dobox is not None and stopped is not None:
                await self.cleanup_sandbox_if_needed(dobox, stopped)
            return
        if terminal_status(existing.status):
            return
        job = await self.repository.claim_job(job_id)
        if job is None:
            return
        try:
            resolver = self.credential_resolver_factory()
            resolver.use_access_token(job.apicred_access_token)
            try:
                authorization = await resolver.authorize(
                    user_id=job.user_id,
                    provider=job.provider,
                    model=job.model,
                    job_id=job.id,
                    max_iterations=job.max_iterations,
                    max_runtime_seconds=job.max_runtime_seconds,
                    max_tool_calls=job.max_tool_calls,
                    max_llm_tokens=job.max_llm_tokens,
                    max_llm_cost=job.max_llm_cost,
                    sandbox_network_mode=job.sandbox_network_mode,
                    artifact_mode=job.artifact_mode,
                )
            except Exception as exc:
                reason = f"apicred_authorize_failed:{exc}"
                await self.repository.add_step(
                    job.id,
                    "system",
                    {
                        "type": "apicred_authorize",
                        "status": "failed",
                        "provider": job.provider,
                        "model": job.model,
                        "error": str(exc),
                    },
                )
                artifact_id = await self.export_failure_artifacts(job, reason)
                await self.repository.update_job(job.id, status=JobStatus.FAILED, failure_reason=reason, artifact_id=artifact_id)
                return
            await self.repository.add_step(
                job.id,
                "system",
                {
                    "type": "apicred_authorize",
                    "allowed": authorization.allowed,
                    "reason": authorization.reason,
                    "budget_tokens": authorization.budget_tokens,
                    "budget_cost": authorization.budget_cost,
                },
            )
            if not authorization.allowed:
                reason = authorization.reason or "apicred_authorization_denied"
                artifact_id = await self.export_failure_artifacts(job, reason)
                await self.repository.update_job(job.id, status=JobStatus.FAILED, failure_reason=reason, artifact_id=artifact_id)
                return
            if await self._is_stopped(job.id):
                await self.finalize_stopped_job(job.id)
                return

            dobox = self.dobox_client_factory()
            project = await dobox.create_project(
                name=f"docode-{job.id}",
                repo_url=job.repo_url,
                branch=job.branch,
                network_mode=job.sandbox_network_mode,
            )
            job = await self.repository.update_job(job.id, dobox_project_id=project.project_id, dobox_sandbox_id=project.sandbox_id)
            if await self._is_stopped(job.id):
                stopped = await self.finalize_stopped_job(job.id, dobox)
                if stopped is not None:
                    await self.cleanup_sandbox_if_needed(dobox, stopped)
                return
            session = await dobox.create_agent_session(project.project_id, name=f"docode-{job.id}")
            job = await self.repository.update_job(job.id, dobox_agent_session_id=session.session_id)
            if await self._is_stopped(job.id):
                stopped = await self.finalize_stopped_job(job.id, dobox)
                if stopped is not None:
                    await self.cleanup_sandbox_if_needed(dobox, stopped)
                return
            tools = DoBoxTools(
                dobox,
                project.project_id,
                agent_session_id=session.session_id,
                command_timeout_seconds=self.config.command_timeout_seconds,
                output_limit_bytes=self.config.output_limit_bytes,
            )
            agent_tools = CompositeAgentTools(tools, self.build_web_tools())
            runtime = await build_docode_runtime(job, resolver, agent_tools)
            await self.repository.add_step(
                job.id,
                "system",
                {
                    "type": "runtime_assembly",
                    "router": type(runtime.router).__name__,
                    "tool_registry": type(runtime.tools).__name__ if runtime.tools is not None else None,
                    "provider": runtime.provider,
                    "model": runtime.model,
                    "sandbox_network_mode": job.sandbox_network_mode,
                    "dobox_agent_session_id": session.session_id,
                    "tools": [definition.name for definition in agent_tools.definitions()],
                },
            )
            if await self._is_stopped(job.id):
                stopped = await self.finalize_stopped_job(job.id, dobox)
                if stopped is not None:
                    await self.cleanup_sandbox_if_needed(dobox, stopped)
                return
            llm = await self.llm_factory(job) if self.llm_factory is not None else runtime.llm
            verifier_judge = (
                WeavVerifierJudge(runtime.provider_client, runtime.model, runtime.usage_meter) if runtime.provider_client is not None else None
            )
            max_llm_tokens = runtime_budget(job.max_llm_tokens, authorization.budget_tokens)
            max_llm_cost = runtime_budget(job.max_llm_cost, authorization.budget_cost)
            loop = CodingAgentLoop(
                llm=llm,
                tools=agent_tools,
                verifier=CodingVerifier(judge=verifier_judge),
                repository=self.repository,
                exporter=ArtifactExporter(
                    self.config.artifact_dir,
                    self.repository,
                    workspace_archive_provider=lambda: dobox.archive_workspace(project.project_id, agent_session_id=session.session_id),
                    commit_provider=lambda message: dobox.git_commit(project.project_id, message, agent_session_id=session.session_id),
                    github_exporter=GitHubExporter(enabled=self.config.github_export_enabled, work_dir=self.config.github_work_dir),
                ),
                stop_policy=StopPolicy(
                    max_iterations=job.max_iterations,
                    max_runtime_seconds=job.max_runtime_seconds,
                    max_consecutive_failures=job.max_consecutive_failures,
                    max_tool_calls=job.max_tool_calls,
                    max_llm_tokens=max_llm_tokens,
                    max_llm_cost=max_llm_cost,
                ),
                usage_meter=runtime.usage_meter,
            )
            completed = await loop.run(job)
            usage = runtime.usage_meter.snapshot()
            await self.report_usage_best_effort(resolver, completed, usage, max_llm_tokens, max_llm_cost)
            await self.cleanup_sandbox_if_needed(dobox, completed)
        except Exception as exc:
            if not await self._is_stopped(job.id):
                current = await self.repository.get_job(job.id) or job
                artifact_id = await self.export_failure_artifacts(current, str(exc), dobox if "dobox" in locals() else None)
                failed = await self.repository.update_job(job.id, status=JobStatus.FAILED, failure_reason=str(exc), artifact_id=artifact_id)
                if "dobox" in locals():
                    await self.cleanup_sandbox_if_needed(dobox, failed)

    async def _is_stopped(self, job_id: str) -> bool:
        current = await self.repository.get_job(job_id)
        return current is not None and current.status == JobStatus.STOPPED

    def build_web_tools(self) -> WebTools | None:
        if not self.config.web_tools_enabled:
            return None
        return WebTools(
            WebToolsConfig(
                openai_api_key=self.config.openai_api_key,
                openai_base_url=self.config.openai_base_url,
                openai_search_model=self.config.openai_search_model,
                openai_search_tool_type=self.config.openai_search_tool_type,
                search_context_size=self.config.web_search_context_size,
                fetch_timeout_seconds=self.config.web_fetch_timeout_seconds,
                output_limit_bytes=self.config.output_limit_bytes,
                allow_private_hosts=self.config.web_fetch_allow_private_hosts,
            )
        )

    async def report_usage_best_effort(
        self,
        resolver: APICredCredentialResolver,
        completed,
        usage: dict[str, object],
        budget_tokens: int | None,
        budget_cost: float | None,
    ) -> None:
        try:
            await resolver.report_usage(
                user_id=completed.user_id,
                provider=completed.provider,
                model=completed.model,
                tokens=int(usage["total_tokens"]),
                cost=float(usage["cost"]),
            )
        except Exception as exc:
            await self.repository.add_step(
                completed.id,
                "system",
                {
                    "type": "apicred_usage_report",
                    "status": "failed",
                    "provider": completed.provider,
                    "model": completed.model,
                    "tokens": usage["total_tokens"],
                    "cost": usage["cost"],
                    "usage": usage,
                    "budget_tokens": budget_tokens,
                    "budget_cost": budget_cost,
                    "error": str(exc),
                },
            )
            return
        await self.repository.add_step(
            completed.id,
            "system",
            {
                "type": "apicred_usage_report",
                "status": "reported",
                "provider": completed.provider,
                "model": completed.model,
                "tokens": usage["total_tokens"],
                "cost": usage["cost"],
                "usage": usage,
                "budget_tokens": budget_tokens,
                "budget_cost": budget_cost,
            },
        )

    async def finalize_stopped_job(self, job_id: str, dobox: DoBoxClient | None = None):
        stopped = await self.repository.get_job(job_id)
        if stopped is None or stopped.status != JobStatus.STOPPED or stopped.artifact_id:
            return stopped
        artifact_id = await self.export_stopped_artifacts(stopped, stopped.failure_reason or "cancelled", dobox)
        return await self.repository.update_job(job_id, artifact_id=artifact_id)

    async def cleanup_sandbox_if_needed(self, dobox: DoBoxClient, job) -> None:
        policy = self.config.sandbox_retention
        should_delete = policy == "delete_always" or (policy == "delete_on_success" and job.status == JobStatus.SUCCEEDED)
        if not job.dobox_project_id:
            await self.repository.add_step(job.id, "system", {"type": "sandbox_cleanup", "policy": policy, "status": "skipped", "reason": "no_project"})
            return
        if not should_delete:
            await self.repository.add_step(job.id, "system", {"type": "sandbox_cleanup", "policy": policy, "status": "kept", "project_id": job.dobox_project_id})
            return
        try:
            await dobox.delete_project(job.dobox_project_id)
        except Exception as exc:
            await self.repository.add_step(
                job.id,
                "system",
                {"type": "sandbox_cleanup", "policy": policy, "status": "failed", "project_id": job.dobox_project_id, "error": str(exc)},
            )
            return
        await self.repository.add_step(
            job.id,
            "system",
            {"type": "sandbox_cleanup", "policy": policy, "status": "deleted", "project_id": job.dobox_project_id},
        )

    async def export_failure_artifacts(self, job, reason: str, dobox: DoBoxClient | None = None) -> str | None:
        git_diff = ""
        git_diff_truncated = False
        if dobox is not None and job.dobox_project_id:
            try:
                if hasattr(dobox, "git_diff_result"):
                    diff_result = await dobox.git_diff_result(job.dobox_project_id, agent_session_id=job.dobox_agent_session_id)
                    git_diff = diff_result.output
                    git_diff_truncated = diff_result.truncated
                else:
                    git_diff = await dobox.git_diff(job.dobox_project_id, agent_session_id=job.dobox_agent_session_id)
            except Exception:
                git_diff = ""
                git_diff_truncated = False
        archive_provider = None
        if dobox is not None and job.dobox_project_id:
            archive_provider = lambda: dobox.archive_workspace(job.dobox_project_id, agent_session_id=job.dobox_agent_session_id)
        try:
            artifacts = await ArtifactExporter(
                self.config.artifact_dir,
                self.repository,
                workspace_archive_provider=archive_provider,
            ).export_failure(
                job,
                reason,
                steps=await self.repository.list_steps(job.id),
                git_diff=git_diff,
                git_diff_truncated=git_diff_truncated,
            )
        except Exception as export_exc:
            await self.repository.add_step(job.id, "system", {"type": "failure_export_failed", "reason": reason, "error": str(export_exc)})
            return None
        artifact_id = terminal_artifact_id(artifacts)
        await self.repository.add_step(job.id, "system", {"type": "failure_artifacts_exported", "reason": reason, "artifact_id": artifact_id})
        return artifact_id

    async def export_stopped_artifacts(self, job, reason: str, dobox: DoBoxClient | None = None) -> str | None:
        git_diff = ""
        git_diff_truncated = False
        if dobox is not None and job.dobox_project_id:
            try:
                if hasattr(dobox, "git_diff_result"):
                    diff_result = await dobox.git_diff_result(job.dobox_project_id, agent_session_id=job.dobox_agent_session_id)
                    git_diff = diff_result.output
                    git_diff_truncated = diff_result.truncated
                else:
                    git_diff = await dobox.git_diff(job.dobox_project_id, agent_session_id=job.dobox_agent_session_id)
            except Exception:
                git_diff = ""
                git_diff_truncated = False
        archive_provider = None
        if dobox is not None and job.dobox_project_id:
            archive_provider = lambda: dobox.archive_workspace(job.dobox_project_id, agent_session_id=job.dobox_agent_session_id)
        try:
            artifacts = await ArtifactExporter(
                self.config.artifact_dir,
                self.repository,
                workspace_archive_provider=archive_provider,
            ).export_stopped(
                job,
                reason,
                steps=await self.repository.list_steps(job.id),
                git_diff=git_diff,
                git_diff_truncated=git_diff_truncated,
            )
        except Exception as export_exc:
            await self.repository.add_step(job.id, "system", {"type": "stopped_export_failed", "reason": reason, "error": str(export_exc)})
            return None
        artifact_id = terminal_artifact_id(artifacts)
        await self.repository.add_step(job.id, "system", {"type": "stopped_artifacts_exported", "reason": reason, "artifact_id": artifact_id})
        return artifact_id


def runtime_budget(job_budget, apicred_budget):
    budgets = [budget for budget in (job_budget, apicred_budget) if budget is not None and budget > 0]
    return min(budgets) if budgets else None
