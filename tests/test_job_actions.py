from __future__ import annotations

from unittest import IsolatedAsyncioTestCase

from docode.api.job_actions import CreateJobInput, JobActionError, create_coding_job
from docode.config import DocodeConfig
from docode.llm.model_policy import ModelPolicyResult
from docode.storage.repository import InMemoryJobRepository


class RecordingQueue:
    def __init__(self) -> None:
        self.enqueued: list[str] = []

    async def enqueue(self, job_id: str) -> None:
        self.enqueued.append(job_id)


class FakeModelPolicy:
    def __init__(self, result: ModelPolicyResult) -> None:
        self.result = result
        self.calls: list[dict[str, str | None]] = []

    async def resolve(
        self,
        *,
        provider: str | None,
        model: str | None,
        quality: str | None = None,
        user_id: str | None = None,
    ) -> ModelPolicyResult:
        self.calls.append({"provider": provider, "model": model, "quality": quality, "user_id": user_id})
        return self.result


class JobActionTests(IsolatedAsyncioTestCase):
    async def test_create_coding_job_applies_defaults_and_enqueues_job(self) -> None:
        repo = InMemoryJobRepository()
        queue = RecordingQueue()
        policy = FakeModelPolicy(ModelPolicyResult(provider="scripted", model="scripted", allowed=True))
        config = DocodeConfig(
            max_iterations=12,
            max_runtime_seconds=900,
            max_tool_calls=33,
            max_llm_tokens=44_000,
            max_llm_cost=1.25,
            github_base_branch="develop",
            sandbox_network_mode="bridge",
        )

        job = await create_coding_job(
            repository=repo,
            queue=queue,  # type: ignore[arg-type]
            config=config,
            model_policy=policy,  # type: ignore[arg-type]
            user_id="user-1",
            apicred_access_token="bp_xat_cross_app",
            request=CreateJobInput(instruction="add settings page", provider="dev"),
        )

        self.assertEqual(job.user_id, "user-1")
        self.assertEqual(job.apicred_access_token, "bp_xat_cross_app")
        self.assertEqual(job.provider, "scripted")
        self.assertEqual(job.model, "scripted")
        self.assertEqual(job.quality, "balanced")
        self.assertEqual(job.max_iterations, 12)
        self.assertEqual(job.max_runtime_seconds, 900)
        self.assertEqual(job.max_consecutive_failures, 5)
        self.assertEqual(job.max_tool_calls, 33)
        self.assertEqual(job.max_llm_tokens, 44_000)
        self.assertEqual(job.max_llm_cost, 1.25)
        self.assertEqual(job.artifact_mode, "patch")
        self.assertEqual(job.base_branch, "develop")
        self.assertEqual(job.sandbox_network_mode, "project")
        self.assertEqual(queue.enqueued, [job.id])
        self.assertIs(await repo.get_job(job.id), job)
        self.assertEqual(policy.calls, [{"provider": "dev", "model": None, "quality": None, "user_id": "user-1"}])

    async def test_create_coding_job_preserves_explicit_runtime_fields(self) -> None:
        repo = InMemoryJobRepository()
        queue = RecordingQueue()
        policy = FakeModelPolicy(ModelPolicyResult(provider="anthropic", model="claude-sonnet-4-5", allowed=True, quality="strong"))

        job = await create_coding_job(
            repository=repo,
            queue=queue,  # type: ignore[arg-type]
            config=DocodeConfig(),
            model_policy=policy,  # type: ignore[arg-type]
            user_id="user-1",
            request=CreateJobInput(
                instruction="fix build",
                repo_url="https://github.com/acme/app",
                branch="feature/input",
                github_repo="acme/app",
                base_branch="release",
                provider="anthropic",
                model="claude-sonnet-4-5",
                quality="strong",
                max_iterations=4,
                max_runtime_seconds=120,
                max_consecutive_failures=7,
                max_tool_calls=8,
                max_llm_tokens=9000,
                max_llm_cost=0.5,
                artifact_mode="pr",
                sandbox_network_mode="offline",
            ),
        )

        self.assertEqual(job.repo_url, "https://github.com/acme/app")
        self.assertEqual(job.branch, "feature/input")
        self.assertEqual(job.github_repo, "acme/app")
        self.assertEqual(job.base_branch, "release")
        self.assertEqual(job.provider, "anthropic")
        self.assertEqual(job.model, "claude-sonnet-4-5")
        self.assertEqual(job.quality, "strong")
        self.assertEqual(job.max_iterations, 4)
        self.assertEqual(job.max_runtime_seconds, 120)
        self.assertEqual(job.max_consecutive_failures, 7)
        self.assertEqual(job.max_tool_calls, 8)
        self.assertEqual(job.max_llm_tokens, 9000)
        self.assertEqual(job.max_llm_cost, 0.5)
        self.assertEqual(job.artifact_mode, "pr")
        self.assertEqual(job.sandbox_network_mode, "no_internet")
        self.assertEqual(queue.enqueued, [job.id])

    async def test_create_coding_job_expands_budget_for_crawler_tasks(self) -> None:
        repo = InMemoryJobRepository()
        queue = RecordingQueue()
        policy = FakeModelPolicy(ModelPolicyResult(provider="openai", model="gpt-4o-mini", allowed=True))

        job = await create_coding_job(
            repository=repo,
            queue=queue,  # type: ignore[arg-type]
            config=DocodeConfig(max_iterations=50, max_tool_calls=100),
            model_policy=policy,  # type: ignore[arg-type]
            user_id="user-1",
            request=CreateJobInput(instruction="帮我联网寻找数据源并生成一个长期每日更新数据的 Python 爬虫"),
        )

        self.assertEqual(job.max_iterations, 100)
        self.assertEqual(job.max_consecutive_failures, 12)
        self.assertEqual(job.max_tool_calls, 250)

    async def test_create_coding_job_passes_quality_to_model_policy(self) -> None:
        repo = InMemoryJobRepository()
        queue = RecordingQueue()
        policy = FakeModelPolicy(ModelPolicyResult(provider="openai", model="gpt-4o-mini", allowed=True, quality="fast"))

        job = await create_coding_job(
            repository=repo,
            queue=queue,  # type: ignore[arg-type]
            config=DocodeConfig(),
            model_policy=policy,  # type: ignore[arg-type]
            user_id="user-1",
            request=CreateJobInput(instruction="fix build", quality="fast"),
        )

        self.assertEqual(job.quality, "fast")
        self.assertEqual(policy.calls, [{"provider": None, "model": None, "quality": "fast", "user_id": "user-1"}])

    async def test_create_coding_job_keeps_explicit_budget_for_crawler_tasks(self) -> None:
        repo = InMemoryJobRepository()
        queue = RecordingQueue()
        policy = FakeModelPolicy(ModelPolicyResult(provider="openai", model="gpt-4o-mini", allowed=True))

        job = await create_coding_job(
            repository=repo,
            queue=queue,  # type: ignore[arg-type]
            config=DocodeConfig(max_iterations=50, max_tool_calls=100),
            model_policy=policy,  # type: ignore[arg-type]
            user_id="user-1",
            request=CreateJobInput(
                instruction="帮我联网寻找数据源并生成一个长期每日更新数据的 Python 爬虫",
                max_iterations=12,
                max_consecutive_failures=4,
                max_tool_calls=30,
            ),
        )

        self.assertEqual(job.max_iterations, 12)
        self.assertEqual(job.max_consecutive_failures, 4)
        self.assertEqual(job.max_tool_calls, 30)

    async def test_create_coding_job_uses_stricter_configured_cost_budget(self) -> None:
        repo = InMemoryJobRepository()
        queue = RecordingQueue()
        policy = FakeModelPolicy(ModelPolicyResult(provider="openai", model="gpt-4o", allowed=True))

        job = await create_coding_job(
            repository=repo,
            queue=queue,  # type: ignore[arg-type]
            config=DocodeConfig(max_llm_cost=0.25),
            model_policy=policy,  # type: ignore[arg-type]
            user_id="user-1",
            request=CreateJobInput(instruction="fix build", provider="openai", model="gpt-4o", max_llm_cost=0.5),
        )

        self.assertEqual(job.max_llm_cost, 0.25)

    async def test_create_coding_job_rejects_model_policy_denial_without_enqueue(self) -> None:
        repo = InMemoryJobRepository()
        queue = RecordingQueue()
        policy = FakeModelPolicy(ModelPolicyResult(provider="openai", model="missing", allowed=False, reason="model_not_available"))

        with self.assertRaises(JobActionError) as raised:
            await create_coding_job(
                repository=repo,
                queue=queue,  # type: ignore[arg-type]
                config=DocodeConfig(),
                model_policy=policy,  # type: ignore[arg-type]
                user_id="user-1",
                request=CreateJobInput(instruction="fix build", provider="openai", model="missing"),
            )

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail, "model_not_available")
        self.assertEqual(await repo.list_jobs(), [])
        self.assertEqual(queue.enqueued, [])

    async def test_create_coding_job_rejects_raw_docker_network_mode_without_enqueue(self) -> None:
        repo = InMemoryJobRepository()
        queue = RecordingQueue()
        policy = FakeModelPolicy(ModelPolicyResult(provider="scripted", model="scripted", allowed=True))

        with self.assertRaises(JobActionError) as raised:
            await create_coding_job(
                repository=repo,
                queue=queue,  # type: ignore[arg-type]
                config=DocodeConfig(),
                model_policy=policy,  # type: ignore[arg-type]
                user_id="user-1",
                request=CreateJobInput(instruction="fix build", sandbox_network_mode="host"),
            )

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail, "sandbox_network_mode must be project or no_internet")
        self.assertEqual(await repo.list_jobs(), [])
        self.assertEqual(queue.enqueued, [])
        self.assertEqual(policy.calls, [])

    async def test_create_coding_job_rejects_invalid_runtime_policy_without_enqueue(self) -> None:
        cases = [
            (CreateJobInput(instruction="fix build", artifact_mode="tar"), "artifact_mode must be patch, zip, commit, or pr"),
            (CreateJobInput(instruction="fix build", max_iterations=0), "max_iterations must be between 1 and 200"),
            (CreateJobInput(instruction="fix build", max_runtime_seconds=29), "max_runtime_seconds must be between 30 and 86400"),
            (CreateJobInput(instruction="fix build", max_consecutive_failures=0), "max_consecutive_failures must be between 1 and 200"),
            (CreateJobInput(instruction="fix build", max_tool_calls=0), "max_tool_calls must be between 1 and 1000"),
            (CreateJobInput(instruction="fix build", max_llm_tokens=0), "max_llm_tokens must be between 1 and 10000000"),
            (CreateJobInput(instruction="fix build", max_llm_cost=0), "max_llm_cost must be greater than 0 and at most 10000"),
        ]

        for request, expected_detail in cases:
            with self.subTest(expected_detail=expected_detail):
                repo = InMemoryJobRepository()
                queue = RecordingQueue()
                policy = FakeModelPolicy(ModelPolicyResult(provider="scripted", model="scripted", allowed=True))

                with self.assertRaises(JobActionError) as raised:
                    await create_coding_job(
                        repository=repo,
                        queue=queue,  # type: ignore[arg-type]
                        config=DocodeConfig(),
                        model_policy=policy,  # type: ignore[arg-type]
                        user_id="user-1",
                        request=request,
                    )

                self.assertEqual(raised.exception.status_code, 400)
                self.assertEqual(raised.exception.detail, expected_detail)
                self.assertEqual(await repo.list_jobs(), [])
                self.assertEqual(queue.enqueued, [])
                self.assertEqual(policy.calls, [])

    async def test_create_coding_job_normalizes_artifact_mode(self) -> None:
        repo = InMemoryJobRepository()
        queue = RecordingQueue()
        policy = FakeModelPolicy(ModelPolicyResult(provider="scripted", model="scripted", allowed=True))

        job = await create_coding_job(
            repository=repo,
            queue=queue,  # type: ignore[arg-type]
            config=DocodeConfig(),
            model_policy=policy,  # type: ignore[arg-type]
            user_id="user-1",
            request=CreateJobInput(instruction="fix build", artifact_mode=" PR "),
        )

        self.assertEqual(job.artifact_mode, "pr")


if __name__ == "__main__":
    import unittest

    unittest.main()
