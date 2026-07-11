from .base import BudgetPolicy, CommandDependencyPolicy, ContextPolicy, RepairPolicy, TaskProfile

REPOSITORY_TASK_PROFILE = TaskProfile(
    name="repository_task",
    source_inspection_required=False,
    allowed_source_schemes=(),
    artifact_contract=None,
    command_dependency_policy=CommandDependencyPolicy(),
    context_policy=ContextPolicy(),
    repair_policy=RepairPolicy(),
    budget_policy=BudgetPolicy(maximum_source_requests=0),
    prompt_guidance=("Use the repository index and maintain a dependency-aware task graph before editing.",),
)
