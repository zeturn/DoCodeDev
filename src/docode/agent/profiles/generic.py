from .base import BudgetPolicy, CommandDependencyPolicy, ContextPolicy, RepairPolicy, TaskProfile

GENERIC_PROFILE = TaskProfile(
    name="generic",
    source_inspection_required=False,
    allowed_source_schemes=(),
    artifact_contract=None,
    command_dependency_policy=CommandDependencyPolicy(),
    context_policy=ContextPolicy(),
    repair_policy=RepairPolicy(),
    budget_policy=BudgetPolicy(maximum_source_requests=0),
    prompt_guidance=("Treat explicit user verification commands as authoritative and preserve them exactly.",),
)
