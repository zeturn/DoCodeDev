from .base import BudgetPolicy, CommandDependencyPolicy, ContextPolicy, RepairPolicy, TaskProfile

CRAWLER_PROFILE = TaskProfile(
    name="crawler",
    source_inspection_required=True,
    allowed_source_schemes=("http", "https"),
    artifact_contract=None,
    command_dependency_policy=CommandDependencyPolicy(),
    context_policy=ContextPolicy(),
    repair_policy=RepairPolicy(),
    budget_policy=BudgetPolicy(),
    prompt_guidance=(
        "Inspect a literal source in the sandbox before the first edit.",
        "Resolve relative links against the response final URL and keep unrelated query parameters.",
        "Derive schemas and pagination only from task requirements and source evidence.",
    ),
)
