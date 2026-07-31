from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docode.agent.artifact_contract import ArtifactSemanticContract


@dataclass(frozen=True, slots=True)
class CommandDependencyPolicy:
    preserve_user_order: bool = True
    invalidate_after_edit: bool = True


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    use_repository_index: bool = True
    bounded_source_excerpts: bool = True
    include_task_graph: bool = True


@dataclass(frozen=True, slots=True)
class RepairPolicy:
    maximum_identical_failures: int = 3
    require_targeted_edit: bool = True


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    maximum_source_requests: int = 4
    maximum_pre_edit_decisions: int = 2


@dataclass(frozen=True, slots=True)
class TaskProfile:
    name: str
    source_inspection_required: bool
    allowed_source_schemes: tuple[str, ...]
    artifact_contract: ArtifactSemanticContract | None
    command_dependency_policy: CommandDependencyPolicy
    context_policy: ContextPolicy
    repair_policy: RepairPolicy
    budget_policy: BudgetPolicy
    prompt_guidance: tuple[str, ...]
