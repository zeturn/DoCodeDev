from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .artifact_contract import ArtifactSemanticContract, extract_artifact_contract
from .finalization_controller import FinalizationController
from .profiles import TaskProfile, select_task_profile
from .repair_coordinator import RepairCoordinator
from .task_contract import TaskContract, task_contract_from_instruction
from .task_graph import TaskGraph, TaskNode
from .verification_scheduler import VerificationScheduler


@dataclass(slots=True)
class RuntimeComponents:
    profile: TaskProfile
    task_contract: TaskContract
    artifact_contract: ArtifactSemanticContract
    verification_scheduler: VerificationScheduler
    repair_coordinator: RepairCoordinator
    repository_context: Any | None
    task_graph: TaskGraph
    finalization_controller: FinalizationController


def build_runtime_components(instruction: str) -> RuntimeComponents:
    profile = select_task_profile(instruction)
    task_contract = task_contract_from_instruction(instruction)
    artifact_contract = extract_artifact_contract(instruction)
    scheduler = VerificationScheduler.from_explicit_commands(task_contract.must_run_commands)
    graph = TaskGraph(
        [
            TaskNode("understand", "Understand relevant repository interfaces and constraints", target_files=list(task_contract.must_modify_files), acceptance_criteria=["relevant repository evidence collected"]),
            TaskNode("plan", "Plan the minimal dependency-aware change", dependencies=["understand"], acceptance_criteria=["repository-specific plan recorded"]),
            TaskNode("implement", "Implement the requested change", target_files=list(task_contract.must_modify_files), dependencies=["plan"], acceptance_criteria=["task-relevant target modified"]),
            TaskNode("verify", "Run required verification", dependencies=["implement"], verification=list(task_contract.must_run_commands), acceptance_criteria=["all required commands fresh at current edit epoch"]),
            TaskNode("review", "Review the changed-file impact", dependencies=["verify"], acceptance_criteria=["quality and semantic review passed"]),
        ]
    )
    return RuntimeComponents(
        profile=profile,
        task_contract=task_contract,
        artifact_contract=artifact_contract,
        verification_scheduler=scheduler,
        repair_coordinator=RepairCoordinator(profile.repair_policy.maximum_identical_failures),
        repository_context=None,
        task_graph=graph,
        finalization_controller=FinalizationController(),
    )
