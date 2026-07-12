"""Deterministic state-progress fingerprints.

The fingerprint captures *only* semantically meaningful state changes
(edit epoch, evidence, task-graph statuses, repair phase, active blocker).
Transient bookkeeping (iteration, message count, timestamps) is excluded
so that repeated no-op iterations do not look like progress.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, TYPE_CHECKING

from docode.git_changes import changed_paths_from_status
from docode.agent.repair_coordinator import RepairPhase

if TYPE_CHECKING:
    from docode.agent.state import AgentState


def state_progress_snapshot(state: AgentState) -> dict[str, object]:
    """Return a dict of semantically meaningful state for fingerprinting."""

    # changed paths from latest git status
    changed: list[str] = []
    if state.latest_git_status is not None:
        changed = sorted(
            changed_paths_from_status(state.latest_git_status.output)
        )

    # task-graph nodes
    tg_nodes: dict[str, object] = {}
    if state.task_graph is not None:
        for nid, node in sorted(state.task_graph.nodes.items()):
            status = node.status
            if hasattr(status, "value"):
                status = status.value
            tg_nodes[nid] = {
                "status": str(status),
                "evidence_refs": sorted(
                    list(getattr(node, "evidence_refs", []) or [])
                ),
            }

    # scheduler
    scheduler: dict[str, object] = {}
    if state.verification_scheduler is not None:
        scheduler["edit_epoch"] = getattr(
            state.verification_scheduler, "edit_epoch", 0
        )
        nxt = state.verification_scheduler.next_command()
        scheduler["next_command"] = nxt if nxt else ""
        commands: dict[str, object] = {}
        if hasattr(state.verification_scheduler, "_command_evidence"):
            for cmd, evidence in sorted(
                state.verification_scheduler._command_evidence.items()
            ):
                commands[cmd] = {
                    "passed": evidence.passed,
                    "edit_epoch": getattr(evidence, "edit_epoch", 0),
                }
        scheduler["commands"] = commands

    # repair
    repair: dict[str, object] = {}
    if state.repair_coordinator is not None:
        repair["phase"] = state.repair_coordinator.phase.value
    else:
        repair["phase"] = ""
    repair["mode"] = state.repair_mode or ""
    repair_sig = state.active_repair_action or {}
    repair["active_signature"] = {
        "action": str(repair_sig.get("action", "")),
        "target": str(repair_sig.get("target", "")),
    }

    # blocker
    blocker_fp = ""
    if hasattr(state, "active_blocker") and state.active_blocker is not None:
        blocker_fp = state.active_blocker.fingerprint()

    return {
        "edit_epoch": state.edit_epoch,
        "changed_paths": changed,
        "task_graph_nodes": tg_nodes,
        "scheduler": scheduler,
        "repair": repair,
        "quality_gate_passed": state.quality_gate_passed,
        "active_blocker_fingerprint": blocker_fp,
        "terminal_no_progress_reason": getattr(
            state, "terminal_no_progress_reason", None
        ) or "",
        "terminal_repair_reason": state.terminal_repair_reason or "",
    }


def state_progress_fingerprint(state: AgentState) -> str:
    """64-char lowercase SHA-256 hex of the progress snapshot."""
    snapshot = state_progress_snapshot(state)
    encoded = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
