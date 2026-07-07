from __future__ import annotations

import importlib.abc
import sys
from types import ModuleType
from typing import Any

from docode.agent.context import ContextPack, clip_text

_HOOK_INSTALLED = False
_PATCHED_MODULE_IDS: set[int] = set()

DEFAULT_REPAIR_CONTEXT_PATHS = (
    "tests/test_parser.py",
    "fixtures/sample.html",
    "fixtures/sample_source.html",
)


def install() -> None:
    """Install repair-context injection for worker and direct loop imports."""

    module = sys.modules.get("docode.agent.loop")
    if isinstance(module, ModuleType) and hasattr(module, "CodingAgentLoop"):
        patch_loop_module(module)
        return
    install_import_hook()


def install_import_hook() -> None:
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    sys.meta_path.insert(0, _RepairContextFinder())
    _HOOK_INSTALLED = True


class _RepairContextFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object | None, target: object | None = None):
        if fullname != "docode.agent.loop":
            return None
        for finder in sys.meta_path:
            if finder is self or is_docode_repair_patch_finder(finder):
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                continue
            spec.loader = _RepairContextLoader(spec.loader)
            return spec
        return None


def is_docode_repair_patch_finder(finder: object) -> bool:
    module = finder.__class__.__module__
    return module.startswith("docode.agent.targeted_repair_")


class _RepairContextLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self.wrapped = wrapped

    def create_module(self, spec):
        create_module = getattr(self.wrapped, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped.exec_module(module)
        patch_loop_module(module)


def patch_loop_module(loop_module: ModuleType) -> None:
    module_id = id(loop_module)
    if module_id in _PATCHED_MODULE_IDS:
        return
    loop_cls = getattr(loop_module, "CodingAgentLoop", None)
    if loop_cls is None:
        return
    if not hasattr(loop_cls, "_docode_original_collect_observation"):
        loop_cls._docode_original_collect_observation = loop_cls.collect_observation
    loop_cls.collect_observation = collect_observation_with_repair_context
    _PATCHED_MODULE_IDS.add(module_id)


async def collect_observation_with_repair_context(self: Any, state: Any) -> ContextPack:
    original = getattr(type(self), "_docode_original_collect_observation")
    pack = await original(self, state)
    repair_context = await build_repair_context(self, state)
    if not repair_context:
        return pack
    latest_evidence = pack.latest_evidence + "\n\n## Repair Context\n" + repair_context
    return ContextPack(
        task_contract=pack.task_contract,
        repo_map=pack.repo_map,
        working_memory=pack.working_memory,
        file_memory=pack.file_memory,
        latest_evidence=clip_text(latest_evidence, 5000),
        recent_messages=pack.recent_messages,
    )


async def build_repair_context(loop: Any, state: Any) -> str:
    action = state.active_repair_action or None
    if state.repair_mode != "targeted_repair" or not isinstance(action, dict):
        return ""

    parts: list[str] = []
    parts.append("The repair planner has already selected the target. Use this context to edit the target file; do not request broad reads repeatedly.")
    parts.append("Repair action summary:\n" + repair_action_summary(action, state))

    failure = latest_failed_tool_result(state)
    if failure:
        parts.append("Latest failing command output:\n" + failure)

    paths = repair_context_paths(action, state)
    for path in paths:
        excerpt = await safe_read_excerpt(loop, path)
        if excerpt:
            parts.append(f"File excerpt: {path}\n{excerpt}")

    return "\n\n".join(parts)


def repair_action_summary(action: dict[str, Any], state: Any) -> str:
    targets = ", ".join(str(path) for path in action.get("target_files") or []) or "<none>"
    reruns = ", ".join(str(command) for command in action.get("rerun_commands") or []) or "<none>"
    phase = state.targeted_repair_phase or action.get("phase") or "inspect_allowed"
    return "\n".join(
        [
            f"- category: {action.get('category')}",
            f"- reason: {action.get('reason')}",
            f"- phase: {phase}",
            f"- target_files: {targets}",
            f"- rerun_after_edit: {reruns}",
            f"- instruction: {clip_text(str(action.get('instruction') or ''), 1200)}",
        ]
    )


def latest_failed_tool_result(state: Any) -> str:
    for message in reversed(state.messages):
        if message.get("role") != "tool" or int(message.get("exit_code") or 0) == 0:
            continue
        tool = str(message.get("tool") or "tool")
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        command = metadata.get("command")
        header = f"{tool} failed"
        if command:
            header += f" command={command}"
        return header + "\n" + clip_text(str(message.get("output") or ""), 1800)
    return ""


def repair_context_paths(action: dict[str, Any], state: Any) -> list[str]:
    paths: list[str] = []
    for path in action.get("target_files") or []:
        add_path(paths, str(path))
    for path in DEFAULT_REPAIR_CONTEXT_PATHS:
        add_path(paths, path)
    for message in reversed(state.messages):
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        path = metadata.get("path")
        if isinstance(path, str):
            add_path(paths, path)
        if len(paths) >= 6:
            break
    return paths[:6]


def add_path(paths: list[str], path: str) -> None:
    normalized = normalize_workspace_path(path)
    if not normalized or normalized.startswith(".docode_probe") or normalized.startswith(".git/"):
        return
    if normalized not in paths:
        paths.append(normalized)


def normalize_workspace_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    if normalized.startswith("/workspace/"):
        normalized = normalized[len("/workspace/") :]
    return normalized.lstrip("/")


async def safe_read_excerpt(loop: Any, path: str) -> str:
    try:
        result = await loop.tools.read_file(path)
    except Exception as exc:
        return f"<unavailable: {exc}>"
    if int(getattr(result, "exit_code", 0) or 0) != 0:
        return ""
    text = str(getattr(result, "output", "") or "")
    return clip_text(text, 1800)
