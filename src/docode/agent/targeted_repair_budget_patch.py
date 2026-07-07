from __future__ import annotations

import importlib.abc
import sys
from types import ModuleType
from typing import Any

_HOOK_INSTALLED = False
_PATCHED_MODULE_IDS: set[int] = set()

CONTEXT_HEAVY_REPAIR_CATEGORIES = {
    "missing_required_field",
    "parsed_value_mismatch",
    "json_semantic_failure",
    "parser_records_empty",
    "parser_records_too_few",
    "parser_record_count_mismatch",
}

MIN_CONTEXT_REPAIR_INSPECTION_BUDGET = 3


def install() -> None:
    module = sys.modules.get("docode.agent.loop")
    if isinstance(module, ModuleType) and hasattr(module, "repair_action_contract"):
        patch_loop_module(module)
        return
    install_import_hook()


def install_import_hook() -> None:
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    sys.meta_path.insert(0, _BudgetPatchFinder())
    _HOOK_INSTALLED = True


class _BudgetPatchFinder(importlib.abc.MetaPathFinder):
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
            spec.loader = _BudgetPatchLoader(spec.loader)
            return spec
        return None


def is_docode_repair_patch_finder(finder: object) -> bool:
    module = finder.__class__.__module__
    return module.startswith("docode.agent.targeted_repair_")


class _BudgetPatchLoader(importlib.abc.Loader):
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
    if not hasattr(loop_module, "_docode_original_repair_action_contract"):
        loop_module._docode_original_repair_action_contract = loop_module.repair_action_contract
    loop_module.repair_action_contract = repair_action_contract
    _PATCHED_MODULE_IDS.add(module_id)


def repair_action_contract(action: Any, state: Any) -> dict[str, Any]:
    loop_module = sys.modules.get("docode.agent.loop")
    original = getattr(loop_module, "_docode_original_repair_action_contract")
    payload = original(action, state)
    category = str(payload.get("category") or "")
    if category in CONTEXT_HEAVY_REPAIR_CATEGORIES:
        payload["initial_inspection_budget"] = max(
            int(payload.get("initial_inspection_budget") or 0),
            MIN_CONTEXT_REPAIR_INSPECTION_BUDGET,
        )
        payload["next_allowed_tools"] = [
            "read_file",
            "read_file_range",
            "read_symbol",
            "edit_file",
            "apply_patch",
            "write_file",
            "replace_in_file",
        ]
    return payload
