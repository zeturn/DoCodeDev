from __future__ import annotations

from docode.agent.task_contract import is_crawler_instruction

from .base import TaskProfile
from .crawler import CRAWLER_PROFILE
from .generic import GENERIC_PROFILE
from .repository_task import REPOSITORY_TASK_PROFILE

REPOSITORY_MARKERS = ("across files", "multi-file", "repository", "codebase", "rename", "migration", "跨文件", "代码仓库")


def select_task_profile(instruction: str) -> TaskProfile:
    if is_crawler_instruction(instruction):
        return CRAWLER_PROFILE
    lowered = instruction.lower()
    if any(marker in lowered for marker in REPOSITORY_MARKERS):
        return REPOSITORY_TASK_PROFILE
    return GENERIC_PROFILE
