from __future__ import annotations

from docode.config import DocodeConfig

from .repository import InMemoryJobRepository, JobRepository
from .sqlite import SQLiteJobRepository


def build_repository(config: DocodeConfig) -> JobRepository:
    if config.database_path == ":memory:":
        return InMemoryJobRepository()
    return SQLiteJobRepository(config.database_path)

