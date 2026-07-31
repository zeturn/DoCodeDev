from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    case_id: str
    run_id: str
    request_id: str
    timestamp: str
    method: str
    path: str
    raw_query: str = ""
    parsed_query: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    headers_subset: Mapping[str, str] = field(default_factory=dict)
    body_hash: str = ""
    response_status: int = 200
    response_fixture_id: str = ""
    cursor_in: str | None = None
    cursor_out: str | None = None

    @classmethod
    def create(cls, **values) -> "RecordedRequest":
        values.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        return cls(**values)

    @property
    def path_with_query(self) -> str:
        return self.path + (f"?{self.raw_query}" if self.raw_query else "")


@dataclass(frozen=True, slots=True)
class RuntimeRequestEvidence:
    case_id: str
    run_id: str
    requests: tuple[RecordedRequest, ...]
    producer_command_id: str
    producer_exit_code: int
    stdout_ref: str | None = None
    stderr_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.producer_command_id:
            raise ValueError("runtime request evidence requires an executed producer command")
        if any(item.case_id != self.case_id or item.run_id != self.run_id for item in self.requests):
            raise ValueError("runtime request evidence cannot mix cases or runs")


@dataclass(frozen=True, slots=True)
class RequestEvidencePolicy:
    maximum_requests: int | None = None
    expected_paths: tuple[str, ...] = ()
    expected_cursor_order: tuple[str, ...] = ()
    reject_duplicates: bool = True


def validate_runtime_requests(evidence: RuntimeRequestEvidence, policy: RequestEvidencePolicy) -> tuple[str, ...]:
    failures: list[str] = []
    if evidence.producer_exit_code != 0:
        failures.append(f"producer_exit_code:{evidence.producer_exit_code}")
    paths = [request.path_with_query for request in evidence.requests]
    if policy.maximum_requests is not None and len(paths) > policy.maximum_requests:
        failures.append(f"request_budget:{len(paths)}>{policy.maximum_requests}")
    for expected in policy.expected_paths:
        if expected not in paths:
            failures.append(f"request_path_missing:{expected}")
    if policy.reject_duplicates and len(paths) != len(set(paths)):
        failures.append("duplicate_request")
    cursor_order = tuple(item.cursor_in for item in evidence.requests if item.cursor_in is not None)
    if policy.expected_cursor_order and cursor_order != policy.expected_cursor_order:
        failures.append(f"cursor_order:{cursor_order!r}!={policy.expected_cursor_order!r}")
    return tuple(failures)
