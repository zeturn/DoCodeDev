from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import asdict
from typing import Mapping
from urllib.parse import parse_qs

from docode.runtime.execution_evidence import RecordedRequest, RuntimeRequestEvidence


class RequestLogStore:
    """Thread-safe runtime request log, isolated by case and producer run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str], list[RecordedRequest]] = {}

    def reset(self, case_id: str, run_id: str) -> None:
        with self._lock:
            self._requests[(case_id, run_id)] = []

    def record(
        self,
        *,
        case_id: str,
        run_id: str,
        method: str,
        path: str,
        raw_query: str = "",
        headers: Mapping[str, str] | None = None,
        body: bytes = b"",
        response_status: int = 200,
        response_fixture_id: str = "",
        cursor_in: str | None = None,
        cursor_out: str | None = None,
    ) -> RecordedRequest:
        key = (case_id, run_id)
        with self._lock:
            if key not in self._requests:
                raise KeyError(f"request log not initialized for {case_id}/{run_id}")
            item = RecordedRequest.create(
                case_id=case_id,
                run_id=run_id,
                request_id=str(uuid.uuid4()),
                method=method.upper(),
                path=path,
                raw_query=raw_query,
                parsed_query={name: tuple(values) for name, values in parse_qs(raw_query, keep_blank_values=True).items()},
                headers_subset=_safe_headers(headers or {}),
                body_hash=hashlib.sha256(body).hexdigest(),
                response_status=response_status,
                response_fixture_id=response_fixture_id,
                cursor_in=cursor_in,
                cursor_out=cursor_out,
            )
            self._requests[key].append(item)
            return item

    def requests(self, case_id: str, run_id: str) -> tuple[RecordedRequest, ...]:
        with self._lock:
            return tuple(self._requests.get((case_id, run_id), ()))

    def evidence(self, case_id: str, run_id: str, *, producer_command_id: str, producer_exit_code: int, stdout_ref: str | None = None, stderr_ref: str | None = None) -> RuntimeRequestEvidence:
        return RuntimeRequestEvidence(case_id, run_id, self.requests(case_id, run_id), producer_command_id, producer_exit_code, stdout_ref, stderr_ref)

    def serialized(self, case_id: str, run_id: str) -> list[dict[str, object]]:
        return [asdict(item) for item in self.requests(case_id, run_id)]


def _safe_headers(headers: Mapping[str, str]) -> dict[str, str]:
    allowed = {"accept", "content-type", "user-agent", "x-runtime-v2-run-id"}
    return {str(name).lower(): str(value) for name, value in headers.items() if str(name).lower() in allowed}
