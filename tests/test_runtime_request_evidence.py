from __future__ import annotations

import unittest
import asyncio
from types import SimpleNamespace

from docode.agent.artifact_contract import ArtifactSemanticContract
from docode.agent.artifact_validator import ExecutionEvidence, validate_remote_artifact
from docode.runtime.execution_evidence import RecordedRequest, RequestEvidencePolicy, RuntimeRequestEvidence, validate_runtime_requests


def request(run_id: str, request_id: str, path: str, query: str = "", cursor: str | None = None) -> RecordedRequest:
    return RecordedRequest.create(case_id="cursor", run_id=run_id, request_id=request_id, method="GET", path=path, raw_query=query, cursor_in=cursor)


class RuntimeRequestEvidenceTests(unittest.TestCase):
    def test_records_exact_runtime_paths_and_queries(self) -> None:
        evidence = RuntimeRequestEvidence("cursor", "run-1", (request("run-1", "1", "/feed", "cursor=a"), request("run-1", "2", "/feed", "cursor=b")), "producer-1", 0)
        self.assertEqual(["/feed?cursor=a", "/feed?cursor=b"], [item.path_with_query for item in evidence.requests])

    def test_cursor_duplicate_and_budget_failures_are_independent(self) -> None:
        evidence = RuntimeRequestEvidence("cursor", "run-1", (request("run-1", "1", "/feed", "cursor=a", "b"), request("run-1", "2", "/feed", "cursor=a", "a")), "producer-1", 0)
        failures = validate_runtime_requests(evidence, RequestEvidencePolicy(maximum_requests=1, expected_cursor_order=("a", "b")))
        self.assertIn("duplicate_request", failures)
        self.assertTrue(any(item.startswith("request_budget:") for item in failures))
        self.assertTrue(any(item.startswith("cursor_order:") for item in failures))

    def test_runs_cannot_be_mixed(self) -> None:
        with self.assertRaises(ValueError):
            RuntimeRequestEvidence("cursor", "run-1", (request("run-2", "1", "/feed"),), "producer-1", 0)

    def test_producer_execution_is_required(self) -> None:
        with self.assertRaises(ValueError):
            RuntimeRequestEvidence("cursor", "run-1", (), "", 0)

    def test_source_inspection_paths_cannot_satisfy_runtime_contract(self) -> None:
        contract = ArtifactSemanticContract(artifact_paths=["out.json"], expected_request_count=1, expected_request_paths=["/feed"])
        reader = SimpleNamespace(read_file=lambda path, limit: async_value("[]"))
        result = asyncio.run(validate_remote_artifact(reader, "out.json", contract, ExecutionEvidence(request_paths=("/feed",))))
        self.assertIn("runtime_request_evidence_missing", result.failures)


async def async_value(value: str) -> str:
    return value


if __name__ == "__main__":
    unittest.main()
