import json
import tempfile
import unittest
from pathlib import Path

from docode.agent.artifact_contract import ArtifactSemanticContract, extract_artifact_contract
from docode.agent.artifact_validator import ExecutionEvidence, validate_artifact, validate_remote_artifact
from docode.agent.workspace_reader import WorkspaceEntry


class ArtifactSemanticTests(unittest.TestCase):
    def test_conservative_contract_extraction(self) -> None:
        contract = extract_artifact_contract("Emit exactly 2 records to out.json with headline, detail_url and score; deduplicate by headline while preserving first-seen order. detail_url must be absolute.")
        self.assertEqual(contract.exact_record_count, 2)
        self.assertEqual(contract.required_fields, ["headline", "detail_url", "score"])
        self.assertEqual(contract.unique_by, ["headline"])
        self.assertTrue(contract.preserve_first_seen_order)
        self.assertEqual(contract.absolute_url_fields, ["detail_url"])

    def test_validator_checks_fields_types_urls_and_dedup(self) -> None:
        contract = ArtifactSemanticContract(container_type="list", exact_record_count=2, required_fields=["name", "url", "score"], non_empty_fields=["name"], field_types={"score": "integer"}, nullable_fields=["score"], absolute_url_fields=["url"], unique_by=["name"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.json"
            path.write_text(json.dumps([{"name": " ", "url": "/one", "score": True}, {"name": " ", "url": "https://example.test/two", "score": None}]), encoding="utf-8")
            result = validate_artifact(path, contract)
        self.assertFalse(result.passed)
        self.assertTrue(any(item.startswith("non_empty_field") for item in result.failures))
        self.assertTrue(any(item.startswith("field_type") for item in result.failures))
        self.assertTrue(any(item.startswith("absolute_url") for item in result.failures))
        self.assertTrue(any(item.startswith("duplicate") for item in result.failures))

    def test_valid_nullable_numeric_records_pass(self) -> None:
        contract = ArtifactSemanticContract(container_type="list", minimum_record_count=2, field_types={"score": "number"}, nullable_fields=["score"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.json"
            path.write_text('[{"score": 1.5}, {"score": null}]', encoding="utf-8")
            result = validate_artifact(path, contract)
        self.assertTrue(result.passed, result.failures)


class RemoteArtifactSemanticTests(unittest.IsolatedAsyncioTestCase):
    async def test_remote_validation_checks_semantics_requests_and_freshness(self) -> None:
        class Reader:
            async def read_file(self, path: str, max_bytes: int = 256_000) -> str:
                return '[{"id":"a","url":"/relative"},{"id":"a","url":"https://example.test/b"}]'
        contract = ArtifactSemanticContract(artifact_paths=["out.json"], container_type="list", exact_record_count=2, absolute_url_fields=["url"], unique_by=["id"], expected_request_count=2, expected_request_paths=["/page/2"])
        evidence = ExecutionEvidence(edit_epoch=2, producer_epoch=1, producer_sequence=2, validator_epoch=2, validator_sequence=1, request_paths=("/page/1",))
        result = await validate_remote_artifact(Reader(), "out.json", contract, evidence)
        self.assertFalse(result.passed)
        self.assertTrue({"producer_stale", "validator_before_producer", "request_count:1!=2", "request_path_missing:/page/2"} <= set(result.failures))
        self.assertTrue(any(item.startswith("absolute_url") for item in result.failures))
        self.assertTrue(any(item.startswith("duplicate") for item in result.failures))


if __name__ == "__main__":
    unittest.main()
