from __future__ import annotations

from docode.agent.reviewer import parse_review_result


def test_parse_review_result_accepts_structured_json() -> None:
    result = parse_review_result(
        """
        review:
        {
          "passed": false,
          "confidence": 0.72,
          "blocking_issues": ["artifact has empty urls"],
          "warnings": "minor risk",
          "repair_plan": ["normalize repository urls"],
          "reason": "bad artifact"
        }
        """
    )

    assert not result.passed
    assert result.confidence == 0.72
    assert result.blocking_issues == ["artifact has empty urls"]
    assert result.warnings == ["minor risk"]
    assert result.repair_plan == ["normalize repository urls"]
    assert result.reason == "bad artifact"
