from pathlib import Path


text = Path("guidebook.md").read_text(encoding="utf-8")
required_sections = ("## Operational Envelope", "## Rollback Signals", "## Example Transcript")
for heading in required_sections:
    assert heading in text, f"missing section: {heading}"
lower = text.lower()
for concept in ("idempotent", "latency", "rollback", "correlation id"):
    assert concept in lower, f"missing concept: {concept}"
assert "```text" in text and "request>" in lower and "response>" in lower
assert Path("engine/quiet_core.py").read_text(encoding="utf-8") == 'def stable_identifier(value: str) -> str:\n    return "-".join(value.lower().split())\n'
print("semantic documentation checks passed")
