from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

ContainerType = Literal["list", "object", "text", "file", "unknown"]


@dataclass(slots=True)
class ArtifactSemanticContract:
    artifact_paths: list[str] = field(default_factory=list)
    container_type: ContainerType = "unknown"
    minimum_record_count: int | None = None
    exact_record_count: int | None = None
    required_fields: list[str] = field(default_factory=list)
    non_empty_fields: list[str] = field(default_factory=list)
    nullable_fields: list[str] = field(default_factory=list)
    field_types: dict[str, str] = field(default_factory=dict)
    absolute_url_fields: list[str] = field(default_factory=list)
    unique_by: list[str] = field(default_factory=list)
    preserve_first_seen_order: bool | None = None
    sorted_by: list[str] = field(default_factory=list)
    expected_request_count: int | None = None
    expected_request_paths: list[str] = field(default_factory=list)
    producer_commands: list[str] = field(default_factory=list)
    validator_commands: list[str] = field(default_factory=list)


def extract_artifact_contract(instruction: str) -> ArtifactSemanticContract:
    """Conservatively extract only constraints explicitly stated by the user."""
    text = instruction or ""
    lowered = text.lower()
    contract = ArtifactSemanticContract()
    paths = re.findall(r"(?<![\w/.-])([\w./-]+\.(?:json|csv|tsv|xml|txt))(?![\w.-])", text, re.IGNORECASE)
    contract.artifact_paths = list(dict.fromkeys(paths))
    if any(path.lower().endswith(".json") for path in paths):
        contract.container_type = "list" if re.search(r"\b(records|entries|items|rows)\b", lowered) else "unknown"
    exact = re.search(r"\b(?:exactly|emit exactly|write exactly)\s+(\d+)\s+(?:records|entries|items|rows)\b", lowered)
    minimum = re.search(r"\b(?:at least|minimum(?: of)?)\s+(\d+)\s+(?:records|entries|items|rows)\b", lowered)
    if exact:
        contract.exact_record_count = int(exact.group(1))
    if minimum:
        contract.minimum_record_count = int(minimum.group(1))
    field_match = re.search(r"\b(?:with|fields?[:=]|containing)\s+([a-z_][\w]*(?:\s*,\s*[a-z_][\w]*)+(?:\s*(?:,?\s*and)\s*[a-z_][\w]*)?)", lowered)
    if field_match:
        fields = re.split(r"\s*,\s*|\s+(?:and)\s+", field_match.group(1))
        contract.required_fields = list(dict.fromkeys(field.strip() for field in fields if field.strip()))
    for match in re.finditer(r"\b([a-z_][\w]*)\s+(?:must be\s+)?(?:non[- ]empty|required and non[- ]empty)\b", lowered):
        contract.non_empty_fields.append(match.group(1))
    for match in re.finditer(r"\bnullable\s+([a-z_][\w]*)|\b([a-z_][\w]*)\s+(?:may be null|is nullable)\b", lowered):
        contract.nullable_fields.append(next(group for group in match.groups() if group))
    for match in re.finditer(r"\b(?:absolute\s+)?([a-z_][\w]*(?:url|uri|link))\s+(?:must be\s+)?absolute\b|\babsolute\s+([a-z_][\w]*(?:url|uri|link))\b", lowered):
        contract.absolute_url_fields.append(next(group for group in match.groups() if group))
    unique = re.search(r"\b(?:deduplicate|unique)\s+(?:by|on)\s+([a-z_][\w]*)", lowered)
    if unique:
        contract.unique_by = [unique.group(1)]
    if "preserv" in lowered and "first-seen order" in lowered:
        contract.preserve_first_seen_order = True
    requests = re.search(r"\bexactly\s+(\d+)\s+(?:http\s+)?requests?\b", lowered)
    if requests:
        contract.expected_request_count = int(requests.group(1))
    return contract
