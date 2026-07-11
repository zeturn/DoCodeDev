from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .artifact_contract import ArtifactSemanticContract


@dataclass(frozen=True, slots=True)
class ArtifactValidationResult:
    passed: bool
    failures: tuple[str, ...] = ()
    sample: tuple[Any, ...] = ()
    record_count: int | None = None


def validate_artifact(path: str | Path, contract: ArtifactSemanticContract) -> ArtifactValidationResult:
    artifact = Path(path)
    failures: list[str] = []
    if not artifact.is_file():
        return ArtifactValidationResult(False, (f"artifact_missing:{artifact}",))
    try:
        value = json.loads(artifact.read_text(encoding="utf-8")) if artifact.suffix.lower() == ".json" else artifact.read_text(encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return ArtifactValidationResult(False, (f"artifact_unparseable:{type(exc).__name__}:{exc}",))
    if contract.container_type == "list" and not isinstance(value, list):
        failures.append("container_type:list")
    elif contract.container_type == "object" and not isinstance(value, dict):
        failures.append("container_type:object")
    records = value if isinstance(value, list) else [value] if isinstance(value, dict) else []
    count = len(records) if isinstance(value, (list, dict)) else None
    if contract.exact_record_count is not None and count != contract.exact_record_count:
        failures.append(f"exact_record_count:{count}!={contract.exact_record_count}")
    if contract.minimum_record_count is not None and (count is None or count < contract.minimum_record_count):
        failures.append(f"minimum_record_count:{count}<{contract.minimum_record_count}")
    seen: set[tuple[Any, ...]] = set()
    previous_sort: tuple[Any, ...] | None = None
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            failures.append(f"record_not_object:{index}")
            continue
        for field_name in contract.required_fields:
            if field_name not in record:
                failures.append(f"required_field:{index}:{field_name}")
        for field_name in contract.non_empty_fields:
            if field_name not in record or _empty(record.get(field_name)):
                failures.append(f"non_empty_field:{index}:{field_name}")
        for field_name, expected in contract.field_types.items():
            item = record.get(field_name)
            if item is None and field_name in contract.nullable_fields:
                continue
            if not _matches_type(item, expected):
                failures.append(f"field_type:{index}:{field_name}:{expected}")
        for field_name in contract.absolute_url_fields:
            parsed = urlparse(str(record.get(field_name) or ""))
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                failures.append(f"absolute_url:{index}:{field_name}")
        if contract.unique_by:
            identity = tuple(_hashable(record.get(field_name)) for field_name in contract.unique_by)
            if identity in seen:
                failures.append(f"duplicate:{index}:{','.join(contract.unique_by)}")
            seen.add(identity)
        if contract.sorted_by:
            current = tuple(record.get(field_name) for field_name in contract.sorted_by)
            if previous_sort is not None and current < previous_sort:
                failures.append(f"ordering:{index}:{','.join(contract.sorted_by)}")
            previous_sort = current
    return ArtifactValidationResult(not failures, tuple(dict.fromkeys(failures)), tuple(records[:3]), count)


def _empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _matches_type(value: Any, expected: str) -> bool:
    types = {"string": str, "str": str, "integer": int, "int": int, "number": (int, float), "float": (int, float), "boolean": bool, "bool": bool, "object": dict, "array": list}
    expected_type = types.get(expected.lower())
    if expected_type is None or isinstance(value, bool) and expected.lower() in {"integer", "int", "number", "float"}:
        return expected_type is None
    return isinstance(value, expected_type)


def _hashable(value: Any) -> Any:
    try:
        hash(value)
        return value
    except TypeError:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
