from __future__ import annotations

import json
from pathlib import Path


def normalize_item(item: dict[str, object]) -> dict[str, object]:
    return {
        "name": item["name"],
        "price": float(item["price"]),
        "available": bool(item["availability"]),
        "category": item["category"],
    }


def load_items(path: str | Path) -> list[dict[str, object]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [normalize_item(item) for item in raw]
