from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_records(text: str) -> list[dict[str, object]]:
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = parse_records(Path(args.source).read_text(encoding="utf-8"))
    print(json.dumps(records))


if __name__ == "__main__":
    main()
