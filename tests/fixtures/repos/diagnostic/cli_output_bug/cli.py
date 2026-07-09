from __future__ import annotations

import argparse
import json


def build_greeting(name: str) -> dict[str, str]:
    return {"greeting": f"Hello, {name}!"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(json.dumps(build_greeting(args.name)))


if __name__ == "__main__":
    main()
