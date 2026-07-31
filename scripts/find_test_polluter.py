from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


def test_ids(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from test_ids(item)
        else:
            yield item.id()


def run_modules(modules: list[str], target: str) -> dict[str, object]:
    command = [sys.executable, "-m", "unittest", *modules, target]
    completed = subprocess.run(command, text=True, capture_output=True, env={**os.environ, "PYTHONPATH": "src"}, check=False)
    return {
        "modules": modules,
        "returncode": completed.returncode,
        "tail": (completed.stdout + completed.stderr).splitlines()[-25:],
        "command": command,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-dir", default="tests")
    parser.add_argument("--top-level", default=".")
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(args.start_dir, top_level_dir=args.top_level)
    target_prefix = args.target + "."
    modules: list[str] = []
    for test_id in test_ids(suite):
        module = test_id.rsplit(".", 2)[0]
        if test_id.startswith(target_prefix):
            break
        if module not in modules and module != args.target:
            modules.append(module)
    direct = run_modules([], args.target)
    polluters: list[str] = []
    commands = [direct]
    for module in modules:
        result = run_modules([module], args.target)
        commands.append(result)
        if direct["returncode"] == 0 and result["returncode"]:
            polluters.append(module)
    report = {
        "target": args.target,
        "direct_result": "passed" if direct["returncode"] == 0 else "failed",
        "minimal_polluting_modules": polluters,
        "required_combination": direct["returncode"] == 0 and not polluters,
        "candidate_modules": modules,
        "commands": commands,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
