from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path


RESULT_ROOT = Path(".docode/evals/sandbox-source-inspection-v1")
RESULT_PATH = RESULT_ROOT / "results.json"
SELECTED_CASES = ("opal_canopy", "violet_prism", "copper_orbit", "cedar_signal")

# The frozen benchmark runner honors this path for its per-job full traces.
os.environ["DOCODE_CRAWLER_BENCHMARK_RESULT_ROOT"] = str(RESULT_ROOT)

from docode.dobox.client import DoBoxClient  # noqa: E402
from tests.crawler_benchmark_v1.definitions import CASE_BY_NAME  # noqa: E402
from tests.crawler_benchmark_v1.harness import sanitize  # noqa: E402
from tests.crawler_benchmark_v1.test_real_benchmark import RealCrawlerBenchmarkV1Tests  # noqa: E402


async def run() -> None:
    runner = RealCrawlerBenchmarkV1Tests(methodName="test_three_real_runs_per_case")
    config = await runner._real_dobox_config()
    client = DoBoxClient(config.dobox_base_url, config.dobox_token)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "traces").mkdir(parents=True, exist_ok=True)
    payload = load_results()
    completed = {str(item.get("case")) for item in payload["runs"]}
    for name in SELECTED_CASES:
        if name in completed:
            continue
        result = await runner._run_case(client, CASE_BY_NAME[name], 1)
        payload["runs"].append(sanitize(result))
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        write_results(payload)
    if {str(item.get("case")) for item in payload["runs"]} != set(SELECTED_CASES):
        raise RuntimeError("limited source-inspection evaluation did not produce exactly the selected four cases")


def load_results() -> dict[str, object]:
    if RESULT_PATH.is_file():
        payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if payload.get("selected_cases") != list(SELECTED_CASES):
            raise RuntimeError("existing source-inspection result has a different case selection")
        return payload
    return {
        "evaluation": "sandbox-source-inspection-v1",
        "provider": "deepseek",
        "model": "deepseek-chat",
        "runs_per_case": 1,
        "selected_cases": list(SELECTED_CASES),
        "generated_at": None,
        "runs": [],
    }


def write_results(payload: dict[str, object]) -> None:
    ordered = sorted(payload["runs"], key=lambda item: SELECTED_CASES.index(str(item["case"])))
    payload["runs"] = ordered
    RESULT_PATH.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(run())
