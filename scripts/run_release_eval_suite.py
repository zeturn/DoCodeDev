"""Deterministic holdout evaluation harness runner (V1).

Runs the 8 fixed evaluation cases through the REAL production path:

    real provider (DecisionLLM) -> real DoBox workspace -> CodingAgentLoop
    -> independent hidden checker -> metrics aggregation

It reuses the vertical-slice runner's single implementation of provider
configuration resolution, DoBox readiness/autostart, the seeded-project client,
and the evidence bundle writer. It does NOT introduce a second provider
parser, a second DoBox autostart, a second redaction path, or a second
JobRunnerService assembly.

Usage:
    python scripts/run_release_eval_suite.py --validate-fixtures
    python scripts/run_release_eval_suite.py --cases all --runs-per-case 1 \
        --start-dobox --output artifacts/release-eval-baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# The vertical-slice runner lives alongside this script and provides the
# shared, already-tested infrastructure (provider/config resolution, DoBox
# readiness, seeded-project client, inspectors, evidence writer).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_release_vertical_slice import (  # noqa: E402
    FixtureSeedingDoBoxClient,
    ensure_dobox_smoke_token,
    local_dobox_checks,
    managed_local_dobox,
    plan_dobox_readiness,
    redact_endpoint,
    resolve_provider_and_config,
    check_http_health,
)

from docode.eval.fixture import load_fixture, validate_all_fixtures
from docode.eval.manifest import load_suite_manifests
from docode.eval.metrics import aggregate_results, suite_exit_code, write_suite_outputs
from docode.eval.models import RunResult
from docode.eval.runner import run_case

FIXTURES_ROOT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "release_eval"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DoCode Runtime V2 deterministic holdout evaluation suite (live).")
    parser.add_argument("--cases", default="all", help="comma-separated case ids or 'all'")
    parser.add_argument("--runs-per-case", type=int, default=1)
    parser.add_argument("--output", default="artifacts/release-eval-baseline")
    parser.add_argument("--start-dobox", action="store_true", help="autostart a local DoBox backend if unreachable")
    parser.add_argument("--keep-dobox", action="store_true", help="keep the autostarted backend after the run (debug only)")
    parser.add_argument("--validate-fixtures", action="store_true", help="only validate fixtures locally (no provider/DoBox)")
    parser.add_argument("--fail-fast", action="store_true", help="stop at the first case failure")
    return parser


def _select_cases(args: argparse.Namespace, manifests: dict[str, Any]) -> list[str]:
    if args.cases.strip().lower() == "all":
        return list(manifests.keys())
    selected = [c.strip() for c in args.cases.split(",") if c.strip()]
    missing = [c for c in selected if c not in manifests]
    if missing:
        raise SystemExit(f"unknown case id(s): {missing}; available: {sorted(manifests)}")
    return selected


async def run_validate_fixtures() -> int:
    reports = await validate_all_fixtures(FIXTURES_ROOT)
    ok = True
    for case_id, report in reports.items():
        status = "OK" if report.ok else "INVALID"
        if not report.ok:
            ok = False
        print(f"[{case_id}] {status}", file=sys.stderr)
        for state, value in report.states.items():
            print(f"    - {state}: {value}", file=sys.stderr)
    if not ok:
        print("FIXTURE VALIDATION FAILED", file=sys.stderr)
        return 3
    print("FIXTURE VALIDATION PASSED", file=sys.stderr)
    return 0


async def main_async(args: argparse.Namespace) -> int:
    if args.validate_fixtures:
        return await run_validate_fixtures()

    manifests = load_suite_manifests(FIXTURES_ROOT)
    selected = _select_cases(args, manifests)

    config, local_credentials, provider, model, reasons = resolve_provider_and_config()

    readiness = await plan_dobox_readiness(config, start_dobox=args.start_dobox)
    if readiness.fail_reason:
        reasons.append(readiness.fail_reason)

    effective_start = args.start_dobox and not readiness.reachable
    async with managed_local_dobox(
        config, check_http_health, effective_start, readiness.autostart_checks, keep=args.keep_dobox
    ) as start_checks:
        if effective_start:
            autostart_failed = next(
                (c for c in start_checks if c.name == "dobox_autostart" and c.status == "failed"),
                None,
            )
            if autostart_failed is not None:
                reasons.append(f"DoBox autostart failed: {autostart_failed.detail}")

        token, _token_check = await ensure_dobox_smoke_token(config)
        if token:
            config.dobox_token = token
        elif not config.dobox_token:
            reasons.append("DoBox token could not be resolved (auth failed)")

        if reasons:
            report = {
                "status": "failed",
                "failure_reason": "environment_failure",
                "details": reasons,
                "note": "SKIPPED != PASSED: missing real infrastructure; no success claimed.",
            }
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "terminal_result.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print("[release-eval-suite] FAIL-CLOSED (environment):", file=sys.stderr)
            for reason in reasons:
                print(f"  - {reason}", file=sys.stderr)
            return 2

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        dobox_runtime = {
            "dobox_mode": readiness.mode,
            "dobox_started_by_runner": effective_start,
            "docker_daemon_available": readiness.docker_daemon_available,
            "sandbox_image_available": readiness.sandbox_image_available,
        }

        suite_run_id = f"release-eval-{provider}-{model}"
        run_results: list[RunResult] = []
        any_failure = False

        for case_id in selected:
            fixture = load_fixture(manifests[case_id].fixture_dir)
            # Seed the fixture's workspace directory (never the checker/gold).
            seeding_client = FixtureSeedingDoBoxClient(
                config.dobox_base_url, config.dobox_token, fixture.workspace_dir
            )
            for index in range(args.runs_per_case):
                print(f"[release-eval-suite] {case_id} run {index + 1}/{args.runs_per_case} ...", file=sys.stderr)
                result = await run_case(
                    suite_run_id=suite_run_id,
                    case_id=case_id,
                    run_index=index,
                    config=config,
                    local_credentials=local_credentials,
                    provider=provider,
                    model=model,
                    fixture=fixture,
                    dobox=seeding_client,
                    output_dir=output_dir,
                    dobox_runtime=dobox_runtime,
                )
                run_results.append(result)
                print(
                    f"[release-eval-suite] {case_id}#{index}: outcome={result.outcome} "
                    f"checker_passed={result.checker_passed} job={result.job_id}",
                    file=sys.stderr,
                )
                if result.outcome not in ("passed", "expected_outcome_pass"):
                    any_failure = True
                    if args.fail_fast:
                        break
            if args.fail_fast and any_failure:
                break

        summary = aggregate_results(run_results, suite_run_id=suite_run_id)
        paths = write_suite_outputs(
            output_dir,
            manifests=manifests,
            run_results=run_results,
            suite_run_id=suite_run_id,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))  # noqa: F821
        print(f"[release-eval-suite] wrote {paths}", file=sys.stderr)

        infra_fail_closed = any(
            r.outcome in ("infrastructure_failure", "provider_failure", "dobox_transport_failure")
            for r in run_results
        ) and not any_failure
        return suite_exit_code(summary, harness_valid=True, infra_fail_closed=infra_fail_closed)


def main() -> int:
    args = build_arg_parser().parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
