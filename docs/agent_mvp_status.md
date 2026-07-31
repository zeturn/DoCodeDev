# Agent MVP Status

This document captures the current validated agent MVP before adding more crawler evals.

## Validated Capabilities

- Scripted LLM with fixture-backed fake tools:
  - README edit smoke.
  - Calculator bugfix with exact required-command gating.
  - Product parser fixture smoke.
  - Product parser repair after an initial failing implementation.
- Real LLM with fixture-backed fake tools:
  - README edit smoke.
  - Calculator bugfix with exact required-command gating.
- Real LLM with real DoBox:
  - README edit smoke.
  - Calculator bugfix smoke.
  - Generic local crawler CLI smoke using fixture HTML and explicit local verification commands.

## Added Optional Evals

- Real LLM diagnostic suite: `tests.test_real_llm_diagnostic_suite`.
  - Optional and skipped by default unless `DOCODE_REAL_LLM_SMOKE=1` is set.
  - Uses a real LLM with local fixture-backed tools that perform real file IO and subprocess command execution in temporary workspaces.
  - Does not use real DoBox; this isolates loop, context, repair, command-selection, final-gate, verifier, and artifact-export behavior from sandbox availability.
  - Designed to diagnose real model/loop failure modes, not to serve as a release gate.
  - Run this suite before changing loop or repair-control logic so failures have comparable structured diagnostics.
- Neutral external-source crawler CLI smoke: `tests.test_real_dobox_external_crawler_smoke`.
  - Implemented, but currently unstable and diagnostic only.
  - Skipped by default and not a release gate.
  - Uses a test-harness mock HTTP source with neutral product records.
  - Requires both real LLM and real DoBox flags.
  - The harness first verifies that the mock source URL is reachable from both host-side `fetch_url` and the DoBox sandbox. If the host-to-sandbox route is unavailable, the test skips with a reachability diagnostic.
  - Current observed integration behavior:
    - source URL selection works;
    - host fetch works;
    - DoBox sandbox fetch works;
    - `fetch_url` evidence works with HTTP 200;
    - `crawler.py` is modified;
    - `out.json` may be produced;
    - remaining failures are model repair quality, incomplete CLI `--output` handling, and multi-stage repair-loop behavior.
  - This eval is useful for collecting real repair-loop traces, but it is not listed as a validated capability until it passes consistently in the integration environment.

## Non-Goals

- No GitHub Trends runtime logic.
- No controller-generated solution code.
- No background crawler or external web crawler eval yet.
- No new crawler evals in this stabilization pass.

## Known Limitations

- Optional integration tests require a reachable DoBox backend and configured LLM credentials.
- DoBox sandbox setup may need a `python` to `python3` alias for smoke commands that intentionally use `python`.
- The external-source crawler smoke may need `DOCODE_EXTERNAL_CRAWLER_SOURCE_HOST` when `host.docker.internal` or the host LAN address is not reachable from the DoBox sandbox.
- The neutral external-source crawler eval is a diagnostic frontier eval. It is expected to expose real model behavior variance and may fail with `max_iterations_exceeded` even when the release-gate deterministic suite and validated local crawler CLI smoke pass.
- Production still has old suggested command hints such as `python3` entrypoint hints for calculator.py and cli.py style tasks.
- Crawler/source policy is still partly embedded in verifier and loop code. It should eventually move into a clearer policy/plugin boundary.
- Optional real LLM tests are intentionally skipped by default and may fail for model-behavior reasons even when the deterministic suite passes.

## Integration Matrix

| Tools | LLM | Status |
| --- | --- | --- |
| Fake tools | Scripted LLM | Passing: README, calculator, parser, parser repair |
| Fake tools | Real LLM | Passing: README, calculator |
| Real DoBox | Scripted LLM | Passing: README |
| Real DoBox | Real LLM | Passing: README, calculator, generic local crawler CLI |

## Test Commands

Run from the `DoCodeDev` repository root on macOS with the workspace venv:

```bash
PATH=/Users/henryzhao/Desktop/workplace/.venv/bin:$PATH PYTHONPATH=src ../.venv/bin/python -m unittest discover -s tests
```

Real LLM with fake tools:

```bash
PATH=/Users/henryzhao/Desktop/workplace/.venv/bin:$PATH DOCODE_REAL_LLM_SMOKE=1 PYTHONPATH=src ../.venv/bin/python -m unittest -v tests.test_real_llm_smoke
```

Real DoBox with scripted LLM:

```bash
PATH=/Users/henryzhao/Desktop/workplace/.venv/bin:$PATH DOCODE_REAL_DOBOX_SMOKE=1 PYTHONPATH=src ../.venv/bin/python -m unittest -v tests.test_real_dobox_smoke.RealDoBoxSmokeTests.test_readme_edit_runs_through_real_dobox
```

Real LLM with real DoBox README and calculator:

```bash
PATH=/Users/henryzhao/Desktop/workplace/.venv/bin:$PATH DOCODE_REAL_LLM_SMOKE=1 DOCODE_REAL_DOBOX_SMOKE=1 PYTHONPATH=src ../.venv/bin/python -m unittest -v tests.test_real_dobox_smoke
```

Real LLM with real DoBox generic local crawler CLI:

```bash
PATH=/Users/henryzhao/Desktop/workplace/.venv/bin:$PATH DOCODE_REAL_LLM_SMOKE=1 DOCODE_REAL_DOBOX_SMOKE=1 PYTHONPATH=src ../.venv/bin/python -m unittest -v tests.test_real_dobox_crawler_smoke
```

Optional diagnostic neutral external-source crawler CLI:

```bash
DOCODE_REAL_LLM_SMOKE=1 DOCODE_REAL_DOBOX_SMOKE=1 PATH=/Users/henryzhao/Desktop/workplace/.venv/bin:$PATH PYTHONPATH=src ../.venv/bin/python -m unittest -v tests.test_real_dobox_external_crawler_smoke
```

This command is diagnostic only. It is not part of the default release gate; the currently validated crawler capability is the generic local crawler CLI smoke.

Optional real LLM diagnostic suite with local fixture-backed tools:

```bash
DOCODE_REAL_LLM_SMOKE=1 PATH=/Users/henryzhao/Desktop/workplace/.venv/bin:$PATH PYTHONPATH=src ../.venv/bin/python -m unittest -v tests.test_real_llm_diagnostic_suite
```

This suite is diagnostic only. It is intended to collect structured failure evidence across bounded realistic tasks before changing loop or repair behavior.

The real LLM commands use the existing provider configuration. By default the optional real LLM helper resolves a DeepSeek model through BasaltPass/APICred when those environment variables are configured; direct OpenAI use requires the existing explicit direct OpenAI environment switches.
# Runtime V2 status (2026-07-11)

Runtime V2 architecture components and real DoBox/DeepSeek canary are experimental. The formal 8 crawler + 3 large-repository frozen evaluation has not run, so the Runtime V2 release gate is **FAILED**. See `docs/evals/runtime-v2-release.md`.
