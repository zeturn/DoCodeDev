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

## Non-Goals

- No GitHub Trends runtime logic.
- No controller-generated solution code.
- No background crawler or external web crawler eval yet.
- No new crawler evals in this stabilization pass.

## Known Limitations

- Optional integration tests require a reachable DoBox backend and configured LLM credentials.
- DoBox sandbox setup may need a `python` to `python3` alias for smoke commands that intentionally use `python`.
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

The real LLM commands use the existing provider configuration. By default the optional real LLM helper resolves a DeepSeek model through BasaltPass/APICred when those environment variables are configured; direct OpenAI use requires the existing explicit direct OpenAI environment switches.
