# DoCode Runtime V2 — Deterministic Holdout Evaluation Harness V1

Baseline summary for the 8-fixed-fixture deterministic holdout harness
(`scripts/run_release_eval_suite.py`). This document is de-sensitized: no
provider API keys, DoBox tokens, or endpoint URLs are included. Endpoint and
workspace identifiers in raw evidence are redacted by the harness.

## Harness outcome taxonomy

`passed`, `expected_outcome_pass`, `agent_failure`, `checker_failure`,
`provider_failure`, `decision_parse_failure`, `dobox_transport_failure`,
`infrastructure_failure`, `budget_exceeded`, `no_progress`, `harness_failure`.

A pre-workspace provisioning failure (`project_id=None` / `iterations=0`) must be
classified as `infrastructure_failure`, **not** `agent_failure`.

## Event 1 — INVALID attempt (preserved, not overwritten)

- `suite_run_id`: `release-eval-openai-gpt-5.4-mini`
- git SHA: `6d8886252f390a1a54f99c3d39ce9c215d4f8d1b`
- status: **INVALID**
- root cause: harness outcome-classification bug — infrastructure/provisioning
  failures were misclassified as `agent_failure` (reported `agent_failure: 8`).
- preserved artifacts (untouched): `artifacts/release-eval-baseline-20260716-170238/`
  containing `INVALID_HARNESS_BUG.md`, `summary.json`, `results.jsonl`,
  `report.md`, and per-run evidence.

## Event 2 — VALID attempt

- `suite_run_id`: `release-eval-openai-gpt-5.4-mini`
- git SHA: `fa7ef4e` (harness classification fix committed & pushed to
  `feature/eval-harness-v1`; all CI checks green)
- output directory: `artifacts/release-eval-baseline-valid-attempt-20260718-134209/`
  (new, distinct from the INVALID attempt; the INVALID artifacts are untouched)
- status: **VALID**

### Outcome distribution

| outcome | count |
|---------|-------|
| agent_failure | 7 |
| infrastructure_failure | 1 |
| (passed / expected_outcome_pass) | 0 |

- total runs: 8 (8 distinct jobs)
- `false_success_count`: 0
- `false_failure_count`: 0
- `provider_failure`: 0, `dobox_transport_failure`: 0, `harness_failure`: 0
- `infrastructure_failure` case: `single_file_bugfix` — genuine pre-workspace
  provisioning failure (`project=None`, `iterations=0`, `elapsed=3.1s`,
  `utf-8` decode error during fixture seeding). Correctly classified as
  `infrastructure_failure`, not `agent_failure`.
- The 7 `agent_failure` cases failed with `max_consecutive_failures_exceeded`
  after real tool activity — genuine agent failures, correctly classified.

### Validity checks

- 8 distinct jobs: ✅
- correct infrastructure classification: ✅ (1 genuine infra, 7 agent)
- no secret leak in outputs: ✅ (scoped scan clean; endpoints redacted)
- false success / false failure both 0: ✅

## Environment gates at run time

- Harness classification fix committed & PR #15 CI green: ✅
- Fixture validation 8/8: ✅
- Focused eval tests green (49 tests): ✅
- No-LLM DoBox lifecycle probe 12/12, 0 address-pool errors, 0 orphan growth: ✅
- DoBox health OK: ✅
- Code tree clean (only untracked `artifacts/`): ✅

## Notes

- The `single_file_bugfix` `infrastructure_failure` is a real fixture/encoding
  provisioning issue (non-UTF-8 byte in a fixture file), surfaced correctly as
  infrastructure rather than agent failure. Fixing the Runtime/fixture encoding
  is out of scope for this harness-validation task.
- The INVALID baseline artifacts remain permanently preserved and were not
  merged, overwritten, or deleted.
