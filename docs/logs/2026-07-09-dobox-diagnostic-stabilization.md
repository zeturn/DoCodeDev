# 2026-07-09 DoBox real LLM diagnostic stabilization log

## Summary

Today we stabilized the real LLM diagnostic suite by moving the main signal to real DoBox/Linux execution and fixing the control-loop paths that were causing agent jobs to stall before meaningful repair.

Final result:

```text
DOCODE_REAL_LLM_SMOKE=1 DOCODE_REAL_DOBOX_SMOKE=1 DOCODE_REAL_LLM_PROVIDER=deepseek DOCODE_REAL_LLM_MODEL=deepseek-chat DOCODE_DEEPSEEK_BASE_URL=https://api.deepseek.com/v1 PYTHONPATH=src python3 -m unittest -v tests.test_real_llm_diagnostic_suite
```

Result:

```text
Ran 12 tests in 901.163s
OK

case | mode | status | iterations | commands run | final attempted | repair actions | likely failure category | short reason
cli_output_bug | real_dobox | succeeded | 6 | 2 | True | 0 | unknown | Completed the requested changes in cli.py, out.json.
multifile_api_mismatch | real_dobox | succeeded | 7 | 2 | True | 1 | unknown | Completed the requested changes in app.py.
parser_edge_case | real_dobox | succeeded | 12 | 3 | True | 2 | unknown | Completed the requested changes in parser.py.
two_stage_repair | real_dobox | succeeded | 15 | 4 | True | 2 | unknown | Completed the requested changes in crawler.py, out.json.
```

Deterministic suite also passed:

```text
PYTHONPATH=src python3 -m unittest discover -s tests
Ran 389 tests in 2.829s
OK (skipped=12)
```

## What changed

### 1. Moved the real diagnostic signal to DoBox

The previous real LLM diagnostics were polluted by local fixture runner and OS behavior, especially Windows shell differences such as Unix-only `/dev/null` redirection. We added/used the `DOCODE_REAL_DOBOX_SMOKE=1` path so the real diagnostic cases run with:

- real LLM;
- real DoBox project/session;
- real Linux command execution;
- real file IO;
- production `DoBoxTools` path.

This confirmed that DoBox itself was not the blocker.

### 2. Fixed duplicate inspection dead loops

Earlier failures were mostly:

```text
0 commands + duplicate_inspection_after_edit_pressure + max_consecutive_failures_exceeded
```

The loop was turning repeated `read_file` attempts into hard rejected decisions. DeepSeek would often repeat the same read request, causing the job to die before any test or repair evidence existed.

The fix changed the behavior to:

- retarget repeated duplicate inspection to the first required verification command when no diff/edit/required command exists yet;
- otherwise return a synthetic successful cached duplicate-read result instead of a hard rejection;
- stop using duplicate inspection as a consecutive-failure sink.

Outcome: the three previously stuck cases started running required commands and producing useful failure evidence.

### 3. Added fallback repair for failed required commands

After duplicate-read retargeting, two cases reached test failure but produced no repair action:

```text
multifile_api_mismatch | commands=1 | repair_actions=0 | test_failure_not_repaired
parser_edge_case | commands=1 | repair_actions=0 | test_failure_not_repaired
```

We added a generic fallback repair path for failed required commands. If a `run_command` is equivalent to one of the task contract's required commands and the specific repair planner cannot classify the output, the loop creates a generic `failed_required_command` repair action with:

- failed command;
- failure summary;
- candidate target files from the task contract;
- instruction to edit before rerunning;
- rerun command.

Outcome: `multifile_api_mismatch` and `parser_edge_case` moved from `repair_actions=0` to successful repair runs.

### 4. Made targeted repair advisory instead of coercive

`two_stage_repair` reached a repair action but could still get stuck behind `targeted_repair_wrong_action` / `forbidden_until_modified` hard gates.

We changed targeted repair control so normal local tools are not hard-blocked by the active repair action:

- read/edit/write/apply_patch/run_command/git_status/git_diff remain usable;
- unsafe external tools can still be blocked;
- targeted repair remains strong guidance in context, not a rigid controller.

Outcome: `two_stage_repair` completed successfully with 2 repair actions and 4 commands.

### 5. Cleaned local diagnostic fixture metadata

The DoBox seeding path uploads fixture files as UTF-8 text. Local macOS metadata or other binary hidden files could break seeding with `UnicodeDecodeError` before the agent loop started.

We added test package cleanup for:

- hidden files;
- `__pycache__`;
- `.pyc` / `.pyo`;
- any non-UTF-8 diagnostic fixture files.

Outcome: real DoBox fixture seeding no longer fails on local metadata files.

### 6. Made diagnostic trace git helpers timeout-safe

A later failure happened after the jobs had already run, while writing diagnostic traces:

```text
write_diagnostic_trace -> tools.git_diff() -> DoBox HTTP ReadTimeout
```

This converted useful real diagnostic results into unittest `ERROR`s.

We made diagnostic git helper calls timeout-safe by returning a `ToolResult(exit_code=124)` instead of raising, so trace export cannot erase the real case summary.

Outcome: the real diagnostic suite now reports true case results instead of failing during trace collection.

## Important commits from the stabilization sequence

Recent key commits include:

```text
da9032b fix(dobox): make diagnostic git trace helpers timeout-safe
26ac2bd fix(agent): install loop runtime fixes via sitecustomize
9a77728 test: clean non-text diagnostic fixture metadata
2cfa65f test: ignore local finder metadata in diagnostic fixtures
8ddecfb revert(agent): disable broad runtime loop monkey patch
```

Earlier related commits included the DoBox real diagnostic mode and duplicate-read retargeting/cached-result behavior.

## Lessons learned

### DoBox was not the real blocker

Once the diagnostics used real DoBox/Linux, `cli_output_bug` passed and the remaining failures became clear control-loop issues.

### Hard rejection loops are dangerous with stochastic LLMs

A model may repeat an invalid or unhelpful tool call after a rejection. If the loop treats every repeated attempt as a hard failure, it can exhaust `max_consecutive_failures` without ever collecting evidence or allowing repair.

Better pattern:

```text
bad repeated inspection -> gather test evidence or return cached guidance
failed required command -> repair episode
repair episode -> advisory guidance + local tools remain available
```

### Failed required commands need a generic fallback

Specific repair planners are useful, but they cannot cover every traceback/assertion shape. If a required verification command fails, the loop must always be able to create an editable repair episode.

### Targeted repair should guide, not trap

The repair action should focus the model, but it should not hard-block normal local tools so aggressively that the agent cannot recover from a slightly different trajectory.

### Trace/export code must not hide the real result

Diagnostics should be robust in the face of auxiliary DoBox timeouts. A trace failure should be recorded as unavailable metadata, not promoted into the primary test failure.

## Current validated state

As of this log, the following passed on the user's machine:

```text
Deterministic suite: 389 tests OK, 12 skipped
Real DoBox + DeepSeek diagnostic suite: 12 tests OK
All four real diagnostic cases succeeded
```

## Follow-up notes

- The runtime patch approach was intentionally narrow and used to avoid risky full replacement of the large `loop.py` through the remote GitHub contents API.
- Once stable, the fallback repair and advisory targeted-repair behavior should be folded directly into `loop.py` with dedicated deterministic tests.
- Keep `DOCODE_REAL_DOBOX_SMOKE=1` as the authoritative real LLM diagnostic path.
- Preserve strict final gates: required commands, non-empty diff, quality gate, verifier, and artifact export should remain strict.
