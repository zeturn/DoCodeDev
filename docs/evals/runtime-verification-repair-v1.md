# Runtime verification and repair v1 evaluation

## Evaluation identity

- Frozen baseline tag: `agent-baseline-eff27a7`
- Frozen baseline commit: `eff27a7cbe70408097591369787105ffc5aea777`
- Starting evaluation commit: `e61984eb9fed7af3503543bca05e739277176d70`
- Runtime branch: `fix/runtime-verification-repair-v1`
- Verification implementation commit: `4cc018edb59b5c78e240b43f9d232f8e1d0e9bb2`
- Repair implementation commit: `93842ab2c8d501ce3edd834c19818915491d0d06`
- Provider/model: DeepSeek `deepseek-chat`
- Formal holdout sample: one unchanged run of each of the eight frozen cases

The baseline tag was not moved. No holdout fixture, prompt, validation command, expected result, iteration budget, or tool budget was changed.

## Verification command audit

Before this pass, verification commands came from multiple independent paths:

1. `task_contract.py` parsed explicit one-line and multiline commands into `TaskContract.must_run_commands`.
2. `inspector.py` detected repository test/build/lint commands, then separately regex-mapped instruction text to conventional test commands and could overwrite detection.
3. `DoBoxTools` detected and executed repository test/build/lint commands, including harness-provided command overrides.
4. `workflow.py` used `TaskContract` commands for phase transitions and the final gate.
5. `loop.py` executed model and controller-owned `run_command` calls.
6. `verifier.py` parsed the instruction again, ran test/build/lint unconditionally, and appended explicit commands to smoke verification, causing fallback and duplicate execution.

The new path is: instruction -> one `TaskContract` parse -> inspector/bootstrap -> context -> workflow/controller -> canonical `metadata.command` evidence -> verifier -> final gate. Repository-detected test/build/lint commands remain supplemental and never replace explicit acceptance commands.

## Implementation

Explicit commands are retained in full in `ProjectInspection` and `VerificationPlan`. Compact plans use a first-line multiline summary. Workflow success is now scoped after the latest successful source edit. Evidence stores every command run with full command, output, exit code, step index, edit epoch, and explicit marker; duplicates remain aligned instead of being deduplicated into parallel arrays.

The verifier reuses the latest fresh exact result, runs only a missing exact command, and fails on a fresh failed exact result. Generic test/build/lint checks run only when the repository detector returns a real command. Missing checks are represented as exit-zero results with `metadata.detected=false` and `metadata.skipped=true`. A genuinely detected failing check still fails verification.

Targeted repair now has these capabilities:

| phase | focused reads | edits | run command | git evidence |
| --- | --- | --- | --- | --- |
| `inspect_allowed` | target `read_file`, `read_file_range`, `read_symbol` | yes | blocked | status/diff |
| `edit_forced` | still present in schema; bounded by policy | yes | blocked | status/diff |
| `rerun_required` | hidden | no | exact repair rerun | status/diff |

Only successful, non-empty target reads consume the budget. Unrelated paths return `repair_read_not_targeted`; identical requests return `repair_read_repeated`; exhausted useful reads return `repair_read_budget_exhausted`. A first useful read remains possible at zero budget, and truncated reads permit a range/symbol follow-up. Generic fallback repair candidates include previously inspected source files and failure-referenced fixture/test context, preventing a multi-file repair from being locked to the first changed file.

## Deterministic results

| suite | result |
| --- | --- |
| Focused contract/context/workflow/verifier/loop/inspector/planner/regression | 213 passed |
| Frozen deterministic holdout | all 8 cases passed; unittest package 5 passed, 1 skipped |
| Full deterministic suite | 463 passed, 13 skipped |

The full baseline was 442 passed and 13 skipped. New neutral regressions cover a repository with no `tests/` directory and two exact `checks/` commands, a failing explicit variant, stale command evidence, complete heredoc identity, detected failing tests, and an `edit_forced` range-read/edit/rerun sequence with no schema loop.

## Real diagnostics

The first unchanged four-case invocation produced 3/4: CLI, multi-file API, and two-stage repair succeeded; parser edge case exposed that a failure-referenced fixture was absent from generic fallback targets. After the generic context-target fixes, targeted confirmation of parser edge case and two-stage repair passed 2/2. The latest per-case confirmation is therefore 4/4, although the last confirmation was split across the full invocation and a targeted two-case invocation rather than one final four-case process.

All diagnostic logs and 11 complete sanitized step traces are under `.docode/evals/runtime-verification-repair-v1/diagnostic-traces/`. These raw diagnostic traces are intentionally not committed.

## Formal holdout comparison

| case | baseline | new | baseline category | new category | functional | iterations | tool calls | explanation |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| `ivory_quill` | failed | failed | repair_loop | repair_loop | no | 36 -> 36 | 36 -> 36 | Incomplete package creation; one edit followed by repeated failing commands. |
| `cobalt_span` | succeeded | failed | success | repository_understanding_failure | no | 21 -> 36 | 20 -> 5 | This sample read context but never produced an edit; no verifier or hidden-read loop. |
| `verdant_port` | succeeded | succeeded | success | success | yes | 20 -> 30 | 17 -> 17 | Retained functional and strict success after three repair actions. |
| `amber_depth` | failed | succeeded | verifier_false_negative | success | yes | 36 -> 7 | 23 -> 6 | Exact explicit check was reused; nonexistent conventional tests no longer invalidate it. |
| `silver_source` | failed | succeeded | repair_loop | success | yes | 35 -> 10 | 8 -> 8 | Repair completed without a hidden-read loop and both explicit commands were reused. |
| `indigo_block` | succeeded | succeeded | success | success | yes | 3 -> 4 | 4 -> 5 | Full heredoc identity remained intact and was reused by the verifier. |
| `sable_manual` | failed | succeeded | repair_loop | success | yes | 33 -> 15 | 6 -> 8 | Structured read-budget feedback led to the missing transcript edit and rerun. |
| `crimson_ladder` | failed | succeeded | verifier_false_negative | success | yes | 37 -> 11 | 35 -> 10 | Fresh exact command evidence passed without a `tests/` fallback. |

## Aggregate metrics

| metric | baseline | new |
| --- | ---: | ---: |
| Strict success | 3/8 (37.5%) | 6/8 (75.0%) |
| Independently functional | 5/8 (62.5%) | 6/8 (75.0%) |
| Verifier false negatives | 2 | 0 |
| Repair loops | 3 | 1 |
| Functionally correct but runtime failed | 2 | 0 |
| Median iterations | 34.0 | 13.0 |
| Median tool calls | 18.5 | 8.0 |
| Successful tasks requiring repair | 2/3 | 5/6 |

All eight new runs read existing files before editing. Two of six successful runs used whole-file writes (`verdant_port`, `sable_manual`); the large-file repair remained targeted. Every successful verifier reused fresh explicit-command evidence, including the full Node heredoc. Verifier finalization executed zero duplicate explicit commands; repeated command executions in traces were repair attempts before the final success.

## Failure and risk analysis

`ivory_quill` remains a genuine code-generation/project-creation repair loop, not a verifier false negative. Its 34 command executions show a separate controller progress issue: failed required commands can be retried repeatedly while the multi-file package is still incomplete.

`cobalt_span` is a regression in this single stochastic sample: it failed before any edit with repository-understanding failure. The deterministic version remains green, so the evidence does not support a task-specific patch; this needs broader reliability measurement.

No run had a repair loop caused by a hidden safe read. There was one `read_file` schema rejection in successful `silver_source` after the source edit, when the phase had intentionally moved to `rerun_required`; it did not repeat or block success. Keeping focused reads visible in that phase is a possible future schema-stability refinement.

The formal targets were met for strict success (>=5/8), functional success (>=6/8), nonexistent-`tests/` verifier false negatives (0), and hidden-safe-read repair loops (0). The target that all three baseline successes remain successful was missed because `cobalt_span` failed.

## Leakage and artifacts

Production sources contain none of the eight holdout case names, holdout-specific filenames, expected outputs, or unique fixture values. The sanitized structured result has zero raw secret-pattern matches. Formal structured results are stored at `.docode/evals/runtime-verification-repair-v1/results.json`; raw holdout and diagnostic traces remain local and uncommitted for debugging.
