# Primary systemic blocker — finalization rejects work the runtime itself verified

Status: identified from baseline V1 forensics
(`docs/eval/baseline-v1-failure-analysis.md`).

## Statement

> Every deterministic-holdout run that reached the finalization path was
> rejected there (4/7), including two runs that the isolated hidden checker
> independently confirmed as functionally correct. The rejections are produced
> by runtime logic that ignores, or structurally contradicts, evidence the
> runtime had already collected.

This is not a model-quality problem. It caps the achievable score of *every*
future case, regardless of provider.

## Affected runs

| Case | Reached finalization | Rejected | Hidden checker verdict |
| --- | --- | --- | --- |
| multi_file_bugfix | yes | yes (5x) | genuinely incomplete (second file unfixed) |
| node_bugfix | yes | yes (5x) | **functionally correct** |
| small_feature | yes | yes (blocked pre-verifier, 8x) | **functionally correct** |
| unsatisfiable_task | yes | yes (5x) | fabricated premise (agent failure) |

4/7 affected, 2/7 provably false failures.

## Mechanisms

The blocker has three concrete mechanisms, all in the finalization path, all
fixture-independent.

### M1 — the LLM judge never sees the required-command results

`src/docode/agent/verifier.py`

```python
explicit_results = await run_or_reuse_explicit_commands(...)   # runs / reuses
evidence = evidence_with_command_results(evidence, explicit_results)
...
test_result = await run_or_reuse_detected_check("test", "run_tests", ...)
...
judgement = await self._judge(job, status_result, verified_diff,
                              test_result, build_result, lint_result, smoke_result)
```

`explicit_results` is used for `commands_ok` bookkeeping, but is never handed to
`self._judge`. Because holdout projects have no auto-detectable test runner,
`run_or_reuse_detected_check` returns

```python
skipped_result("run_tests", "no test command detected")   # exit 0, 24 bytes
```

so the judge's only "test evidence" literally reads *no test command detected*
and it rejects with high confidence:

* node_bugfix — *"the required verification command `node tests/calc.test.js`
  was not run or its passing output was not provided"* (0.97)
* multi_file_bugfix — *"the required verification command `python -m unittest
  -q` was not shown as run/passing"*

Recorded evidence contradicts both: `explicit_commands[0].exit_code == 0`,
output `ALL TESTS PASSED` / `OK`.

**Expected behaviour:** the judge must receive the results of the job's declared
verification commands. When explicit commands exist and passed, the judge must
not be able to claim they were not run.

### M2 — `require_test_change` is unsatisfiable for non-Python test runners

`build_verification_plan()` sets `require_test_change=True` for bugfix tasks.
`bugfix_test_evidence_ok()` waives it only when a command matches

```python
markers = ("pytest", "unittest", "npm test", "go test", "cargo test")
```

`node tests/calc.test.js` matches none, so the plan demands "add or update a
related test" while the task instruction says *"Modify the implementation, not
the tests"* and the hidden checker fails any run whose tests changed. The agent
cannot satisfy both.

**Expected behaviour:** a successfully executed declared verification command
counts as test evidence, regardless of which runner it invokes; and the
requirement must be waived when the instruction forbids test modification.

### M3 — placeholder detection fires on *removed* placeholders

`src/docode/agent/quality_gate.py::detect_placeholder_code()` lowercases the
whole per-file diff (added lines, removed lines and hunk headers) and searches
for `todo`, `stub`, `raise notimplementederror`, `pass  # ...` markers.

A correct implementation of a stubbed function *necessarily* removes those
markers, producing `-` lines that contain them, so the blocker
`placeholder_left_in_diff` is guaranteed. `small_feature` was blocked this way
on every one of its eight finalization attempts while its tests passed.

**Expected behaviour:** scan only added lines (`+`, excluding the `+++` header).

## Why the loop becomes terminal

Each rejection is delivered without new actionable information (M1) or with an
instruction the agent must not follow (M2) or cannot satisfy (M3). The agent
resubmits the same `final_candidate`; the finalization controller counts it as
another consecutive failure; the run dies on
`max_consecutive_failures_exceeded`.

## Minimal fix

1. Pass explicit/required command results into `CodingVerifier._judge` and into
   the judge prompt; when a declared command exists and passed, state that
   explicitly and forbid the judge from claiming it was not run.
2. Treat any successful declared verification command as bugfix test evidence;
   waive `require_test_change` when the instruction forbids modifying tests.
3. Restrict `detect_placeholder_code` to added diff lines.

No fixture, checker, gold solution, budget, prompt-difficulty or model change is
involved. No `if case_id == ...` special-casing.

## Regression tests

* verifier judge receives explicit command results (unit).
* verifier passes when the declared command exited 0 and the diff is
  non-trivial, with a judge stub that fails if explicit evidence is absent.
* `bugfix_test_evidence_ok` accepts `node tests/x.test.js`,
  `python -m unittest -q`, `go test ./...`, `deno test`, `bun test`, and a bare
  script path that the job declared as a verification command.
* `require_test_change` waived when the instruction says tests must not change.
* `detect_placeholder_code` returns no issue for a diff that only *removes* a
  `# TODO` / `raise NotImplementedError` line, and still fires when the marker
  is added.

## Secondary blocker (tracked separately)

`apply_patch` accepts only `git apply` unified diffs and answers the OpenAI
apply-patch envelope with `error: No valid patches in input` (exit 128), with no
description of the accepted format. 3/7 runs produced zero bytes of work because
of this. See `docs/eval/baseline-v1-failure-analysis.md` §2.
