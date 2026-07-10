# DoCode unseen holdout: `eff27a7`

## Evaluation identity

- Frozen commit: `eff27a7cbe70408097591369787105ffc5aea777`
- Annotated tag: `agent-baseline-eff27a7`
- Local tag object: `076eef7f85a1e0319c94bf0241cbfea7837d76f7`
- Local and remote peeled commit: `eff27a7cbe70408097591369787105ffc5aea777`
- Evaluation branch: `eval/unseen-holdout-eff27a7`
- Provider/model for every real sample: `deepseek` / `deepseek-chat`
- DoBox mode: real isolated project, `no_internet`
- Real sample size: one valid run per case (`n=1`); pass@3 is not available

The production tree under `src/docode` was never changed during this evaluation. Harness-only defects found during the first collection attempt were corrected before accepting samples: the DoBox `FileResult` contract was handled correctly, the independent checker received the sandbox Python alias, and the Node 18 heredoc used dynamic ESM import. Invalid harness samples were replaced, not counted as extra model attempts.

## Baseline regression evidence

The direct Windows command ran 437 tests with 12 skips but reported three failures in existing calculator/product-parser smoke fixtures. All three used POSIX `shlex.quote()` around the Windows Python executable and consequently passed an invalid single-quoted path to `cmd.exe`. A read-only diagnostic showed the generated workspace changes were correct. Re-running the same suite with the test subprocess executable set to the PATH-resolved `python` produced:

```text
Ran 437 tests in 27.848s
OK (skipped=12)
```

No repository file was changed for this host compatibility adjustment.

The new deterministic holdout harness passed all eight cases, including final quality gate, verifier, and artifact export:

```text
Ran 4 tests in 51.937s
OK
```

The existing real DeepSeek + real DoBox diagnostic suite also remained green:

| known diagnostic | status | LLM decisions | commands | repairs |
| --- | --- | ---: | ---: | ---: |
| cli_output_bug | succeeded | 8 | 3 | 1 |
| multifile_api_mismatch | succeeded | 7 | 2 | 1 |
| parser_edge_case | succeeded | 12 | 3 | 2 |
| two_stage_repair | succeeded | 13 | 4 | 2 |

The complete diagnostic module ran 12 tests in 197.346 seconds and passed.

## Real unseen holdout results

| case | language | run | status | iterations | LLM decisions | commands | repairs | final | category | short reason |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| ivory_quill | Python | 1 | failed | 36 | 36 | 5 | 5 | no | repair_loop | Package creation stopped after one incomplete file; max iterations |
| cobalt_span | TypeScript | 1 | succeeded | 21 | 21 | 3 | 2 | yes | success | Both consumers repaired; public API preserved |
| verdant_port | Go | 1 | succeeded | 20 | 20 | 3 | 2 | yes | success | Endpoint and invalid-JSON behavior passed `go test ./...` |
| amber_depth | Python | 1 | failed | 36 | 36 | 11 | 1 | yes | verifier_false_negative | Correct workspace; verifier ran the wrong default test directory |
| silver_source | Python/HTTP | 1 | failed | 35 | 35 | 1 | 1 | no | repair_loop | Adapter hardcoded the wrong port/path/output, then lost repair progress |
| indigo_block | Node.js | 1 | succeeded | 3 | 3 | 2 | 0 | yes | success | Full heredoc was one controller-executed command and passed |
| sable_manual | Markdown | 1 | failed | 33 | 33 | 1 | 1 | no | repair_loop | Semantic check failed, followed by unavailable-read loop |
| crimson_ladder | Python | 1 | failed | 37 | 37 | 20 | 3 | yes | verifier_false_negative | Both staged defects repaired; verifier ran the wrong default test directory |

Structured results are in `.docode/evals/eff27a7-unseen-holdout/results.json`. The eight accepted result rows reference ignored raw traces under `.docode/evals/eff27a7-unseen-holdout/traces/`; raw traces are not committed.

## Aggregate metrics

| metric | value |
| --- | ---: |
| valid real runs | 8 |
| strict task success rate | 37.5% (3/8) |
| pass@1 | 37.5% (3/8) |
| pass@3 | not available (`n=1`) |
| independently functionally correct | 62.5% (5/8) |
| median iterations | 34 |
| median tool calls | 18.5 |
| successful tasks requiring repair | 66.7% (2/3) |
| runs failed before any edit | 0% |
| runs functionally correct but failed before runtime finalization | 25% (2/8) |

Every run read source or repository material before its first successful edit. The Go case and the docs case rewrote an existing whole file; neither fact alone explains their status because Go succeeded and docs failed on semantic content.

## Failure classification and independent workspace inspection

### Functionally correct but runtime failed

`amber_depth` is correct outside the loop. The explicit required command and independent command both passed all three checks. The verifier nevertheless invoked `python3 -m unittest discover -s tests`, which failed because this fixture intentionally uses `checks/`. Its own smoke path then ran `python -m unittest discover -s checks` successfully. The repeated disagreement consumed the remaining iterations. Category: `verifier_false_negative`.

`crimson_ladder` is also correct outside the loop. It progressed through a syntax failure and a changed value-mismatch signature, made two targeted edits, and repeatedly passed the two-test `checks/` suite. The verifier again invoked the nonexistent default `tests/` directory while its smoke command passed `checks/`. Category: `verifier_false_negative`.

These are runtime failures, not code-generation failures.

### Functionally incorrect failures

`ivory_quill` created only `nexora/__init__.py`, which imported nonexistent `tokenizer` and `ledger` objects. It then repeated repository listings and failing verification without creating the remaining modules or tests. Category: `repair_loop`. The architecture-level issue is weak multi-file project-creation progress, not the randomized package name.

`silver_source` wrote an adapter that ignored the supplied CLI URL and output path, instead hardcoding `localhost:8080`, `/`, and `artifact.json`. The local server therefore refused the connection. After the first failing command, 27 requested `read_file` actions were rejected because that tool was absent from the active repair schema. Category: `repair_loop` with an initiating code-generation error.

`sable_manual` made a Markdown-only edit and preserved source code, but omitted the required fenced request/response transcript. After the semantic command failed, 27 requested `read_file` actions were rejected by the active repair schema. Category: `repair_loop`.

Artifact export ran for all terminal jobs, including failure artifacts. This does not turn a failed job into a passing task.

## Evidence for and against overfitting

Evidence against solution leakage:

- The holdout uses new randomized entities, filenames, schemas, record values, and languages.
- The anti-overfit tests scan all Python production sources for holdout markers, filenames, payload values, and an expected implementation snippet; they pass.
- A direct diff check confirms no path under `src/docode` changed from the frozen tag.
- TypeScript, Go, and Node cases succeeded without known Python diagnostic vocabulary.
- Credential scanning found zero raw secret matches in structured results or raw traces.

Evidence that known-task performance overstates general reliability:

- All four known real diagnostics passed, while strict unseen pass@1 was only 37.5%.
- Production contains direct crawler, GitHub repository, `calculator.py`, and `cli.py` policies developed around historical tasks.
- Two unseen functionally correct workspaces were rejected because generic verifier command selection did not preserve the explicit task command.
- Two unrelated unseen failures entered the same unavailable-`read_file` repair loop.

The current agent is broad—it can solve unseen TypeScript, Go, Node, large-file Python, and staged Python code at least functionally—but it should not yet be called reliably generally capable. Strict final success is below half at `n=1`, and repeated runtime state failures materially reduce reliability.

## Production special-case inventory

| item | location | classification | assessment |
| --- | --- | --- | --- |
| `crawler_contract_requirements` | `agent/context.py` | reusable task profile candidate | Dependency, dry-run, artifact, and fixture guidance is useful for crawler tasks but should not live in generic context assembly. |
| GitHub repository JSON validation | `agent/quality_gate.py` | domain-specific validator | Validates GitHub URLs and repository identifiers. Legitimate when explicitly selected, but it is directly keyword-dispatched inside the general quality gate. |
| crawler source-domain controls | `agent/loop.py` | reusable task profile candidate | Constrains fetch/search/edit domains and corrects off-domain fetches. Useful policy, but tightly couples the generic loop to crawler semantics. |
| CISA/CIS drift blockers | `agent/loop.py` | likely historical-eval residue | Literal `cisa`, `cis benchmark`, `cis control`, and security-advisory blockers plus a CISA/CIS prompt sentence are too specific for a generic loop. |
| `calculator.py` repair hint | `agent/loop.py` | likely historical-eval residue | Filename-dispatched `python-bugfix` advice is harmless in isolation but directly encodes a known fixture. |
| `cli.py` repair hint | `agent/loop.py` | likely historical-eval residue | Filename-dispatched `python-cli` advice is another known-task heuristic and belongs in a selectable profile, if retained. |
| crawler verifier requirements | `agent/verifier.py` | domain-specific validator / reusable profile candidate | Adds crawler entrypoint, dependency declaration, output artifact, dry-run, public-source evidence, and duplicate-implementation rules. These are coherent as a profile, not as generic verifier branches. |
| crawler/CLI-expanded default budgets | `api/job_actions.py` | harmless heuristic / reusable profile candidate | Keyword-based budget expansion is not a solution, but it changes evaluation resources by task wording and should be profile-owned and reported. |
| existing production anti-overfit scanner | `tests/test_production_anti_overfit.py` | general invariant | Useful guard, but its permanent literal denylist covers known fixtures only and cannot detect unseen leakage automatically. |
| new holdout anti-overfit scanner | `tests/holdout/test_deterministic_holdout.py` | general invariant | Scans the full production tree for the new fixture manifest, exact values, task filenames, payload, and solution snippet without modifying production. |

No exact dangerous hardcoded holdout solution was found in production. The concern is accumulated historical-evaluation policy in generic code, not leakage of this holdout's answers.

## Smallest architecture-level improvements supported by multiple cases

1. Make explicit required commands authoritative end to end. The inspector, `DoBoxTools` detection, verifier, and smoke path should share one command registry. A successful exact required command must not be invalidated by a later fallback to `python3 -m unittest discover -s tests`. This addresses both `amber_depth` and `crimson_ladder`.
2. Replace phase-specific tool hiding with explicit capabilities and progress transitions. Targeted repair must either expose the safe read needed for the target edit or controller-execute/retarget it once. Repeated requests for an unavailable safe tool should trigger a state correction, not consume 27 decisions. This addresses both `silver_source` and `sable_manual`.
3. Add a project-creation workflow that can establish a multi-file change set before switching to `TEST_REQUIRED`. An empty repository should not become trapped after the first package file. This is supported directly by `ivory_quill` and by the high repair cost of otherwise successful multi-file TypeScript/Go tasks.
4. Move domain behavior behind registered task profiles/policies rather than keyword branches in the generic loop.

A minimal future layout is:

```text
src/docode/task_profiles/
    base.py
    crawler.py
    github_repository.py
    cli.py
```

Each profile should provide context requirements, allowed source policy, verification-plan additions, quality validators, and optional budget defaults through interfaces. The generic loop should consume those interfaces without containing domain literals. A registry/plugin design is equivalent if it keeps activation explicit and traceable.

## One-off failures that should not receive production patches

- Do not add `nexora` package/module hints or expected file lists for `ivory_quill`.
- Do not add local mosaic selectors, record values, port/path guesses, or output schema strings for `silver_source`.
- Do not add Sable-specific headings or transcript text to prompts/verifiers.
- Do not add `zephyr_lattice.py` or `lumen_quota.py` exceptions; their code was already functionally correct.
- Do not add Node heredoc fixture vocabulary; the generic multiline controller path already passed.

The next production pass should address only the shared command-authority and repair-capability state problems after review of this frozen report.
