# DoCode crawler benchmark v1: `d9579ed`

## Evaluation identity and protocol

- Frozen runtime: `d9579ed12ea7f116a296363de39fa3b329d81e41`
- Annotated tag: `agent-runtime-v1-d9579ed`
- Tag object: `9beeeaf8d0bee4c5b1caf092d940a43566855851`
- Local and remote peeled commit: `d9579ed12ea7f116a296363de39fa3b329d81e41`
- Evaluation branch: `eval/crawler-benchmark-v1-d9579ed`
- Definition SHA-256: `ed5b7c3b1329163bd04e45133fbf9091d4b29fe7412c546d0e3be16d9b9d0142`
- Provider/model/quality: `deepseek` / `deepseek-chat` / `balanced`
- Limits: `max_iterations=36`, `max_tool_calls=80`, `artifact_mode=patch`
- Sample size: three valid runs per case, 18 valid runs total
- Isolation: every valid run created a new DoBox project and deleted it afterward
- Network: cases A-E used `no_internet`; case F used project networking for its public HTTPS feed

The prompts, fixtures, checker, budgets, and aggregation code were frozen before the first formal model call. The formal runner stored a sanitized summary after every valid run and resumed by missing `(case, run)` slot. One `copper_orbit` run-2 collection attempt hit a DoBox exec `ReadTimeout` during post-run inspection. It is recorded as an invalid harness attempt, is excluded from all denominators, and the missing slot was recollected without repeating the ten already valid samples.

Raw traces remain under ignored `.docode/evals/crawler-benchmark-v1-d9579ed/traces/`. The committed assets are the structured result, 18 sanitized run summaries, deterministic harness, fixtures/checkers, and this report.

Credential scanning covered the structured result, all run summaries, raw traces, tests, and report. It found zero matches for the configured secret and zero serialized bearer credentials. The only source-level `sk-...`-shaped string is the deliberately synthetic sanitizer unit-test input; it is not a credential, and its test asserts replacement with `[REDACTED]`.

## Case design

| case | source and difficulty | workspace shape | independent check |
| --- | --- | --- | --- |
| `opal_canopy` | local HTML; 9 target cards plus decoys; standard library | empty implementation | exact hidden-variant payload and schema |
| `flint_harbor` | local irregular table; 8 rows, reordered/nested cells, commas, whitespace, missing value | partial scaffold | exact hidden-variant payload and normalization |
| `marble_tide` | local two-page HTML; 7+6 rows, two duplicates, discovered next link | README plus absent implementation | exact hidden-variant order, deduplication, and payload |
| `violet_prism` | local RSS; namespaces, entities, relative/absolute links, multiline and missing summary | empty implementation plus existing `checks/` directory | exact hidden-variant payload |
| `copper_orbit` | local cursor JSON; 6+5 rows, one duplicate, numeric strings/integers | partial scaffold | exact hidden-variant pagination, normalization, and payload |
| `cedar_signal` | documented public CNEOS RSS over HTTPS | README plus absent implementation | live structural validation, at least five rows |

The controlled cases have an undisclosed variant with different values and, where relevant, different cardinality/cursor. Formal functional correctness required the generated collector to run against that variant after the runtime ended; passing only the prompt-visible artifact was insufficient.

The public source was documented at `https://cneos.jpl.nasa.gov/feed/` and the feed was `https://cneos.jpl.nasa.gov/feed/news.xml`. Before freezing, host and DoBox probes both returned HTTP 200, `text/xml`, 10 items, and 8,346 bytes. No run was classified `source_unavailable`.

## Pre-collection evidence

The focused deterministic suite passed all six reference collectors against both base and hidden variants. It covered fixture startup/shutdown, response content types, schema, output paths, whitespace and numeric normalization, deduplication, pagination, and exact request order/count:

```text
Ran 6 tests in 8.835s
OK
```

A separate real DoBox lifecycle probe started the same fixture service, fetched both ledger pages, read `{"count": 2, "requests": ["/ledger/start", "/ledger/next"]}`, stopped the service, and deleted the project.

The direct full-suite command without `PYTHONPATH=src` failed at import time because this repository uses a `src/` layout. With `PYTHONPATH=src`, the suite ran 470 tests: 466 passed, 14 were skipped, and four historical tests failed. Three were the already documented Windows `shlex.quote(sys.executable)` problem; all three passed when the test subprocess executable was set to PATH-resolved `python`. The remaining failure was the previous unseen holdout's `ivory_quill` deterministic case. It independently produced a correct workspace but the runtime stopped at `max_consecutive_failures_exceeded`. The same failure reproduced in a detached worktree at the unmodified annotated tag, so it is a frozen `d9579ed` baseline behavior, not a crawler-benchmark regression.

The production tree remained unchanged from the frozen tag throughout collection.

## Formal results

| case | run | runtime | category | iterations | tools | repairs | functional | exact commands | strict |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `opal_canopy` | 1 | failed | repair loop | 31 | 19 | 4 | no | no | no |
| `flint_harbor` | 1 | failed | repair loop | 30 | 22 | 6 | no | no | no |
| `marble_tide` | 1 | failed | repair loop | 31 | 22 | 5 | no | no | no |
| `violet_prism` | 1 | succeeded | code generation failure | 7 | 8 | 1 | no | yes | no |
| `copper_orbit` | 1 | failed | repair loop | 36 | 28 | 10 | no | no | no |
| `cedar_signal` | 1 | failed | runtime failed after functional verification | 31 | 21 | 5 | yes | no | no |
| `opal_canopy` | 2 | failed | repair loop | 32 | 19 | 4 | no | no | no |
| `flint_harbor` | 2 | failed | repair loop | 32 | 19 | 16 | no | yes | no |
| `marble_tide` | 2 | failed | repair loop | 30 | 18 | 4 | no | no | no |
| `violet_prism` | 2 | succeeded | code generation failure | 7 | 8 | 1 | no | yes | no |
| `copper_orbit` | 2 | failed | repair loop | 30 | 20 | 6 | no | no | no |
| `cedar_signal` | 2 | succeeded | success | 4 | 5 | 0 | yes | yes | yes |
| `opal_canopy` | 3 | failed | repair loop | 32 | 17 | 4 | no | no | no |
| `flint_harbor` | 3 | failed | repair loop | 33 | 20 | 17 | no | yes | no |
| `marble_tide` | 3 | failed | repair loop | 31 | 21 | 5 | no | no | no |
| `violet_prism` | 3 | failed | repair loop | 35 | 16 | 4 | no | no | no |
| `copper_orbit` | 3 | failed | repair loop | 29 | 21 | 7 | no | no | no |
| `cedar_signal` | 3 | succeeded | success | 4 | 5 | 0 | yes | yes | yes |

Structured results are in `.docode/evals/crawler-benchmark-v1-d9579ed/results.json`.

## Aggregate metrics

| metric | value |
| --- | ---: |
| valid runs | 18 |
| invalid harness attempts | 1, excluded |
| overall strict run success | 11.1% (2/18) |
| pass@1 | 0.0% (0/6) |
| pass@3 | 16.7% (1/6 cases had at least one strict pass) |
| independently functional | 16.7% (3/18) |
| median iterations | 31 |
| median tool calls | 19 |
| successful runs requiring repair | 0.0% (0/2) |
| failed before any edit | 0.0% |
| functional but runtime failed | 5.6% (1/18) |
| HTTP source inspected before first edit | 0.0% (0/18) |
| source unavailable | 0 |

By case, only `cedar_signal` passed: it was functionally correct in all three runs and strictly successful in runs 2 and 3. All 15 controlled runs failed independent hidden-variant payload equality. `violet_prism` runs 1 and 2 are especially important: the runtime and both prompt-visible commands succeeded, but the hidden feed exposed wrong `source` and missing-summary normalization.

## Failure analysis

### Source inspection did not happen

All 18 runs read a workspace file before editing, but none fetched its HTTP source before the first edit. Empty implementations were read as zero-byte files and then replaced with guessed parsers. The task explicitly required source inspection. The runtime's generic crawler prompt strongly prefers `fetch_url`, while these DoBox-local sources were reachable through sandbox commands and the active benchmark toolset did not expose a sandbox-backed `fetch_url`. The model still had `run_command` and could inspect with standard-library HTTP, but it never did so before coding.

This explains several concrete mistakes:

- `opal_canopy` produced no hidden cards.
- `flint_harbor` treated positional/decorative cells as fields, emitting `x` as sector in all three hidden variants.
- `marble_tide` produced an empty hidden artifact.
- `violet_prism` missed the namespaced creator and represented missing summary incorrectly.
- `copper_orbit` run 1 treated the cursor token as a URL; runs 2 and 3 produced empty hidden artifacts.

### Producer/validator dependency was not preserved during repair

The first exact command generated the artifact; the second exact heredoc validated it. After a validation failure, targeted repair repeatedly edited source and reran only the dependent heredoc. It did not consistently rerun the producer first, so validation often observed stale output. This pattern is supported across `opal_canopy`, `marble_tide`, `copper_orbit`, and `cedar_signal` run 1. The latter is decisive: independent inspection reran the final collector and obtained a structurally correct live artifact, but the runtime exhausted iterations after five failed heredoc attempts against earlier output.

`d9579ed` correctly preserved the exact multiline command itself; it did not preserve the dependency between two exact commands.

### Placeholder quality repair loop

`flint_harbor` began from a partial scaffold containing `raise NotImplementedError`. Across its three runs, 33 quality-repair actions reported the same `placeholder_left_in_diff` issue. The generated collectors could pass the visible commands in runs 2 and 3 while leaving the marker in the diff, and the runtime continued repair until max iterations. The hidden parser was still wrong, so this is not a false positive, but the repeated identical quality repair consumed budget without converging.

### Runtime success was not sufficient evidence

Two of 18 runs were runtime `succeeded` yet independently incorrect (`violet_prism` runs 1 and 2). Conversely, one run was independently correct but runtime failed (`cedar_signal` run 1). This benchmark therefore distinguishes runtime finalization from functional correctness and requires both for strict success.

### Post-collection request-metric limitation

The hidden-variant payload check ran for all 15 controlled samples. The subsequent hidden-variant request-metrics probe invoked the sandbox Python alias outside the shell PATH where the alias was configured, so those formal summaries do not contain usable hidden request counts. This does not change a classification: every controlled sample had already failed exact hidden-payload equality. Prompt-visible request counts were enforced whenever the second exact heredoc passed. Separately, all deterministic base/hidden reference checks and the real DoBox lifecycle probe passed exact request order/count.

No claims in this report rely on the unavailable hidden request metric.

## Generalization and overfitting assessment

The crawler capability in `d9579ed` is not robustly general.

- Pass@1 is 0/6 and pass@3 is 1/6.
- None of 15 controlled HTML/RSS/JSON samples generalized to an undisclosed variant.
- All three live RSS workspaces were functional, and two finalized quickly, showing a useful narrow capability.
- The earlier known diagnostic suite was 4/4 and the prior broad unseen holdout had higher strict pass@1, but this focused crawler benchmark is materially worse. The evaluations differ in task mix, so the delta is evidence of a crawler frontier weakness, not a causal estimate of regression.
- The exact-command work in `d9579ed` generalized partially: complete heredocs were preserved and visible commands passed in several runs. It did not solve command dependency ordering, mandatory source inspection, hidden-data generalization, or repeated quality-repair convergence.

The result is more consistent with strong performance on familiar/visible crawler shapes than with a generally reliable crawler-building agent. The live RSS success is real, but it is insufficient to offset 0/15 controlled hidden-variant correctness.

## Production special-case inventory

| location | current special case | assessment |
| --- | --- | --- |
| `agent/prompts.py` | hard-coded crawler instructions plus historical GitHub-trending parser symbols, fields, CLI flags, and `crawler.py` advice | mixed reusable policy and obvious historical-task residue; generic prompts should not prescribe one site's schema |
| `agent/context.py` | keyword-triggered crawler contract requirements and source-first instructions | reusable as an explicit crawler profile, but capability/tool availability must be checked before requiring `fetch_url` |
| `agent/loop.py` | crawler source-domain correction/blocking, CISA/CIS drift strings, and filename-specific calculator/CLI hints | source-domain policy is reusable; CISA/CIS and filename branches are historical residue and should leave the generic loop |
| `agent/quality_gate.py` | generic JSON artifact checks plus GitHub repository URL/name validation | generic artifact checks are useful; GitHub validation belongs in a selected domain validator |
| `agent/verifier.py` | crawler entrypoint, dependency, artifact, dry-run, source-evidence, and duplicate-implementation rules; preferred filenames include historical names | coherent crawler-profile material, but not generic verifier branches |
| `api/job_actions.py` | keyword-based crawler budget expansion | harmless heuristic but should be profile-owned and reported explicitly |

No new benchmark marker, filename, fixture value, or reference solution was found in `src/docode`. The concern is accumulated old-task policy, not leakage of these six cases.

## Smallest architecture-level improvements supported by multiple cases

1. Introduce a source-inspection capability owned by the sandbox/tool profile. For local/private fixture URLs it should execute in the DoBox network namespace; for public URLs it may use the web tool. Context and initial workflow must ask only for tools actually present. Record successful source inspection as a required pre-edit state transition. This is supported by all six cases and the observed 0/18 HTTP-inspection rate.
2. Model verification commands as a dependency graph. When command 2 validates an artifact produced by command 1, any source edit must invalidate that artifact and rerun command 1 before command 2. Keep exact heredoc atomicity, but preserve producer-before-validator ordering. This is supported by at least `opal_canopy`, `marble_tide`, `copper_orbit`, and `cedar_signal` run 1.
3. Add repeated-repair convergence handling. A repeated quality signature such as `placeholder_left_in_diff` should target the exact marker, verify its removal, and stop requesting semantically identical edits. This is supported by all three `flint_harbor` runs and the 33 repeated quality repairs.
4. Move crawler and domain behavior behind registered task profiles. A crawler profile should provide source-inspection tools, verification dependencies, artifact validators, dependency policy, and budget defaults. GitHub-specific fields/URLs should be a separate explicit profile, not global crawler prompt text.

Do not add case filenames, selectors, cursor values, expected records, or hidden fixtures to production. Those would be one-off patches and would invalidate the benchmark's purpose.

## Exact frozen prompts and required commands

### `opal_canopy`

```text
Build a standard-library HTML card collector.

The local page contains exactly nine target cards among decoys. Write a CLI that accepts SOURCE_URL and OUTPUT_PATH. Extract sigil, label, group, integer rank, and an absolute detail_url. Use only the Python standard library.

Inspect the supplied source before editing. Implement the collector; do not merely describe it. Both verification commands are required, and the second is one atomic multiline heredoc.

Verification commands:
1. python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/__reset', timeout=5).read()" && python aurora_index.py http://127.0.0.1:8765/aurora/cards aurora_records.json
2. python - <<'PY'
import json
import urllib.request
from pathlib import Path
records = json.loads(Path('aurora_records.json').read_text(encoding='utf-8'))
assert isinstance(records, list) and len(records) == 9, len(records)
assert all(set(record) == set(('sigil', 'label', 'group', 'rank', 'detail_url')) for record in records)

with urllib.request.urlopen('http://127.0.0.1:8765/__metrics', timeout=5) as response:
    metrics = json.load(response)
assert metrics['count'] == 1, metrics
assert metrics['requests'] == ['/aurora/cards'], metrics
print('benchmark validation passed')
PY
```

### `flint_harbor`

```text
Complete the irregular observation-table collector.

The target table has eight records plus irrelevant rows/tables, reordered cells, nested markup, whitespace, comma-formatted values, and one missing reading. Emit sector, station, reading (integer or null), and observed_at.

Inspect the supplied source before editing. Implement the collector; do not merely describe it. Both verification commands are required, and the second is one atomic multiline heredoc.

Verification commands:
1. python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/__reset', timeout=5).read()" && python kiln_reader.py http://127.0.0.1:8765/kiln/observations kiln_snapshot.json
2. python - <<'PY'
import json
import urllib.request
from pathlib import Path
records = json.loads(Path('kiln_snapshot.json').read_text(encoding='utf-8'))
assert isinstance(records, list) and len(records) == 8, len(records)
assert all(set(record) == set(('sector', 'station', 'reading', 'observed_at')) for record in records)

with urllib.request.urlopen('http://127.0.0.1:8765/__metrics', timeout=5) as response:
    metrics = json.load(response)
assert metrics['count'] == 1, metrics
assert metrics['requests'] == ['/kiln/observations'], metrics
print('benchmark validation passed')
PY
```

### `marble_tide`

```text
Implement a two-page deduplicating ledger collector.

Start at the supplied URL, discover the rel=next link, request exactly two pages, and preserve first-seen order. The pages contain 7 and 6 rows with two repeated marks, so emit 11 unique records with mark, caption, numeric amount, and absolute source_url.

Inspect the supplied source before editing. Implement the collector; do not merely describe it. Both verification commands are required, and the second is one atomic multiline heredoc.

Verification commands:
1. python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/__reset', timeout=5).read()" && python tide_collector.py http://127.0.0.1:8765/ledger/start tide_ledger.json
2. python - <<'PY'
import json
import urllib.request
from pathlib import Path
records = json.loads(Path('tide_ledger.json').read_text(encoding='utf-8'))
assert isinstance(records, list) and len(records) == 11, len(records)
assert all(set(record) == set(('mark', 'caption', 'amount', 'source_url')) for record in records)

with urllib.request.urlopen('http://127.0.0.1:8765/__metrics', timeout=5) as response:
    metrics = json.load(response)
assert metrics['count'] == 2, metrics
assert metrics['requests'] == ['/ledger/start', '/ledger/next'], metrics
print('benchmark validation passed')
PY
```

### `violet_prism`

```text
Build a namespace-tolerant RSS collector.

Parse at least the supplied seven entries and emit headline, absolute link, published, source, and summary. Handle XML entities, namespaces, relative and absolute links, multiline text, and a missing summary using the Python standard library.

Inspect the supplied source before editing. Implement the collector; do not merely describe it. Both verification commands are required, and the second is one atomic multiline heredoc.

Verification commands:
1. python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/__reset', timeout=5).read()" && python prism_feed.py http://127.0.0.1:8765/prism/feed prism_entries.json
2. python - <<'PY'
import json
import urllib.request
from pathlib import Path
records = json.loads(Path('prism_entries.json').read_text(encoding='utf-8'))
assert isinstance(records, list) and len(records) == 7, len(records)
assert all(set(record) == set(('headline', 'link', 'published', 'source', 'summary')) for record in records)

with urllib.request.urlopen('http://127.0.0.1:8765/__metrics', timeout=5) as response:
    metrics = json.load(response)
assert metrics['count'] == 1, metrics
assert metrics['requests'] == ['/prism/feed'], metrics
print('benchmark validation passed')
PY
```

### `copper_orbit`

```text
Complete a cursor-based JSON API collector.

Follow next_cursor until null, make exactly two API requests, deduplicate by identity while preserving first-seen order, and emit 10 records with identity, title, integer measure, and absolute origin. Inputs mix numeric strings and integers.

Inspect the supplied source before editing. Implement the collector; do not merely describe it. Both verification commands are required, and the second is one atomic multiline heredoc.

Verification commands:
1. python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/__reset', timeout=5).read()" && python orbit_cursor.py http://127.0.0.1:8765/orbit/measurements?cursor= orbit_measurements.json
2. python - <<'PY'
import json
import urllib.request
from pathlib import Path
records = json.loads(Path('orbit_measurements.json').read_text(encoding='utf-8'))
assert isinstance(records, list) and len(records) == 10, len(records)
assert all(set(record) == set(('identity', 'title', 'measure', 'origin')) for record in records)

with urllib.request.urlopen('http://127.0.0.1:8765/__metrics', timeout=5) as response:
    metrics = json.load(response)
assert metrics['count'] == 2, metrics
assert metrics['requests'] == ['/orbit/measurements?cursor=', '/orbit/measurements?cursor=phase-amber-2'], metrics
print('benchmark validation passed')
PY
```

### `cedar_signal`

```text
Build a collector for the documented CNEOS news feed.

Fetch the supplied public HTTPS RSS feed without third-party packages. Emit at least five entries with headline, absolute link, published, source, and summary. Live values may change, so do not hardcode titles or dates.

Inspect the supplied source before editing. Implement the collector; do not merely describe it. Both verification commands are required, and the second is one atomic multiline heredoc.

Verification commands:
1. python cedar_digest.py https://cneos.jpl.nasa.gov/feed/news.xml cedar_brief.json
2. python - <<'PY'
import json
import urllib.request
from pathlib import Path
records = json.loads(Path('cedar_brief.json').read_text(encoding='utf-8'))
assert isinstance(records, list) and len(records) >= 5, len(records)
assert all(set(record) == set(('headline', 'link', 'published', 'source', 'summary')) for record in records)
print('benchmark validation passed')
PY
```
