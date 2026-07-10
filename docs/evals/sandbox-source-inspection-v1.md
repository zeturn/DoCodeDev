# Sandbox source inspection v1

## Scope

This change adds `inspect_source`, a DoBox-native HTTP/HTTPS inspection tool for crawler jobs. It does not alter the host-side `fetch_url` behavior or its localhost/private-address rejection.

Implementation commits:

- `e1323c2` - `feat(dobox): add sandbox source inspection tool`
- `790165b` - `fix(agent): inspect crawler sources before editing`

The runtime now selects literal source candidates, excluding control endpoints such as `/__reset` and `/__metrics`, and auto-runs the primary source once before the first model edit. A successful result requires an HTTP response from `execution_scope=sandbox`; `web_search` and host `fetch_url` cannot satisfy this stage. The source body is excerpted in one observation, then represented by compact evidence.

## Verification

Deterministic top-level suite plus the frozen crawler deterministic suite:

```text
466 passed, 13 skipped, 35 subtests passed
```

Focused source-inspection, agent, verifier, runner, and tool tests:

```text
221 passed, 4 subtests passed
```

A real DoBox smoke started a fixture server in a live project sandbox and called `inspect_source` against `http://127.0.0.1:18765/redirect`. The result followed the redirect to `/feed?cursor=next`, returned HTTP 200 JSON, and recorded `execution_scope=sandbox`.

Direct `pytest -q` is not the repository's valid full-suite command because it recursively collects fixture repositories under `tests/fixtures/**`; those fixtures intentionally have isolated import roots. The recorded suite selected the top-level tests and added `tests/crawler_benchmark_v1/test_deterministic.py`.

## Limited real-LLM evaluation

Exactly four existing benchmark case definitions were selected for formal results: one local HTML source, one local RSS/XML source, one local cursor JSON source, and one public RSS source. Each formal case ran once with real `deepseek-chat` and real DoBox. The old 18-run benchmark was not launched.

One earlier HTML collection completed its model job but failed in the new wrapper while writing to an uncreated `traces/` directory. It is recorded as an excluded invalid harness attempt; the directory bug was fixed and the case was recollected. It is not included in any result denominator.

| case | source | source before edit | runtime | functional | required commands | strict |
| --- | --- | --- | --- | --- | --- | --- |
| `opal_canopy` | local HTML | yes | failed | no | 1/2 | no |
| `violet_prism` | local RSS/XML | yes | succeeded | no | 2/2 | no |
| `copper_orbit` | local cursor JSON | yes | failed | no | 1/2 | no |
| `cedar_signal` | public RSS | yes | failed | yes | 1/2 | no |

Aggregate results:

- Source inspection before first edit: 4/4 (100%).
- First source responses: 4/4 HTTP 200 with `execution_scope=sandbox`.
- Strict success: 0/4.
- Independently functional: 1/4.
- Runtime succeeded: 1/4.
- All exact required commands passed: 1/4.

Structured results are in `.docode/evals/sandbox-source-inspection-v1/results.json`. Full sanitized traces are retained locally in `.docode/evals/sandbox-source-inspection-v1/traces/`; they contain every persisted observation, LLM decision, tool call/result, verifier step, and artifact step.

`deepseek-chat` returned no separate `reasoning` or `reasoning_records` fields in these runs: 0 explicit reasoning records across 97 decisions. The traces therefore preserve the complete reasoning surface exposed by the provider, but do not claim access to hidden chain-of-thought.

## Findings

The targeted architecture problem is fixed: local/private source inspection now occurs from the same network namespace as the generated crawler, and all four jobs inspected real source bytes before editing. The prompt included each raw source excerpt once rather than duplicating the full body in working memory.

The initial formal traces also showed models manually re-calling a successfully inspected source several times. After collection, the runtime was tightened to remove `inspect_source` from that job's available tools once successful evidence exists; deterministic coverage verifies edits and verification commands remain unlocked.

The limited evaluation does not establish general crawler success. The remaining failures match out-of-scope baseline issues:

- The HTML hidden variant produced no records.
- The RSS/XML result normalized timestamps and missing summaries incorrectly and used the channel title instead of the item source.
- The cursor JSON result did not resolve relative origins and did not generalize pagination to the hidden variant.
- The public RSS implementation was independently functional, but the runtime exhausted iterations after dependent verification/repair behavior.

No benchmark selector, fixture value, hidden record, or case-specific parser behavior was added to production code.
