# Runtime V2 release evaluation

Status: **FAILED — release gates not met**

Commit under development: `release/runtime-v2` (record final SHA in the PR).

## Configuration exercised

- Provider/model: `deepseek` / `deepseek-chat`
- DoBox: real backend at `http://localhost:3000`, Docker project sandbox
- Canary limits: 24 iterations, 48 tool calls, 300 seconds, no-internet project

## Evidence

- Focused architecture suites passed: profiles 4/4, source pipeline 28/28, artifact semantics 3/3, controllers 6/6, repository understanding 2/2, architecture/quality 102/102.
- Real DoBox scripted README smoke and real sandbox source inspection: 2/2 passed.
- Real LLM + real DoBox generic crawler canary: 1/1 strict test passed in 76.375 seconds.
- Host deterministic discovery: 528 tests, 504 passed, 15 skipped, 9 failed before follow-up fixes. The semantic `silver_source` regression and three obsolete fixed-schema assertions were subsequently fixed and focused tests passed.
- Docker deterministic attempt was invalid as a release result: the first image could not access a private dependency; the repository image later timed out while installing Git/Node/Go before tests started.

## Missing release evidence

- Frozen crawler matrix: not created/run (0/8).
- Frozen large-repository matrix: not created/run (0/3).
- Hidden checker hashes, prompt hash, baseline tag, aggregate medians: unavailable.
- Full deterministic 100% pass on the final commit: unavailable.

The real canary is diagnostic evidence only and must not be reported as the Runtime V2 release suite.
