# DoCode

DoCode is a headless autonomous coding runtime. It accepts a natural language software task, creates one DoBox project sandbox for the job, drives the sandbox through project-level tools, verifies the result, and exports artifacts such as a patch and final report.

The service is intentionally separate from DoBox:

- DoBox owns sandbox/container orchestration.
- DoCode owns the coding job, agent loop, verification, and artifacts.
- Weav AI Runtime/Providers assemble model clients.
- APICred/BasaltPass provide authentication, credential resolution, and usage accounting.

## Runtime Integrations

- **DoBox**: DoCode only calls project-level sandbox APIs, creates one DoBox agent session per job, and never accepts a Docker container id.
- **APICred**: DoCode supports APICred runtime-credit mode and APICred OpenAI-compatible proxy mode. APICred identity is taken from the inbound BasaltPass cross-app bearer token and stored with the job for worker use, so no global APICred API token is required. In runtime mode DoCode calls APICred authorization before a provider-backed job, resolves provider credentials only in memory, builds a transient weav provider/router runtime, and reports measured usage after the loop. In proxy mode it sends model calls directly to APICred's `/v1/chat/completions` endpoint using the job's BasaltPass cross-app token as the OpenAI-compatible `api_key`, so usage is billed by APICred during the chat completion request and DoCode does not require `/runtime/authorize`. The local `scripted` runtime remains available for smoke tests without provider credentials.
- **BasaltPass**: DoCode reads upstream-authenticated user identity from forwarded headers such as `X-Basalt-User-ID`.
- **GitHub**: DoCode currently exports local patch/report/archive artifacts and includes a GitHub exporter adapter surface for branch/PR creation once a GitHub app or connector is configured.

## MVP Flow

```text
POST /v1/jobs
  -> store CodingJob
  -> enqueue job
  -> worker creates DoBox project sandbox
  -> worker assembles APICred-backed weav router + DoBox tool registry
  -> agent loop observes, plans, acts, verifies
  -> exporter stores terminal result/report artifacts
```

## Local Development

```bash
python -m pip install -e ".[dev]"
uvicorn docode.main:app --reload --port 8110
```

Recommended deterministic test command:

```bash
PATH=/Users/henryzhao/Desktop/workplace/.venv/bin:$PATH PYTHONPATH=src ../.venv/bin/python -m unittest discover -s tests
```

Use `python` for host-side local tests so the command works on platforms where `python3` is not installed. Agent tasks still preserve and enforce user-declared verification commands exactly.

Create a deterministic smoke-test job against the configured DoBox API without calling a real LLM:

```bash
docode scripted-job "create a result file"
```

Generate runtime evidence reports:

```bash
docode smoke-check --report .docode/smoke-check.json
docode smoke-run --report .docode/smoke-run.json
docode smoke-run --start-dobox --report .docode/smoke-run.json
docode eval scaffold .docode/eval-suite --force
docode eval jobs .docode/eval-suite/manifest.json --results-dir .docode/eval-results --quality balanced
docode eval run .docode/eval-results --report .docode/eval-report.json
docode eval assert .docode/eval-report.json --min-success-rate 0.8 --max-avg-tool-calls 30 --max-cost 1.00
```

Generic smoke ladder:

1. README edit smoke: run `python -m unittest tests.test_smoke_readme_job`; the agent must edit `README.md`, produce a non-empty diff, record loop/tool/verifier steps, and export terminal artifacts.
2. Calculator bugfix smoke: a repo with `calculator.py` and `tests/test_calculator.py`; the instruction explicitly includes the verification command, and the agent must edit source, run that exact command, and submit `final_candidate`.
3. Product parser fixture smoke: run `python -m unittest tests.test_smoke_product_parser_job`; a repo with `parser.py`, `fixtures/products.html`, and `tests/test_parser.py` validates reading the fixture/test/source, implementing parser logic, and passing the explicit verification command.
4. External GitHub Trends eval: run only after the first three generic cases pass. GitHub Trends is not a production shortcut or embedded runtime capability.

`smoke-check` verifies configured DoBox health, local DoBox backend path, Docker CLI/daemon access, APICred model access, local `gh` availability, database path, and artifact directory. `smoke-run` first runs those checks and then executes a `provider=scripted` end-to-end job when DoBox is reachable. Pass `--start-dobox` to temporarily run `go run ./cmd/server` from the configured local DoBox backend directory for the duration of the smoke check or smoke job.
`eval scaffold` creates ten small git repositories covering Python bugfix, Python CLI, crawler, API adapter, README-only, JS bugfix, no-test project, bad web source repair, large command output, and GitHub PR artifact export scenarios. `eval jobs` runs manifest cases through DoCode jobs and writes per-case result JSON files. `eval run` aggregates those saved eval results into a report with success rate, iterations, tool calls, token/cost totals, failure reasons, and verification-plan failures. `eval run` and `eval assert` accept `--min-success-rate`, `--max-avg-tool-calls`, and `--max-cost`; when a threshold fails, the report includes `regression`, `thresholds`, and `threshold_failures`, and the command exits non-zero.

### Optional Smoke Tests

Optional smoke tests are skipped unless their environment flags are set. The commands below use the macOS workspace venv pattern.

Default deterministic tests:

```bash
PATH=/Users/henryzhao/Desktop/workplace/.venv/bin:$PATH PYTHONPATH=src ../.venv/bin/python -m unittest discover -s tests
```

Real LLM with fake fixture tools:

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

Generic local crawler CLI smoke with real LLM and real DoBox:

```bash
PATH=/Users/henryzhao/Desktop/workplace/.venv/bin:$PATH DOCODE_REAL_LLM_SMOKE=1 DOCODE_REAL_DOBOX_SMOKE=1 PYTHONPATH=src ../.venv/bin/python -m unittest -v tests.test_real_dobox_crawler_smoke
```

Neutral external-source crawler CLI smoke with real LLM and real DoBox:

```bash
DOCODE_REAL_LLM_SMOKE=1 DOCODE_REAL_DOBOX_SMOKE=1 PATH=/Users/henryzhao/Desktop/workplace/.venv/bin:$PATH PYTHONPATH=src ../.venv/bin/python -m unittest -v tests.test_real_dobox_external_crawler_smoke
```

The real LLM smoke exchanges the BasaltPass subject token for an APICred token and selects a `deepseek` model from the APICred model catalog by default when that configuration is present. Set `DOCODE_REAL_LLM_PROVIDER` or `DOCODE_REAL_LLM_MODEL` to override the selected catalog entry. The real DoBox smoke uses local fixtures; the combined real LLM + real DoBox tests use the same credential settings as `tests.test_real_llm_smoke`. The external-source crawler smoke uses a test-harness mock HTTP source and may need `DOCODE_EXTERNAL_CRAWLER_SOURCE_HOST` when the default host route is not reachable from the DoBox sandbox.

Integration matrix:

| Tools | LLM | Current status |
| --- | --- | --- |
| Fake tools | Scripted LLM | Passing: README, calculator, parser, parser repair |
| Fake tools | Real LLM | Passing: README, calculator |
| Real DoBox | Scripted LLM | Passing: README |
| Real DoBox | Real LLM | Passing: README, calculator, generic local crawler CLI |

See [docs/agent_mvp_status.md](docs/agent_mvp_status.md) for the current MVP status, non-goals, and known limitations.

Workers claim queued jobs by atomically moving them to `preparing` before APICred authorization or DoBox project creation, so duplicate queue deliveries do not start duplicate sandboxes for the same job. On API startup, jobs interrupted in `preparing`, `running`, or `verifying` are requeued with an audit step before the worker begins claiming jobs.

Or through the API:

```json
{
  "instruction": "create a result file",
  "quality": "balanced",
  "max_iterations": 5,
  "max_tool_calls": 10,
  "max_llm_cost": 1.25,
  "sandbox_network_mode": "project",
  "artifact_mode": "pr",
  "github_repo": "zeturn/example",
  "base_branch": "main"
}
```

`provider` and `model` are optional. When omitted, DoCode resolves a concrete provider/model through the runtime model catalog using `quality`, which can be `fast`, `balanced`, or `strong`. Explicit provider/model pairs are still accepted and validated against the same catalog. The local `provider=scripted`, `model=scripted` runtime remains available for smoke tests.

Clients can discover the currently allowed provider/model choices before creating a job:

```text
GET /v1/runtime/providers
```

The response includes catalog options plus the concrete `fast`, `balanced`, and `strong` defaults that job creation will use for omitted provider/model fields.

Useful environment variables:

- `DOCODE_DOBOX_BASE_URL`: DoBox API base URL, defaults to `http://localhost:3000`
- `DOCODE_DOBOX_TOKEN`: bearer token for DoBox API
- `DOCODE_DOBOX_BACKEND_DIR`: local DoBox backend path used by `--start-dobox`, auto-detected from common workspace layouts when unset.
- `DOCODE_DOBOX_START_TIMEOUT_SECONDS`: seconds to wait for a local DoBox backend started by smoke tooling, defaults to `20`.
- `DOCODE_APICRED_BASE_URL`: APICred runtime/API base URL, for example `http://localhost:8103/v1`
- `DOCODE_APICRED_TOKEN`: optional legacy service token for APICred runtime/auth endpoints. Normal BasaltPass cross-app deployments do not need this.
- `DOCODE_APICRED_MODE`: `auto` (default), `proxy`, or `runtime`. Use `proxy` for APICred's OpenAI-compatible relay service.
- `DOCODE_AUTH_REQUIRED`: set to `true` to require APICred/BasaltPass session verification instead of local/forwarded-user fallback.
- `DOCODE_DATABASE_PATH`: SQLite job database path, defaults to `.docode/docode.db`
- `DOCODE_ARTIFACT_DIR`: local artifact directory, defaults to `.docode_artifacts`
- `DOCODE_MAX_TOOL_CALLS`: default per-job tool-call budget, defaults to `100`
- `DOCODE_MAX_LLM_TOKENS`: default estimated LLM token budget per job, defaults to `100000`.
- `DOCODE_DEFAULT_MODEL`: balanced fallback model when the runtime catalog cannot provide a more specific model, defaults to `gpt-5.4`.
- `DOCODE_MAX_LLM_COST`: optional default LLM cost budget per job. DoCode enforces the stricter of this value, any job-level `max_llm_cost`, and APICred's authorization cost budget when runtime mode returns one.
- `DOCODE_SANDBOX_RETENTION`: DoBox sandbox retention policy, one of `keep`, `delete_on_success`, or `delete_always`; defaults to `keep`.
- `DOCODE_SANDBOX_NETWORK_MODE`: DoBox project network policy, either `project` or `no_internet`; defaults to `project`. Job creation can override this with `sandbox_network_mode`.
- `DOCODE_GITHUB_EXPORT_ENABLED`: set to `true` to let `artifact_mode=pr` use the local `gh` CLI.
- `DOCODE_GITHUB_WORK_DIR`: temporary clone directory for GitHub PR export, defaults to `.docode/github`.
- `DOCODE_WEB_TOOLS_ENABLED`: set to `false` to hide hosted web tools from the agent; defaults to `true`.
- `DOCODE_OPENAI_API_KEY`: OpenAI API key used by the MVP `web_search` agent tool. Falls back to `OPENAI_API_KEY` when unset.
- `DOCODE_OPENAI_BASE_URL`: OpenAI-compatible base URL for `web_search`, defaults to `https://api.openai.com/v1`.
- `DOCODE_OPENAI_SEARCH_MODEL`: model used with OpenAI hosted web search, defaults to `gpt-4o-mini`.
- `DOCODE_OPENAI_SEARCH_TOOL_TYPE`: hosted search tool type sent to the Responses API, defaults to `web_search`.
- `DOCODE_WEB_SEARCH_CONTEXT_SIZE`: OpenAI web search context size, defaults to `low`.
- `DOCODE_WEB_FETCH_TIMEOUT_SECONDS`: timeout for the `fetch_url` agent tool, defaults to `20`.
- `DOCODE_WEB_FETCH_ALLOW_PRIVATE_HOSTS`: set to `true` only in trusted development environments to allow `fetch_url` to access private or local hosts; defaults to blocked.

Job-level `artifact_mode` values:

- `patch`: export patch/report/log/result/archive/zip artifacts.
- `zip`: same local artifact export path, with `workspace.zip` emphasized for clients.
- `commit`: export local artifacts and ask DoBox to create a git commit after verification.
- `pr`: export local artifacts, create a sandbox commit, and, when GitHub export is enabled, clone the repo with `gh`, apply `patch.diff`, push a branch, and open a PR whose body uses the same final report evidence: changed files, summary, verification reason, and checks.

Every terminal job writes `result.json`, a machine-readable final result with status, summary or terminal reason, changed files when available, verification checks for successful final candidates, and artifact filenames. Successful jobs require a non-empty agent final summary before verification can complete, then write `final_report.md` plus patch/test artifacts; failed jobs write `failure_report.md` and `failure_log.txt`; stopped jobs write `stopped_report.md` and `stopped_log.txt`. Terminal bundles always include `result.json` and `workspace.zip`, with `patch.diff` included when a diff exists and `workspace.tar` included when a sandbox archive was exported. `GET /v1/jobs/{job_id}` includes this result payload after it has been exported.

For non-scripted provider runs, DoCode also creates an independent verifier judge from the assembled weav provider client. Command checks remain mandatory: `git status`, a complete non-empty `git diff`, and detected test/build/lint commands must pass before completion. The verifier builds a task-aware `VerificationPlan`: bugfixes prefer related test changes, CLI/script/crawler work must run an entrypoint smoke check, crawler/API tasks require external-source evidence, and docs-only tasks avoid meaningless test gates. The verifier judge receives the status, diff, and command outputs, then can add structured `required_fixes` or veto completion when the change does not satisfy the original instruction.

DoCode records LLM prompt/completion usage for provider-backed agent and verifier calls, prefers provider-reported usage metadata when available, and falls back to conservative text-size estimates otherwise. Runtime assembly now passes a Weav `RuntimePolicy` with purpose, token/cost budgets, allowed providers, and fallback model into `AIRuntime`, while provider call adaptation accepts the Weav `LLMCallResult` shape when providers expose it. Runtime mode enforces the stricter of the job token budget and APICred authorization budget and reports measured usage after the run. Proxy mode relies on APICred's OpenAI-compatible request path for billing and skips the separate runtime usage report.

Malformed or temporarily failing model decisions are audited as `llm_error` steps and fed back into the loop. Repeated unusable model output is stopped by the same consecutive-failure policy as failing tools and verifier repairs. Tool results shown to the agent and verifier are prompt-safe summaries capped to the first 300 lines, with original size metadata recorded when output is clipped. Job event, step-listing APIs, and failed/stopped step-log artifacts omit full tool output, verifier status/diff text, and verifier command output, exposing size metadata instead. Failed/stopped jobs omit `patch.diff` when DoBox reports a truncated diff so terminal artifacts do not publish partial patches.

The sandbox tool registry exposes only project-level DoBox operations: command execution, file read/write/list/search, exact `edit_file`, unified-diff `apply_patch`, git status/diff/commit, detected test/build/lint commands, preview URL creation, and recent sandbox logs. Agents never receive a Docker container id. `edit_file` requires an exact `old_text` match and returns a diff preview; ambiguous or missing matches return repairable context instead of overwriting the file. DoCode can also expose host-side web tools for data-source discovery: `web_search` uses OpenAI hosted web search as an MVP search backend, and `fetch_url` reads public HTTP/HTTPS pages while rejecting private/local hosts by default. `fetch_url` accepts an optional `goal` and `max_sections`, then returns structured JSON with title, summary, relevant sections, original byte size, returned byte size, and truncation metadata instead of dumping an entire page into context. Each DoCode job creates a DoBox agent session, persists the session id, and attaches sandbox tool calls to that session for project-scoped auditability. DoCode rejects model-supplied file paths and command working directories that resolve outside `/workspace` before calling DoBox. DoBox project sandboxes use the standard `dobox/code-sandbox:latest` image, run as non-root UID/GID `1000:1000`, and apply backend-capped CPU, memory, and PID limits. Each project gets its own Docker bridge network by default; DoBox also accepts a `no_internet` project network policy that creates an internal Docker network, while raw Docker modes such as `host` or `container:...` are rejected. DoBox enforces command timeouts plus output caps for command and file-read project tools, preserves sanitized project tool-call audit rows, and DoCode preserves returned `truncated` flags in tool results.
