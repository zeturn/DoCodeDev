import { type Dispatch, type FormEvent, type ReactNode, type SetStateAction, useCallback, useEffect, useMemo, useState } from 'react';
import {
  Job,
  JobStatus,
  StepEventPayload,
  apiBase,
  cancelJob,
  createJob,
  getJob,
  getSteps,
  listJobs,
  streamJobEvents
} from './api';

const STORAGE_TOKEN_KEY = 'docode.authToken';
const STORAGE_THEME_KEY = 'docode-theme';

function getInitialTheme(): 'light' | 'dark' {
  if (typeof window === 'undefined') return 'dark';
  const stored = localStorage.getItem(STORAGE_THEME_KEY);
  if (stored === 'light' || stored === 'dark') return stored;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function ThemeToggle({ theme, onToggle }: { theme: 'light' | 'dark'; onToggle: () => void }) {
  return (
    <button type="button" className="theme-toggle" onClick={onToggle} aria-label="Toggle theme" title={theme === 'dark' ? 'Switch to light' : 'Switch to dark'}>
      {theme === 'dark' ? (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
        </svg>
      ) : (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      )}
    </button>
  );
}

const TERMINAL_STATUSES: JobStatus[] = ['succeeded', 'failed', 'stopped'];
const INLINE_VALUE_MAX_LENGTH = 180;
const OUTPUT_PREVIEW_MAX_LENGTH = 2400;

interface JobFormState {
  instruction: string;
  github_repo: string;
  repo_url: string;
  base_branch: string;
  branch: string;
  quality: 'fast' | 'balanced' | 'strong';
  artifact_mode: 'patch' | 'zip' | 'commit' | 'pr';
  sandbox_network_mode: string;
  max_iterations: string;
  max_tool_calls: string;
  max_runtime_seconds: string;
  max_llm_cost: string;
}

const initialJobForm: JobFormState = {
  instruction: '',
  github_repo: '',
  repo_url: '',
  base_branch: 'main',
  branch: '',
  quality: 'balanced',
  artifact_mode: 'patch',
  sandbox_network_mode: 'project',
  max_iterations: '',
  max_tool_calls: '',
  max_runtime_seconds: '',
  max_llm_cost: ''
};

function App() {
  const [authToken, setAuthToken] = useState(() => localStorage.getItem(STORAGE_TOKEN_KEY) ?? '');
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [steps, setSteps] = useState<StepEventPayload[]>([]);
  const [form, setForm] = useState<JobFormState>(initialJobForm);
  const [statusFilter, setStatusFilter] = useState<JobStatus | ''>('');
  const [loadingJobs, setLoadingJobs] = useState(false);
  const [creatingJob, setCreatingJob] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);

  const [theme, setTheme] = useState<'light' | 'dark'>(getInitialTheme);
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    localStorage.setItem(STORAGE_THEME_KEY, theme);
  }, [theme]);
  const toggleTheme = () => setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));

  const selectedJobFromList = useMemo(() => jobs.find((job) => job.id === selectedJobId) ?? null, [jobs, selectedJobId]);

  const refreshJobs = useCallback(async () => {
    setLoadingJobs(true);
    setError(null);
    try {
      const loaded = await listJobs(authToken, statusFilter || undefined);
      setJobs(loaded);
      setSelectedJobId((current) => {
        if (!current && loaded.length > 0) {
          return loaded[0].id;
        }
        if (current && !loaded.some((job) => job.id === current)) {
          return loaded[0]?.id ?? null;
        }
        return current;
      });
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoadingJobs(false);
    }
  }, [authToken, statusFilter]);

  useEffect(() => {
    localStorage.setItem(STORAGE_TOKEN_KEY, authToken);
  }, [authToken]);

  useEffect(() => {
    void refreshJobs();
  }, [refreshJobs]);

  useEffect(() => {
    const selectedJob = selectedJobId;

    if (!selectedJob) {
      setSelectedJob(null);
      setSteps([]);
      return;
    }

    const jobId: string = selectedJob;

    const controller = new AbortController();
    setSelectedJob(null);
    setSteps([]);
    setStreamError(null);
    setStreaming(true);

    async function loadAndStream() {
      try {
        const [job, existingSteps] = await Promise.all([getJob(jobId, authToken), getSteps(jobId, authToken)]);
        if (!controller.signal.aborted) {
          setSelectedJob(job);
          setSteps(dedupeSteps(existingSteps));
          upsertJob(setJobs, job);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setStreamError(errorMessage(err));
          setStreaming(false);
        }
        return;
      }

      try {
        await streamJobEvents(
          jobId,
          authToken,
          ({ event, data }) => {
            if (event === 'status' && isRecord(data)) {
              const nextStatus = data.status;
              if (typeof nextStatus === 'string') {
                patchJobStatus(jobId, nextStatus as JobStatus);
              }
            }
            if (event === 'step' && isRecord(data)) {
              setSteps((current) => dedupeSteps([...current, data as StepEventPayload]));
            }
            if (event === 'done' && isRecord(data)) {
              const nextStatus = typeof data.status === 'string' ? (data.status as JobStatus) : undefined;
              if (nextStatus) {
                patchJobStatus(jobId, nextStatus);
              }
              setStreaming(false);
              void getJob(jobId, authToken).then((job) => {
                if (!controller.signal.aborted) {
                  setSelectedJob(job);
                  upsertJob(setJobs, job);
                }
              }).catch(() => undefined);
            }
          },
          controller.signal
        );
      } catch (err) {
        if (!controller.signal.aborted) {
          setStreamError(errorMessage(err));
        }
      } finally {
        if (!controller.signal.aborted) {
          setStreaming(false);
        }
      }
    }

    void loadAndStream();
    return () => controller.abort();
  }, [authToken, selectedJobId]);

  const activeJob = selectedJob ?? selectedJobFromList;

  async function onCreateJob(event: FormEvent) {
    event.preventDefault();
    if (!form.instruction.trim()) {
      setError('Instruction is required.');
      return;
    }
    setCreatingJob(true);
    setError(null);
    try {
      const created = await createJob(
        {
          instruction: form.instruction.trim(),
          github_repo: form.github_repo.trim(),
          repo_url: form.repo_url.trim(),
          base_branch: form.base_branch.trim(),
          branch: form.branch.trim(),
          quality: form.quality,
          artifact_mode: form.artifact_mode,
          sandbox_network_mode: form.sandbox_network_mode.trim(),
          max_iterations: numberOrUndefined(form.max_iterations),
          max_tool_calls: numberOrUndefined(form.max_tool_calls),
          max_runtime_seconds: numberOrUndefined(form.max_runtime_seconds),
          max_llm_cost: numberOrUndefined(form.max_llm_cost)
        },
        authToken
      );
      setForm({ ...initialJobForm, github_repo: form.github_repo, repo_url: form.repo_url, base_branch: form.base_branch || 'main' });
      setStatusFilter('');
      setSelectedJobId(created.job_id);
      setJobs(await listJobs(authToken));
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setCreatingJob(false);
    }
  }

  async function onCancelJob() {
    if (!activeJob || isTerminalStatus(activeJob.status)) {
      return;
    }
    setError(null);
    try {
      await cancelJob(activeJob.id, authToken);
      await refreshJobs();
    } catch (err) {
      setError(errorMessage(err));
    }
  }

  function patchJobStatus(jobId: string, status: JobStatus) {
    setJobs((current) => current.map((job) => (job.id === jobId ? { ...job, status, updated_at: new Date().toISOString() } : job)));
    setSelectedJob((current) => (current?.id === jobId ? { ...current, status, updated_at: new Date().toISOString() } : current));
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">DC</span>
          <div>
            <h1>DoCode</h1>
            <p>Jobs dashboard</p>
          </div>
        </div>

        <label className="field compact">
          <span>Bearer token</span>
          <input
            type="password"
            value={authToken}
            placeholder="Optional in local mode"
            onChange={(event) => setAuthToken(event.target.value)}
          />
        </label>

        <div className="toolbar">
          <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value as JobStatus | '')}>
            <option value="">All statuses</option>
            <option value="queued">Queued</option>
            <option value="preparing">Preparing</option>
            <option value="running">Running</option>
            <option value="verifying">Verifying</option>
            <option value="succeeded">Succeeded</option>
            <option value="failed">Failed</option>
            <option value="stopped">Stopped</option>
          </select>
          <button type="button" className="secondary" onClick={() => void refreshJobs()} disabled={loadingJobs}>
            {loadingJobs ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>

        <div className="job-list" aria-label="Jobs">
          {jobs.length === 0 && <p className="empty">No jobs yet.</p>}
          {jobs.map((job) => (
            <button
              key={job.id}
              type="button"
              className={`job-card ${job.id === selectedJobId ? 'selected' : ''}`}
              onClick={() => setSelectedJobId(job.id)}
            >
              <span className={`status-dot ${job.status}`} />
              <span className="job-card-main">
                <strong>{shortInstruction(job.instruction)}</strong>
                <small>{job.github_repo || job.repo_url || job.id}</small>
              </span>
              <span className={`status-pill ${job.status}`}>{job.status}</span>
            </button>
          ))}
        </div>
      </aside>

      <section className="content">
        <header className="topbar">
          <div>
            <p className="eyebrow">API</p>
            <h2>{apiBase() || 'same origin'}</h2>
          </div>
          <div className="topbar-actions">
            <ThemeToggle theme={theme} onToggle={toggleTheme} />
            {activeJob && (
              <button type="button" className="danger" onClick={() => void onCancelJob()} disabled={isTerminalStatus(activeJob.status)}>
                Cancel job
              </button>
            )}
          </div>
        </header>

        {error && <div className="alert error">{error}</div>}

        <section className="panel create-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Create</p>
              <h2>New coding job</h2>
            </div>
          </div>
          <form onSubmit={(event) => void onCreateJob(event)} className="job-form">
            <label className="field full">
              <span>Instruction</span>
              <textarea
                rows={5}
                value={form.instruction}
                placeholder="Example: Fix the calculator bug and run python -m unittest discover -s tests"
                onChange={(event) => setForm((current) => ({ ...current, instruction: event.target.value }))}
              />
            </label>
            <label className="field">
              <span>GitHub repo</span>
              <input
                value={form.github_repo}
                placeholder="owner/repo"
                onChange={(event) => setForm((current) => ({ ...current, github_repo: event.target.value }))}
              />
            </label>
            <label className="field">
              <span>Repo URL</span>
              <input
                value={form.repo_url}
                placeholder="https://github.com/owner/repo.git"
                onChange={(event) => setForm((current) => ({ ...current, repo_url: event.target.value }))}
              />
            </label>
            <label className="field">
              <span>Base branch</span>
              <input value={form.base_branch} onChange={(event) => setForm((current) => ({ ...current, base_branch: event.target.value }))} />
            </label>
            <label className="field">
              <span>Work branch</span>
              <input
                value={form.branch}
                placeholder="optional"
                onChange={(event) => setForm((current) => ({ ...current, branch: event.target.value }))}
              />
            </label>
            <label className="field">
              <span>Quality</span>
              <select value={form.quality} onChange={(event) => setForm((current) => ({ ...current, quality: event.target.value as JobFormState['quality'] }))}>
                <option value="fast">Fast</option>
                <option value="balanced">Balanced</option>
                <option value="strong">Strong</option>
              </select>
            </label>
            <label className="field">
              <span>Artifact mode</span>
              <select
                value={form.artifact_mode}
                onChange={(event) => setForm((current) => ({ ...current, artifact_mode: event.target.value as JobFormState['artifact_mode'] }))}
              >
                <option value="patch">Patch</option>
                <option value="zip">Zip</option>
                <option value="commit">Commit</option>
                <option value="pr">PR</option>
              </select>
            </label>
            <label className="field">
              <span>Network mode</span>
              <input value={form.sandbox_network_mode} onChange={(event) => setForm((current) => ({ ...current, sandbox_network_mode: event.target.value }))} />
            </label>
            <label className="field">
              <span>Max iterations</span>
              <input type="number" min="1" value={form.max_iterations} onChange={(event) => setForm((current) => ({ ...current, max_iterations: event.target.value }))} />
            </label>
            <label className="field">
              <span>Max tool calls</span>
              <input type="number" min="1" value={form.max_tool_calls} onChange={(event) => setForm((current) => ({ ...current, max_tool_calls: event.target.value }))} />
            </label>
            <label className="field">
              <span>Max runtime seconds</span>
              <input
                type="number"
                min="30"
                value={form.max_runtime_seconds}
                onChange={(event) => setForm((current) => ({ ...current, max_runtime_seconds: event.target.value }))}
              />
            </label>
            <label className="field">
              <span>Max LLM cost</span>
              <input type="number" min="0" step="0.01" value={form.max_llm_cost} onChange={(event) => setForm((current) => ({ ...current, max_llm_cost: event.target.value }))} />
            </label>
            <div className="form-actions full">
              <button type="submit" disabled={creatingJob}>{creatingJob ? 'Creating...' : 'Create job'}</button>
            </div>
          </form>
        </section>

        <section className="panel output-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Realtime</p>
              <h2>{activeJob ? shortJobId(activeJob.id) : 'Select a job'}</h2>
            </div>
            {activeJob && <span className={`status-pill large ${activeJob.status}`}>{activeJob.status}</span>}
          </div>

          {activeJob ? (
            <>
              <dl className="job-meta">
                <div>
                  <dt>Created</dt>
                  <dd>{formatDate(activeJob.created_at)}</dd>
                </div>
                <div>
                  <dt>Updated</dt>
                  <dd>{formatDate(activeJob.updated_at)}</dd>
                </div>
                <div>
                  <dt>Provider</dt>
                  <dd>{activeJob.provider} / {activeJob.model}</dd>
                </div>
                <div>
                  <dt>Artifacts</dt>
                  <dd>{activeJob.artifact_id || 'pending'}</dd>
                </div>
              </dl>
              <p className="instruction-preview">{activeJob.instruction}</p>
              {streamError && <div className="alert warning">{streamError}</div>}
              <div className="stream-header">
                <span>{streaming && !isTerminalStatus(activeJob.status) ? 'Connected to job stream' : 'Stream output'}</span>
                <span>{steps.length} steps</span>
              </div>
              <div className="timeline">
                {steps.length === 0 && <p className="empty">No steps recorded yet.</p>}
                {steps.map((step) => (
                  <article className="step" key={step.step_id ?? `${step.job_id}-${step.step_index}`}>
                    <header>
                      <span className="step-index">#{step.step_index}</span>
                      <strong>{stepTitle(step)}</strong>
                      <time>{formatTime(step.created_at)}</time>
                    </header>
                    {renderStepBody(step)}
                  </article>
                ))}
              </div>
            </>
          ) : (
            <p className="empty large-empty">Create or select a job to watch its output.</p>
          )}
        </section>
      </section>
    </main>
  );
}

export default App;

function upsertJob(setJobs: Dispatch<SetStateAction<Job[]>>, job: Job) {
  setJobs((current) => {
    const exists = current.some((item) => item.id === job.id);
    const next = exists ? current.map((item) => (item.id === job.id ? job : item)) : [job, ...current];
    return [...next].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
  });
}

function dedupeSteps(items: StepEventPayload[]): StepEventPayload[] {
  const byKey = new Map<string, StepEventPayload>();
  for (const item of items) {
    const key = item.step_id || `${item.job_id}:${item.step_index}`;
    byKey.set(key, item);
  }
  return [...byKey.values()].sort((a, b) => a.step_index - b.step_index);
}

function renderStepBody(step: StepEventPayload) {
  const outputPreview = firstTextValue(step, ['output', 'stdout', 'stderr', 'logs', 'git_diff', 'git_status']);
  const metadata = isRecord(step.metadata) ? step.metadata : undefined;

  if (step.type === 'tool_call') {
    return (
      <StepCard raw={step}>
        <p className="step-summary">{toolCallSummary(step)}</p>
        {renderKeyValueCard('Call details', compactEntries([
          ['tool', step.tool],
          ['command', commandFromArgs(step.args)],
          ['path', valueFromRecord(step.args, ['path', 'file', 'filename'])],
          ['cwd', valueFromRecord(step.args, ['cwd', 'workdir', 'working_directory'])],
          ['timeout', valueFromRecord(step.args, ['timeout', 'timeout_seconds'])]
        ]))}
        {renderRecordCard('Arguments', step.args, ['command', 'cmd', 'args', 'path', 'file', 'filename', 'cwd', 'workdir', 'working_directory', 'timeout', 'timeout_seconds'])}
      </StepCard>
    );
  }

  if (step.type === 'tool_result') {
    const exitCode = typeof step.exit_code === 'number' ? step.exit_code : undefined;
    return (
      <StepCard raw={step}>
        <p className={`step-summary ${exitCode && exitCode !== 0 ? 'failed' : ''}`}>{toolResultSummary(step)}</p>
        {renderKeyValueCard('Result details', compactEntries([
          ['tool', step.tool],
          ['exit', step.exit_code ?? 'n/a'],
          ['truncated', step.truncated],
          ['status', valueFromRecord(metadata, ['status', 'state'])],
          ['artifact', valueFromRecord(metadata, ['artifact_id', 'artifact'])]
        ]))}
        {outputPreview && renderOutputPreview(outputPreview)}
        {renderRecordCard('Metadata', metadata)}
      </StepCard>
    );
  }

  return (
    <StepCard raw={step}>
      <p className="step-summary">{genericStepSummary(step)}</p>
      {renderThinkingCard(step)}
      {renderKeyValueCard('Highlights', compactEntries([
        ['type', step.type ?? step.kind],
        ['decision', step.decision_type],
        ['reason', step.reason],
        ['detail', step.detail],
        ['summary', step.summary]
      ]))}
      {outputPreview && renderOutputPreview(outputPreview)}
      {renderRecordCard('Metadata', metadata)}
    </StepCard>
  );
}

function renderThinkingCard(step: StepEventPayload) {
  const reasoningRecords = Array.isArray(step.reasoning_records)
    ? step.reasoning_records.filter(isRecord)
    : [];
  const reasoningText = typeof step.reasoning === 'string' ? step.reasoning.trim() : '';
  if (!reasoningText && reasoningRecords.length === 0) {
    return null;
  }

  return (
    <section className="info-card thinking-card">
      <h4>Thinking</h4>
      {reasoningText && <pre>{truncateMiddle(reasoningText, OUTPUT_PREVIEW_MAX_LENGTH)}</pre>}
      {reasoningRecords.length > 0 && (
        <div className="thinking-records">
          {reasoningRecords.slice(0, 6).map((record, index) => {
            const text = typeof record.text === 'string' ? record.text : stringifyJson(record);
            const label = [record.type, record.source].filter((value) => typeof value === 'string' && value.trim()).join(' / ');
            return (
              <article className="thinking-record" key={`${index}-${label || 'record'}`}>
                {label && <span>{label}</span>}
                <pre>{truncateMiddle(text, OUTPUT_PREVIEW_MAX_LENGTH)}</pre>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}

function StepCard({ children, raw }: { children: ReactNode; raw: StepEventPayload }) {
  return (
    <div className="step-body">
      {children}
      <details className="raw-json">
        <summary>View raw JSON</summary>
        <pre>{stringifyJson(raw)}</pre>
      </details>
    </div>
  );
}

function renderKeyValueCard(title: string, entries: Array<[string, unknown]>) {
  if (entries.length === 0) {
    return null;
  }
  return (
    <section className="info-card">
      <h4>{title}</h4>
      <div className="step-grid">
        {entries.map(([label, value]) => (
          <div className="step-grid-row" key={label}>
            <span>{humanizeKey(label)}</span>
            {renderInlineValue(value)}
          </div>
        ))}
      </div>
    </section>
  );
}

function renderRecordCard(title: string, record?: Record<string, unknown>, omittedKeys: string[] = []) {
  if (!record) {
    return null;
  }
  const omitted = new Set(omittedKeys);
  const entries = Object.entries(record).filter(([, value]) => value !== undefined && value !== null && value !== '' && !omitted.has(String(value)));
  const visibleEntries = Object.entries(record).filter(([key, value]) => !omitted.has(key) && value !== undefined && value !== null && value !== '');
  if (visibleEntries.length === 0 && entries.length === 0) {
    return null;
  }
  return renderKeyValueCard(title, visibleEntries.slice(0, 8));
}

function renderOutputPreview(output: string) {
  const clipped = truncateMiddle(output, OUTPUT_PREVIEW_MAX_LENGTH);
  return (
    <section className="info-card output-card">
      <h4>Output preview</h4>
      <pre>{clipped}</pre>
    </section>
  );
}

function renderInlineValue(value: unknown) {
  if (typeof value === 'boolean') {
    return <code>{value ? 'yes' : 'no'}</code>;
  }
  if (typeof value === 'number') {
    return <code>{value}</code>;
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed.length > INLINE_VALUE_MAX_LENGTH ? <p>{truncateMiddle(trimmed, INLINE_VALUE_MAX_LENGTH)}</p> : <code>{trimmed || 'n/a'}</code>;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <code>[]</code>;
    }
    if (value.every((item) => typeof item === 'string' || typeof item === 'number' || typeof item === 'boolean')) {
      return <p>{value.map(String).join(' ')}</p>;
    }
    return <pre>{truncateMiddle(stringifyJson(value), OUTPUT_PREVIEW_MAX_LENGTH)}</pre>;
  }
  if (isRecord(value)) {
    return <pre>{truncateMiddle(stringifyJson(value), OUTPUT_PREVIEW_MAX_LENGTH)}</pre>;
  }
  return <code>n/a</code>;
}

function toolCallSummary(step: StepEventPayload): string {
  const command = commandFromArgs(step.args);
  if (command) {
    return `Running ${step.tool ?? 'tool'}: ${command}`;
  }
  const target = valueFromRecord(step.args, ['path', 'file', 'filename', 'query']);
  if (typeof target === 'string' && target.trim()) {
    return `Calling ${step.tool ?? 'tool'} for ${target}`;
  }
  return `Calling ${step.tool ?? 'tool'} with structured arguments.`;
}

function toolResultSummary(step: StepEventPayload): string {
  if (step.summary?.trim()) {
    return step.summary.trim();
  }
  if (typeof step.exit_code === 'number') {
    return step.exit_code === 0 ? `${step.tool ?? 'Tool'} completed successfully.` : `${step.tool ?? 'Tool'} exited with code ${step.exit_code}.`;
  }
  return `${step.tool ?? 'Tool'} returned a result.`;
}

function genericStepSummary(step: StepEventPayload): string {
  if (step.summary?.trim()) {
    return step.summary.trim();
  }
  if (step.reason || step.detail) {
    return [step.reason, step.detail].filter(Boolean).join(': ');
  }
  if (step.decision_type) {
    return `Decision: ${step.decision_type}`;
  }
  return `Recorded ${step.type ? step.type.split('_').join(' ') : step.kind}.`;
}

function commandFromArgs(args?: Record<string, unknown>): string | undefined {
  if (!args) {
    return undefined;
  }
  const command = valueFromRecord(args, ['command', 'cmd']);
  if (typeof command === 'string') {
    return command;
  }
  const commandArgs = args.args;
  if (Array.isArray(commandArgs) && commandArgs.every((item) => typeof item === 'string')) {
    return commandArgs.join(' ');
  }
  return undefined;
}

function valueFromRecord(record: Record<string, unknown> | undefined, keys: string[]): unknown {
  if (!record) {
    return undefined;
  }
  for (const key of keys) {
    const value = record[key];
    if (value !== undefined && value !== null && value !== '') {
      return value;
    }
  }
  return undefined;
}

function firstTextValue(record: Record<string, unknown>, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === 'string' && value.trim()) {
      return value;
    }
  }
  return undefined;
}

function compactEntries(entries: Array<[string, unknown]>): Array<[string, unknown]> {
  return entries.filter(([, value]) => value !== undefined && value !== null && value !== '');
}

function humanizeKey(key: string): string {
  return key.split('_').join(' ');
}

function truncateMiddle(value: string, maxLength: number): string {
  if (value.length <= maxLength) {
    return value;
  }
  const edge = Math.floor((maxLength - 15) / 2);
  return `${value.slice(0, edge)}\n... clipped ...\n${value.slice(-edge)}`;
}

function stringifyJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function stepTitle(step: StepEventPayload): string {
  if (step.type === 'tool_call') {
    return `Tool call: ${step.tool ?? 'unknown'}`;
  }
  if (step.type === 'tool_result') {
    return `Tool result: ${step.tool ?? 'unknown'}`;
  }
  if (step.type) {
    return step.type.split('_').join(' ');
  }
  return step.kind;
}

function shortInstruction(instruction: string): string {
  return instruction.length > 80 ? `${instruction.slice(0, 77)}...` : instruction;
}

function shortJobId(jobId: string): string {
  return jobId.length > 18 ? `${jobId.slice(0, 18)}...` : jobId;
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value));
}

function formatTime(value: string): string {
  return new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(new Date(value));
}

function numberOrUndefined(value: string): number | undefined {
  const trimmed = value.trim();
  if (!trimmed) {
    return undefined;
  }
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function isTerminalStatus(status: JobStatus): boolean {
  return TERMINAL_STATUSES.includes(status);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
