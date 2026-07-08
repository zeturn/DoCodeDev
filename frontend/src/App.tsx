import React, { FormEvent, useCallback, useEffect, useMemo, useState } from 'react';
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
const TERMINAL_STATUSES: JobStatus[] = ['succeeded', 'failed', 'stopped'];

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

  const selectedJobFromList = useMemo(() => jobs.find((job) => job.id === selectedJobId) ?? null, [jobs, selectedJobId]);

  const refreshJobs = useCallback(async () => {
    setLoadingJobs(true);
    setError(null);
    try {
      const loaded = await listJobs(authToken, statusFilter || undefined);
      setJobs(loaded);
      if (!selectedJobId && loaded.length > 0) {
        setSelectedJobId(loaded[0].id);
      }
      if (selectedJobId && !loaded.some((job) => job.id === selectedJobId)) {
        setSelectedJobId(loaded[0]?.id ?? null);
      }
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoadingJobs(false);
    }
  }, [authToken, selectedJobId, statusFilter]);

  useEffect(() => {
    localStorage.setItem(STORAGE_TOKEN_KEY, authToken);
  }, [authToken]);

  useEffect(() => {
    void refreshJobs();
  }, [refreshJobs]);

  useEffect(() => {
    if (!selectedJobId) {
      setSelectedJob(null);
      setSteps([]);
      return;
    }

    const controller = new AbortController();
    setSelectedJob(selectedJobFromList);
    setSteps([]);
    setStreamError(null);
    setStreaming(true);

    async function loadAndStream() {
      try {
        const [job, existingSteps] = await Promise.all([getJob(selectedJobId, authToken), getSteps(selectedJobId, authToken)]);
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
          selectedJobId,
          authToken,
          ({ event, data }) => {
            if (event === 'status' && isRecord(data)) {
              const nextStatus = data.status;
              if (typeof nextStatus === 'string') {
                patchJobStatus(selectedJobId, nextStatus as JobStatus);
              }
            }
            if (event === 'step' && isRecord(data)) {
              setSteps((current) => dedupeSteps([...current, data as StepEventPayload]));
            }
            if (event === 'done' && isRecord(data)) {
              const nextStatus = typeof data.status === 'string' ? (data.status as JobStatus) : undefined;
              if (nextStatus) {
                patchJobStatus(selectedJobId, nextStatus);
              }
              setStreaming(false);
              void refreshJobs();
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
  }, [authToken, refreshJobs, selectedJobFromList, selectedJobId]);

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
      setSelectedJobId(created.job_id);
      await refreshJobs();
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
          {activeJob && (
            <button type="button" className="danger" onClick={() => void onCancelJob()} disabled={isTerminalStatus(activeJob.status)}>
              Cancel job
            </button>
          )}
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

function upsertJob(setJobs: React.Dispatch<React.SetStateAction<Job[]>>, job: Job) {
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
  if (step.type === 'tool_call') {
    return <pre>{JSON.stringify(step.args ?? {}, null, 2)}</pre>;
  }
  if (step.type === 'tool_result') {
    return (
      <div className="step-grid">
        <span>tool</span><code>{step.tool}</code>
        <span>exit</span><code>{step.exit_code ?? 'n/a'}</code>
        <span>summary</span><p>{String(step.summary ?? '') || 'No summary available.'}</p>
        {step.truncated !== undefined && <><span>truncated</span><code>{String(step.truncated)}</code></>}
      </div>
    );
  }
  if (step.reason || step.detail) {
    return <p>{[step.reason, step.detail].filter(Boolean).join(': ')}</p>;
  }
  return <pre>{JSON.stringify(redactLargeValues(step), null, 2)}</pre>;
}

function stepTitle(step: StepEventPayload): string {
  if (step.type === 'tool_call') {
    return `Tool call: ${step.tool ?? 'unknown'}`;
  }
  if (step.type === 'tool_result') {
    return `Tool result: ${step.tool ?? 'unknown'}`;
  }
  if (step.type) {
    return step.type.replaceAll('_', ' ');
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

function redactLargeValues(value: Record<string, unknown>): Record<string, unknown> {
  const copy: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(value)) {
    if (key === 'output' || key === 'git_diff' || key === 'git_status') {
      copy[key] = '[redacted]';
    } else {
      copy[key] = item;
    }
  }
  return copy;
}
