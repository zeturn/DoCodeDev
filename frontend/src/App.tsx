import { type Dispatch, type FormEvent, type ReactNode, type SetStateAction, useCallback, useEffect, useMemo, useState } from 'react';
import { Routes, Route, useLocation, useNavigate } from 'react-router-dom';
import { Button, Input, Select, Card, Alert } from '@zeturn/watercolor-react';
import { isLoggedIn, beginLogin, finishLogin, logout, userProfile } from './auth';
import BasaltPassLogin from './BasaltPassLogin';
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

const APP = 'docode';
const STORAGE_TOKEN_KEY = 'docode.authToken';

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

  // BasaltPass auth
  const [loggedIn, setLoggedIn] = useState(isLoggedIn(APP));
  const [authError, setAuthError] = useState('');
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    if (location.pathname !== '/auth/callback') return;
    finishLogin(APP, location.search)
      .then(() => { setLoggedIn(true); navigate('/', { replace: true }); })
      .catch((err: any) => setAuthError(err.message));
  }, [location.pathname]);

  const selectedJobFromList = useMemo(() => jobs.find((job) => job.id === selectedJobId) ?? null, [jobs, selectedJobId]);

  const refreshJobs = useCallback(async () => {
    setLoadingJobs(true);
    setError(null);
    try {
      const loaded = await listJobs(authToken, statusFilter || undefined);
      setJobs(loaded);
      setSelectedJobId((current) => {
        if (!current && loaded.length > 0) return loaded[0].id;
        if (current && !loaded.some((job) => job.id === current)) return loaded[0]?.id ?? null;
        return current;
      });
    } catch (err) { setError(errorMessage(err)); }
    finally { setLoadingJobs(false); }
  }, [authToken, statusFilter]);

  useEffect(() => { localStorage.setItem(STORAGE_TOKEN_KEY, authToken); }, [authToken]);
  useEffect(() => { void refreshJobs(); }, [refreshJobs]);

  useEffect(() => {
    const sid = selectedJobId;
    if (!sid) { setSelectedJob(null); setSteps([]); return; }
    const controller = new AbortController();
    setSelectedJob(null); setSteps([]); setStreamError(null); setStreaming(true);
    async function loadAndStream() {
      try {
        const [job, existingSteps] = await Promise.all([getJob(sid!, authToken), getSteps(sid!, authToken)]);
        if (!controller.signal.aborted) { setSelectedJob(job); setSteps(dedupeSteps(existingSteps)); upsertJob(setJobs, job); }
      } catch (err) {
        if (!controller.signal.aborted) { setStreamError(errorMessage(err)); setStreaming(false); }
        return;
      }
      try {
        await streamJobEvents(sid!, authToken, ({ event, data }) => {
          if (event === 'status' && isRecord(data) && typeof data.status === 'string') patchJobStatus(sid!, data.status as JobStatus);
          if (event === 'step' && isRecord(data)) setSteps((current) => dedupeSteps([...current, data as StepEventPayload]));
          if (event === 'done' && isRecord(data)) {
            if (typeof data.status === 'string') patchJobStatus(sid!, data.status as JobStatus);
            setStreaming(false);
            void getJob(sid!, authToken).then((j) => { if (!controller.signal.aborted) { setSelectedJob(j); upsertJob(setJobs, j); } }).catch(() => undefined);
          }
        }, controller.signal);
      } catch (err) {
        if (!controller.signal.aborted) setStreamError(errorMessage(err));
      } finally { if (!controller.signal.aborted) setStreaming(false); }
    }
    void loadAndStream();
    return () => controller.abort();
  }, [authToken, selectedJobId]);

  const activeJob = selectedJob ?? selectedJobFromList;

  async function onCreateJob(event: FormEvent) {
    event.preventDefault();
    if (!form.instruction.trim()) { setError('Instruction is required.'); return; }
    setCreatingJob(true); setError(null);
    try {
      const created = await createJob({
        instruction: form.instruction.trim(), github_repo: form.github_repo.trim(), repo_url: form.repo_url.trim(),
        base_branch: form.base_branch.trim(), branch: form.branch.trim(), quality: form.quality,
        artifact_mode: form.artifact_mode, sandbox_network_mode: form.sandbox_network_mode.trim(),
        max_iterations: numberOrUndefined(form.max_iterations), max_tool_calls: numberOrUndefined(form.max_tool_calls),
        max_runtime_seconds: numberOrUndefined(form.max_runtime_seconds), max_llm_cost: numberOrUndefined(form.max_llm_cost)
      }, authToken);
      setForm({ ...initialJobForm, github_repo: form.github_repo, repo_url: form.repo_url, base_branch: form.base_branch || 'main' });
      setStatusFilter(''); setSelectedJobId(created.job_id);
      setJobs(await listJobs(authToken));
    } catch (err) { setError(errorMessage(err)); }
    finally { setCreatingJob(false); }
  }

  async function onCancelJob() {
    if (!activeJob || isTerminalStatus(activeJob.status)) return;
    setError(null);
    try { await cancelJob(activeJob.id, authToken); await refreshJobs(); }
    catch (err) { setError(errorMessage(err)); }
  }

  function patchJobStatus(jobId: string, status: JobStatus) {
    setJobs((current) => current.map((job) => (job.id === jobId ? { ...job, status, updated_at: new Date().toISOString() } : job)));
    setSelectedJob((current) => (current?.id === jobId ? { ...current, status, updated_at: new Date().toISOString() } : current));
  }

  // Callback route
  if (location.pathname === '/auth/callback') {
    return <div className="min-h-screen flex items-center justify-center text-muted">{authError || '正在完成 BasaltPass 登录…'}</div>;
  }
  if (!loggedIn) {
    return <BasaltPassLogin app={APP} brand="DoCode" description="登录以访问 DoCode Jobs Dashboard。" />;
  }

  const profile = userProfile(APP);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">DC</span>
          <div>
            <h1>DoCode</h1>
            <p>Jobs dashboard</p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            {profile.name !== 'BasaltPass 用户' && <span className="text-xs text-muted">{profile.name}</span>}
            <Button variant="text" size="sm" onClick={() => { logout(APP); setLoggedIn(false); }}>退出</Button>
          </div>
        </div>

        <Input value={authToken} onChange={(e) => setAuthToken(e.target.value)} placeholder="Bearer token (optional)" type="password" label="Auth token" />

        <div className="toolbar">
          <Select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value as JobStatus | '')}
            options={[
              { label: 'All statuses', value: '' }, { label: 'Queued', value: 'queued' }, { label: 'Preparing', value: 'preparing' },
              { label: 'Running', value: 'running' }, { label: 'Verifying', value: 'verifying' },
              { label: 'Succeeded', value: 'succeeded' }, { label: 'Failed', value: 'failed' }, { label: 'Stopped', value: 'stopped' },
            ]} />
          <Button variant="secondary" onClick={() => void refreshJobs()} disabled={loadingJobs}>
            {loadingJobs ? 'Refreshing...' : 'Refresh'}
          </Button>
        </div>

        <div className="job-list" aria-label="Jobs">
          {jobs.length === 0 && <p className="empty">No jobs yet.</p>}
          {jobs.map((job) => (
            <button key={job.id} type="button" className={`job-card ${job.id === selectedJobId ? 'selected' : ''}`} onClick={() => setSelectedJobId(job.id)}>
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
            {activeJob && (
              <Button variant="error" onClick={() => void onCancelJob()} disabled={isTerminalStatus(activeJob.status)}>
                Cancel job
              </Button>
            )}
          </div>
        </header>

        {error && <Alert type="error" closable onClose={() => setError(null)}>{error}</Alert>}

        <Card variant="minimal" className="panel create-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Create</p>
              <h2>New coding job</h2>
            </div>
          </div>
          <form onSubmit={(event) => void onCreateJob(event)} className="job-form">
            <div className="field full">
              <Input multiline rows={5} label="Instruction" value={form.instruction} onChange={(e) => setForm((c) => ({ ...c, instruction: e.target.value }))} placeholder="Example: Fix the calculator bug and run python -m unittest discover -s tests" />
            </div>
            <div className="field"><Input label="GitHub repo" value={form.github_repo} onChange={(e) => setForm((c) => ({ ...c, github_repo: e.target.value }))} placeholder="owner/repo" /></div>
            <div className="field"><Input label="Repo URL" value={form.repo_url} onChange={(e) => setForm((c) => ({ ...c, repo_url: e.target.value }))} placeholder="https://github.com/owner/repo.git" /></div>
            <div className="field"><Input label="Base branch" value={form.base_branch} onChange={(e) => setForm((c) => ({ ...c, base_branch: e.target.value }))} /></div>
            <div className="field"><Input label="Work branch" value={form.branch} onChange={(e) => setForm((c) => ({ ...c, branch: e.target.value }))} placeholder="optional" /></div>
            <div className="field">
              <Select label="Quality" value={form.quality} onChange={(e) => setForm((c) => ({ ...c, quality: e.target.value as JobFormState['quality'] }))}
                options={[{ label: 'Fast', value: 'fast' }, { label: 'Balanced', value: 'balanced' }, { label: 'Strong', value: 'strong' }]} />
            </div>
            <div className="field">
              <Select label="Artifact mode" value={form.artifact_mode} onChange={(e) => setForm((c) => ({ ...c, artifact_mode: e.target.value as JobFormState['artifact_mode'] }))}
                options={[{ label: 'Patch', value: 'patch' }, { label: 'Zip', value: 'zip' }, { label: 'Commit', value: 'commit' }, { label: 'PR', value: 'pr' }]} />
            </div>
            <div className="field"><Input label="Network mode" value={form.sandbox_network_mode} onChange={(e) => setForm((c) => ({ ...c, sandbox_network_mode: e.target.value }))} /></div>
            <div className="field"><Input label="Max iterations" type="number" value={form.max_iterations} onChange={(e) => setForm((c) => ({ ...c, max_iterations: e.target.value }))} /></div>
            <div className="field"><Input label="Max tool calls" type="number" value={form.max_tool_calls} onChange={(e) => setForm((c) => ({ ...c, max_tool_calls: e.target.value }))} /></div>
            <div className="field"><Input label="Max runtime seconds" type="number" value={form.max_runtime_seconds} onChange={(e) => setForm((c) => ({ ...c, max_runtime_seconds: e.target.value }))} /></div>
            <div className="field"><Input label="Max LLM cost" type="number" value={form.max_llm_cost} onChange={(e) => setForm((c) => ({ ...c, max_llm_cost: e.target.value }))} /></div>
            <div className="form-actions full">
              <Button variant="primary" type="submit" disabled={creatingJob} loading={creatingJob}>{creatingJob ? 'Creating...' : 'Create job'}</Button>
            </div>
          </form>
        </Card>

        <Card variant="minimal" className="panel output-panel">
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
                <div><dt>Created</dt><dd>{formatDate(activeJob.created_at)}</dd></div>
                <div><dt>Updated</dt><dd>{formatDate(activeJob.updated_at)}</dd></div>
                <div><dt>Provider</dt><dd>{activeJob.provider} / {activeJob.model}</dd></div>
                <div><dt>Artifacts</dt><dd>{activeJob.artifact_id || 'pending'}</dd></div>
              </dl>
              <p className="instruction-preview">{activeJob.instruction}</p>
              {streamError && <Alert type="warning">{streamError}</Alert>}
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
        </Card>
      </section>
    </main>
  );
}

export default App;

// === All helper functions preserved below (identical to original) ===

function upsertJob(setJobs: Dispatch<SetStateAction<Job[]>>, job: Job) {
  setJobs((current) => {
    const exists = current.some((item) => item.id === job.id);
    const next = exists ? current.map((item) => (item.id === job.id ? job : item)) : [job, ...current];
    return [...next].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
  });
}

function dedupeSteps(items: StepEventPayload[]): StepEventPayload[] {
  const byKey = new Map<string, StepEventPayload>();
  for (const item of items) { const key = item.step_id || `${item.job_id}:${item.step_index}`; byKey.set(key, item); }
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
  const reasoningRecords = Array.isArray(step.reasoning_records) ? step.reasoning_records.filter(isRecord) : [];
  const reasoningText = typeof step.reasoning === 'string' ? step.reasoning.trim() : '';
  if (!reasoningText && reasoningRecords.length === 0) return null;
  return (<section className="info-card thinking-card"><h4>Thinking</h4>{reasoningText && <pre>{truncateMiddle(reasoningText, OUTPUT_PREVIEW_MAX_LENGTH)}</pre>}{reasoningRecords.length > 0 && (<div className="thinking-records">{reasoningRecords.slice(0, 6).map((record, index) => { const text = typeof record.text === 'string' ? record.text : stringifyJson(record); const label = [record.type, record.source].filter((value) => typeof value === 'string' && value.trim()).join(' / '); return (<article className="thinking-record" key={`${index}-${label || 'record'}`}>{label && <span>{label}</span>}<pre>{truncateMiddle(text, OUTPUT_PREVIEW_MAX_LENGTH)}</pre></article>); })}</div>)}</section>);
}

function StepCard({ children, raw }: { children: ReactNode; raw: StepEventPayload }) {
  return (<div className="step-body">{children}<details className="raw-json"><summary>View raw JSON</summary><pre>{stringifyJson(raw)}</pre></details></div>);
}

function renderKeyValueCard(title: string, entries: Array<[string, unknown]>) {
  if (entries.length === 0) return null;
  return (<section className="info-card"><h4>{title}</h4><div className="step-grid">{entries.map(([label, value]) => (<div className="step-grid-row" key={label}><span>{humanizeKey(label)}</span>{renderInlineValue(value)}</div>))}</div></section>);
}

function renderRecordCard(title: string, record?: Record<string, unknown>, omittedKeys: string[] = []) {
  if (!record) return null;
  const omitted = new Set(omittedKeys);
  const visibleEntries = Object.entries(record).filter(([key, value]) => !omitted.has(key) && value !== undefined && value !== null && value !== '');
  if (visibleEntries.length === 0) return null;
  return renderKeyValueCard(title, visibleEntries.slice(0, 8));
}

function renderOutputPreview(output: string) {
  return (<section className="info-card output-card"><h4>Output preview</h4><pre>{truncateMiddle(output, OUTPUT_PREVIEW_MAX_LENGTH)}</pre></section>);
}

function renderInlineValue(value: unknown) {
  if (typeof value === 'boolean') return <code>{value ? 'yes' : 'no'}</code>;
  if (typeof value === 'number') return <code>{value}</code>;
  if (typeof value === 'string') return value.trim().length > INLINE_VALUE_MAX_LENGTH ? <p>{truncateMiddle(value.trim(), INLINE_VALUE_MAX_LENGTH)}</p> : <code>{value.trim() || 'n/a'}</code>;
  if (Array.isArray(value)) return value.length === 0 ? <code>[]</code> : (value.every((item) => typeof item === 'string' || typeof item === 'number' || typeof item === 'boolean') ? <p>{value.map(String).join(' ')}</p> : <pre>{truncateMiddle(stringifyJson(value), OUTPUT_PREVIEW_MAX_LENGTH)}</pre>);
  if (isRecord(value)) return <pre>{truncateMiddle(stringifyJson(value), OUTPUT_PREVIEW_MAX_LENGTH)}</pre>;
  return <code>n/a</code>;
}

function toolCallSummary(step: StepEventPayload): string { const command = commandFromArgs(step.args); if (command) return `Running ${step.tool ?? 'tool'}: ${command}`; const target = valueFromRecord(step.args, ['path', 'file', 'filename', 'query']); if (typeof target === 'string' && target.trim()) return `Calling ${step.tool ?? 'tool'} for ${target}`; return `Calling ${step.tool ?? 'tool'} with structured arguments.`; }
function toolResultSummary(step: StepEventPayload): string { if (step.summary?.trim()) return step.summary.trim(); if (typeof step.exit_code === 'number') return step.exit_code === 0 ? `${step.tool ?? 'Tool'} completed successfully.` : `${step.tool ?? 'Tool'} exited with code ${step.exit_code}.`; return `${step.tool ?? 'Tool'} returned a result.`; }
function genericStepSummary(step: StepEventPayload): string { if (step.summary?.trim()) return step.summary.trim(); if (step.reason || step.detail) return [step.reason, step.detail].filter(Boolean).join(': '); if (step.decision_type) return `Decision: ${step.decision_type}`; return `Recorded ${step.type ? step.type.split('_').join(' ') : step.kind}.`; }
function commandFromArgs(args?: Record<string, unknown>): string | undefined { if (!args) return undefined; const command = valueFromRecord(args, ['command', 'cmd']); if (typeof command === 'string') return command; const commandArgs = args.args; if (Array.isArray(commandArgs) && commandArgs.every((item) => typeof item === 'string')) return commandArgs.join(' '); return undefined; }
function valueFromRecord(record: Record<string, unknown> | undefined, keys: string[]): unknown { if (!record) return undefined; for (const key of keys) { const value = record[key]; if (value !== undefined && value !== null && value !== '') return value; } return undefined; }
function firstTextValue(record: Record<string, unknown>, keys: string[]): string | undefined { for (const key of keys) { const value = record[key]; if (typeof value === 'string' && value.trim()) return value; } return undefined; }
function compactEntries(entries: Array<[string, unknown]>): Array<[string, unknown]> { return entries.filter(([, value]) => value !== undefined && value !== null && value !== ''); }
function humanizeKey(key: string): string { return key.split('_').join(' '); }
function truncateMiddle(value: string, maxLength: number): string { if (value.length <= maxLength) return value; const edge = Math.floor((maxLength - 15) / 2); return `${value.slice(0, edge)}\n... clipped ...\n${value.slice(-edge)}`; }
function stringifyJson(value: unknown): string { try { return JSON.stringify(value, null, 2); } catch { return String(value); } }
function stepTitle(step: StepEventPayload): string { if (step.type === 'tool_call') return `Tool call: ${step.tool ?? 'unknown'}`; if (step.type === 'tool_result') return `Tool result: ${step.tool ?? 'unknown'}`; if (step.type) return step.type.split('_').join(' '); return step.kind; }
function shortInstruction(instruction: string): string { return instruction.length > 80 ? `${instruction.slice(0, 77)}...` : instruction; }
function shortJobId(jobId: string): string { return jobId.length > 18 ? `${jobId.slice(0, 18)}...` : jobId; }
function formatDate(value: string): string { return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value)); }
function formatTime(value: string): string { return new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(new Date(value)); }
function numberOrUndefined(value: string): number | undefined { const trimmed = value.trim(); if (!trimmed) return undefined; const parsed = Number(trimmed); return Number.isFinite(parsed) ? parsed : undefined; }
function isRecord(value: unknown): value is Record<string, unknown> { return Boolean(value) && typeof value === 'object' && !Array.isArray(value); }
function isTerminalStatus(status: JobStatus): boolean { return TERMINAL_STATUSES.includes(status); }
function errorMessage(error: unknown): string { return error instanceof Error ? error.message : String(error); }
