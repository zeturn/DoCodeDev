export type JobStatus = 'queued' | 'preparing' | 'running' | 'verifying' | 'succeeded' | 'failed' | 'stopped';

export interface Job {
  id: string;
  user_id: string;
  instruction: string;
  repo_url?: string | null;
  branch?: string | null;
  github_repo?: string | null;
  base_branch: string;
  dobox_project_id?: string | null;
  dobox_sandbox_id?: string | null;
  dobox_agent_session_id?: string | null;
  provider: string;
  model: string;
  quality: string;
  status: JobStatus;
  max_iterations: number;
  max_runtime_seconds: number;
  max_consecutive_failures: number;
  max_tool_calls: number;
  max_llm_tokens: number;
  max_llm_cost?: number | null;
  artifact_mode: string;
  sandbox_network_mode: string;
  result_summary?: string | null;
  failure_reason?: string | null;
  artifact_id?: string | null;
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
  result?: Record<string, unknown> | null;
}

export interface CreateJobRequest {
  instruction: string;
  repo_url?: string;
  branch?: string;
  github_repo?: string;
  base_branch?: string;
  provider?: string;
  model?: string;
  quality?: 'fast' | 'balanced' | 'strong';
  max_iterations?: number;
  max_runtime_seconds?: number;
  max_consecutive_failures?: number;
  max_tool_calls?: number;
  max_llm_tokens?: number;
  max_llm_cost?: number;
  artifact_mode?: 'patch' | 'zip' | 'commit' | 'pr';
  sandbox_network_mode?: string;
}

export interface CreateJobResponse {
  job_id: string;
  status: JobStatus;
}

export interface StepEventPayload {
  step_id: string;
  job_id: string;
  step_index: number;
  kind: string;
  created_at: string;
  type?: string;
  tool?: string;
  args?: Record<string, unknown>;
  exit_code?: number;
  summary?: string;
  truncated?: boolean;
  metadata?: Record<string, unknown>;
  reason?: string;
  detail?: string;
  decision_type?: string;
  [key: string]: unknown;
}

export interface StreamEvent {
  event: string;
  data: unknown;
}

const API_BASE = (import.meta.env.VITE_DOCODE_API_BASE_URL ?? '').replace(/\/$/, '');

export function apiBase(): string {
  return API_BASE;
}

function authHeaders(token: string): HeadersInit {
  return token.trim() ? { Authorization: `Bearer ${token.trim()}` } : {};
}

async function apiRequest<T>(path: string, options: RequestInit & { token?: string } = {}): Promise<T> {
  const { token = '', headers: optionHeaders, ...requestInit } = options;
  const headers: HeadersInit = {
    Accept: 'application/json',
    ...authHeaders(token),
    ...(requestInit.body ? { 'Content-Type': 'application/json' } : {}),
    ...(optionHeaders ?? {})
  };
  const response = await fetch(`${API_BASE}${path}`, { ...requestInit, headers });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        detail = typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload.detail);
      }
    } catch {
      // Keep the HTTP status text when the body is not JSON.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export async function listJobs(token: string, status?: JobStatus): Promise<Job[]> {
  const query = status ? `?status=${encodeURIComponent(status)}` : '';
  return apiRequest<Job[]>(`/v1/jobs${query}`, { token });
}

export async function getJob(jobId: string, token: string): Promise<Job> {
  return apiRequest<Job>(`/v1/jobs/${encodeURIComponent(jobId)}`, { token });
}

export async function getSteps(jobId: string, token: string): Promise<StepEventPayload[]> {
  return apiRequest<StepEventPayload[]>(`/v1/jobs/${encodeURIComponent(jobId)}/steps`, { token });
}

export async function createJob(request: CreateJobRequest, token: string): Promise<CreateJobResponse> {
  return apiRequest<CreateJobResponse>('/v1/jobs', {
    method: 'POST',
    token,
    body: JSON.stringify(removeEmptyValues(request))
  });
}

export async function cancelJob(jobId: string, token: string): Promise<{ job_id: string; status: string }> {
  return apiRequest<{ job_id: string; status: string }>(`/v1/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: 'POST',
    token
  });
}

export async function streamJobEvents(
  jobId: string,
  token: string,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const headers: HeadersInit = {
    Accept: 'text/event-stream',
    ...authHeaders(token)
  };
  const response = await fetch(`${API_BASE}/v1/jobs/${encodeURIComponent(jobId)}/events`, { headers, signal });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  if (!response.body) {
    throw new Error('Streaming is not supported by this browser');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split(/\r?\n\r?\n/);
    buffer = parts.pop() ?? '';
    for (const part of parts) {
      const parsed = parseSseBlock(part);
      if (parsed) {
        onEvent(parsed);
      }
    }
  }

  buffer += decoder.decode();
  const parsed = parseSseBlock(buffer);
  if (parsed) {
    onEvent(parsed);
  }
}

function parseSseBlock(block: string): StreamEvent | null {
  if (!block.trim()) {
    return null;
  }
  let event = 'message';
  const dataLines: string[] = [];
  for (const rawLine of block.split(/\r?\n/)) {
    const line = rawLine.trimEnd();
    if (!line || line.startsWith(':')) {
      continue;
    }
    const separatorIndex = line.indexOf(':');
    const field = separatorIndex >= 0 ? line.slice(0, separatorIndex) : line;
    const value = separatorIndex >= 0 ? line.slice(separatorIndex + 1).replace(/^ /, '') : '';
    if (field === 'event') {
      event = value;
    } else if (field === 'data') {
      dataLines.push(value);
    }
  }
  if (dataLines.length === 0) {
    return { event, data: null };
  }
  const dataText = dataLines.join('\n');
  try {
    return { event, data: JSON.parse(dataText) };
  } catch {
    return { event, data: dataText };
  }
}

function removeEmptyValues(value: object): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(value).filter(([, item]) => item !== undefined && item !== null && item !== '')
  );
}
