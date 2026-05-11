import axios from 'axios';

import { getStoredIdentity } from './identity';

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? '';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: API_KEY ? { 'X-API-Key': API_KEY } : {},
});

// Read the chatbot identity from localStorage on every outgoing request
// so identity changes (sign in / rename / clear) take effect immediately
// without rebuilding the axios instance. ``getStoredIdentity`` is
// SSR-safe — returns null on the server, so this interceptor is a no-op
// inside Next.js server components.
apiClient.interceptors.request.use((config) => {
  const identity = getStoredIdentity();
  if (identity) {
    // `config.headers` is an `AxiosHeaders` instance in request
    // interceptors (axios v1) — use its `.set()` rather than index
    // assignment so the types line up under every tsconfig.
    config.headers.set('X-User-Id', identity.id);
  }
  return config;
});

/**
 * Coerce any backend error into a single string suitable for rendering.
 *
 * FastAPI returns 422 with ``detail`` as an **array of Pydantic validation
 * objects** (each ``{type, loc, msg, input, ctx}``), not a string. Other
 * errors return ``detail`` as a string. Some non-API failures (network,
 * CORS) only have an ``Error.message``. Without this normaliser, code that
 * does ``setError(err.response.data.detail)`` accidentally puts an array of
 * objects into JSX and React crashes with "Objects are not valid as a
 * React child".
 */
export function formatApiError(err: any, fallback = 'Request failed'): string {
  const detail = err?.response?.data?.detail;
  let out: string | null = null;

  if (typeof detail === 'string' && detail) {
    out = detail;
  } else if (Array.isArray(detail) && detail.length > 0) {
    out = detail
      .map((d: any) => {
        if (typeof d === 'string') return d;
        const where = Array.isArray(d?.loc) ? d.loc.slice(1).join('.') : null;
        const msg = d?.msg ?? 'Validation error';
        return where ? `${where}: ${msg}` : msg;
      })
      .join('; ');
  } else if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    out = detail.msg ?? JSON.stringify(detail);
  } else if (typeof err?.message === 'string' && err.message) {
    out = err.message;
  }

  // Always return *something* readable. An empty string would render
  // nothing at the call site and confuse the user.
  return out && out.trim() ? out : fallback;
}

// ── Graph ─────────────────────────────────────────────────────────────────

export interface Entity {
  id: string;
  name: string;
  entity_type: string;
  description?: string;
  properties: Record<string, unknown>;
  tags: string[];
  source_trust?: string | null;
  source_chunk_ids: string[];
  source_document_ids: string[];
  created_at?: string;
}

export interface Relationship {
  id: string;
  source_entity_id: string;
  target_entity_id: string;
  relationship_type: string;
  description?: string;
  strength: number;
  source_chunk_ids: string[];
  source_document_ids: string[];
}

export interface GraphStats {
  total_entities: number;
  total_relationships: number;
  entity_type_counts: Record<string, number>;
  relationship_type_counts: Record<string, number>;
}

export const getGraphStats = () =>
  apiClient.get<GraphStats>('/graph/stats').then((r) => r.data);

export const getEntities = (params?: { entity_type?: string; limit?: number; offset?: number }) =>
  apiClient.get<{ items: Entity[]; total: number; limit: number; offset: number }>('/graph/entities', { params }).then((r) => r.data);

export const getRelationships = (params?: { limit?: number; offset?: number }) =>
  apiClient.get<{ items: Relationship[]; total: number; limit: number; offset: number }>('/graph/relationships', { params }).then((r) => r.data);

export interface Subgraph {
  entities: Entity[];
  relationships: Relationship[];
  seed_count: number;
  expanded_count: number;
  seed_per_type: Record<string, number>;
  total_entities: number;
  total_relationships: number;
}

/** Fetch a self-consistent slice of the graph for visualization.
 *  Seeds are picked **per entity type** (newest `per_type_limit` of each),
 *  excluding any types listed in `exclude_types` (default `"Document"`).
 *  Every relationship in the response has both endpoints in the entity
 *  list, so the caller doesn't need a bipartite filter.
 *
 *  When `include_types` is set (comma-separated), only those types seed
 *  the slice — used by the graph page's progressive loader to fan out
 *  one request per type so the canvas paints in batches instead of
 *  staying blank until the full payload arrives. Cross-type neighbours
 *  are still returned for each per-type slice, so dedup-and-merge on
 *  the client keeps the graph coherent. */
export const getSubgraph = (params?: {
  per_type_limit?: number;
  exclude_types?: string;
  include_types?: string;
  max_neighbors?: number;
}) =>
  apiClient.get<Subgraph>('/graph/subgraph', { params }).then((r) => r.data);

// ── Documents ────────────────────────────────────────────────────────────

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
export type StageStatus = 'pending' | 'running' | 'completed' | 'skipped' | 'failed';

export interface JobEvent {
  ts: string;
  stage: string | null;
  level: 'info' | 'warn' | 'error';
  message: string;
  data?: Record<string, unknown>;
}

export interface Job {
  job_id: string;
  kind: string;
  status: JobStatus;
  message?: string;
  progress: number;
  stages: string[];
  current_stage: string | null;
  stage_progress: Record<string, StageStatus>;
  events: JobEvent[];
  result?: Record<string, unknown>;
  error?: string;
  cancel_requested: boolean;
  created_at: string;
  updated_at: string;
}

export interface JobSummary {
  job_id: string;
  kind: string;
  status: JobStatus;
  message?: string;
  current_stage: string | null;
  progress: number;
  created_at: string;
  updated_at: string;
}

export const processDocument = (body: {
  url?: string;
  text?: string;
  source_label?: string;
  tags?: string[];
  chunk_size?: number;
  chunk_overlap?: number;
}) => apiClient.post<Job>('/documents/process', body).then((r) => r.data);

export const getJob = (jobId: string) =>
  apiClient.get<Job>(`/documents/jobs/${jobId}`).then((r) => r.data);

export const cancelJob = (jobId: string) =>
  apiClient.post<Job>(`/documents/jobs/${jobId}/cancel`).then((r) => r.data);

export const listJobs = (limit = 30) =>
  apiClient
    .get<JobSummary[]>('/documents/jobs', { params: { limit } })
    .then((r) => r.data);

export const getJobStreamUrl = (jobId: string) =>
  `${BASE_URL}/documents/jobs/${jobId}/stream${API_KEY ? `?api_key=${API_KEY}` : ''}`;

// ── Pipeline metrics ─────────────────────────────────────────────────────

export interface PipelineMetrics {
  uptime_seconds: number;
  llm: {
    calls: number;
    calls_by_type: Record<string, number>;
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    avg_latency_ms: number;
    cache_hits: number;
    cache_hit_rate: number;
  };
  embedding: {
    calls: number;
    cache_hits: number;
    cache_hit_rate: number;
  };
  pipeline: {
    documents_processed: number;
    chunks_processed: number;
    entities_saved: number;
    relationships_saved: number;
  };
  cache_sizes: {
    dedup_entries: number;
    embedding_entries: number;
  };
}

export const getMetrics = () =>
  apiClient.get<PipelineMetrics>('/health/metrics').then((r) => r.data);

// ── Ingest ───────────────────────────────────────────────────────────────

export interface IngestResponse { job_id: string; source: string; status: string; }

export type OpenTargetsKind = 'disease' | 'target' | 'drug' | 'variant' | 'study';

export const ingestOpenTargets = (body: {
  entity_id: string;
  entity_type?: OpenTargetsKind;
  max_associations?: number;
  max_known_drugs?: number;
  min_association_score?: number;
  tag?: string;
}) =>
  apiClient.post<IngestResponse>('/ingest/open-targets', body).then((r) => r.data);

export const ingestPubMed = (body: { query: string; max_articles?: number; email?: string; tag?: string }) =>
  apiClient.post<IngestResponse>('/ingest/pubmed', body).then((r) => r.data);

export const ingestCrawl = (body: { urls: string[]; max_pages?: number; allowed_domains?: string[]; tag?: string }) =>
  apiClient.post<IngestResponse>('/ingest/crawl', body).then((r) => r.data);

// ── Curation ─────────────────────────────────────────────────────────────

export interface CurationEvent {
  entity_id?: string;
  relationship_id?: string;
  action: 'approve' | 'reject' | 'flag' | 'correct';
  curator_id?: string;
  notes?: string;
  /** Required for action=correct. For entity: name/description/properties.
   *  For relationship: relationship_type/description/strength. */
  corrections?: Record<string, unknown>;
}

export interface CurationAuditEntry {
  ts: string;
  action: string;
  target_id: string | null;
  curator: string;
  reason?: string;
  corrections?: Record<string, unknown>;
  success: boolean;
  message?: string;
  error?: string;
}

export interface CurationQueueItem {
  type: 'entity' | 'relationship';
  id: string;
  // Common
  description?: string | null;
  verification_status: 'unverified' | 'flagged' | 'rejected' | string;
  notes?: string | null;
  source_chunk_count: number;
  source_document_count: number;
  source_trust?: string | null;
  created_at?: string | null;
  // Entity-only
  name?: string;
  entity_type?: string;
  tags?: string[];
  // Relationship-only
  source_entity_id?: string;
  source_entity_name?: string | null;
  source_entity_type?: string | null;
  target_entity_id?: string;
  target_entity_name?: string | null;
  target_entity_type?: string | null;
  relationship_type?: string;
  strength?: number | null;
}

export const getCurationQueue = (params?: { status?: string; type?: string; limit?: number; offset?: number }) =>
  apiClient
    .get<{ total: number; items: CurationQueueItem[]; limit: number; offset: number }>(
      '/curation/queue',
      { params },
    )
    .then((r) => r.data);

export interface CurationQueueCounts {
  total: number;
  rejected: number;
  flagged: number;
  unverified: number;
}

export const getCurationQueueCounts = (params?: { type?: string }) =>
  apiClient
    .get<CurationQueueCounts>('/curation/queue/counts', { params })
    .then((r) => r.data);

export const submitCurationEvents = (events: CurationEvent[]) =>
  apiClient
    .post<{ processed: number; failed: number; errors: string[] }>(
      '/curation/events',
      { events },
    )
    .then((r) => r.data);

export const getCurationAudit = (limit = 100) =>
  apiClient
    .get<{ total: number; items: CurationAuditEntry[] }>('/curation/audit', {
      params: { limit },
    })
    .then((r) => r.data);

// ── Chunks (source text behind extractions) ──────────────────────────────

export interface ChunkRecord {
  id: string;
  document_id: string;
  chunk_index: number;
  content: string;
  character_count: number;
  token_count?: number | null;
}

export const getChunks = (ids: string[]) =>
  apiClient
    .get<{ items: ChunkRecord[]; missing: string[] }>('/graph/chunks', {
      params: { ids: ids.join(',') },
    })
    .then((r) => r.data);

// ── Type catalogs (drives the Correct-form dropdowns) ────────────────────

export const getEntityTypes = () =>
  apiClient.get<{ items: string[] }>('/graph/types/entities').then((r) => r.data.items);

export const getRelationshipTypes = () =>
  apiClient.get<{ items: string[] }>('/graph/types/relationships').then((r) => r.data.items);

// ── Verification ─────────────────────────────────────────────────────────

export interface VerificationReport {
  total: number;
  passed: number;
  failed: number;
  skipped: number;
  report: {
    relationship_id: string;
    source_entity_id: string;
    target_entity_id: string;
    relationship_type: string;
    status: string;
    confidence: number;
    reasoning: string;
    stage_results: { stage: string; status: string; confidence: number; reasoning: string }[];
  }[];
}

export const runVerification = (body: {
  relationship_ids: string[];
  enable_embedding?: boolean;
  enable_llm?: boolean;
  embedding_threshold?: number;
  early_exit_on_pass?: boolean;
  early_exit_on_fail?: boolean;
  context_map?: Record<string, string>;
}) => apiClient.post<VerificationReport>('/verification/run', body).then((r) => r.data);

// ── Text Verification ────────────────────────────────────────────────────

export interface TextVerificationEntry {
  relationship_id: string;
  source_entity_id: string;
  target_entity_id: string;
  source_entity_name: string;
  target_entity_name: string;
  relationship_type: string;
  relationship_description: string;
  status: string;
  confidence: number;
  reasoning: string;
  stage_results: { stage: string; status: string; confidence: number; reasoning: string }[];
}

export interface TextVerificationResponse {
  query_text: string;
  total_candidates: number;
  verified: number;
  not_verified: number;
  skipped: number;
  best_confidence: number;
  entries: TextVerificationEntry[];
}

export const verifyText = (body: {
  text: string;
  enable_embedding?: boolean;
  enable_llm?: boolean;
  embedding_threshold?: number;
  early_exit_on_pass?: boolean;
  early_exit_on_fail?: boolean;
  max_candidates?: number;
}) => apiClient.post<TextVerificationResponse>('/verification/text', body).then((r) => r.data);

// ── Export ───────────────────────────────────────────────────────────────

export const getExportUrl = (format: string) =>
  `${BASE_URL}/export?format=${format}${API_KEY ? `&api_key=${API_KEY}` : ''}`;

// ── Conflict Detection ───────────────────────────────────────────────────

export interface ConflictEntry {
  conflict_type: string;
  severity: string;
  existing_relationship_id: string;
  existing_relationship_type: string;
  existing_description: string;
  existing_source_chunk_ids: string[];
  existing_source_trust: string | null;
  new_relationship_type: string;
  new_description: string;
  new_source_chunk_ids: string[];
  new_source_trust: string | null;
  source_entity_name: string;
  target_entity_name: string;
  reasoning: string;
  requires_review: boolean;
}

export interface ConflictCheckResponse {
  total_checked: number;
  conflicts_found: number;
  conflicts: ConflictEntry[];
}

export const checkConflicts = (body: { text: string; use_llm?: boolean }) =>
  apiClient.post<ConflictCheckResponse>('/verification/conflicts', body).then((r) => r.data);

// ── Pending Reviews ──────────────────────────────────────────────────────

export interface PendingReviewItem {
  review_id: string;
  conflict: ConflictEntry;
  submitted_at: string;
  status: string;
}

export interface PendingReviewListResponse {
  total: number;
  items: PendingReviewItem[];
}

export const getPendingReviews = (status = 'pending') =>
  apiClient.get<PendingReviewListResponse>('/verification/reviews', { params: { status } }).then((r) => r.data);

export const decideReview = (body: { review_id: string; decision: 'approve' | 'reject'; notes?: string }) =>
  apiClient.post<{ review_id: string; status: string; notes: string | null }>('/verification/reviews/decide', body).then((r) => r.data);

// ── Chat / QA ────────────────────────────────────────────────────────────

export type RetrievalChannel =
  | 'cypher'
  | 'vector_entity'
  | 'vector_relationship'
  | 'bm25';

export interface ChatSource {
  kind: 'entity' | 'relationship' | 'chunk';
  id: string;
  label: string;
  score_vector?: number | null;
  score_bm25?: number | null;
  score_cypher?: number | null;
  score_rrf: number;
  score_rerank?: number | null;
  final_confidence: number;
  source_url?: string | null;
  source_doc_id?: string | null;
  source_chunk_id?: string | null;
  source_chunk_ids: string[];
  chunk_preview?: string | null;
  /** The entity/relationship's own description from the graph node.
   *  Often the only prose for Open-Targets-ingested entities — shown as
   *  a fallback when there's no hydrated chunk preview. */
  description?: string | null;
  contributing_channels: RetrievalChannel[];
  reasoning: string;
}

export interface ChannelTrace {
  channel: RetrievalChannel;
  hits: number;
  latency_ms: number;
  error?: string | null;
}

export interface RetrievalTrace {
  query: string;
  extracted_terms: string[];
  channels: ChannelTrace[];
  rrf_top_n: number;
  final_top_k: number;
  hydrated_chunks: number;
  total_latency_ms: number;
}

/** Compact view of which memory layers fed this turn (§5 of RAG_QA_PLAN.md).
 *  v1 surfaces only the trace counts — the actual rendered block stays on the
 *  server. Frontend may render this in a debug pane in a follow-up iteration. */
export interface MemoryTrace {
  working_turns: number;
  summary_chars: number;
  episodic_hit?: { turn_id: string; score: number } | null;
  summary_regenerated: boolean;
}

/** One read-only or mutating tool call the LLM made during a turn (P9/P10). */
export interface ToolCall {
  tool: string;
  args: Record<string, unknown>;
  result?: Record<string, unknown> | null;
  error?: string | null;
  latency_ms: number;
  tool_call_id?: string | null;
}

/** Per-claim faithfulness verdict from the post-generation check (P8). */
export interface ClaimVerification {
  claim_text: string;
  source_indices: number[];
  score: number;
  verdict: 'supported' | 'borderline' | 'unsupported' | string;
  matched_chunk?: string | null;
}

export interface FaithfulnessResult {
  overall_score?: number | null;
  claims: ClaimVerification[];
  failed_claims: number;
}

export interface AskResponse {
  session_id: string;
  turn_id: string;
  answer: string;
  sources: ChatSource[];
  cited_source_indices: number[];
  retrieval_trace: RetrievalTrace;
  memory_trace?: MemoryTrace | null;
  request_id?: string | null;
  latency_ms: number;
  /** Tool calls the LLM made during this turn (P9 read / P10 write).
   *  Empty when enable_tools + enable_mutations are both off. */
  tool_calls?: ToolCall[];
  /** Per-claim verdicts + aggregate score (P8). */
  faithfulness?: FaithfulnessResult | null;
}

/** Compact retrieval snapshot persisted on a turn's metadata so a
 *  reopened session re-renders the same source cards + trace pane as a
 *  live ask. Written by QAService._append_turn → _snapshot_sources. */
export interface RetrievalSnapshot {
  sources: ChatSource[];
  cited_source_indices: number[];
  retrieval_trace?: RetrievalTrace;
}

export interface ChatTurn {
  id: string;
  session_id: string;
  idx: number;
  user_query: string;
  llm_answer: string;
  request_id?: string | null;
  cited_entity_ids: string[];
  cited_relationship_ids: string[];
  cited_chunk_ids: string[];
  feedback_rating?: number | null;
  feedback_comment?: string | null;
  prompt_tokens: number;
  completion_tokens: number;
  latency_ms: number;
  created_at: string;
  metadata?: { retrieval_snapshot?: RetrievalSnapshot } & Record<string, unknown>;
}

export interface ChatSession {
  id: string;
  user_id?: string | null;
  title?: string | null;
  summary: string;
  turn_count: number;
  created_at: string;
  last_active_at: string;
}

export interface AskRequestBody {
  query: string;
  session_id?: string;
  user_id?: string | null;
  top_k?: number;
  enable_tools?: boolean;
  enable_mutations?: boolean;
  model?: string | null;
}

export const askQuestion = (body: AskRequestBody) =>
  apiClient.post<AskResponse>('/qa/ask', body).then((r) => r.data);

// ── Streaming /qa/ask (SSE, P11) ─────────────────────────────────────────
//
// `/qa/ask/stream` is a POST with a JSON body, so `EventSource` (GET-only,
// can't set headers) is out — we drive it with `fetch()` + a manual SSE
// frame parser over the response body's ReadableStream. The auth headers
// are constructed by hand here to mirror the axios `apiClient` interceptor
// above (X-API-Key from env, X-User-Id from localStorage); keep them in
// sync if either changes.

/** Payload of the `retrieval` SSE event — sources + traces, emitted once
 *  retrieval + memory complete, *before* the first answer `delta`. */
export interface AskStreamRetrieval {
  sources: ChatSource[];
  retrieval_trace: RetrievalTrace;
  memory_trace?: MemoryTrace | null;
  intent?: string | null;
}

/** Payload of the terminal `done` SSE event. */
export interface AskStreamDone {
  session_id: string;
  turn_id: string;
  answer: string;
  cited_source_indices: number[];
  faithfulness?: FaithfulnessResult | null;
  tool_calls?: ToolCall[];
  request_id?: string | null;
  latency_ms: number;
}

export interface AskStreamHandlers {
  /** Coarse progress signal: "retrieving" | "tools" | "generating". */
  onPhase?(phase: string, requestId?: string): void;
  /** Sources + retrieval/memory traces — fires once, before any delta. */
  onRetrieval?(d: AskStreamRetrieval): void;
  /** One read-only or mutating tool call (only when tools are enabled). */
  onToolCall?(call: ToolCall): void;
  /** A chunk of answer text — append to the running answer. May fire many
   *  times; with tool-use enabled, fires once with the full final answer. */
  onDelta?(text: string): void;
  /** Stream finished cleanly. */
  onDone?(d: AskStreamDone): void;
  /** Stream failed (or never started). `kind` ∈ session_not_found |
   *  retrieval_failed | llm_failed | internal_error | http_error | network. */
  onError?(message: string, kind: string): void;
}

/** One parsed SSE frame. */
interface SseEvent {
  event: string;
  data: string;
}

/** Parse a (possibly partial) chunk of SSE wire text. Returns the complete
 *  events found plus the leftover incomplete tail to prepend to the next
 *  chunk. Skips comment lines (`: ping …` heartbeats from sse-starlette).
 *
 *  Exported for unit testing — the streaming client uses it internally.
 */
export function parseSseChunk(buffer: string): { events: SseEvent[]; rest: string } {
  // Normalise CRLF so frame splitting on "\n\n" works regardless of how
  // the server (or an intermediary) terminates lines.
  buffer = buffer.replace(/\r\n/g, '\n');
  const events: SseEvent[] = [];
  let idx: number;
  while ((idx = buffer.indexOf('\n\n')) !== -1) {
    const frame = buffer.slice(0, idx);
    buffer = buffer.slice(idx + 2);
    const parsed = parseSseFrame(frame);
    if (parsed) events.push(parsed);
  }
  return { events, rest: buffer };
}

function parseSseFrame(frame: string): SseEvent | null {
  let event = 'message';
  const dataLines: string[] = [];
  for (const line of frame.split('\n')) {
    // Blank line (shouldn't happen post-split) or comment / heartbeat.
    if (!line || line.startsWith(':')) continue;
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') event = value;
    else if (field === 'data') dataLines.push(value);
  }
  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join('\n') };
}

function dispatchSseEvent(ev: SseEvent, h: AskStreamHandlers): boolean {
  let data: any;
  try {
    data = JSON.parse(ev.data);
  } catch {
    return false; // malformed frame — ignore, keep reading
  }
  switch (ev.event) {
    case 'phase':
      h.onPhase?.(data.phase, data.request_id);
      return false;
    case 'retrieval':
      h.onRetrieval?.({
        sources: data.sources ?? [],
        retrieval_trace: data.retrieval_trace,
        memory_trace: data.memory_trace ?? null,
        intent: data.intent ?? null,
      });
      return false;
    case 'tool_call':
      h.onToolCall?.(data as ToolCall);
      return false;
    case 'delta':
      if (typeof data.text === 'string' && data.text) h.onDelta?.(data.text);
      return false;
    case 'done':
      h.onDone?.(data as AskStreamDone);
      return true;
    case 'error':
      h.onError?.(
        typeof data.message === 'string' ? data.message : 'Stream error',
        typeof data.kind === 'string' ? data.kind : 'unknown',
      );
      return true;
    default:
      return false;
  }
}

/** Open a streaming `/qa/ask/stream` request. Returns an `AbortController`
 *  the caller can use to cancel the in-flight stream (e.g. on unmount or a
 *  new submit). All callbacks are optional; `onError` is invoked for both
 *  transport failures and server-emitted `error` events. */
export function askQuestionStream(
  body: AskRequestBody,
  handlers: AskStreamHandlers,
): AbortController {
  const controller = new AbortController();
  void runAskStream(body, handlers, controller.signal);
  return controller;
}

async function runAskStream(
  body: AskRequestBody,
  handlers: AskStreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (API_KEY) headers['X-API-Key'] = API_KEY;
    const identity = getStoredIdentity();
    if (identity) headers['X-User-Id'] = identity.id;
    res = await fetch(`${BASE_URL}/qa/ask/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal,
    });
  } catch (err: any) {
    if (signal.aborted) return;
    handlers.onError?.(err?.message ?? 'Network error', 'network');
    return;
  }

  if (!res.ok || !res.body) {
    // FastAPI returns { detail: ... } on validation / auth errors. Reuse
    // the same normaliser the axios path uses so the message is readable.
    let msg = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      msg = formatApiError({ response: { data } }, msg);
    } catch {
      /* non-JSON body — keep the status-code fallback */
    }
    handlers.onError?.(msg, 'http_error');
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let closed = false;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const { events, rest } = parseSseChunk(buffer);
      buffer = rest;
      for (const ev of events) {
        if (dispatchSseEvent(ev, handlers)) {
          closed = true;
          break;
        }
      }
      if (closed) break;
    }
    // Flush any trailing frame the server sent without a final blank line.
    if (!closed && buffer.trim()) {
      const { events } = parseSseChunk(buffer + '\n\n');
      for (const ev of events) {
        if (dispatchSseEvent(ev, handlers)) {
          closed = true;
          break;
        }
      }
    }
    if (!closed) {
      // Stream ended without a `done`/`error` event — surface it so the
      // caller stops the spinner instead of hanging forever.
      handlers.onError?.('Stream ended unexpectedly', 'incomplete');
    }
  } catch (err: any) {
    if (signal.aborted) return; // caller cancelled — not an error
    handlers.onError?.(err?.message ?? 'Stream read failed', 'network');
  } finally {
    try {
      await reader.cancel();
    } catch {
      /* already closed */
    }
  }
}

export const getChatSession = (sessionId: string) =>
  apiClient
    .get<{ session: ChatSession; turns: ChatTurn[] }>(`/qa/sessions/${sessionId}`)
    .then((r) => r.data);

export const listChatSessions = (params?: { user_id?: string | null; limit?: number; offset?: number }) =>
  apiClient
    .get<{ sessions: ChatSession[] }>('/qa/sessions', { params })
    .then((r) => r.data);

export const deleteChatSession = (sessionId: string) =>
  apiClient.delete(`/qa/sessions/${sessionId}`).then((r) => r.status === 204);

export const sendChatFeedback = (
  turnId: string,
  body: { rating: -1 | 0 | 1; comment?: string },
) =>
  apiClient
    .post<{ turn_id: string; accepted: boolean }>(`/qa/turns/${turnId}/feedback`, body)
    .then((r) => r.data);

// ── Chatbot users (lightweight browser identity, §14.1) ──────────────────

export interface ChatUser {
  id: string;
  display_name: string;
  metadata: Record<string, unknown>;
  created_at: string;
  last_seen_at: string;
}

export const registerChatUser = (body: { display_name: string }) =>
  apiClient.post<ChatUser>('/users', body).then((r) => r.data);

export const getChatUser = (userId: string) =>
  apiClient.get<ChatUser>(`/users/${userId}`).then((r) => r.data);

export const updateChatUser = (
  userId: string,
  body: { display_name?: string; metadata?: Record<string, unknown> },
) => apiClient.patch<ChatUser>(`/users/${userId}`, body).then((r) => r.data);

// ── Chatbot mutation proposals (P10) ─────────────────────────────────────

export interface ProposedMutation {
  id: string;
  tool: string;
  args: Record<string, unknown>;
  proposer_user_id?: string | null;
  status: 'pending' | 'approved' | 'rejected';
  created_at: string;
  decided_at?: string | null;
  applied_target_id?: string | null;
  error?: string | null;
  notes?: string | null;
}

export const listProposals = (params?: { status?: string; limit?: number }) =>
  apiClient
    .get<{ total: number; items: ProposedMutation[] }>('/qa/proposals', { params })
    .then((r) => r.data);

export const applyProposal = (proposalId: string, body?: { notes?: string }) =>
  apiClient
    .post<{ proposal_id: string; status: string; target_id?: string | null; error?: string | null }>(
      `/qa/proposals/${proposalId}/apply`,
      body ?? {},
    )
    .then((r) => r.data);

export const rejectProposal = (proposalId: string, body?: { notes?: string }) =>
  apiClient
    .post<{ proposal_id: string; status: string }>(
      `/qa/proposals/${proposalId}/reject`,
      body ?? {},
    )
    .then((r) => r.data);
