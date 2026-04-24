import axios from 'axios';

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8001';
const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? '';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: API_KEY ? { 'X-API-Key': API_KEY } : {},
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
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d: any) => {
        if (typeof d === 'string') return d;
        const where = Array.isArray(d?.loc) ? d.loc.slice(1).join('.') : null;
        const msg = d?.msg ?? 'Validation error';
        return where ? `${where}: ${msg}` : msg;
      })
      .join('; ');
  }
  if (detail && typeof detail === 'object') {
    return detail.msg ?? JSON.stringify(detail);
  }
  if (typeof err?.message === 'string' && err.message) return err.message;
  return fallback;
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

export const ingestOpenTargets = (body: { disease_id: string; max_associations?: number; min_association_score?: number; tag?: string }) =>
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

export const getCurationQueue = (params?: { status?: string; limit?: number; offset?: number }) =>
  apiClient.get<{ total: number; items: CurationQueueItem[] }>('/curation/queue', { params }).then((r) => r.data);

export const submitCurationEvents = (events: CurationEvent[]) =>
  apiClient.post('/curation/events', { events }).then((r) => r.data);

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
