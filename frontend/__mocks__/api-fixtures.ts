/**
 * Mock backend payloads + edge-case error shapes.
 *
 * Used by the contract check (frontend/scripts/contract-check.mjs) to
 * prove the frontend's helpers don't throw on any reasonable backend
 * response — both happy-path and the messy edges (empty arrays, missing
 * fields, FastAPI 422 validation arrays, network errors, plain strings).
 *
 * If a backend route ever changes shape, add the new shape here and the
 * CI build will catch any frontend code that can't handle it.
 */

import type {
  Entity,
  GraphStats,
  Job,
  JobSummary,
  PipelineMetrics,
  Relationship,
} from '@/lib/api';

// ──────────────────────────────────────────────────────────────────────
// Happy-path payloads
// ──────────────────────────────────────────────────────────────────────

export const mockGraphStats: GraphStats = {
  total_entities: 42,
  total_relationships: 17,
  entity_type_counts: { GENE: 12, DISEASE: 8, DRUG: 5 },
  relationship_type_counts: { RELATED_TO: 10, INFLUENCES: 7 },
};

export const mockEntity: Entity = {
  id: 'ent-001',
  name: 'BRCA1',
  entity_type: 'GENE',
  description: 'Breast cancer susceptibility gene 1',
  properties: {},
  tags: ['cancer'],
  source_trust: 'extracted',
  source_chunk_ids: ['chunk-1'],
  source_document_ids: ['doc-1'],
  created_at: '2026-04-23T16:00:00Z',
};

export const mockRelationship: Relationship = {
  id: 'rel-001',
  source_entity_id: 'ent-001',
  target_entity_id: 'ent-002',
  relationship_type: 'RELATED_TO',
  description: 'BRCA1 mutations associated with breast cancer',
  strength: 0.93,
  source_chunk_ids: ['chunk-1'],
  source_document_ids: ['doc-1'],
};

export const mockJob: Job = {
  job_id: 'job-001',
  kind: 'document',
  status: 'completed',
  message: 'Extracted 4 entities and 3 relationships from 1 chunks',
  progress: 1.0,
  stages: ['fetch', 'chunk', 'entities', 'relationships', 'finalize'],
  current_stage: 'finalize',
  stage_progress: {
    fetch: 'completed',
    chunk: 'completed',
    entities: 'completed',
    relationships: 'completed',
    finalize: 'completed',
  },
  events: [
    {
      ts: '2026-04-23T16:00:00Z',
      stage: 'fetch',
      level: 'info',
      message: 'Resolving content',
      data: {},
    },
  ],
  result: { entities_extracted: 4, relationships_extracted: 3 },
  cancel_requested: false,
  created_at: '2026-04-23T16:00:00Z',
  updated_at: '2026-04-23T16:00:30Z',
};

export const mockJobSummary: JobSummary = {
  job_id: 'job-001',
  kind: 'document',
  status: 'completed',
  message: 'Done',
  current_stage: 'finalize',
  progress: 1.0,
  created_at: '2026-04-23T16:00:00Z',
  updated_at: '2026-04-23T16:00:30Z',
};

export const mockMetrics: PipelineMetrics = {
  uptime_seconds: 1234,
  llm: {
    calls: 12,
    calls_by_type: { entity_extraction: 6, relationship_extraction: 6 },
    prompt_tokens: 4200,
    completion_tokens: 3100,
    total_tokens: 7300,
    avg_latency_ms: 850,
    cache_hits: 2,
    cache_hit_rate: 0.16,
  },
  embedding: { calls: 18, cache_hits: 4, cache_hit_rate: 0.22 },
  pipeline: {
    documents_processed: 3,
    chunks_processed: 6,
    entities_saved: 24,
    relationships_saved: 18,
  },
  cache_sizes: { dedup_entries: 12, embedding_entries: 18 },
};

// ──────────────────────────────────────────────────────────────────────
// Edge cases — payload shapes the frontend must not crash on
// ──────────────────────────────────────────────────────────────────────

/** Empty graph (cold-start) */
export const mockEmptyGraphStats: GraphStats = {
  total_entities: 0,
  total_relationships: 0,
  entity_type_counts: {},
  relationship_type_counts: {},
};

/** Entity with every optional field missing */
export const mockMinimalEntity: Entity = {
  id: 'ent-min',
  name: 'X',
  entity_type: 'CONCEPT',
  properties: {},
  tags: [],
  source_chunk_ids: [],
  source_document_ids: [],
};

/** Job that's still pending (no stages started) */
export const mockPendingJob: Job = {
  job_id: 'job-pending',
  kind: 'document',
  status: 'pending',
  progress: 0,
  stages: ['fetch', 'chunk', 'entities', 'relationships', 'finalize'],
  current_stage: null,
  stage_progress: {
    fetch: 'pending',
    chunk: 'pending',
    entities: 'pending',
    relationships: 'pending',
    finalize: 'pending',
  },
  events: [],
  cancel_requested: false,
  created_at: '2026-04-23T16:00:00Z',
  updated_at: '2026-04-23T16:00:00Z',
};

/** Failed job with error message */
export const mockFailedJob: Job = {
  ...mockPendingJob,
  job_id: 'job-failed',
  status: 'failed',
  progress: 0.4,
  message: 'OpenAI API key invalid',
  error: 'OpenAI API key invalid',
  events: [
    {
      ts: '2026-04-23T16:00:01Z',
      stage: 'entities',
      level: 'error',
      message: 'LLM call failed',
      data: {},
    },
  ],
};

// ──────────────────────────────────────────────────────────────────────
// Error shapes for formatApiError to swallow gracefully
// ──────────────────────────────────────────────────────────────────────

export const errorShapes = [
  { label: 'null', err: null },
  { label: 'undefined', err: undefined },
  { label: 'plain string', err: { message: 'Network error' } },
  {
    label: 'FastAPI 4xx with string detail',
    err: { response: { data: { detail: 'Disease not found' } } },
  },
  {
    label: 'FastAPI 422 validation array',
    err: {
      response: {
        data: {
          detail: [
            {
              type: 'string_type',
              loc: ['body', 'email'],
              msg: 'Input should be a valid string',
              input: null,
              ctx: {},
            },
            {
              type: 'missing',
              loc: ['body', 'query'],
              msg: 'Field required',
              input: {},
              ctx: {},
            },
          ],
        },
      },
    },
  },
  {
    label: 'detail as nested object',
    err: { response: { data: { detail: { msg: 'Custom error', code: 42 } } } },
  },
  { label: 'completely empty', err: {} },
  {
    label: 'detail is empty array',
    err: { response: { data: { detail: [] } } },
  },
  {
    label: 'detail is array of strings',
    err: { response: { data: { detail: ['err1', 'err2'] } } },
  },
];
