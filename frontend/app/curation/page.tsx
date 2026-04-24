'use client';

import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  ClipboardCheck,
  ExternalLink,
  FileText,
  GitBranch,
  Layers,
  Loader2,
  Network,
  RefreshCw,
  ShieldAlert,
  Tag,
  XCircle,
} from 'lucide-react';
import {
  CurationQueueItem,
  formatApiError,
  getCurationQueue,
  submitCurationEvents,
} from '@/lib/api';

/* ─────────────────────────────────────────────────────────────────
   Constants — kept in sync with the Graph page palette so the
   coloured dots mean the same thing across the app.
   ───────────────────────────────────────────────────────────────── */

const TYPE_COLORS: Record<string, string> = {
  DISEASE:  '#d5212c',
  GENE:     '#1d4ed8',
  PROTEIN:  '#0e7490',
  DRUG:     '#f59e0b',
  PATHWAY:  '#7c3aed',
  COMPOUND: '#475569',
  CONCEPT:  '#0891b2',
  ORGANISM: '#15803d',
};
const entityColor = (t?: string | null) =>
  TYPE_COLORS[(t ?? '').toUpperCase()] ?? '#94a3b8';

const STATUS_TONE: Record<string, { bg: string; fg: string; border: string; label: string }> = {
  rejected:   { bg: '#fef2f2', fg: '#991b1b', border: 'rgba(153,27,27,0.30)', label: 'Rejected' },
  flagged:    { bg: '#fffbeb', fg: '#b45309', border: 'rgba(180,83,9,0.30)',  label: 'Flagged'  },
  unverified: { bg: '#eff6ff', fg: '#1d4ed8', border: 'rgba(29,78,216,0.25)', label: 'Unverified' },
};

const STATUS_HELP: Record<string, string> = {
  rejected:   'A verifier or conflict-detection step found this conflicts with trusted data.',
  flagged:    'Verifier confidence is low — please double-check sources before approving.',
  unverified: 'Newly extracted by the LLM. No verifier has weighed in yet.',
};

/* ─────────────────────────────────────────────────────────────────
   Helpers
   ───────────────────────────────────────────────────────────────── */

function StatusBadge({ status }: { status: string }) {
  const tone = STATUS_TONE[status] ?? {
    bg: 'var(--bg-muted)',
    fg: 'var(--text-secondary)',
    border: 'var(--border-default)',
    label: status,
  };
  return (
    <span
      className="badge"
      style={{
        background: tone.bg,
        color: tone.fg,
        border: `1px solid ${tone.border}`,
      }}
    >
      {tone.label}
    </span>
  );
}

function TypeChip({ type }: { type?: string | null }) {
  if (!type) return null;
  const color = entityColor(type);
  return (
    <span
      className="badge"
      style={{
        background: `${color}15`,
        color,
        border: `1px solid ${color}40`,
      }}
    >
      {type}
    </span>
  );
}

function relativeTime(iso?: string | null) {
  if (!iso) return null;
  const d = new Date(iso);
  const ms = Date.now() - d.getTime();
  if (ms < 60_000) return 'just now';
  if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m ago`;
  if (ms < 86_400_000) return `${Math.floor(ms / 3_600_000)}h ago`;
  return `${Math.floor(ms / 86_400_000)}d ago`;
}

/* ─────────────────────────────────────────────────────────────────
   Page
   ───────────────────────────────────────────────────────────────── */

export default function CurationPage() {
  const qc = useQueryClient();
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [typeFilter, setTypeFilter] = useState<string>('');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const { data, isLoading, isFetching, refetch } = useQuery({
    queryKey: ['curation-queue', statusFilter, typeFilter],
    queryFn: () =>
      getCurationQueue({
        status: statusFilter || undefined,
        // Backend accepts ?type=entity|relationship via the alias.
        ...(typeFilter ? ({ type: typeFilter } as any) : {}),
        limit: 200,
      }),
    refetchInterval: 8000,
  });

  // Counts per bucket for the filter chips
  const counts = useMemo(() => {
    const all = data?.items ?? [];
    return {
      total: all.length,
      rejected: all.filter((i) => i.verification_status === 'rejected').length,
      flagged: all.filter((i) => i.verification_status === 'flagged').length,
      unverified: all.filter((i) => i.verification_status === 'unverified').length,
    };
  }, [data]);

  // Auto-select first item when the list loads or filter changes
  useEffect(() => {
    if (!data?.items?.length) {
      setSelectedId(null);
      return;
    }
    if (!data.items.find((i) => i.id === selectedId)) {
      setSelectedId(data.items[0].id);
    }
  }, [data, selectedId]);

  const selected = data?.items.find((i) => i.id === selectedId) ?? null;

  const mutation = useMutation({
    mutationFn: submitCurationEvents,
    onMutate: () => setActionError(null),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['curation-queue'] }),
    onError: (err: any) =>
      setActionError(formatApiError(err, 'Could not record curation event')),
  });

  function act(item: CurationQueueItem, action: 'approve' | 'reject') {
    const event =
      item.type === 'entity'
        ? { entity_id: item.id, action }
        : { relationship_id: item.id, action };
    mutation.mutate([event as any]);
  }

  /* ── Filters bar ───────────────────────────────────────────────── */
  const statusFilters = [
    { value: '',           label: 'All',        count: counts.total,      tone: 'var(--accent)'  },
    { value: 'rejected',   label: 'Rejected',   count: counts.rejected,   tone: '#991b1b' },
    { value: 'flagged',    label: 'Flagged',    count: counts.flagged,    tone: '#b45309' },
    { value: 'unverified', label: 'Unverified', count: counts.unverified, tone: '#1d4ed8' },
  ];

  return (
    <div className="space-y-6">
      <header className="flex items-end justify-between flex-wrap gap-3">
        <div className="space-y-2">
          <span
            className="badge badge-brand inline-flex"
            style={{ background: 'var(--accent-soft)', border: '1px solid rgba(213,33,44,0.25)' }}
          >
            <ClipboardCheck size={10} className="opacity-70" />
            Human-in-the-loop review
          </span>
          <h1 className="page-title">Curation Queue</h1>
          <p className="page-desc">
            Review extracted items before they're promoted to trusted graph
            content. Pick a row on the left to see full evidence on the right.
          </p>
        </div>
        <button
          onClick={() => refetch()}
          className="btn-ghost"
          disabled={isFetching}
        >
          <RefreshCw size={13} className={isFetching ? 'animate-spin' : ''} />
          Refresh
        </button>
      </header>

      {/* ── Filter chips ─────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-2">
        {statusFilters.map((f) => {
          const active = statusFilter === f.value;
          return (
            <button
              key={f.value || 'all'}
              onClick={() => setStatusFilter(f.value)}
              className="text-[12px] font-semibold px-3 py-1.5 rounded-full transition-all flex items-center gap-2"
              style={{
                background: active ? `${f.tone}15` : 'var(--bg-card)',
                color: active ? f.tone : 'var(--text-secondary)',
                border: `1px solid ${active ? `${f.tone}45` : 'var(--border-default)'}`,
              }}
            >
              {f.label}
              <span
                className="text-[10.5px] tabular-nums px-1.5 py-0.5 rounded-full"
                style={{
                  background: active ? `${f.tone}25` : 'var(--bg-muted)',
                  color: active ? f.tone : 'var(--text-muted)',
                }}
              >
                {f.count}
              </span>
            </button>
          );
        })}
        <span className="mx-2 h-4 w-px" style={{ background: 'var(--border-default)' }} />
        {[
          { value: '', label: 'Both', icon: Layers },
          { value: 'entity', label: 'Entities', icon: Network },
          { value: 'relationship', label: 'Relationships', icon: GitBranch },
        ].map((t) => {
          const active = typeFilter === t.value;
          const Icon = t.icon;
          return (
            <button
              key={t.value || 'both'}
              onClick={() => setTypeFilter(t.value)}
              className="text-[12px] font-semibold px-3 py-1.5 rounded-full transition-all flex items-center gap-1.5"
              style={{
                background: active ? 'var(--accent-muted)' : 'var(--bg-card)',
                color: active ? 'var(--accent)' : 'var(--text-secondary)',
                border: `1px solid ${active ? 'rgba(213,33,44,0.45)' : 'var(--border-default)'}`,
              }}
            >
              <Icon size={11} />
              {t.label}
            </button>
          );
        })}
      </div>

      {/* ── Master-detail ────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        {/* Left: list */}
        <div className="lg:col-span-2 card overflow-hidden flex flex-col" style={{ maxHeight: 720 }}>
          <div
            className="px-4 py-3 border-b flex items-center justify-between"
            style={{ borderColor: 'var(--border-subtle)' }}
          >
            <p className="field-label !mb-0">Queue</p>
            <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
              {data?.items.length ?? 0} items
            </span>
          </div>
          <div className="overflow-y-auto flex-1">
            {isLoading ? (
              <div className="p-6 text-sm flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
                <Loader2 size={14} className="animate-spin" /> Loading queue
              </div>
            ) : !data?.items.length ? (
              <div className="empty-state">
                <span className="empty-icon">
                  <CheckCircle2 size={20} />
                </span>
                <p className="text-[12.5px] font-medium" style={{ color: 'var(--text-secondary)' }}>
                  Queue is empty — nothing to review
                </p>
                <p className="text-[10.5px]">
                  Items appear here when a verifier marks them as
                  <StatusBadge status="flagged" /> or <StatusBadge status="rejected" />.
                  Run a Process or Ingest job to populate the queue with
                  <StatusBadge status="unverified" /> items.
                </p>
              </div>
            ) : (
              <ul className="divide-y" style={{ borderColor: 'var(--border-subtle)' }}>
                {data.items.map((item) => (
                  <QueueRow
                    key={item.id}
                    item={item}
                    selected={item.id === selectedId}
                    onClick={() => setSelectedId(item.id)}
                  />
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* Right: detail */}
        <div className="lg:col-span-3">
          {selected ? (
            <DetailPanel
              item={selected}
              onApprove={() => act(selected, 'approve')}
              onReject={() => act(selected, 'reject')}
              acting={mutation.isPending}
              actionError={actionError}
            />
          ) : (
            <div className="card p-8">
              <div className="empty-state">
                <span className="empty-icon">
                  <ClipboardCheck size={20} />
                </span>
                <p className="text-[12.5px] font-medium" style={{ color: 'var(--text-secondary)' }}>
                  Pick an item on the left to review
                </p>
                <p className="text-[10.5px]">
                  Full description, source counts, and approve/reject actions appear here.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────
   QueueRow — compact card with all the at-a-glance info
   ───────────────────────────────────────────────────────────────── */
function QueueRow({
  item,
  selected,
  onClick,
}: {
  item: CurationQueueItem;
  selected: boolean;
  onClick: () => void;
}) {
  const isEntity = item.type === 'entity';
  const tone = STATUS_TONE[item.verification_status];
  const headColor = isEntity
    ? entityColor(item.entity_type)
    : entityColor(item.source_entity_type);

  return (
    <li>
      <button
        onClick={onClick}
        className="w-full text-left px-4 py-3 transition-all flex flex-col gap-1.5"
        style={{
          background: selected ? 'var(--accent-muted)' : 'transparent',
          borderLeft: `3px solid ${selected ? 'var(--accent)' : 'transparent'}`,
        }}
      >
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <span
              className="h-2 w-2 rounded-full shrink-0"
              style={{ background: headColor }}
            />
            {isEntity ? (
              <span
                className="text-[13px] font-semibold truncate"
                style={{ color: 'var(--text-primary)' }}
                title={item.name}
              >
                {item.name}
              </span>
            ) : (
              <span
                className="text-[13px] font-semibold truncate flex items-center gap-1.5"
                style={{ color: 'var(--text-primary)' }}
              >
                <span className="truncate">{item.source_entity_name ?? '?'}</span>
                <ArrowRight size={11} className="shrink-0" style={{ color: 'var(--text-muted)' }} />
                <span className="truncate">{item.target_entity_name ?? '?'}</span>
              </span>
            )}
          </div>
          {tone && <StatusBadge status={item.verification_status} />}
        </div>
        <div className="flex items-center gap-1.5 text-[10.5px]" style={{ color: 'var(--text-muted)' }}>
          {isEntity ? (
            <>
              <Network size={10} />
              <span>{item.entity_type}</span>
            </>
          ) : (
            <>
              <GitBranch size={10} />
              <span className="font-mono opacity-80">{item.relationship_type}</span>
            </>
          )}
          <span className="opacity-50">·</span>
          <FileText size={10} />
          <span>{item.source_document_count} doc{item.source_document_count === 1 ? '' : 's'}</span>
          {item.created_at && (
            <>
              <span className="opacity-50">·</span>
              <span>{relativeTime(item.created_at)}</span>
            </>
          )}
        </div>
        {item.description && (
          <p
            className="text-[11.5px] line-clamp-2 leading-snug"
            style={{ color: 'var(--text-secondary)' }}
          >
            {item.description}
          </p>
        )}
      </button>
    </li>
  );
}

/* ─────────────────────────────────────────────────────────────────
   DetailPanel — the full review surface for the selected item
   ───────────────────────────────────────────────────────────────── */
function DetailPanel({
  item,
  onApprove,
  onReject,
  acting,
  actionError,
}: {
  item: CurationQueueItem;
  onApprove: () => void;
  onReject: () => void;
  acting: boolean;
  actionError: string | null;
}) {
  const isEntity = item.type === 'entity';
  const statusHelp = STATUS_HELP[item.verification_status];

  return (
    <div className="card p-6 space-y-5 fade-up">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p
            className="text-[10px] uppercase tracking-wider font-semibold mb-1"
            style={{ color: 'var(--text-muted)' }}
          >
            {isEntity ? 'Entity under review' : 'Relationship under review'}
          </p>
          {isEntity ? (
            <h2
              className="text-[20px] font-semibold tracking-tight"
              style={{ color: 'var(--text-primary)' }}
            >
              {item.name}
            </h2>
          ) : (
            <RelationshipTitle item={item} />
          )}
        </div>
        <StatusBadge status={item.verification_status} />
      </div>

      {/* Type / strength chips */}
      <div className="flex flex-wrap items-center gap-2">
        {isEntity && <TypeChip type={item.entity_type} />}
        {!isEntity && (
          <span
            className="badge"
            style={{
              background: 'var(--accent-soft)',
              color: 'var(--accent)',
              border: '1px solid rgba(213,33,44,0.30)',
            }}
          >
            {item.relationship_type}
          </span>
        )}
        {!isEntity && typeof item.strength === 'number' && (
          <span
            className="badge badge-neutral"
            title="Extractor confidence at the time the relationship was created"
          >
            strength {item.strength.toFixed(2)}
          </span>
        )}
        {item.source_trust && (
          <span className="badge badge-neutral" title="Trust level of the source the item came from">
            trust: {item.source_trust}
          </span>
        )}
      </div>

      {/* Why-it's-here explainer */}
      {statusHelp && (
        <div
          className="rounded-lg p-3 flex items-start gap-2.5 text-[12.5px]"
          style={{
            background: STATUS_TONE[item.verification_status]?.bg ?? 'var(--bg-muted)',
            border: `1px solid ${STATUS_TONE[item.verification_status]?.border ?? 'var(--border-default)'}`,
          }}
        >
          <ShieldAlert
            size={14}
            className="shrink-0 mt-0.5"
            style={{ color: STATUS_TONE[item.verification_status]?.fg }}
          />
          <div>
            <p
              className="font-semibold leading-tight"
              style={{ color: STATUS_TONE[item.verification_status]?.fg }}
            >
              Why this item is in the queue
            </p>
            <p className="mt-1" style={{ color: 'var(--text-secondary)' }}>
              {statusHelp}
            </p>
            {item.notes && (
              <p
                className="mt-1.5 italic"
                style={{ color: 'var(--text-secondary)' }}
              >
                Verifier note: {item.notes}
              </p>
            )}
          </div>
        </div>
      )}

      {/* Description */}
      {item.description && (
        <div>
          <p
            className="text-[10px] uppercase tracking-wider font-semibold mb-1.5"
            style={{ color: 'var(--text-muted)' }}
          >
            Description
          </p>
          <p
            className="text-[13px] leading-relaxed"
            style={{ color: 'var(--text-primary)' }}
          >
            {item.description}
          </p>
        </div>
      )}

      {/* Tags (entity only) */}
      {isEntity && item.tags && item.tags.length > 0 && (
        <div>
          <p
            className="text-[10px] uppercase tracking-wider font-semibold mb-1.5 flex items-center gap-1.5"
            style={{ color: 'var(--text-muted)' }}
          >
            <Tag size={10} /> Tags
          </p>
          <div className="flex flex-wrap gap-1.5">
            {item.tags.map((t) => (
              <span key={t} className="badge badge-neutral">
                {t}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Provenance */}
      <div className="grid grid-cols-3 gap-3">
        <ProvenanceCell
          icon={FileText}
          label="Source documents"
          value={item.source_document_count}
        />
        <ProvenanceCell
          icon={Layers}
          label="Source chunks"
          value={item.source_chunk_count}
        />
        <ProvenanceCell
          icon={ChevronRight}
          label="Created"
          value={relativeTime(item.created_at) ?? '—'}
        />
      </div>

      {/* Identifier (collapsed footer) */}
      <details className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
        <summary className="cursor-pointer select-none font-semibold uppercase tracking-wider">
          Technical details
        </summary>
        <dl className="mt-2 space-y-1 font-mono">
          <div className="flex gap-2">
            <dt className="opacity-60">id:</dt>
            <dd className="break-all">{item.id}</dd>
          </div>
          {!isEntity && (
            <>
              <div className="flex gap-2">
                <dt className="opacity-60">source_entity_id:</dt>
                <dd className="break-all">{item.source_entity_id}</dd>
              </div>
              <div className="flex gap-2">
                <dt className="opacity-60">target_entity_id:</dt>
                <dd className="break-all">{item.target_entity_id}</dd>
              </div>
            </>
          )}
        </dl>
      </details>

      {/* Action footer */}
      <div
        className="flex items-center gap-2 pt-4"
        style={{ borderTop: '1px solid var(--border-subtle)' }}
      >
        <button
          onClick={onApprove}
          disabled={acting}
          className="btn-primary"
          title="Mark this item as human-verified. It drops out of the queue."
        >
          {acting ? <Loader2 size={14} className="animate-spin" /> : <CheckCircle2 size={14} />}
          Approve
        </button>
        <button
          onClick={onReject}
          disabled={acting}
          className="btn-ghost"
          style={{
            color: 'var(--danger)',
            borderColor: 'rgba(153,27,27,0.30)',
          }}
          title="Soft-delete: keeps the item in the graph for audit but marks it as bad data."
        >
          <XCircle size={13} />
          Reject
        </button>
        {actionError && (
          <span className="text-[12px] ml-auto" style={{ color: 'var(--danger)' }}>
            {actionError}
          </span>
        )}
      </div>
    </div>
  );
}

function RelationshipTitle({ item }: { item: CurationQueueItem }) {
  const srcColor = entityColor(item.source_entity_type);
  const tgtColor = entityColor(item.target_entity_type);
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span
        className="text-[18px] font-semibold tracking-tight"
        style={{ color: 'var(--text-primary)' }}
      >
        {item.source_entity_name ?? <span style={{ color: 'var(--text-muted)' }}>?</span>}
      </span>
      {item.source_entity_type && (
        <span
          className="text-[10px] px-1.5 py-0.5 rounded font-semibold"
          style={{
            background: `${srcColor}15`,
            color: srcColor,
            border: `1px solid ${srcColor}40`,
          }}
        >
          {item.source_entity_type}
        </span>
      )}
      <ArrowRight size={16} style={{ color: 'var(--text-muted)' }} />
      <span
        className="text-[18px] font-semibold tracking-tight"
        style={{ color: 'var(--text-primary)' }}
      >
        {item.target_entity_name ?? <span style={{ color: 'var(--text-muted)' }}>?</span>}
      </span>
      {item.target_entity_type && (
        <span
          className="text-[10px] px-1.5 py-0.5 rounded font-semibold"
          style={{
            background: `${tgtColor}15`,
            color: tgtColor,
            border: `1px solid ${tgtColor}40`,
          }}
        >
          {item.target_entity_type}
        </span>
      )}
    </div>
  );
}

function ProvenanceCell({
  icon: Icon,
  label,
  value,
}: {
  icon: any;
  label: string;
  value: number | string;
}) {
  return (
    <div
      className="rounded-lg p-3"
      style={{ background: 'var(--bg-muted)' }}
    >
      <div
        className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider font-semibold"
        style={{ color: 'var(--text-muted)' }}
      >
        <Icon size={10} />
        {label}
      </div>
      <p
        className="text-[18px] font-semibold tabular-nums mt-1"
        style={{ color: 'var(--text-primary)' }}
      >
        {value}
      </p>
    </div>
  );
}
