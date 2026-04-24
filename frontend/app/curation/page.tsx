'use client';

import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  ArrowRight,
  CheckCircle2,
  CheckSquare,
  ChevronRight,
  ClipboardCheck,
  Eye,
  EyeOff,
  FileText,
  GitBranch,
  History,
  Layers,
  Loader2,
  Network,
  Pencil,
  RefreshCw,
  ShieldAlert,
  Square,
  Tag,
  X,
  XCircle,
} from 'lucide-react';
import {
  apiClient,
  ChunkRecord,
  CurationAuditEntry,
  CurationEvent,
  CurationQueueItem,
  formatApiError,
  getChunks,
  getCurationAudit,
  getCurationQueue,
  getRelationshipTypes,
  submitCurationEvents,
} from '@/lib/api';

/* ─────────────────────────────────────────────────────────────────
   Constants — kept in sync with the Graph page palette
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
   Tiny presentational helpers
   ───────────────────────────────────────────────────────────────── */

function StatusBadge({ status }: { status: string }) {
  const tone = STATUS_TONE[status] ?? {
    bg: 'var(--bg-muted)', fg: 'var(--text-secondary)',
    border: 'var(--border-default)', label: status,
  };
  return (
    <span className="badge"
      style={{ background: tone.bg, color: tone.fg, border: `1px solid ${tone.border}` }}>
      {tone.label}
    </span>
  );
}

function TypeChip({ type }: { type?: string | null }) {
  if (!type) return null;
  const c = entityColor(type);
  return (
    <span className="badge" style={{ background: `${c}15`, color: c, border: `1px solid ${c}40` }}>
      {type}
    </span>
  );
}

function relativeTime(iso?: string | null) {
  if (!iso) return null;
  const ms = Date.now() - new Date(iso).getTime();
  if (ms < 60_000) return 'just now';
  if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m ago`;
  if (ms < 86_400_000) return `${Math.floor(ms / 3_600_000)}h ago`;
  return `${Math.floor(ms / 86_400_000)}d ago`;
}

function buildEvent(
  item: CurationQueueItem,
  action: 'approve' | 'reject' | 'correct',
  extras: Partial<CurationEvent> = {},
): CurationEvent {
  return item.type === 'entity'
    ? { entity_id: item.id, action, ...extras }
    : { relationship_id: item.id, action, ...extras };
}

/* ─────────────────────────────────────────────────────────────────
   Main page
   ───────────────────────────────────────────────────────────────── */

export default function CurationPage() {
  const qc = useQueryClient();
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [typeFilter, setTypeFilter] = useState<string>('');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [bulkIds, setBulkIds] = useState<Set<string>>(new Set());
  const [actionError, setActionError] = useState<string | null>(null);
  const [auditOpen, setAuditOpen] = useState(false);
  const [editing, setEditing] = useState(false);

  const { data, isLoading, isFetching, refetch } = useQuery({
    queryKey: ['curation-queue', statusFilter, typeFilter],
    queryFn: () =>
      getCurationQueue({
        status: statusFilter || undefined,
        ...(typeFilter ? ({ type: typeFilter } as any) : {}),
        limit: 200,
      }),
    refetchInterval: 8000,
  });

  // Filter-bucket counts
  const counts = useMemo(() => {
    const all = data?.items ?? [];
    return {
      total: all.length,
      rejected: all.filter((i) => i.verification_status === 'rejected').length,
      flagged: all.filter((i) => i.verification_status === 'flagged').length,
      unverified: all.filter((i) => i.verification_status === 'unverified').length,
    };
  }, [data]);

  // Auto-select on load / filter change
  useEffect(() => {
    if (!data?.items?.length) {
      setSelectedId(null);
      return;
    }
    if (!data.items.find((i) => i.id === selectedId)) {
      setSelectedId(data.items[0].id);
    }
  }, [data, selectedId]);

  // Drop selection IDs that no longer match the current filter
  useEffect(() => {
    if (!data?.items) return;
    const visible = new Set(data.items.map((i) => i.id));
    setBulkIds((prev) => {
      const next = new Set<string>();
      prev.forEach((id) => visible.has(id) && next.add(id));
      return next;
    });
  }, [data]);

  const selected = data?.items.find((i) => i.id === selectedId) ?? null;

  const mutation = useMutation({
    mutationFn: submitCurationEvents,
    onMutate: () => setActionError(null),
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ['curation-queue'] });
      qc.invalidateQueries({ queryKey: ['curation-audit'] });
      if (res.failed > 0) {
        setActionError(`${res.failed} of ${res.processed + res.failed} events failed: ${res.errors.join('; ')}`);
      }
      setEditing(false);
    },
    onError: (err: any) =>
      setActionError(formatApiError(err, 'Could not record curation event')),
  });

  function actOne(item: CurationQueueItem, action: 'approve' | 'reject') {
    mutation.mutate([buildEvent(item, action)]);
  }

  function actBulk(action: 'approve' | 'reject') {
    if (!data?.items?.length) return;
    const events = data.items
      .filter((i) => bulkIds.has(i.id))
      .map((i) => buildEvent(i, action, { curator_id: 'bulk' }));
    if (events.length === 0) return;
    mutation.mutate(events);
    setBulkIds(new Set());
  }

  function toggleSelect(id: string) {
    setBulkIds((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  function selectAllVisible() {
    if (!data?.items) return;
    if (bulkIds.size === data.items.length) {
      setBulkIds(new Set());
    } else {
      setBulkIds(new Set(data.items.map((i) => i.id)));
    }
  }

  function applyCorrection(corrections: Record<string, unknown>, reason: string) {
    if (!selected) return;
    mutation.mutate([buildEvent(selected, 'correct', { corrections, notes: reason })]);
  }

  /* ── Filter chips ─────────────────────────────────────────────── */
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
          <span className="badge badge-brand inline-flex"
            style={{ background: 'var(--accent-soft)', border: '1px solid rgba(213,33,44,0.25)' }}>
            <ClipboardCheck size={10} className="opacity-70" />
            Human-in-the-loop review
          </span>
          <h1 className="page-title">Curation Queue</h1>
          <p className="page-desc">
            Review extracted items before they're promoted to trusted graph
            content. Pick a row on the left to see full evidence on the right.
            Use the checkboxes to act on many at once.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => setAuditOpen(true)} className="btn-ghost">
            <History size={13} />
            Audit log
          </button>
          <button onClick={() => refetch()} className="btn-ghost" disabled={isFetching}>
            <RefreshCw size={13} className={isFetching ? 'animate-spin' : ''} />
            Refresh
          </button>
        </div>
      </header>

      {/* Filter chips */}
      <div className="flex flex-wrap items-center gap-2">
        {statusFilters.map((f) => {
          const active = statusFilter === f.value;
          return (
            <button key={f.value || 'all'} onClick={() => setStatusFilter(f.value)}
              className="text-[12px] font-semibold px-3 py-1.5 rounded-full transition-all flex items-center gap-2"
              style={{
                background: active ? `${f.tone}15` : 'var(--bg-card)',
                color: active ? f.tone : 'var(--text-secondary)',
                border: `1px solid ${active ? `${f.tone}45` : 'var(--border-default)'}`,
              }}>
              {f.label}
              <span className="text-[10.5px] tabular-nums px-1.5 py-0.5 rounded-full"
                style={{
                  background: active ? `${f.tone}25` : 'var(--bg-muted)',
                  color: active ? f.tone : 'var(--text-muted)',
                }}>
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
            <button key={t.value || 'both'} onClick={() => setTypeFilter(t.value)}
              className="text-[12px] font-semibold px-3 py-1.5 rounded-full transition-all flex items-center gap-1.5"
              style={{
                background: active ? 'var(--accent-muted)' : 'var(--bg-card)',
                color: active ? 'var(--accent)' : 'var(--text-secondary)',
                border: `1px solid ${active ? 'rgba(213,33,44,0.45)' : 'var(--border-default)'}`,
              }}>
              <Icon size={11} />
              {t.label}
            </button>
          );
        })}
      </div>

      {/* Bulk action bar — appears only when there's a selection */}
      {bulkIds.size > 0 && (
        <div className="card p-3 flex items-center justify-between gap-3 fade-up"
          style={{ background: 'var(--accent-muted)', borderColor: 'rgba(213,33,44,0.30)' }}>
          <div className="flex items-center gap-2 text-[13px] font-semibold" style={{ color: 'var(--accent-hover)' }}>
            <CheckSquare size={14} />
            {bulkIds.size} item{bulkIds.size === 1 ? '' : 's'} selected
            <button onClick={() => setBulkIds(new Set())}
              className="ml-2 text-[11px] font-medium underline opacity-70 hover:opacity-100">
              clear
            </button>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => actBulk('approve')} disabled={mutation.isPending} className="btn-success">
              {mutation.isPending ? <Loader2 size={16} className="animate-spin" /> : <CheckCircle2 size={16} strokeWidth={2.6} />}
              Approve {bulkIds.size}
            </button>
            <button onClick={() => actBulk('reject')} disabled={mutation.isPending} className="btn-danger">
              <XCircle size={16} strokeWidth={2.6} />
              Reject {bulkIds.size}
            </button>
          </div>
        </div>
      )}

      {/* Master-detail */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        {/* Left list */}
        <div className="lg:col-span-2 card overflow-hidden flex flex-col" style={{ maxHeight: 720 }}>
          <div className="px-4 py-3 border-b flex items-center justify-between"
            style={{ borderColor: 'var(--border-subtle)' }}>
            <button onClick={selectAllVisible}
              className="flex items-center gap-2 text-[12px] font-semibold transition-colors hover:text-[var(--accent)]"
              style={{ color: 'var(--text-secondary)' }}
              title={bulkIds.size === data?.items.length ? 'Clear selection' : 'Select all visible'}>
              {bulkIds.size > 0 && data?.items && bulkIds.size === data.items.length
                ? <CheckSquare size={14} style={{ color: 'var(--accent)' }} />
                : <Square size={14} />}
              {data?.items.length ?? 0} items
            </button>
            <p className="field-label !mb-0">Queue</p>
          </div>
          <div className="overflow-y-auto flex-1">
            {isLoading ? (
              <div className="p-6 text-sm flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
                <Loader2 size={14} className="animate-spin" /> Loading queue
              </div>
            ) : !data?.items.length ? (
              <div className="empty-state">
                <span className="empty-icon"><CheckCircle2 size={20} /></span>
                <p className="text-[12.5px] font-medium" style={{ color: 'var(--text-secondary)' }}>
                  Queue is empty — nothing to review
                </p>
                <p className="text-[10.5px]">
                  Items appear here when a verifier marks them as <StatusBadge status="flagged" />{' '}
                  or <StatusBadge status="rejected" />, or after a Process / Ingest run with{' '}
                  <StatusBadge status="unverified" /> defaults.
                </p>
              </div>
            ) : (
              <ul className="divide-y" style={{ borderColor: 'var(--border-subtle)' }}>
                {data.items.map((item) => (
                  <QueueRow
                    key={item.id}
                    item={item}
                    selected={item.id === selectedId}
                    checked={bulkIds.has(item.id)}
                    onSelect={() => setSelectedId(item.id)}
                    onToggle={() => toggleSelect(item.id)}
                  />
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* Right detail */}
        <div className="lg:col-span-3">
          {selected ? (
            <DetailPanel
              item={selected}
              onApprove={() => actOne(selected, 'approve')}
              onReject={() => actOne(selected, 'reject')}
              onEdit={() => setEditing(true)}
              acting={mutation.isPending}
              actionError={actionError}
            />
          ) : (
            <div className="card p-8">
              <div className="empty-state">
                <span className="empty-icon"><ClipboardCheck size={20} /></span>
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

      {/* Modals */}
      {editing && selected && (
        <CorrectModal
          item={selected}
          onClose={() => setEditing(false)}
          onSubmit={applyCorrection}
          submitting={mutation.isPending}
        />
      )}
      {auditOpen && <AuditDrawer onClose={() => setAuditOpen(false)} />}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────
   QueueRow — checkbox + at-a-glance card
   ───────────────────────────────────────────────────────────────── */
function QueueRow({
  item,
  selected,
  checked,
  onSelect,
  onToggle,
}: {
  item: CurationQueueItem;
  selected: boolean;
  checked: boolean;
  onSelect: () => void;
  onToggle: () => void;
}) {
  const isEntity = item.type === 'entity';
  const headColor = isEntity ? entityColor(item.entity_type) : entityColor(item.source_entity_type);
  return (
    <li>
      <div
        className="px-3 py-3 transition-all flex items-start gap-2"
        style={{
          background: selected ? 'var(--accent-muted)' : 'transparent',
          borderLeft: `3px solid ${selected ? 'var(--accent)' : 'transparent'}`,
        }}
      >
        <button
          onClick={onToggle}
          className="mt-0.5 shrink-0"
          title={checked ? 'Deselect' : 'Select for bulk action'}
        >
          {checked
            ? <CheckSquare size={15} style={{ color: 'var(--accent)' }} />
            : <Square size={15} style={{ color: 'var(--text-muted)' }} />}
        </button>
        <button
          onClick={onSelect}
          className="text-left flex-1 min-w-0 flex flex-col gap-1.5"
        >
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 min-w-0">
              <span className="h-2 w-2 rounded-full shrink-0" style={{ background: headColor }} />
              {isEntity ? (
                <span className="text-[13px] font-semibold truncate"
                  style={{ color: 'var(--text-primary)' }} title={item.name}>
                  {item.name}
                </span>
              ) : (
                <span className="text-[13px] font-semibold truncate flex items-center gap-1.5"
                  style={{ color: 'var(--text-primary)' }}>
                  <span className="truncate">{item.source_entity_name ?? '?'}</span>
                  <ArrowRight size={11} className="shrink-0" style={{ color: 'var(--text-muted)' }} />
                  <span className="truncate">{item.target_entity_name ?? '?'}</span>
                </span>
              )}
            </div>
            <StatusBadge status={item.verification_status} />
          </div>
          <div className="flex items-center gap-1.5 text-[10.5px]" style={{ color: 'var(--text-muted)' }}>
            {isEntity ? (
              <><Network size={10} /><span>{item.entity_type}</span></>
            ) : (
              <><GitBranch size={10} /><span className="font-mono opacity-80">{item.relationship_type}</span></>
            )}
            <span className="opacity-50">·</span>
            <FileText size={10} />
            <span>{item.source_document_count} doc{item.source_document_count === 1 ? '' : 's'}</span>
            {item.created_at && (<><span className="opacity-50">·</span><span>{relativeTime(item.created_at)}</span></>)}
          </div>
          {item.description && (
            <p className="text-[11.5px] line-clamp-2 leading-snug" style={{ color: 'var(--text-secondary)' }}>
              {item.description}
            </p>
          )}
        </button>
      </div>
    </li>
  );
}

/* ─────────────────────────────────────────────────────────────────
   DetailPanel — full review surface (with source-chunk reveal)
   ───────────────────────────────────────────────────────────────── */
function DetailPanel({
  item, onApprove, onReject, onEdit, acting, actionError,
}: {
  item: CurationQueueItem;
  onApprove: () => void;
  onReject: () => void;
  onEdit: () => void;
  acting: boolean;
  actionError: string | null;
}) {
  const isEntity = item.type === 'entity';
  const statusHelp = STATUS_HELP[item.verification_status];
  const [showChunks, setShowChunks] = useState(false);

  return (
    <div className="card p-6 space-y-5 fade-up">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-[10px] uppercase tracking-wider font-semibold mb-1"
            style={{ color: 'var(--text-muted)' }}>
            {isEntity ? 'Entity under review' : 'Relationship under review'}
          </p>
          {isEntity
            ? <h2 className="text-[20px] font-semibold tracking-tight" style={{ color: 'var(--text-primary)' }}>{item.name}</h2>
            : <RelationshipTitle item={item} />}
        </div>
        <StatusBadge status={item.verification_status} />
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {isEntity && <TypeChip type={item.entity_type} />}
        {!isEntity && (
          <span className="badge"
            style={{ background: 'var(--accent-soft)', color: 'var(--accent)', border: '1px solid rgba(213,33,44,0.30)' }}>
            {item.relationship_type}
          </span>
        )}
        {!isEntity && typeof item.strength === 'number' && (
          <span className="badge badge-neutral" title="Extractor confidence at extraction time">
            strength {item.strength.toFixed(2)}
          </span>
        )}
        {item.source_trust && (
          <span className="badge badge-neutral" title="Trust level of the source">trust: {item.source_trust}</span>
        )}
      </div>

      {statusHelp && (
        <div className="rounded-lg p-3 flex items-start gap-2.5 text-[12.5px]"
          style={{
            background: STATUS_TONE[item.verification_status]?.bg ?? 'var(--bg-muted)',
            border: `1px solid ${STATUS_TONE[item.verification_status]?.border ?? 'var(--border-default)'}`,
          }}>
          <ShieldAlert size={14} className="shrink-0 mt-0.5"
            style={{ color: STATUS_TONE[item.verification_status]?.fg }} />
          <div>
            <p className="font-semibold leading-tight" style={{ color: STATUS_TONE[item.verification_status]?.fg }}>
              Why this item is in the queue
            </p>
            <p className="mt-1" style={{ color: 'var(--text-secondary)' }}>{statusHelp}</p>
            {item.notes && <p className="mt-1.5 italic" style={{ color: 'var(--text-secondary)' }}>Verifier note: {item.notes}</p>}
          </div>
        </div>
      )}

      {item.description && (
        <div>
          <p className="text-[10px] uppercase tracking-wider font-semibold mb-1.5"
            style={{ color: 'var(--text-muted)' }}>Description</p>
          <p className="text-[13px] leading-relaxed" style={{ color: 'var(--text-primary)' }}>{item.description}</p>
        </div>
      )}

      {isEntity && item.tags && item.tags.length > 0 && (
        <div>
          <p className="text-[10px] uppercase tracking-wider font-semibold mb-1.5 flex items-center gap-1.5"
            style={{ color: 'var(--text-muted)' }}>
            <Tag size={10} /> Tags
          </p>
          <div className="flex flex-wrap gap-1.5">
            {item.tags.map((t) => <span key={t} className="badge badge-neutral">{t}</span>)}
          </div>
        </div>
      )}

      <div className="grid grid-cols-3 gap-3">
        <ProvenanceCell icon={FileText} label="Source documents" value={item.source_document_count} />
        <ProvenanceCell icon={Layers} label="Source chunks" value={item.source_chunk_count} />
        <ProvenanceCell icon={ChevronRight} label="Created" value={relativeTime(item.created_at) ?? '—'} />
      </div>

      {/* Source chunk reveal — fetches actual text on demand */}
      {item.source_chunk_count > 0 && (
        <SourceChunksSection
          itemId={item.id}
          itemType={item.type}
          show={showChunks}
          onToggle={() => setShowChunks((v) => !v)}
        />
      )}

      <details className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
        <summary className="cursor-pointer select-none font-semibold uppercase tracking-wider">
          Technical details
        </summary>
        <dl className="mt-2 space-y-1 font-mono">
          <div className="flex gap-2"><dt className="opacity-60">id:</dt><dd className="break-all">{item.id}</dd></div>
          {!isEntity && (<>
            <div className="flex gap-2"><dt className="opacity-60">source_entity_id:</dt><dd className="break-all">{item.source_entity_id}</dd></div>
            <div className="flex gap-2"><dt className="opacity-60">target_entity_id:</dt><dd className="break-all">{item.target_entity_id}</dd></div>
          </>)}
        </dl>
      </details>

      <div className="flex items-center gap-2 pt-4 flex-wrap" style={{ borderTop: '2px solid var(--border-subtle)' }}>
        <button onClick={onApprove} disabled={acting} className="btn-success"
          title="Mark as human-verified — drops out of the queue. +XP!">
          {acting ? <Loader2 size={16} className="animate-spin" /> : <CheckCircle2 size={16} strokeWidth={2.6} />}
          Approve
        </button>
        <button onClick={onReject} disabled={acting} className="btn-danger"
          title="Soft-delete — keeps the item for audit but marks it as bad data.">
          <XCircle size={16} strokeWidth={2.6} />
          Reject
        </button>
        <button onClick={onEdit} disabled={acting} className="btn-ghost"
          title="Open the inline correct form to edit this item before approval.">
          <Pencil size={14} strokeWidth={2.6} />
          Correct
        </button>
        {actionError && (
          <span className="text-[12px] font-bold ml-auto" style={{ color: 'var(--danger)' }}>
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
      <span className="text-[18px] font-semibold tracking-tight" style={{ color: 'var(--text-primary)' }}>
        {item.source_entity_name ?? <span style={{ color: 'var(--text-muted)' }}>?</span>}
      </span>
      {item.source_entity_type && (
        <span className="text-[10px] px-1.5 py-0.5 rounded font-semibold"
          style={{ background: `${srcColor}15`, color: srcColor, border: `1px solid ${srcColor}40` }}>
          {item.source_entity_type}
        </span>
      )}
      <ArrowRight size={16} style={{ color: 'var(--text-muted)' }} />
      <span className="text-[18px] font-semibold tracking-tight" style={{ color: 'var(--text-primary)' }}>
        {item.target_entity_name ?? <span style={{ color: 'var(--text-muted)' }}>?</span>}
      </span>
      {item.target_entity_type && (
        <span className="text-[10px] px-1.5 py-0.5 rounded font-semibold"
          style={{ background: `${tgtColor}15`, color: tgtColor, border: `1px solid ${tgtColor}40` }}>
          {item.target_entity_type}
        </span>
      )}
    </div>
  );
}

function ProvenanceCell({ icon: Icon, label, value }: { icon: any; label: string; value: number | string }) {
  return (
    <div className="rounded-lg p-3" style={{ background: 'var(--bg-muted)' }}>
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider font-semibold"
        style={{ color: 'var(--text-muted)' }}>
        <Icon size={10} />{label}
      </div>
      <p className="text-[18px] font-semibold tabular-nums mt-1" style={{ color: 'var(--text-primary)' }}>{value}</p>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────
   SourceChunksSection — fetches & displays chunk text on demand
   ───────────────────────────────────────────────────────────────── */
function SourceChunksSection({
  itemId, itemType, show, onToggle,
}: {
  itemId: string;
  itemType: 'entity' | 'relationship';
  show: boolean;
  onToggle: () => void;
}) {
  // Use the matching list endpoint to grab the source_chunk_ids since
  // the queue payload only carries the count.
  const { data: chunkIds } = useQuery({
    queryKey: ['source-chunk-ids', itemType, itemId],
    enabled: show,
    queryFn: async () => {
      // For an entity, /graph/entities/{id} returns source_chunk_ids directly.
      // For a relationship, we don't have a single-rel endpoint — fall back to
      // listing and filtering. (Future: add /graph/relationships/{id}.)
      if (itemType === 'entity') {
        const r = await apiClient.get(`/graph/entities/${itemId}`);
        return (r.data?.source_chunk_ids as string[]) ?? [];
      }
      const r = await apiClient.get('/graph/relationships', { params: { limit: 2000 } });
      const list = (r.data?.items ?? []) as Array<{ id: string; source_chunk_ids: string[] }>;
      return list.find((x) => x.id === itemId)?.source_chunk_ids ?? [];
    },
  });

  const { data: chunks, isFetching } = useQuery<{
    items: ChunkRecord[];
    missing: string[];
  }>({
    queryKey: ['source-chunks', chunkIds],
    enabled: show && Array.isArray(chunkIds) && chunkIds.length > 0,
    queryFn: () => getChunks((chunkIds ?? []).slice(0, 20)),
  });

  return (
    <div>
      <button onClick={onToggle}
        className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider font-semibold transition-colors hover:text-[var(--accent)]"
        style={{ color: 'var(--text-muted)' }}>
        {show ? <EyeOff size={11} /> : <Eye size={11} />}
        {show ? 'Hide source text' : 'Reveal source text'}
      </button>
      {show && (
        <div className="mt-2 space-y-2">
          {isFetching && (
            <div className="flex items-center gap-2 text-[12px]" style={{ color: 'var(--text-muted)' }}>
              <Loader2 size={12} className="animate-spin" /> Loading source chunks…
            </div>
          )}
          {chunks?.items.length === 0 && !isFetching && (
            <p className="text-[11.5px] italic" style={{ color: 'var(--text-muted)' }}>
              Source chunks aren't available — likely from a /dev/seed or an OT/PubMed ingest
              (no document text was stored). Process a document to see this populated.
            </p>
          )}
          {chunks?.items.map((c) => (
            <div key={c.id} className="rounded-lg p-3"
              style={{ background: 'var(--bg-muted)', border: '1px solid var(--border-subtle)' }}>
              <p className="text-[10px] uppercase tracking-wider font-semibold mb-1"
                style={{ color: 'var(--text-muted)' }}>
                chunk {c.chunk_index + 1} · {c.character_count} chars
              </p>
              <p className="text-[12.5px] leading-relaxed whitespace-pre-wrap"
                style={{ color: 'var(--text-primary)' }}>
                {c.content}
              </p>
            </div>
          ))}
          {chunks && chunks.missing.length > 0 && (
            <p className="text-[10.5px] italic" style={{ color: 'var(--text-muted)' }}>
              {chunks.missing.length} chunk{chunks.missing.length === 1 ? '' : 's'} could not be loaded.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────
   CorrectModal — inline edit for entity / relationship
   ───────────────────────────────────────────────────────────────── */
function CorrectModal({
  item, onClose, onSubmit, submitting,
}: {
  item: CurationQueueItem;
  onClose: () => void;
  onSubmit: (corrections: Record<string, unknown>, reason: string) => void;
  submitting: boolean;
}) {
  const isEntity = item.type === 'entity';
  const [name, setName] = useState(item.name ?? '');
  const [description, setDescription] = useState(item.description ?? '');
  const [relType, setRelType] = useState(item.relationship_type ?? '');
  const [strength, setStrength] = useState<string>(item.strength?.toString() ?? '');
  const [reason, setReason] = useState('');

  const { data: relTypes } = useQuery({
    queryKey: ['relationship-types'],
    queryFn: getRelationshipTypes,
    enabled: !isEntity,
    staleTime: 60_000,
  });

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const corrections: Record<string, unknown> = {};
    if (isEntity) {
      if (name !== item.name) corrections.name = name;
      if (description !== (item.description ?? '')) corrections.description = description;
    } else {
      if (relType !== item.relationship_type) corrections.relationship_type = relType;
      if (description !== (item.description ?? '')) corrections.description = description;
      const s = parseFloat(strength);
      if (!isNaN(s) && s !== item.strength) corrections.strength = s;
    }
    onSubmit(corrections, reason);
  }

  const noChanges = Object.keys(
    isEntity
      ? { ...(name !== item.name && { name }), ...(description !== (item.description ?? '') && { description }) }
      : {
          ...(relType !== item.relationship_type && { relType }),
          ...(description !== (item.description ?? '') && { description }),
          ...(parseFloat(strength) !== item.strength && !isNaN(parseFloat(strength)) && { strength }),
        },
  ).length === 0;

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center p-4"
      style={{ background: 'rgba(15,23,42,0.55)' }}
      onClick={onClose}>
      <form onSubmit={submit} onClick={(e) => e.stopPropagation()}
        className="card p-6 w-full max-w-lg space-y-4 fade-up">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-[10px] uppercase tracking-wider font-semibold mb-1"
              style={{ color: 'var(--text-muted)' }}>Correct {isEntity ? 'entity' : 'relationship'}</p>
            <h2 className="text-[18px] font-semibold tracking-tight" style={{ color: 'var(--text-primary)' }}>
              {isEntity ? item.name : `${item.source_entity_name} → ${item.target_entity_name}`}
            </h2>
          </div>
          <button onClick={onClose} type="button" className="text-slate-400 hover:text-slate-700"><X size={16} /></button>
        </div>

        {isEntity ? (
          <>
            <div>
              <label className="field-label">Name</label>
              <input className="input" value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div>
              <label className="field-label">Description</label>
              <textarea className="input resize-none" rows={4} value={description}
                onChange={(e) => setDescription(e.target.value)} />
            </div>
          </>
        ) : (
          <>
            <div>
              <label className="field-label">Relationship type</label>
              <select className="input" value={relType} onChange={(e) => setRelType(e.target.value)}>
                {(relTypes ?? [item.relationship_type ?? '']).map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="field-label">Description</label>
              <textarea className="input resize-none" rows={3} value={description}
                onChange={(e) => setDescription(e.target.value)} />
            </div>
            <div>
              <label className="field-label">Strength (0.0 – 1.0)</label>
              <input className="input" type="number" step={0.05} min={0} max={1}
                value={strength} onChange={(e) => setStrength(e.target.value)} />
            </div>
          </>
        )}

        <div>
          <label className="field-label">Reason / note (optional)</label>
          <input className="input" value={reason} onChange={(e) => setReason(e.target.value)}
            placeholder="Why this correction?" />
        </div>

        <div className="flex items-center gap-2 pt-3"
          style={{ borderTop: '1px solid var(--border-subtle)' }}>
          <button type="submit" disabled={submitting || noChanges} className="btn-primary"
            title={noChanges ? 'Change at least one field' : 'Apply correction and approve in one step'}>
            {submitting ? <Loader2 size={14} className="animate-spin" /> : <CheckCircle2 size={14} />}
            Apply correction
          </button>
          <button type="button" onClick={onClose} className="btn-ghost">Cancel</button>
          {noChanges && (
            <span className="text-[11px] ml-auto" style={{ color: 'var(--text-muted)' }}>
              Edit at least one field to enable
            </span>
          )}
        </div>
      </form>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────
   AuditDrawer — recent curation events from logs/curation_audit.jsonl
   ───────────────────────────────────────────────────────────────── */
function AuditDrawer({ onClose }: { onClose: () => void }) {
  const { data, isLoading } = useQuery({
    queryKey: ['curation-audit'],
    queryFn: () => getCurationAudit(200),
    refetchInterval: 5000,
  });

  return (
    <div className="fixed inset-0 z-40 flex" style={{ background: 'rgba(15,23,42,0.55)' }} onClick={onClose}>
      <div className="ml-auto h-full w-full max-w-lg bg-white shadow-2xl flex flex-col fade-up"
        onClick={(e) => e.stopPropagation()}>
        <div className="px-5 py-4 border-b flex items-center justify-between"
          style={{ borderColor: 'var(--border-subtle)' }}>
          <div>
            <p className="text-[10px] uppercase tracking-wider font-semibold"
              style={{ color: 'var(--text-muted)' }}>Persistent log</p>
            <h2 className="text-[16px] font-semibold tracking-tight flex items-center gap-2"
              style={{ color: 'var(--text-primary)' }}>
              <History size={14} style={{ color: 'var(--accent)' }} />
              Curation audit log
              {data && <span className="text-[11px] font-normal" style={{ color: 'var(--text-muted)' }}>
                ({data.total} entries)
              </span>}
            </h2>
            <p className="text-[10.5px] mt-0.5" style={{ color: 'var(--text-muted)' }}>
              Backed by <code>logs/curation_audit.jsonl</code>. Survives backend restarts.
            </p>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-700"><X size={16} /></button>
        </div>

        <div className="overflow-y-auto flex-1 p-5">
          {isLoading ? (
            <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}>
              <Loader2 size={14} className="animate-spin" /> Loading log
            </div>
          ) : !data?.items.length ? (
            <p className="text-[12.5px]" style={{ color: 'var(--text-muted)' }}>No curation events recorded yet.</p>
          ) : (
            <ul className="space-y-2">
              {data.items.map((rec, i) => <AuditRow key={i} rec={rec} />)}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

function AuditRow({ rec }: { rec: CurationAuditEntry }) {
  const ok = rec.success;
  const tone = ok ? '#15803d' : '#991b1b';
  return (
    <li className="rounded-lg p-3 flex items-start gap-2.5"
      style={{
        background: ok ? '#f0fdf4' : '#fef2f2',
        border: `1px solid ${ok ? 'rgba(21,128,61,0.20)' : 'rgba(153,27,27,0.20)'}`,
      }}>
      {ok
        ? <CheckCircle2 size={14} className="mt-0.5 shrink-0" style={{ color: tone }} />
        : <XCircle size={14} className="mt-0.5 shrink-0" style={{ color: tone }} />}
      <div className="min-w-0 flex-1">
        <p className="text-[12px] font-semibold flex items-center gap-2">
          <span style={{ color: 'var(--text-primary)' }}>{rec.action.replace(/_/g, ' ')}</span>
          <span className="font-mono opacity-60 text-[10.5px]"
            style={{ color: 'var(--text-secondary)' }}>{rec.target_id?.slice(0, 8)}</span>
        </p>
        <p className="text-[10.5px] flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
          <span>{new Date(rec.ts).toLocaleString()}</span>
          <span className="opacity-50">·</span>
          <span>by {rec.curator}</span>
        </p>
        {rec.reason && <p className="text-[11.5px] mt-1 italic" style={{ color: 'var(--text-secondary)' }}>"{rec.reason}"</p>}
        {rec.error && <p className="text-[11.5px] mt-1 font-mono" style={{ color: tone }}>{rec.error}</p>}
        {rec.corrections && Object.keys(rec.corrections).length > 0 && (
          <pre className="text-[10.5px] mt-1.5 p-2 rounded font-mono overflow-x-auto"
            style={{ background: 'rgba(0,0,0,0.04)', color: 'var(--text-secondary)' }}>
            {JSON.stringify(rec.corrections, null, 2)}
          </pre>
        )}
      </div>
    </li>
  );
}
