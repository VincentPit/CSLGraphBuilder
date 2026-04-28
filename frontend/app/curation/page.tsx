'use client';

import { useEffect, useState } from 'react';
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
  Sparkles,
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
  getCurationQueueCounts,
  getRelationshipTypes,
  submitCurationEvents,
} from '@/lib/api';
import { Button } from '@/components/ui/Button';
import { StatusBadge as StatusBadgePrimitive, TypeChip as TypeChipPrimitive } from '@/components/ui/Badge';
import { EmptyState } from '@/components/ui/EmptyState';
import { ErrorBanner } from '@/components/ui/ErrorBanner';
import { LoadingState } from '@/components/ui/LoadingState';
import { Dialog } from '@/components/ui/Dialog';
import { KeyboardHint, ShortcutList, type Shortcut } from '@/components/ui/KeyboardHint';
import { Field, FieldLabel, TextareaField } from '@/components/ui/Field';
import { useHotkeys } from '@/lib/hooks/useHotkeys';

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

/* Background-tinting tones used by the "Why this is in the queue"
   callout — kept here (rather than in the Badge primitive) because
   they're a panel, not a badge. */
const STATUS_PANEL: Record<string, { bg: string; fg: string; border: string }> = {
  rejected:   { bg: '#fef2f2', fg: '#921414', border: 'rgba(153,27,27,0.30)' },
  flagged:    { bg: '#fffbeb', fg: '#6f3300', border: 'rgba(180,83,9,0.30)'  },
  unverified: { bg: '#eff6ff', fg: '#0a4a72', border: 'rgba(29,78,216,0.25)' },
};

const STATUS_HELP: Record<string, string> = {
  rejected:   'A verifier or conflict-detection step found this conflicts with trusted data.',
  flagged:    'Verifier confidence is low — please double-check sources before approving.',
  unverified: 'Newly extracted by the LLM. No verifier has weighed in yet.',
};

/* ─────────────────────────────────────────────────────────────────
   Local thin wrappers around the shared primitives. These keep the
   rest of this file readable while routing through the design-system
   components.
   ───────────────────────────────────────────────────────────────── */

function StatusBadge({ status }: { status: string }) {
  return <StatusBadgePrimitive status={status} />;
}

function TypeChip({ type }: { type?: string | null }) {
  if (!type) return null;
  return <TypeChipPrimitive type={type} color={entityColor(type)} />;
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
  const [helpOpen, setHelpOpen] = useState(false);
  // Mobile master/detail toggle. Defaults to "queue" so the user lands
  // on the list; opening an item flips to "detail".
  const [mobileTab, setMobileTab] = useState<'queue' | 'detail'>('queue');
  // ID of the item currently being corrected (any queue item, not just
  // the singly-selected one) — drives the CorrectModal.
  const [correctingId, setCorrectingId] = useState<string | null>(null);
  // Transient feedback after a successful approve / reject / correct,
  // shown as a banner above the action buttons. Auto-clears after 2s.
  const [lastAction, setLastAction] = useState<
    { kind: 'approve' | 'reject' | 'correct'; itemName: string } | null
  >(null);
  // Pagination — small page size keeps refetches fast even when there
  // are thousands of items in the queue. Resets to 0 when filters change.
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 25;

  // Drop the page back to 0 whenever a filter changes — otherwise the
  // user could be sitting on page 5 of "rejected" when they switch to
  // "flagged" and it'd be empty.
  useEffect(() => {
    setPage(0);
  }, [statusFilter, typeFilter]);

  const { data, isLoading, isFetching, refetch } = useQuery({
    queryKey: ['curation-queue', statusFilter, typeFilter, page],
    queryFn: () =>
      getCurationQueue({
        status: statusFilter || undefined,
        ...(typeFilter ? ({ type: typeFilter } as any) : {}),
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      }),
    // Background polling removed — approve/reject now updates the cache
    // optimistically, and the manual Refresh button covers the rare case
    // where the user wants a fresh pull. Polling on top of optimistic
    // updates is mostly noise that re-fetches the same slow endpoint.
  });

  // Filter-chip counts come from a separate cheap endpoint so they stay
  // accurate across the full queue, not just the current page.
  const { data: countsData } = useQuery({
    queryKey: ['curation-counts', typeFilter],
    queryFn: () =>
      getCurationQueueCounts(typeFilter ? ({ type: typeFilter } as any) : undefined),
  });
  const counts = countsData ?? { total: 0, rejected: 0, flagged: 0, unverified: 0 };

  const totalPages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  // If items get curated away beneath our feet and the current page no
  // longer exists, fall back to the last valid page.
  useEffect(() => {
    if (data && page >= totalPages) {
      setPage(Math.max(0, totalPages - 1));
    }
  }, [data, page, totalPages]);

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

  // Auto-clear the success banner 2s after it shows.
  useEffect(() => {
    if (!lastAction) return;
    const id = window.setTimeout(() => setLastAction(null), 2000);
    return () => window.clearTimeout(id);
  }, [lastAction]);

  const mutation = useMutation({
    mutationFn: submitCurationEvents,
    // ── Optimistic update ────────────────────────────────────────────
    // The /curation/queue endpoint can take 1–3 s on graphs with
    // thousands of items. Without this, the user clicks "Approve" and
    // sits looking at the unchanged row until the refetch lands —
    // exactly the "did my click register?" feeling we want to kill.
    //
    // We snapshot the current queue + counts cache, splice out the
    // affected IDs immediately, and reconcile in onSettled. onError
    // rolls back to the snapshot.
    onMutate: async (events) => {
      setActionError(null);
      const affectedIds = new Set(
        events.map((e) => e.entity_id ?? e.relationship_id).filter(Boolean) as string[],
      );
      if (affectedIds.size === 0) return undefined;

      // Cancel in-flight refetches so they don't overwrite our optimistic write.
      await qc.cancelQueries({ queryKey: ['curation-queue'] });
      await qc.cancelQueries({ queryKey: ['curation-counts'] });

      // Snapshot every cached queue page so we can roll back precisely.
      const queueSnapshots = qc.getQueriesData<{
        total: number;
        items: CurationQueueItem[];
        limit: number;
        offset: number;
      }>({ queryKey: ['curation-queue'] });
      const countsSnapshots = qc.getQueriesData<{
        total: number;
        rejected: number;
        flagged: number;
        unverified: number;
      }>({ queryKey: ['curation-counts'] });

      // Remove affected items from every cached queue page.
      for (const [key, value] of queueSnapshots) {
        if (!value) continue;
        const removed = value.items.filter((i) => affectedIds.has(i.id));
        if (removed.length === 0) continue;
        qc.setQueryData(key, {
          ...value,
          items: value.items.filter((i) => !affectedIds.has(i.id)),
          total: Math.max(0, value.total - removed.length),
        });
      }

      // Decrement the chip counters so they don't lag behind the row removal.
      // We tally by status so e.g. removing 3 "flagged" items only drops the
      // flagged count, not the others.
      const removedByStatus: Record<string, number> = {
        rejected: 0,
        flagged: 0,
        unverified: 0,
      };
      for (const [, value] of queueSnapshots) {
        if (!value) continue;
        for (const item of value.items) {
          if (affectedIds.has(item.id)) {
            const s = item.verification_status;
            if (s in removedByStatus) removedByStatus[s] += 1;
          }
        }
      }
      // De-duplicate per ID across pages (an item can appear on multiple
      // cached pages from different filters; we should only count it once).
      const seenStatuses = new Map<string, string>();
      for (const [, value] of queueSnapshots) {
        if (!value) continue;
        for (const item of value.items) {
          if (affectedIds.has(item.id) && !seenStatuses.has(item.id)) {
            seenStatuses.set(item.id, item.verification_status);
          }
        }
      }
      const dedupedByStatus: Record<string, number> = {
        rejected: 0,
        flagged: 0,
        unverified: 0,
      };
      seenStatuses.forEach((s) => {
        if (s in dedupedByStatus) dedupedByStatus[s] += 1;
      });
      const totalRemoved = Object.values(dedupedByStatus).reduce((a, b) => a + b, 0);

      for (const [key, value] of countsSnapshots) {
        if (!value) continue;
        qc.setQueryData(key, {
          rejected: Math.max(0, value.rejected - dedupedByStatus.rejected),
          flagged: Math.max(0, value.flagged - dedupedByStatus.flagged),
          unverified: Math.max(0, value.unverified - dedupedByStatus.unverified),
          total: Math.max(0, value.total - totalRemoved),
        });
      }

      return { queueSnapshots, countsSnapshots };
    },
    onError: (err: any, _events, context) => {
      // Roll back the optimistic state.
      if (context?.queueSnapshots) {
        for (const [key, value] of context.queueSnapshots) {
          qc.setQueryData(key, value);
        }
      }
      if (context?.countsSnapshots) {
        for (const [key, value] of context.countsSnapshots) {
          qc.setQueryData(key, value);
        }
      }
      setActionError(formatApiError(err, 'Could not record curation event'));
    },
    onSuccess: (res) => {
      if (res.failed > 0) {
        setActionError(`${res.failed} of ${res.processed + res.failed} events failed: ${res.errors.join('; ')}`);
      }
      setCorrectingId(null);
    },
    onSettled: () => {
      // Reconcile with the server — covers partial failures, drift from
      // background changes, and the audit log.
      qc.invalidateQueries({ queryKey: ['curation-queue'] });
      qc.invalidateQueries({ queryKey: ['curation-counts'] });
      qc.invalidateQueries({ queryKey: ['curation-audit'] });
    },
  });

  function actOne(item: CurationQueueItem, action: 'approve' | 'reject') {
    // Auto-advance to the next visible item BEFORE the queue refetches,
    // so the user sees the next thing to review immediately.
    const items = data?.items ?? [];
    const idx = items.findIndex((i) => i.id === item.id);
    const next = items[idx + 1] ?? items[idx - 1] ?? null;
    setLastAction({
      kind: action,
      itemName: item.name ?? `${item.source_entity_name} → ${item.target_entity_name}`,
    });
    mutation.mutate([buildEvent(item, action)], {
      onSuccess: () => {
        if (next) setSelectedId(next.id);
      },
    });
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

  function applyCorrection(
    item: CurationQueueItem,
    corrections: Record<string, unknown>,
    reason: string,
  ) {
    setLastAction({
      kind: 'correct',
      itemName: item.name ?? `${item.source_entity_name} → ${item.target_entity_name}`,
    });
    mutation.mutate([buildEvent(item, 'correct', { corrections, notes: reason })]);
  }

  /** Approve every queue item that's already been auto-verified — the
      "low-risk batch" power action that lets curators clear noise fast. */
  function bulkApproveVerified() {
    const verified = (data?.items ?? []).filter(
      (i) => i.verification_status === 'verified',
    );
    if (verified.length === 0) return;
    setLastAction({ kind: 'approve', itemName: `${verified.length} verified items` });
    mutation.mutate(
      verified.map((i) => buildEvent(i, 'approve', { curator_id: 'bulk-verified' })),
    );
  }

  /* ── Keyboard shortcuts ───────────────────────────────────────────
     j / k navigate, a / r approve / reject, c open the correct dialog,
     ? toggles help, Esc clears selection.  Disabled while a modal is
     open so its own focus + Esc handlers stay authoritative. */
  const items = data?.items ?? [];
  const selectedIndex = items.findIndex((i) => i.id === selectedId);

  function moveSelection(delta: number) {
    if (items.length === 0) return;
    const base = selectedIndex < 0 ? 0 : selectedIndex;
    const next = (base + delta + items.length) % items.length;
    setSelectedId(items[next].id);
  }

  useHotkeys(
    {
      j: () => moveSelection(1),
      k: () => moveSelection(-1),
      a: () => selected && actOne(selected, 'approve'),
      r: () => selected && actOne(selected, 'reject'),
      c: () => selected && setCorrectingId(selected.id),
      '?': () => setHelpOpen((v) => !v),
      escape: () => {
        if (helpOpen) setHelpOpen(false);
        else if (bulkIds.size > 0) setBulkIds(new Set());
      },
    },
    correctingId === null && !auditOpen,
  );

  const verifiedCount = items.filter((i) => i.verification_status === 'verified').length;
  const shortcuts: Shortcut[] = [
    { keys: ['J'], description: 'Next item' },
    { keys: ['K'], description: 'Previous item' },
    { keys: ['A'], description: 'Approve selected' },
    { keys: ['R'], description: 'Reject selected' },
    { keys: ['C'], description: 'Correct selected' },
    { keys: ['?'], description: 'Toggle this help' },
    { keys: ['Esc'], description: 'Clear bulk selection' },
  ];

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
          <h1 className="page-title">Curation queue</h1>
          <p className="page-desc">
            Review extracted entities and relationships before they're
            promoted to trusted graph content. Click a row on the left to
            open it in the panel on the right — you'll see the full
            description, the source documents and chunks it was extracted
            from, and the verifier's reason for flagging it. Use the
            checkboxes to approve or reject many at once, or press <kbd>?</kbd>{' '}
            for keyboard shortcuts.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {verifiedCount > 0 && (
            <Button
              variant="success"
              onClick={bulkApproveVerified}
              disabled={mutation.isPending}
              title={`Approve all ${verifiedCount} items already auto-verified`}
            >
              <Sparkles size={14} aria-hidden="true" strokeWidth={2.6} />
              Approve {verifiedCount} verified
            </Button>
          )}
          <Button onClick={() => setAuditOpen(true)} aria-label="Open audit log">
            <History size={13} aria-hidden="true" />
            Audit log
          </Button>
          <Button
            onClick={() => refetch()}
            disabled={isFetching}
            aria-label="Refresh queue"
          >
            <RefreshCw size={13} className={isFetching ? 'animate-spin' : ''} aria-hidden="true" />
            Refresh
          </Button>
          <KeyboardHint onShowHelp={() => setHelpOpen(true)} className="hidden sm:inline-flex" />
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

      {/* Mobile tab switcher — collapses master/detail into one viewport at a time. */}
      <div role="tablist" aria-label="Curation view" className="flex gap-2 lg:hidden">
        <button
          role="tab"
          aria-selected={mobileTab === 'queue'}
          onClick={() => setMobileTab('queue')}
          className={`flex-1 px-3 py-2 rounded-md text-[13px] font-bold transition-all ${mobileTab === 'queue' ? 'btn-ghost active' : 'btn-ghost'}`}
        >
          Queue ({items.length})
        </button>
        <button
          role="tab"
          aria-selected={mobileTab === 'detail'}
          onClick={() => setMobileTab('detail')}
          disabled={!selected}
          className={`flex-1 px-3 py-2 rounded-md text-[13px] font-bold transition-all ${mobileTab === 'detail' ? 'btn-ghost active' : 'btn-ghost'}`}
        >
          Detail
        </button>
      </div>

      {/* Master-detail */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        {/* Left list */}
        <div
          className={`lg:col-span-2 card overflow-hidden flex flex-col ${mobileTab === 'detail' ? 'hidden lg:flex' : ''}`}
          style={{ maxHeight: 720 }}
        >
          <div className="px-4 py-3 border-b flex items-center justify-between"
            style={{ borderColor: 'var(--border-subtle)' }}>
            <button
              onClick={selectAllVisible}
              className="flex items-center gap-2 text-[12px] font-semibold transition-colors hover:text-[var(--accent)]"
              style={{ color: 'var(--text-secondary)' }}
              aria-label={bulkIds.size === items.length && items.length > 0 ? 'Clear selection on this page' : 'Select all on this page'}
            >
              {bulkIds.size > 0 && items.length > 0 && bulkIds.size === items.length
                ? <CheckSquare size={14} style={{ color: 'var(--accent)' }} aria-hidden="true" />
                : <Square size={14} aria-hidden="true" />}
              Select page
            </button>
            <p
              className="text-[11.5px] font-semibold tabular-nums"
              style={{ color: 'var(--text-muted)' }}
            >
              {data ? (
                <>
                  {data.total === 0
                    ? '0'
                    : `${page * PAGE_SIZE + 1}–${Math.min((page + 1) * PAGE_SIZE, data.total)}`}
                  {' of '}
                  {data.total.toLocaleString()}
                </>
              ) : '—'}
            </p>
          </div>
          <div className="overflow-y-auto flex-1">
            {isLoading ? (
              <LoadingState>Loading queue</LoadingState>
            ) : items.length === 0 ? (
              <EmptyState
                icon={CheckCircle2}
                title="Queue is empty — nice work!"
                description={
                  <>
                    The verifier hasn't flagged anything for human review.
                    To populate this queue, run a job from the{' '}
                    <strong>Process</strong> or <strong>Ingest</strong> page —
                    items the verifier isn't confident about will appear here.
                  </>
                }
              />
            ) : (
              <ul className="divide-y" style={{ borderColor: 'var(--border-subtle)' }}>
                {items.map((item) => (
                  <QueueRow
                    key={item.id}
                    item={item}
                    selected={item.id === selectedId}
                    checked={bulkIds.has(item.id)}
                    onSelect={() => {
                      setSelectedId(item.id);
                      setMobileTab('detail');
                    }}
                    onToggle={() => toggleSelect(item.id)}
                  />
                ))}
              </ul>
            )}
          </div>
          {/* Pagination footer */}
          {data && data.total > PAGE_SIZE && (
            <div
              className="px-4 py-2.5 flex items-center justify-between border-t"
              style={{ borderColor: 'var(--border-subtle)' }}
            >
              <button
                type="button"
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0 || isFetching}
                className="text-[12px] font-semibold px-2.5 py-1 rounded-md transition-colors disabled:opacity-40 disabled:cursor-not-allowed hover:text-[var(--accent)]"
                style={{ color: 'var(--text-secondary)' }}
                aria-label="Previous page"
              >
                ← Prev
              </button>
              <span
                className="text-[11.5px] font-semibold tabular-nums"
                style={{ color: 'var(--text-muted)' }}
              >
                Page {page + 1} of {totalPages}
              </span>
              <button
                type="button"
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1 || isFetching}
                className="text-[12px] font-semibold px-2.5 py-1 rounded-md transition-colors disabled:opacity-40 disabled:cursor-not-allowed hover:text-[var(--accent)]"
                style={{ color: 'var(--text-secondary)' }}
                aria-label="Next page"
              >
                Next →
              </button>
            </div>
          )}
        </div>

        {/* Right detail — shows the bulk-review panel when checkboxes
            are selected, otherwise the single-item detail. */}
        <div className={`lg:col-span-3 ${mobileTab === 'queue' ? 'hidden lg:block' : ''}`}>
          {bulkIds.size > 0 ? (
            <BulkReviewPanel
              items={items.filter((i) => bulkIds.has(i.id))}
              onApproveItem={(it) => actOne(it, 'approve')}
              onRejectItem={(it) => actOne(it, 'reject')}
              onCorrectItem={(it) => setCorrectingId(it.id)}
              onClear={() => setBulkIds(new Set())}
              acting={mutation.isPending}
            />
          ) : selected ? (
            <DetailPanel
              item={selected}
              onApprove={() => actOne(selected, 'approve')}
              onReject={() => actOne(selected, 'reject')}
              onEdit={() => setCorrectingId(selected.id)}
              acting={mutation.isPending}
              actionError={actionError}
              onDismissError={() => setActionError(null)}
              lastAction={lastAction}
            />
          ) : (
            <div className="card p-8">
              <EmptyState
                icon={ClipboardCheck}
                title="Pick a row on the left to review"
                description="The selected item's full description, entity / relationship type, source documents, source chunks (the actual extracted text), and the verifier's reason for flagging will appear here — along with Approve / Reject / Correct actions. Tip: press J / K to step through items without the mouse."
              />
            </div>
          )}
        </div>
      </div>

      {/* Keyboard help dialog */}
      <Dialog
        open={helpOpen}
        onClose={() => setHelpOpen(false)}
        labelledBy="curation-shortcuts-title"
        className="card p-6 w-full max-w-sm space-y-4 fade-up"
      >
        <div>
          <p
            className="text-[10px] uppercase tracking-wider font-semibold mb-1"
            style={{ color: 'var(--text-muted)' }}
          >
            Curation shortcuts
          </p>
          <h2
            id="curation-shortcuts-title"
            className="text-[18px] font-semibold tracking-tight"
            style={{ color: 'var(--text-primary)' }}
          >
            Keyboard mode
          </h2>
        </div>
        <ShortcutList shortcuts={shortcuts} />
        <div className="pt-2 flex justify-end">
          <Button onClick={() => setHelpOpen(false)} aria-label="Close shortcuts help">
            Got it
          </Button>
        </div>
      </Dialog>

      {/* Modals */}
      {correctingId && (() => {
        const item = items.find((i) => i.id === correctingId);
        if (!item) return null;
        return (
          <CorrectModal
            item={item}
            onClose={() => setCorrectingId(null)}
            onSubmit={(corrections, reason) => applyCorrection(item, corrections, reason)}
            submitting={mutation.isPending}
          />
        );
      })()}
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
        className="px-3 py-3 transition-colors flex items-start gap-2"
        style={{
          background: selected ? 'var(--accent-muted)' : 'transparent',
          borderLeft: `3px solid ${selected ? 'var(--accent)' : 'transparent'}`,
        }}
      >
        <button
          onClick={onToggle}
          className="mt-0.5 shrink-0"
          aria-label={checked ? 'Deselect for bulk action' : 'Select for bulk action'}
          aria-pressed={checked}
        >
          {checked
            ? <CheckSquare size={15} style={{ color: 'var(--accent)' }} aria-hidden="true" />
            : <Square size={15} style={{ color: 'var(--text-muted)' }} aria-hidden="true" />}
        </button>
        <button
          onClick={onSelect}
          className="text-left flex-1 min-w-0 flex flex-col gap-1.5"
        >
          <div className="flex items-center gap-2 min-w-0">
            <span className="h-2 w-2 shrink-0" style={{ background: headColor, borderRadius: 'var(--radius-sm)' }} aria-hidden="true" />
            {isEntity ? (
              <span className="text-[13px] font-semibold truncate"
                style={{ color: 'var(--text-primary)' }} title={item.name}>
                {item.name}
              </span>
            ) : (
              <span className="text-[13px] font-semibold truncate flex items-center gap-1.5"
                style={{ color: 'var(--text-primary)' }}>
                <span className="truncate">{item.source_entity_name ?? '?'}</span>
                <ArrowRight size={11} className="shrink-0" style={{ color: 'var(--text-muted)' }} aria-hidden="true" />
                <span className="truncate">{item.target_entity_name ?? '?'}</span>
              </span>
            )}
          </div>
          <div className="flex items-center gap-1.5 text-[10.5px]" style={{ color: 'var(--text-muted)' }}>
            {isEntity ? (
              <><Network size={10} aria-hidden="true" /><span>{item.entity_type}</span></>
            ) : (
              <><GitBranch size={10} aria-hidden="true" /><span className="font-mono opacity-80">{item.relationship_type}</span></>
            )}
            <span className="opacity-50">·</span>
            <FileText size={10} aria-hidden="true" />
            <span
              title={
                item.source_document_count === 0
                  ? 'No source documents are stored for this item — likely from a /dev/seed or an Open Targets / PubMed ingest where document text was not retained.'
                  : `Extracted from ${item.source_document_count} source document${item.source_document_count === 1 ? '' : 's'}.`
              }
            >
              {item.source_document_count} doc{item.source_document_count === 1 ? '' : 's'}
            </span>
            {item.created_at && (<><span className="opacity-50">·</span><span>{relativeTime(item.created_at)}</span></>)}
          </div>
          {item.description && (
            <p className="text-[11.5px] line-clamp-2 leading-snug" style={{ color: 'var(--text-secondary)' }}>
              {item.description}
            </p>
          )}
        </button>
        {/* Status badge — pinned to the far right, separate from the
            click target so it reads as a label, not a tag on the title. */}
        <span className="ml-1 shrink-0 self-start mt-0.5">
          <StatusBadge status={item.verification_status} />
        </span>
      </div>
    </li>
  );
}

/* ─────────────────────────────────────────────────────────────────
   DetailPanel — full review surface (with source-chunk reveal)
   ───────────────────────────────────────────────────────────────── */
function DetailPanel({
  item, onApprove, onReject, onEdit, acting, actionError, onDismissError, lastAction,
}: {
  item: CurationQueueItem;
  onApprove: () => void;
  onReject: () => void;
  onEdit: () => void;
  acting: boolean;
  actionError: string | null;
  onDismissError: () => void;
  lastAction: { kind: 'approve' | 'reject' | 'correct'; itemName: string } | null;
}) {
  const isEntity = item.type === 'entity';
  const statusHelp = STATUS_HELP[item.verification_status];
  const [showChunks, setShowChunks] = useState(true);

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
            background: STATUS_PANEL[item.verification_status]?.bg ?? 'var(--bg-muted)',
            border: `1px solid ${STATUS_PANEL[item.verification_status]?.border ?? 'var(--border-default)'}`,
          }}>
          <ShieldAlert size={14} className="shrink-0 mt-0.5"
            style={{ color: STATUS_PANEL[item.verification_status]?.fg }}
            aria-hidden="true" />
          <div>
            <p className="font-semibold leading-tight" style={{ color: STATUS_PANEL[item.verification_status]?.fg }}>
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

      {actionError && (
        <ErrorBanner title="Couldn't record curation event" onDismiss={onDismissError}>
          {actionError}
        </ErrorBanner>
      )}

      {lastAction && !actionError && (
        <div
          role="status"
          className="flex items-center gap-2 px-3 py-2 fade-up text-[13px] font-bold"
          style={{
            background:
              lastAction.kind === 'reject' ? 'var(--danger-muted)' : 'var(--success-muted)',
            color:
              lastAction.kind === 'reject' ? 'var(--danger-shadow)' : 'var(--success-shadow)',
            border: `1px solid ${
              lastAction.kind === 'reject'
                ? 'rgba(217,41,41,0.32)'
                : 'rgba(47,158,63,0.32)'
            }`,
            borderRadius: 'var(--radius-md)',
          }}
        >
          {lastAction.kind === 'reject'
            ? <XCircle size={16} strokeWidth={2.6} aria-hidden="true" />
            : <CheckCircle2 size={16} strokeWidth={2.6} aria-hidden="true" />}
          <span className="truncate">
            {lastAction.kind === 'approve' && 'Approved'}
            {lastAction.kind === 'reject' && 'Rejected'}
            {lastAction.kind === 'correct' && 'Correction applied to'}
            {' '}
            <span style={{ color: 'var(--text-primary)' }}>{lastAction.itemName}</span>
          </span>
        </div>
      )}

      <div className="flex items-center gap-2 pt-4 flex-wrap" style={{ borderTop: '2px solid var(--border-subtle)' }}>
        <Button
          variant="success"
          onClick={onApprove}
          disabled={acting}
          title="Mark as human-verified — drops out of the queue. +XP!"
          aria-keyshortcuts="A"
        >
          {acting
            ? <Loader2 size={16} className="animate-spin" aria-hidden="true" />
            : <CheckCircle2 size={16} strokeWidth={2.6} aria-hidden="true" />}
          Approve
        </Button>
        <Button
          variant="danger"
          onClick={onReject}
          disabled={acting}
          title="Soft-delete — keeps the item for audit but marks it as bad data."
          aria-keyshortcuts="R"
        >
          <XCircle size={16} strokeWidth={2.6} aria-hidden="true" />
          Reject
        </Button>
        <Button
          onClick={onEdit}
          disabled={acting}
          title="Open the inline correct form to edit this item before approval."
          aria-keyshortcuts="C"
        >
          <Pencil size={14} strokeWidth={2.6} aria-hidden="true" />
          Correct
        </Button>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────
   BulkReviewPanel — replaces the single-item DetailPanel when one
   or more queue rows are checkbox-selected. Shows a summary of every
   selected item (name, type, status badge, the verifier's reason for
   flagging it, and the description) with a single Approve all /
   Reject all action footer.
   ───────────────────────────────────────────────────────────────── */
function BulkReviewPanel({
  items,
  onApproveItem,
  onRejectItem,
  onCorrectItem,
  onClear,
  acting,
}: {
  items: CurationQueueItem[];
  onApproveItem: (item: CurationQueueItem) => void;
  onRejectItem: (item: CurationQueueItem) => void;
  onCorrectItem: (item: CurationQueueItem) => void;
  onClear: () => void;
  acting: boolean;
}) {
  return (
    <div className="card p-6 space-y-5 fade-up">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <p
            className="text-[10px] uppercase tracking-wider font-semibold mb-1"
            style={{ color: 'var(--text-muted)' }}
          >
            Bulk review
          </p>
          <h2
            className="text-[20px] font-semibold tracking-tight"
            style={{ color: 'var(--text-primary)' }}
          >
            {items.length} item{items.length === 1 ? '' : 's'} selected
          </h2>
          <p
            className="text-[12.5px] mt-1"
            style={{ color: 'var(--text-secondary)' }}
          >
            Review the reasons below, then approve or reject the whole batch.
          </p>
        </div>
        <button
          type="button"
          onClick={onClear}
          className="text-[12px] font-semibold underline"
          style={{ color: 'var(--text-secondary)' }}
        >
          Clear selection
        </button>
      </div>

      <ul className="space-y-3">
        {items.map((it) => {
          const isEntity = it.type === 'entity';
          const reason = STATUS_HELP[it.verification_status];
          const headColor = isEntity
            ? entityColor(it.entity_type)
            : entityColor(it.source_entity_type);
          return (
            <li
              key={it.id}
              className="p-4"
              style={{
                background: 'var(--bg-muted)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
              }}
            >
              <div className="flex items-start justify-between gap-3 flex-wrap">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span
                      className="h-2 w-2 shrink-0"
                      style={{
                        background: headColor,
                        borderRadius: 'var(--radius-sm)',
                      }}
                      aria-hidden="true"
                    />
                    {isEntity ? (
                      <span
                        className="text-[14px] font-bold truncate"
                        style={{ color: 'var(--text-primary)' }}
                      >
                        {it.name}
                      </span>
                    ) : (
                      <span
                        className="text-[14px] font-bold truncate flex items-center gap-1.5"
                        style={{ color: 'var(--text-primary)' }}
                      >
                        <span className="truncate">{it.source_entity_name ?? '?'}</span>
                        <ArrowRight
                          size={12}
                          className="shrink-0"
                          style={{ color: 'var(--text-muted)' }}
                          aria-hidden="true"
                        />
                        <span className="truncate">{it.target_entity_name ?? '?'}</span>
                      </span>
                    )}
                  </div>
                  <div
                    className="flex items-center gap-1.5 text-[11px] mt-1.5"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    {isEntity ? (
                      <TypeChip type={it.entity_type} />
                    ) : (
                      <span
                        className="badge"
                        style={{
                          background: 'var(--accent-soft)',
                          color: 'var(--accent)',
                          border: '1px solid rgba(213,33,44,0.30)',
                        }}
                      >
                        {it.relationship_type}
                      </span>
                    )}
                  </div>
                </div>
                <StatusBadge status={it.verification_status} />
              </div>

              {reason && (
                <div
                  className="rounded-lg p-2.5 flex items-start gap-2 text-[12px] mt-3"
                  style={{
                    background:
                      STATUS_PANEL[it.verification_status]?.bg ?? 'var(--bg-card)',
                    border: `1px solid ${
                      STATUS_PANEL[it.verification_status]?.border ?? 'var(--border-default)'
                    }`,
                  }}
                >
                  <ShieldAlert
                    size={13}
                    className="shrink-0 mt-0.5"
                    style={{ color: STATUS_PANEL[it.verification_status]?.fg }}
                    aria-hidden="true"
                  />
                  <div>
                    <p
                      className="font-semibold leading-tight"
                      style={{ color: STATUS_PANEL[it.verification_status]?.fg }}
                    >
                      Why this is in the queue
                    </p>
                    <p className="mt-0.5" style={{ color: 'var(--text-secondary)' }}>
                      {reason}
                    </p>
                    {it.notes && (
                      <p
                        className="mt-1 italic"
                        style={{ color: 'var(--text-secondary)' }}
                      >
                        Verifier note: {it.notes}
                      </p>
                    )}
                  </div>
                </div>
              )}

              {it.description && (
                <p
                  className="text-[12px] mt-2 line-clamp-3"
                  style={{ color: 'var(--text-primary)' }}
                >
                  {it.description}
                </p>
              )}

              {/* Per-item action buttons — each selected item can be
                  approved, rejected, or corrected on its own. */}
              <div
                className="flex items-center gap-2 mt-3 pt-3 flex-wrap"
                style={{ borderTop: '1px solid var(--border-subtle)' }}
              >
                <Button
                  variant="success"
                  onClick={() => onApproveItem(it)}
                  disabled={acting}
                  aria-label={`Approve ${it.name ?? it.id}`}
                >
                  <CheckCircle2 size={14} strokeWidth={2.6} aria-hidden="true" />
                  Approve
                </Button>
                <Button
                  variant="danger"
                  onClick={() => onRejectItem(it)}
                  disabled={acting}
                  aria-label={`Reject ${it.name ?? it.id}`}
                >
                  <XCircle size={14} strokeWidth={2.6} aria-hidden="true" />
                  Reject
                </Button>
                <Button
                  onClick={() => onCorrectItem(it)}
                  disabled={acting}
                  aria-label={`Correct ${it.name ?? it.id}`}
                >
                  <Pencil size={13} strokeWidth={2.6} aria-hidden="true" />
                  Correct
                </Button>
              </div>
            </li>
          );
        })}
      </ul>
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
      <div className="flex items-center justify-between mb-2">
        <p className="text-[10px] uppercase tracking-wider font-semibold flex items-center gap-1.5"
          style={{ color: 'var(--text-muted)' }}>
          <FileText size={11} /> Source text — extracted from
        </p>
        <button
          onClick={onToggle}
          className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider font-semibold transition-colors hover:text-[var(--accent)]"
          style={{ color: 'var(--text-muted)' }}
          aria-expanded={show}
          aria-label={show ? 'Hide source chunks' : 'Reveal source chunks'}
        >
          {show ? <EyeOff size={11} aria-hidden="true" /> : <Eye size={11} aria-hidden="true" />}
          {show ? 'Hide' : 'Reveal'}
        </button>
      </div>
      {show && (
        <div className="space-y-2">
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
    <Dialog
      open
      onClose={onClose}
      labelledBy="correct-title"
      className="card p-6 w-full max-w-lg fade-up"
    >
      <form onSubmit={submit} className="space-y-4">
        <div className="flex items-start justify-between">
          <div>
            <p
              className="text-[10px] uppercase tracking-wider font-semibold mb-1"
              style={{ color: 'var(--text-muted)' }}
            >
              Correct {isEntity ? 'entity' : 'relationship'}
            </p>
            <h2
              id="correct-title"
              className="text-[18px] font-semibold tracking-tight"
              style={{ color: 'var(--text-primary)' }}
            >
              {isEntity ? item.name : `${item.source_entity_name} → ${item.target_entity_name}`}
            </h2>
          </div>
          <Button variant="icon" onClick={onClose} aria-label="Close correction dialog">
            <X size={14} aria-hidden="true" />
          </Button>
        </div>

        {isEntity ? (
          <>
            <Field
              label="Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
            <TextareaField
              label="Description"
              rows={4}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="[&_textarea]:resize-none"
            />
          </>
        ) : (
          <>
            <div>
              <FieldLabel htmlFor="rel-type">Relationship type</FieldLabel>
              <select
                id="rel-type"
                className="input"
                value={relType}
                onChange={(e) => setRelType(e.target.value)}
              >
                {(relTypes ?? [item.relationship_type ?? '']).map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
            <TextareaField
              label="Description"
              rows={3}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />
            <Field
              label="Strength (0.0 – 1.0)"
              type="number"
              step={0.05}
              min={0}
              max={1}
              value={strength}
              onChange={(e) => setStrength(e.target.value)}
            />
          </>
        )}

        <Field
          label="Reason / note (optional)"
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          placeholder="Why this correction?"
        />

        <div
          className="flex items-center gap-2 pt-3"
          style={{ borderTop: '1px solid var(--border-subtle)' }}
        >
          <Button
            variant="primary"
            type="submit"
            disabled={submitting || noChanges}
            title={noChanges ? 'Change at least one field' : 'Apply correction and approve in one step'}
          >
            {submitting
              ? <Loader2 size={14} className="animate-spin" aria-hidden="true" />
              : <CheckCircle2 size={14} aria-hidden="true" />}
            Apply correction
          </Button>
          <Button type="button" onClick={onClose}>Cancel</Button>
          {noChanges && (
            <span className="text-[11px] ml-auto" style={{ color: 'var(--text-muted)' }}>
              Edit at least one field to enable
            </span>
          )}
        </div>
      </form>
    </Dialog>
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
    <Dialog
      open
      onClose={onClose}
      labelledBy="audit-title"
      backdropClassName="flex justify-end"
      className="h-full w-full max-w-lg bg-white shadow-2xl flex flex-col fade-up"
    >
      <div
        className="px-5 py-4 border-b flex items-center justify-between"
        style={{ borderColor: 'var(--border-subtle)' }}
      >
        <div>
          <p
            className="text-[10px] uppercase tracking-wider font-semibold"
            style={{ color: 'var(--text-muted)' }}
          >
            Persistent log
          </p>
          <h2
            id="audit-title"
            className="text-[16px] font-semibold tracking-tight flex items-center gap-2"
            style={{ color: 'var(--text-primary)' }}
          >
            <History size={14} style={{ color: 'var(--accent)' }} aria-hidden="true" />
            Curation audit log
            {data && (
              <span className="text-[11px] font-normal" style={{ color: 'var(--text-muted)' }}>
                ({data.total} entries)
              </span>
            )}
          </h2>
          <p className="text-[10.5px] mt-0.5" style={{ color: 'var(--text-muted)' }}>
            Backed by <code>logs/curation_audit.jsonl</code>. Survives backend restarts.
          </p>
        </div>
        <Button variant="icon" onClick={onClose} aria-label="Close audit log">
          <X size={14} aria-hidden="true" />
        </Button>
      </div>

      <div className="overflow-y-auto flex-1 p-5">
        {isLoading ? (
          <LoadingState>Loading log</LoadingState>
        ) : !data?.items.length ? (
          <p className="text-[12.5px]" style={{ color: 'var(--text-muted)' }}>
            No curation events recorded yet.
          </p>
        ) : (
          <ul className="space-y-2">
            {data.items.map((rec, i) => <AuditRow key={i} rec={rec} />)}
          </ul>
        )}
      </div>
    </Dialog>
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
