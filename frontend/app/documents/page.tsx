'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Activity, Layers, RefreshCw } from 'lucide-react';
import { cancelJob, listJobs } from '@/lib/api';
import JobTimeline from '@/components/JobTimeline';
import { useJobStream } from '@/lib/useJobStream';

const KIND_TONE: Record<string, string> = {
  document: 'var(--accent)',
  'web-crawl': '#06b6d4',
  pubmed: '#10b981',
  'open-targets': '#f59e0b',
};

function statusBadge(status: string) {
  if (status === 'completed') return 'badge badge-success';
  if (status === 'failed') return 'badge badge-danger';
  if (status === 'cancelled') return 'badge badge-warning';
  return 'badge badge-neutral';
}

export default function DocumentsPage() {
  const [selected, setSelected] = useState<string | null>(null);
  const { data, isLoading, refetch, isFetching } = useQuery({
    queryKey: ['jobs-list'],
    queryFn: () => listJobs(50),
    refetchInterval: 4000,
  });

  const liveJob = useJobStream(selected);

  async function handleCancel() {
    if (!selected) return;
    try {
      await cancelJob(selected);
    } catch {
      /* surfaced via stream */
    }
  }

  return (
    <div className="space-y-8">
      <header className="flex items-end justify-between flex-wrap gap-3">
        <div className="space-y-2">
          <span
            className="badge badge-brand inline-flex"
            style={{ background: 'var(--accent-soft)', border: '1px solid rgba(99,102,241,0.25)' }}
          >
            <Layers size={10} className="opacity-70" />
            Unified job model
          </span>
          <h1 className="page-title">Job History</h1>
          <p className="page-desc">
            Every document, ingest, and crawl shares the same shape — pick a row
            on the left to see its stage rail, weighted progress, and live event
            log on the right.
          </p>
        </div>
        <button
          onClick={() => refetch()}
          className="btn-ghost"
          disabled={isFetching}
        >
          <RefreshCw size={13} className={isFetching ? 'animate-spin' : ''} /> Refresh
        </button>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        <div className="lg:col-span-2 card overflow-hidden">
          <div className="px-4 py-3 border-b" style={{ borderColor: 'var(--border-subtle)' }}>
            <p className="field-label !mb-0 flex items-center gap-2">
              <Layers size={12} /> All jobs
              {data && (
                <span className="text-[11px] font-normal" style={{ color: 'var(--text-muted)' }}>
                  ({data.length})
                </span>
              )}
            </p>
          </div>
          {isLoading ? (
            <div className="p-6 flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}>
              <Activity size={14} className="animate-pulse" /> Loading…
            </div>
          ) : !data || data.length === 0 ? (
            <p className="p-6 text-sm" style={{ color: 'var(--text-muted)' }}>
              No jobs yet. Run a document or ingest from the sidebar.
            </p>
          ) : (
            <ul className="divide-y" style={{ borderColor: 'var(--border-subtle)' }}>
              {data.map((j) => {
                const active = j.job_id === selected;
                return (
                  <li key={j.job_id}>
                    <button
                      onClick={() => setSelected(j.job_id)}
                      className="w-full text-left px-4 py-3 flex flex-col gap-1.5"
                      style={{
                        background: active ? 'var(--accent-muted)' : 'transparent',
                        borderLeft: `3px solid ${active ? 'var(--accent)' : 'transparent'}`,
                        transition: 'all 0.15s',
                      }}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2 min-w-0">
                          <span
                            className="h-2 w-2 rounded-full"
                            style={{ background: KIND_TONE[j.kind] ?? 'var(--text-muted)' }}
                          />
                          <span
                            className="text-[13px] font-medium truncate"
                            style={{ color: 'var(--text-primary)' }}
                          >
                            {j.kind}
                          </span>
                          <span className="text-[11px] font-mono" style={{ color: 'var(--text-muted)' }}>
                            {j.job_id.slice(0, 8)}
                          </span>
                        </div>
                        <span className={statusBadge(j.status)}>{j.status}</span>
                      </div>
                      <p className="text-[11px] truncate" style={{ color: 'var(--text-secondary)' }}>
                        {j.message ?? j.current_stage ?? '—'}
                      </p>
                      <div className="flex items-center justify-between text-[11px]" style={{ color: 'var(--text-muted)' }}>
                        <span>{new Date(j.created_at).toLocaleString()}</span>
                        <span className="tabular-nums">{Math.round((j.progress ?? 0) * 100)}%</span>
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        <div className="lg:col-span-3">
          {liveJob ? (
            <JobTimeline job={liveJob} onCancel={handleCancel} />
          ) : (
            <div
              className="card p-10 flex flex-col items-center justify-center gap-2 text-center"
              style={{ color: 'var(--text-muted)' }}
            >
              <Layers size={20} />
              <p className="text-sm">Pick a job from the list to see its full timeline.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
