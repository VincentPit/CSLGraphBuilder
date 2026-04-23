'use client';

import {
  CheckCircle2,
  Circle,
  Loader2,
  XCircle,
  MinusCircle,
  Sparkles,
} from 'lucide-react';
import type { Job, StageStatus } from '@/lib/api';

const STAGE_LABELS: Record<string, string> = {
  fetch: 'Fetch',
  chunk: 'Chunk',
  entities: 'Entities',
  relationships: 'Relationships',
  finalize: 'Finalize',
  crawl: 'Crawl',
  process: 'Extract',
  persist: 'Persist',
};

function StageIcon({ status }: { status: StageStatus }) {
  switch (status) {
    case 'completed':
      return <CheckCircle2 size={15} style={{ color: 'var(--success)' }} />;
    case 'running':
      return <Loader2 size={15} className="animate-spin" style={{ color: 'var(--accent)' }} />;
    case 'failed':
      return <XCircle size={15} style={{ color: 'var(--danger)' }} />;
    case 'skipped':
      return <MinusCircle size={15} style={{ color: 'var(--text-muted)' }} />;
    default:
      return <Circle size={15} style={{ color: 'var(--text-muted)', opacity: 0.5 }} />;
  }
}

function stageTone(status: StageStatus): string {
  if (status === 'completed') return 'var(--success)';
  if (status === 'running') return 'var(--accent)';
  if (status === 'failed') return 'var(--danger)';
  return 'var(--text-muted)';
}

/**
 * Visualises a pipeline job: header (status + global progress bar),
 * a per-stage timeline rail, and the recent event log. Renders
 * uniformly for any job kind since they all share the same Job shape.
 */
export default function JobTimeline({
  job,
  onCancel,
}: {
  job: Job;
  onCancel?: () => void;
}) {
  const cancellable =
    !!onCancel &&
    !job.cancel_requested &&
    (job.status === 'pending' || job.status === 'running');

  const isRunning = job.status === 'running' || job.status === 'pending';

  return (
    <div className="card p-5 space-y-5 relative overflow-hidden fade-up">
      {/* Top accent bar — animated when running */}
      <div
        className={`absolute top-0 inset-x-0 h-[3px] ${isRunning ? 'bg-animated' : ''}`}
        style={{
          background:
            job.status === 'failed'
              ? 'var(--danger)'
              : job.status === 'cancelled'
                ? 'var(--warning)'
                : job.status === 'completed'
                  ? 'var(--grad-success)'
                  : 'var(--grad-brand)',
        }}
      />

      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <span className="relative inline-flex h-2.5 w-2.5">
            {isRunning && (
              <span
                className="absolute inline-flex h-full w-full rounded-full opacity-60 pulse-soft"
                style={{ background: 'var(--accent)' }}
              />
            )}
            <span
              className="relative inline-flex h-2.5 w-2.5 rounded-full"
              style={{ background: stageTone(job.status as StageStatus) }}
            />
          </span>
          <div className="min-w-0">
            <p className="text-[13px] font-semibold flex items-center gap-2 flex-wrap" style={{ color: 'var(--text-primary)' }}>
              <span className="font-mono opacity-60">{job.job_id.slice(0, 8)}</span>
              <span className="badge badge-brand normal-case">{job.kind}</span>
            </p>
            {job.message && (
              <p
                className="text-[12px] mt-1 truncate"
                style={{ color: 'var(--text-secondary)' }}
              >
                {job.message}
              </p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge status={job.status} cancelRequested={job.cancel_requested} />
          {cancellable && (
            <button onClick={onCancel} className="btn-ghost">
              Cancel
            </button>
          )}
        </div>
      </header>

      {/* Global progress */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
            Overall Progress
          </span>
          <span className="text-xs tabular-nums font-semibold" style={{ color: 'var(--text-secondary)' }}>
            {Math.round((job.progress ?? 0) * 100)}%
          </span>
        </div>
        <div
          className="h-2 rounded-full overflow-hidden relative"
          style={{ background: 'var(--bg-muted)' }}
        >
          <div
            className="h-full rounded-full relative overflow-hidden"
            style={{
              width: `${Math.max(2, Math.round((job.progress ?? 0) * 100))}%`,
              // Use the longhand `backgroundImage` (paired with backgroundSize)
              // so React can diff each independently. Mixing the `background`
              // shorthand with `backgroundSize` triggers a dev warning because
              // the shorthand resets size on every update.
              backgroundImage:
                job.status === 'failed'
                  ? 'linear-gradient(90deg, var(--danger), var(--danger))'
                  : job.status === 'cancelled'
                    ? 'linear-gradient(90deg, var(--warning), var(--warning))'
                    : 'var(--grad-brand)',
              backgroundSize: '200% 100%',
              boxShadow:
                isRunning ? '0 0 16px rgba(213,33,44,0.45)' : 'none',
              transition: 'width 0.4s cubic-bezier(.2,.7,.2,1)',
            }}
          >
            {isRunning && (
              <div
                className="absolute inset-0 opacity-40"
                style={{
                  backgroundImage:
                    'linear-gradient(90deg, transparent, rgba(255,255,255,.55), transparent)',
                  backgroundSize: '200% 100%',
                  animation: 'shimmer 2s linear infinite',
                }}
              />
            )}
          </div>
        </div>
      </div>

      {/* Stage rail */}
      {job.stages.length > 0 && (
        <ol
          className="grid gap-2"
          style={{ gridTemplateColumns: `repeat(${job.stages.length}, minmax(0,1fr))` }}
        >
          {job.stages.map((stage) => {
            const status = (job.stage_progress[stage] ?? 'pending') as StageStatus;
            const isCurrent = job.current_stage === stage && status === 'running';
            const isDone = status === 'completed';
            return (
              <li
                key={stage}
                className="rounded-xl p-3 flex flex-col gap-1.5 text-xs relative overflow-hidden"
                style={{
                  border: `1px solid ${
                    isCurrent
                      ? 'var(--accent)'
                      : isDone
                        ? 'rgba(16,185,129,0.30)'
                        : 'var(--border-subtle)'
                  }`,
                  background: isCurrent
                    ? 'linear-gradient(135deg, rgba(99,102,241,0.10), rgba(139,92,246,0.06))'
                    : isDone
                      ? 'linear-gradient(135deg, rgba(16,185,129,0.07), transparent)'
                      : 'var(--bg-card)',
                  transition: 'all 0.25s ease',
                }}
              >
                {isCurrent && (
                  <div
                    className="absolute -top-6 -right-6 h-12 w-12 rounded-full opacity-40 blur-2xl pointer-events-none"
                    style={{ background: 'var(--accent)' }}
                  />
                )}
                <div className="flex items-center gap-1.5 relative">
                  <StageIcon status={status} />
                  <span
                    className="font-semibold text-[12.5px]"
                    style={{ color: 'var(--text-primary)' }}
                  >
                    {STAGE_LABELS[stage] ?? stage}
                  </span>
                </div>
                <span
                  className="capitalize text-[11px] font-medium relative"
                  style={{ color: stageTone(status) }}
                >
                  {status}
                </span>
              </li>
            );
          })}
        </ol>
      )}

      {/* Activity log */}
      {job.events.length > 0 && (
        <div>
          <p className="field-label flex items-center gap-1.5">
            <Sparkles size={11} style={{ color: 'var(--accent)' }} />
            Activity
          </p>
          <div
            className="rounded-xl p-3 max-h-64 overflow-y-auto font-mono text-[11px] space-y-1"
            style={{
              background: 'linear-gradient(180deg, var(--bg-muted), transparent)',
              border: '1px solid var(--border-subtle)',
            }}
          >
            {job.events
              .slice(-30)
              .reverse()
              .map((event, i) => (
                <div
                  key={`${event.ts}-${i}`}
                  className="flex gap-2 leading-relaxed"
                  style={{
                    color:
                      event.level === 'error'
                        ? 'var(--danger)'
                        : event.level === 'warn'
                          ? '#b45309'
                          : 'var(--text-secondary)',
                  }}
                >
                  <span style={{ color: 'var(--text-muted)' }}>
                    {new Date(event.ts).toLocaleTimeString()}
                  </span>
                  {event.stage && (
                    <span
                      className="font-semibold"
                      style={{ color: 'var(--accent)' }}
                    >
                      [{event.stage}]
                    </span>
                  )}
                  <span className="flex-1">{event.message}</span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}

function StatusBadge({
  status,
  cancelRequested,
}: {
  status: string;
  cancelRequested: boolean;
}) {
  if (cancelRequested && status !== 'cancelled')
    return <span className="badge badge-warning">cancelling…</span>;
  switch (status) {
    case 'completed':
      return <span className="badge badge-success">completed</span>;
    case 'failed':
      return <span className="badge badge-danger">failed</span>;
    case 'cancelled':
      return <span className="badge badge-warning">cancelled</span>;
    case 'running':
      return <span className="badge badge-brand">running</span>;
    default:
      return <span className="badge badge-neutral">{status}</span>;
  }
}
