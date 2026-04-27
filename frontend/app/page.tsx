'use client';

import Link from 'next/link';
import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  ArrowUpRight,
  ChevronDown,
  ChevronRight,
  Cpu,
  Database,
  GitBranch,
  Layers,
  Link2,
  Network,
  Shapes,
  Sparkles,
  Zap,
} from 'lucide-react';
import { getCurationAudit, getGraphStats, getMetrics, listJobs } from '@/lib/api';
import AnimatedNumber from '@/components/AnimatedNumber';
import { StatCard } from '@/components/ui/StatCard';
import { ErrorBanner } from '@/components/ui/ErrorBanner';
import { EmptyState } from '@/components/ui/EmptyState';
import { SkeletonGrid } from '@/components/ui/LoadingState';
import {
  ENTITY_TYPE_DESCRIPTIONS,
  RELATIONSHIP_TYPE_DESCRIPTIONS,
} from '@/lib/biomedical';

/* ─────────────────────────────────────────────────────────────────
   Distribution breakdown — neutral palette + scan bars + collapsible
   "Other" bucket for rows under the visibility threshold.
   ───────────────────────────────────────────────────────────────── */

const OTHER_THRESHOLD_PCT = 1;

function Breakdown({
  title,
  counts,
  kind,
}: {
  title: string;
  counts: Record<string, number>;
  kind: 'entity' | 'relationship';
}) {
  const [otherOpen, setOtherOpen] = useState(false);

  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const total = sorted.reduce((s, [, v]) => s + v, 0);

  // Anything under the threshold is collected into the "Other" bucket so
  // the breakdown stays scannable for graphs with a long tail of rare
  // labels.
  const visible: [string, number][] = [];
  const hidden: [string, number][] = [];
  for (const [name, value] of sorted) {
    const pct = total > 0 ? (value / total) * 100 : 0;
    if (pct >= OTHER_THRESHOLD_PCT || visible.length < 4) {
      visible.push([name, value]);
    } else {
      hidden.push([name, value]);
    }
  }
  const otherTotal = hidden.reduce((s, [, v]) => s + v, 0);

  const lookup =
    kind === 'entity' ? ENTITY_TYPE_DESCRIPTIONS : RELATIONSHIP_TYPE_DESCRIPTIONS;

  return (
    <div className="card p-7 sm:p-8">
      <h2
        className="text-[14px] font-bold mb-4"
        style={{ color: 'var(--text-primary)' }}
      >
        {title}
      </h2>
      {sorted.length === 0 ? (
        <p
          className="text-[13px]"
          style={{ color: 'var(--text-muted)' }}
        >
          No data yet — start a Process or Ingest job.
        </p>
      ) : (
        <ul className="divide-y" style={{ borderColor: 'var(--border-subtle)' }}>
          {visible.map(([name, value]) => (
            <BreakdownRow
              key={name}
              name={name}
              value={value}
              total={total}
              tooltip={lookup[name.toUpperCase()]}
            />
          ))}
          {hidden.length > 0 && (
            <li>
              <button
                type="button"
                onClick={() => setOtherOpen((v) => !v)}
                className="w-full flex items-center justify-between gap-3 py-2 text-left"
                aria-expanded={otherOpen}
              >
                <span className="flex items-center gap-1.5">
                  {otherOpen
                    ? <ChevronDown size={14} aria-hidden="true" style={{ color: 'var(--text-muted)' }} />
                    : <ChevronRight size={14} aria-hidden="true" style={{ color: 'var(--text-muted)' }} />}
                  <span
                    className="text-[13px] font-semibold"
                    style={{ color: 'var(--text-secondary)' }}
                  >
                    Other ({hidden.length})
                  </span>
                </span>
                <BreakdownMeta value={otherTotal} total={total} />
              </button>
              {otherOpen && (
                <ul className="pl-5 pb-2">
                  {hidden.map(([name, value]) => (
                    <BreakdownRow
                      key={name}
                      name={name}
                      value={value}
                      total={total}
                      tooltip={lookup[name.toUpperCase()]}
                      compact
                    />
                  ))}
                </ul>
              )}
            </li>
          )}
        </ul>
      )}
    </div>
  );
}

function BreakdownRow({
  name,
  value,
  total,
  tooltip,
  compact,
}: {
  name: string;
  value: number;
  total: number;
  tooltip?: string;
  compact?: boolean;
}) {
  return (
    <li
      className={`flex items-center justify-between gap-3 ${compact ? 'py-1.5' : 'py-2.5'}`}
    >
      <span
        className={`${compact ? 'text-[12px]' : 'text-[13px]'} font-semibold truncate`}
        style={{ color: 'var(--text-primary)' }}
        title={tooltip ?? name}
      >
        {name}
      </span>
      <BreakdownMeta value={value} total={total} />
    </li>
  );
}

function BreakdownMeta({ value, total }: { value: number; total: number }) {
  const pct = total > 0 ? (value / total) * 100 : 0;
  return (
    <span className="flex items-center gap-2.5 shrink-0">
      <span
        aria-hidden="true"
        className="inline-block h-1.5 w-12 overflow-hidden"
        style={{ background: 'var(--bg-muted)', borderRadius: 'var(--radius-sm)' }}
      >
        <span
          className="block h-full"
          style={{
            width: `${Math.max(2, pct)}%`,
            background: 'var(--info)',
          }}
        />
      </span>
      <span
        className="text-[11.5px] tabular-nums font-semibold w-10 text-right"
        style={{ color: 'var(--text-muted)' }}
      >
        {pct.toFixed(pct < 1 ? 1 : 0)}%
      </span>
      <span
        className="text-[11.5px] tabular-nums font-semibold w-14 text-right"
        style={{ color: 'var(--text-secondary)' }}
      >
        {value.toLocaleString()}
      </span>
    </span>
  );
}

/* ─────────────────────────────────────────────────────────────────
   Pipeline metrics
   ───────────────────────────────────────────────────────────────── */
function MetricsPanel() {
  const { data, isLoading } = useQuery({
    queryKey: ['metrics'],
    queryFn: getMetrics,
    refetchInterval: 5000,
    staleTime: 4000,
  });

  if (isLoading || !data) {
    return (
      <div className="card p-7 sm:p-8">
        <div className="skeleton h-5 w-44 mb-5" />
        <SkeletonGrid />
      </div>
    );
  }

  const { llm, embedding, pipeline, cache_sizes } = data;
  type Cell = {
    label: string;
    value: number;
    suffix?: string;
    sub: string;
    icon: typeof Cpu;
    tip: string;
  };
  const cells: Cell[] = [
    {
      label: 'LLM Calls',
      value: llm.calls,
      sub: `${llm.cache_hits} cached`,
      icon: Cpu,
      tip: 'Number of completion calls sent to the language model. Lower = more was served from cache.',
    },
    {
      label: 'Total Tokens',
      value: llm.total_tokens,
      sub: `${llm.prompt_tokens.toLocaleString()} prompt · ${llm.completion_tokens.toLocaleString()} completion`,
      icon: Sparkles,
      tip: 'Tokens consumed by the LLM since the API process started.',
    },
    {
      label: 'Avg Latency',
      value: llm.avg_latency_ms,
      suffix: ' ms',
      sub: 'per non-cached call',
      icon: Activity,
      tip: 'Average wall-clock time per non-cached LLM call.',
    },
    {
      label: 'Cache Hit Rate',
      value: Math.round(llm.cache_hit_rate * 100),
      suffix: '%',
      sub: `${cache_sizes.dedup_entries} dedup entries`,
      icon: Zap,
      tip: 'Share of LLM calls served from cache. Higher = cheaper, faster.',
    },
    {
      label: 'Embeddings',
      value: embedding.calls,
      sub: `${Math.round(embedding.cache_hit_rate * 100)}% hit rate`,
      icon: Network,
      tip: 'Embedding vectors generated (used for entity verification and similarity).',
    },
    {
      label: 'Documents',
      value: pipeline.documents_processed,
      sub: `${pipeline.chunks_processed.toLocaleString()} chunks`,
      icon: Layers,
      tip: 'Documents processed end-to-end since startup.',
    },
  ];

  return (
    <div className="card">
      <div
        className="px-7 py-5 sm:px-8 flex items-center justify-between"
        style={{ borderBottom: '1px solid var(--border-subtle)' }}
      >
        <div>
          <h2
            className="text-[14px] font-bold"
            style={{ color: 'var(--text-primary)' }}
          >
            Pipeline performance
          </h2>
          <p
            className="text-[11px] mt-0.5"
            style={{ color: 'var(--text-muted)' }}
          >
            Live · refreshes every 5s
          </p>
        </div>
        <span
          className="badge badge-neutral tabular-nums"
          title="Time since the API process started"
        >
          uptime {formatUptime(data.uptime_seconds)}
        </span>
      </div>
      <div className="p-7 sm:p-8">
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
          {cells.map((c) => (
            <div
              key={c.label}
              className="p-6"
              style={{
                background: 'var(--bg-muted)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
              }}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="flex items-center gap-1.5">
                  <p
                    className="text-[10px] uppercase tracking-wider font-bold"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    {c.label}
                  </p>
                  <span className="help-icon" title={c.tip}>?</span>
                </span>
                <c.icon
                  size={13}
                  style={{ color: 'var(--text-muted)' }}
                  strokeWidth={2.2}
                  aria-hidden="true"
                />
              </div>
              <p
                className="text-[22px] font-bold tabular-nums leading-none"
                style={{ color: 'var(--text-primary)' }}
              >
                <AnimatedNumber value={c.value} />
                {c.suffix && (
                  <span
                    className="text-base font-semibold"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    {c.suffix}
                  </span>
                )}
              </p>
              {c.sub && (
                <p
                  className="text-[10.5px] mt-1.5 font-medium truncate"
                  style={{ color: 'var(--text-secondary)' }}
                  title={c.sub}
                >
                  {c.sub}
                </p>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────
   Curation activity panel — replaces the streak/XP chrome with a
   plain "today / this week / approval rate" summary derived from the
   audit log. No gamification.
   ───────────────────────────────────────────────────────────────── */
function CurationActivityCard() {
  const { data } = useQuery({
    queryKey: ['curation-audit-stats'],
    queryFn: () => getCurationAudit(500),
    refetchInterval: 15000,
  });

  const stats = useMemo(() => {
    if (!data?.items?.length) return { today: 0, week: 0, approvalRate: 0 };
    const now = Date.now();
    const ONE_DAY = 86_400_000;
    let today = 0;
    let week = 0;
    let success = 0;
    for (const rec of data.items) {
      const t = new Date(rec.ts).getTime();
      if (now - t < ONE_DAY) today++;
      if (now - t < 7 * ONE_DAY) week++;
      if (rec.success) success++;
    }
    const approvalRate = data.items.length
      ? Math.round((success / data.items.length) * 100)
      : 0;
    return { today, week, approvalRate };
  }, [data]);

  return (
    <div className="card p-7 sm:p-8 h-full">
      <h2
        className="text-[14px] font-bold mb-4"
        style={{ color: 'var(--text-primary)' }}
      >
        Curation activity
      </h2>
      <dl className="space-y-3">
        <ActivityRow label="Today" value={stats.today} />
        <ActivityRow label="Past 7 days" value={stats.week} />
        <ActivityRow label="Approval rate" value={stats.approvalRate} suffix="%" />
      </dl>
      <Link
        href="/curation"
        className="mt-5 inline-flex items-center gap-1 text-[12px] font-semibold"
        style={{ color: 'var(--accent)' }}
      >
        Open curation queue
        <ArrowUpRight size={12} strokeWidth={2.4} aria-hidden="true" />
      </Link>
    </div>
  );
}

function ActivityRow({
  label,
  value,
  suffix,
}: {
  label: string;
  value: number;
  suffix?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="text-[12.5px]" style={{ color: 'var(--text-secondary)' }}>
        {label}
      </dt>
      <dd
        className="text-[18px] font-bold tabular-nums"
        style={{ color: 'var(--text-primary)' }}
      >
        <AnimatedNumber value={value} />
        {suffix}
      </dd>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────
   Recent jobs
   ───────────────────────────────────────────────────────────────── */
function RecentJobsPanel() {
  const { data } = useQuery({
    queryKey: ['recent-jobs'],
    queryFn: () => listJobs(5),
    refetchInterval: 5000,
  });

  return (
    <div className="card p-7 sm:p-8 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h2
          className="text-[14px] font-bold"
          style={{ color: 'var(--text-primary)' }}
        >
          Recent jobs
        </h2>
        <Link
          href="/documents"
          className="text-[11.5px] font-semibold inline-flex items-center gap-1"
          style={{ color: 'var(--accent)' }}
        >
          See all
          <ArrowUpRight size={11} strokeWidth={2.4} aria-hidden="true" />
        </Link>
      </div>
      {!data || data.length === 0 ? (
        <div className="flex-1 flex items-center justify-center">
          <EmptyState
            icon={Layers}
            title="No jobs yet"
            description="Start a Process or Ingest run to see it appear here."
          />
        </div>
      ) : (
        <ul className="space-y-1.5">
          {data.map((j) => (
            <li
              key={j.job_id}
              className="px-4 py-3 flex items-center justify-between"
              style={{
                background: 'var(--bg-muted)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
              }}
            >
              <div className="min-w-0 flex items-center gap-2">
                <StatusDot status={j.status} animated={j.status === 'running'} />
                <div className="min-w-0">
                  <p
                    className="text-[12.5px] font-bold truncate"
                    style={{ color: 'var(--text-primary)' }}
                  >
                    <span
                      className="font-mono font-medium"
                      style={{ color: 'var(--text-muted)' }}
                    >
                      {j.job_id.slice(0, 7)}
                    </span>
                    <span className="mx-1.5" style={{ color: 'var(--text-muted)' }}>·</span>
                    {j.kind}
                  </p>
                  <p
                    className="text-[10.5px] truncate font-medium"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    {j.message ?? j.current_stage ?? '—'}
                  </p>
                </div>
              </div>
              <span
                className="text-[11px] font-bold tabular-nums"
                style={{ color: 'var(--text-secondary)' }}
              >
                {Math.round((j.progress ?? 0) * 100)}%
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function StatusDot({ status, animated }: { status: string; animated?: boolean }) {
  const color =
    status === 'completed' ? 'var(--success)'
    : status === 'failed' ? 'var(--danger)'
    : status === 'cancelled' ? 'var(--warning)'
    : 'var(--accent)';
  return (
    <span className="relative inline-flex h-2 w-2 shrink-0" aria-hidden="true">
      {animated && (
        <span
          className="absolute inline-flex h-full w-full rounded-full opacity-60 pulse-soft"
          style={{ background: color }}
        />
      )}
      <span
        className="relative inline-flex h-2 w-2 rounded-full"
        style={{ background: color }}
      />
    </span>
  );
}

function formatUptime(s: number): string {
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}

/* ─────────────────────────────────────────────────────────────────
   Quick action tile — flat rectangle.
   ───────────────────────────────────────────────────────────────── */
function ActionTile({
  href, icon: Icon, title, desc,
}: {
  href: string; icon: any; title: string; desc: string;
}) {
  return (
    <Link href={href} className="group block">
      <div
        className="card p-7 sm:p-8 flex items-center gap-3 h-full transition-colors"
        style={{ borderColor: 'var(--border-default)' }}
      >
        <div
          className="h-10 w-10 flex items-center justify-center shrink-0"
          style={{
            background: 'var(--accent-soft)',
            border: '1px solid var(--accent-muted)',
            borderRadius: 'var(--radius-md)',
            color: 'var(--accent)',
          }}
        >
          <Icon size={18} strokeWidth={2.2} aria-hidden="true" />
        </div>
        <div className="flex-1 min-w-0">
          <p
            className="text-[14px] font-bold flex items-center gap-1"
            style={{ color: 'var(--text-primary)' }}
          >
            {title}
            <ArrowUpRight
              size={13}
              className="opacity-0 transition-opacity group-hover:opacity-70"
              style={{ color: 'var(--accent)' }}
              strokeWidth={2.4}
              aria-hidden="true"
            />
          </p>
          <p
            className="text-[11.5px] font-medium mt-0.5"
            style={{ color: 'var(--text-muted)' }}
          >
            {desc}
          </p>
        </div>
      </div>
    </Link>
  );
}

/* ─────────────────────────────────────────────────────────────────
   Page
   ───────────────────────────────────────────────────────────────── */
export default function Dashboard() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['stats'],
    queryFn: getGraphStats,
    refetchInterval: 10000,
  });

  if (error) {
    return (
      <ErrorBanner title="API unreachable">
        Could not reach the backend at localhost:8000. Make sure uvicorn is running.
      </ErrorBanner>
    );
  }

  return (
    <div className="space-y-10 lg:space-y-12">
      {/* Hero header */}
      <header className="space-y-3">
        <h1 className="page-title">Dashboard</h1>
        <p className="page-desc">
          An overview of the knowledge graph — totals, throughput,
          curation activity, and recent jobs.
        </p>
      </header>

      {/* Top stats */}
      <section className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-5 lg:gap-6">
        <StatCard
          label="Entities"
          value={isLoading ? 0 : data!.total_entities}
          icon={Network}
          accent="#d5212c"
          sub="across all sources"
          href="/graph"
        />
        <StatCard
          label="Relationships"
          value={isLoading ? 0 : data!.total_relationships}
          icon={GitBranch}
          accent="#1d8ec7"
          sub="extracted edges"
          href="/graph"
        />
        <StatCard
          label="Entity types"
          value={isLoading ? 0 : Object.keys(data!.entity_type_counts).length}
          icon={Shapes}
          accent="#6b4f44"
          sub="unique node labels"
        />
        <StatCard
          label="Relation types"
          value={isLoading ? 0 : Object.keys(data!.relationship_type_counts).length}
          icon={Link2}
          accent="#6b4f44"
          sub="unique edge labels"
        />
      </section>

      <MetricsPanel />

      <section className="grid grid-cols-1 lg:grid-cols-3 gap-5 lg:gap-6">
        <div className="lg:col-span-2 grid grid-cols-1 lg:grid-cols-2 gap-5 lg:gap-6">
          <Breakdown
            title="Entity types"
            counts={data?.entity_type_counts ?? {}}
            kind="entity"
          />
          <Breakdown
            title="Relationship types"
            counts={data?.relationship_type_counts ?? {}}
            kind="relationship"
          />
        </div>
        <div className="grid grid-cols-1 gap-5 lg:gap-6">
          <CurationActivityCard />
          <RecentJobsPanel />
        </div>
      </section>

      <section className="grid grid-cols-1 sm:grid-cols-3 gap-5 lg:gap-6">
        <ActionTile
          href="/process"
          icon={Zap}
          title="Process a document"
          desc="Paste a URL or text and run live extraction"
        />
        <ActionTile
          href="/ingest"
          icon={Database}
          title="Ingest sources"
          desc="Open Targets · PubMed · Web Crawl"
        />
        <ActionTile
          href="/documents"
          icon={Layers}
          title="Replay a job"
          desc="Inspect any past run with stage timeline"
        />
      </section>
    </div>
  );
}
