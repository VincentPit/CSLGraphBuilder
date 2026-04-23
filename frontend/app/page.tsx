'use client';

import Link from 'next/link';
import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  ArrowUpRight,
  Cpu,
  Database,
  GitBranch,
  Layers,
  Link2,
  Network,
  Shapes,
  Sparkles,
  TrendingUp,
  Zap,
} from 'lucide-react';
import { getGraphStats, getMetrics, listJobs } from '@/lib/api';
import AnimatedNumber from '@/components/AnimatedNumber';

const PALETTE = [
  '#6366f1',
  '#8b5cf6',
  '#ec4899',
  '#f59e0b',
  '#10b981',
  '#06b6d4',
  '#f97316',
  '#64748b',
];

/* ─────────────────────────────────────────────────────────────────
   Stat card — animated number, gradient halo, hover lift
   ───────────────────────────────────────────────────────────────── */
function StatCard({
  label,
  value,
  icon: Icon,
  accent,
  sub,
  href,
}: {
  label: string;
  value: number | string;
  icon: any;
  accent: string;
  sub?: string;
  href?: string;
}) {
  const inner = (
    <div className="card card-hover p-5 flex flex-col gap-3 relative overflow-hidden h-full">
      {/* Decorative gradient blob */}
      <div
        className="absolute -top-8 -right-8 h-24 w-24 rounded-full opacity-40 blur-2xl pointer-events-none"
        style={{ background: accent }}
      />
      <div className="flex items-center justify-between relative">
        <span className="field-label !mb-0">{label}</span>
        <div
          className="h-9 w-9 rounded-xl flex items-center justify-center"
          style={{
            background: `linear-gradient(135deg, ${accent}26, ${accent}14)`,
            boxShadow: `inset 0 0 0 1px ${accent}33`,
          }}
        >
          <Icon size={16} style={{ color: accent }} />
        </div>
      </div>
      <div className="flex items-end gap-2 relative">
        <p
          className="text-[32px] font-bold tracking-tight tabular-nums leading-none"
          style={{ color: 'var(--text-primary)' }}
        >
          {typeof value === 'number' ? (
            <AnimatedNumber value={value} />
          ) : (
            value
          )}
        </p>
        {href && (
          <ArrowUpRight
            size={14}
            className="mb-1 opacity-0 group-hover:opacity-60 transition-opacity"
            style={{ color: accent }}
          />
        )}
      </div>
      {sub && (
        <p className="text-[11px] relative" style={{ color: 'var(--text-muted)' }}>
          {sub}
        </p>
      )}
    </div>
  );
  if (href) return <Link href={href} className="group block h-full">{inner}</Link>;
  return inner;
}

/* ─────────────────────────────────────────────────────────────────
   Distribution breakdown — animated bars with stagger
   ───────────────────────────────────────────────────────────────── */
function Breakdown({ title, counts }: { title: string; counts: Record<string, number> }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 10);
  const total = entries.reduce((s, [, v]) => s + v, 0);

  return (
    <div className="card p-5">
      <div className="flex items-center gap-2 mb-5">
        <div
          className="h-7 w-7 rounded-lg flex items-center justify-center"
          style={{ background: 'var(--accent-muted)' }}
        >
          <TrendingUp size={13} style={{ color: 'var(--accent)' }} />
        </div>
        <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          {title}
        </h2>
      </div>
      {entries.length === 0 ? (
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          No data yet.
        </p>
      ) : (
        <div className="space-y-3 stagger">
          {entries.map(([name, value], i) => {
            const pct = total > 0 ? Math.round((value / total) * 100) : 0;
            const color = PALETTE[i % PALETTE.length];
            return (
              <div key={name}>
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-2">
                    <div
                      className="h-2 w-2 rounded-full"
                      style={{
                        background: color,
                        boxShadow: `0 0 0 3px ${color}22`,
                      }}
                    />
                    <span
                      className="text-[13px] font-medium"
                      style={{ color: 'var(--text-primary)' }}
                    >
                      {name}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className="text-xs tabular-nums"
                      style={{ color: 'var(--text-muted)' }}
                    >
                      {pct}%
                    </span>
                    <span
                      className="text-xs font-semibold tabular-nums"
                      style={{ color: 'var(--text-secondary)' }}
                    >
                      {value.toLocaleString()}
                    </span>
                  </div>
                </div>
                <div
                  className="h-1.5 rounded-full overflow-hidden"
                  style={{ background: 'var(--bg-muted)' }}
                >
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${pct}%`,
                      background: `linear-gradient(90deg, ${color}, ${color}cc)`,
                      boxShadow: `0 0 12px ${color}55`,
                      transition: 'width 0.6s cubic-bezier(.2,.7,.2,1)',
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────
   Pipeline metrics — gradient header bar, sparkly cells
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
      <div className="card p-5">
        <div className="skeleton h-4 w-40 mb-5" />
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="skeleton h-20 rounded-xl" />
          ))}
        </div>
      </div>
    );
  }

  const { llm, embedding, pipeline, cache_sizes } = data;
  const cells = [
    {
      label: 'LLM Calls',
      value: llm.calls,
      sub: `${llm.cache_hits} cached`,
      icon: Cpu,
      accent: '#6366f1',
    },
    {
      label: 'Total Tokens',
      value: llm.total_tokens,
      sub: `${llm.prompt_tokens.toLocaleString()} prompt · ${llm.completion_tokens.toLocaleString()} completion`,
      icon: Sparkles,
      accent: '#8b5cf6',
    },
    {
      label: 'Avg Latency',
      value: llm.avg_latency_ms,
      suffix: ' ms',
      sub: 'per non-cached call',
      icon: Activity,
      accent: '#ec4899',
    },
    {
      label: 'Cache Hit Rate',
      value: Math.round(llm.cache_hit_rate * 100),
      suffix: '%',
      sub: `${cache_sizes.dedup_entries} dedup entries`,
      icon: Zap,
      accent: '#f59e0b',
    },
    {
      label: 'Embeddings',
      value: embedding.calls,
      sub: `${Math.round(embedding.cache_hit_rate * 100)}% hit rate`,
      icon: Network,
      accent: '#10b981',
    },
    {
      label: 'Documents',
      value: pipeline.documents_processed,
      sub: `${pipeline.chunks_processed.toLocaleString()} chunks`,
      icon: Layers,
      accent: '#06b6d4',
    },
  ];

  return (
    <div className="card relative overflow-hidden">
      {/* Top accent strip */}
      <div
        className="absolute top-0 inset-x-0 h-[3px] bg-animated"
        style={{ background: 'var(--grad-brand)' }}
      />
      <div className="p-5 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div
            className="h-8 w-8 rounded-xl flex items-center justify-center"
            style={{
              background: 'linear-gradient(135deg, var(--accent-soft), transparent)',
              boxShadow: 'inset 0 0 0 1px rgba(99,102,241,0.20)',
            }}
          >
            <Cpu size={15} style={{ color: 'var(--accent)' }} />
          </div>
          <div>
            <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
              Pipeline Performance
            </h2>
            <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
              Live · refreshes every 5s
            </p>
          </div>
        </div>
        <span
          className="badge badge-neutral tabular-nums"
          title="Time since the API process started"
        >
          uptime {formatUptime(data.uptime_seconds)}
        </span>
      </div>
      <div className="px-5 pb-5">
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 stagger">
          {cells.map((c) => (
            <div
              key={c.label}
              className="rounded-xl p-3.5 relative overflow-hidden card-hover"
              style={{
                background: `linear-gradient(135deg, ${c.accent}0a, transparent)`,
                border: `1px solid ${c.accent}22`,
              }}
            >
              <div className="flex items-center justify-between mb-2">
                <p
                  className="text-[10px] uppercase tracking-wider font-semibold"
                  style={{ color: 'var(--text-muted)' }}
                >
                  {c.label}
                </p>
                <c.icon size={13} style={{ color: c.accent, opacity: 0.7 }} />
              </div>
              <p
                className="text-xl font-bold tabular-nums"
                style={{ color: 'var(--text-primary)' }}
              >
                <AnimatedNumber value={typeof c.value === 'number' ? c.value : 0} />
                {c.suffix && (
                  <span className="text-sm font-semibold opacity-70">{c.suffix}</span>
                )}
              </p>
              {c.sub && (
                <p
                  className="text-[10.5px] mt-1 truncate"
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
   Recent jobs panel
   ───────────────────────────────────────────────────────────────── */
function RecentJobsPanel() {
  const { data } = useQuery({
    queryKey: ['recent-jobs'],
    queryFn: () => listJobs(5),
    refetchInterval: 5000,
  });

  return (
    <div className="card p-5 h-full flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div
            className="h-7 w-7 rounded-lg flex items-center justify-center"
            style={{ background: 'var(--accent-muted)' }}
          >
            <Layers size={13} style={{ color: 'var(--accent)' }} />
          </div>
          <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            Recent Jobs
          </h2>
        </div>
        <Link
          href="/documents"
          className="text-[11px] font-semibold flex items-center gap-1 group"
          style={{ color: 'var(--accent)' }}
        >
          See all
          <ArrowUpRight
            size={11}
            className="transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5"
          />
        </Link>
      </div>
      {!data || data.length === 0 ? (
        <div
          className="flex-1 flex flex-col items-center justify-center text-center gap-2 py-8"
          style={{ color: 'var(--text-muted)' }}
        >
          <Layers size={18} className="opacity-60" />
          <p className="text-[12px]">No jobs yet — try the Process or Ingest pages.</p>
        </div>
      ) : (
        <ul className="space-y-1.5 stagger">
          {data.map((j) => (
            <li
              key={j.job_id}
              className="rounded-lg px-3 py-2 flex items-center justify-between hover:bg-white/60 transition-colors"
              style={{ background: 'var(--bg-muted)' }}
            >
              <div className="min-w-0 flex items-center gap-2">
                <StatusDot status={j.status} animated={j.status === 'running'} />
                <div className="min-w-0">
                  <p
                    className="text-[12.5px] font-semibold truncate"
                    style={{ color: 'var(--text-primary)' }}
                  >
                    <span className="font-mono opacity-60">{j.job_id.slice(0, 7)}</span>
                    <span className="mx-1.5 opacity-40">·</span>
                    {j.kind}
                  </p>
                  <p
                    className="text-[10.5px] truncate"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    {j.message ?? j.current_stage ?? '—'}
                  </p>
                </div>
              </div>
              <span
                className="text-[11px] font-semibold tabular-nums"
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
    status === 'completed'
      ? 'var(--success)'
      : status === 'failed'
        ? 'var(--danger)'
        : status === 'cancelled'
          ? 'var(--warning)'
          : 'var(--accent)';
  return (
    <span className="relative inline-flex h-2 w-2">
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
   Quick action tile
   ───────────────────────────────────────────────────────────────── */
function ActionTile({
  href,
  icon: Icon,
  title,
  desc,
  accent,
}: {
  href: string;
  icon: any;
  title: string;
  desc: string;
  accent: string;
}) {
  return (
    <Link href={href} className="group">
      <div
        className="card card-hover p-5 flex items-center gap-3 relative overflow-hidden h-full"
      >
        <div
          className="absolute -bottom-8 -right-8 h-20 w-20 rounded-full opacity-30 blur-2xl pointer-events-none transition-opacity group-hover:opacity-60"
          style={{ background: accent }}
        />
        <div
          className="h-10 w-10 rounded-xl flex items-center justify-center shrink-0 relative"
          style={{
            background: `linear-gradient(135deg, ${accent}26, ${accent}10)`,
            boxShadow: `inset 0 0 0 1px ${accent}33`,
          }}
        >
          <Icon size={17} style={{ color: accent }} />
        </div>
        <div className="flex-1 min-w-0">
          <p
            className="text-[13.5px] font-semibold flex items-center gap-1"
            style={{ color: 'var(--text-primary)' }}
          >
            {title}
            <ArrowUpRight
              size={13}
              className="opacity-0 transition-all group-hover:opacity-60 group-hover:translate-x-0.5 group-hover:-translate-y-0.5"
              style={{ color: accent }}
            />
          </p>
          <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
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
      <div
        className="card p-6 text-sm flex items-start gap-3"
        style={{ color: 'var(--danger)', background: 'var(--danger-muted)', borderColor: 'rgba(239,68,68,0.30)' }}
      >
        <span className="mt-0.5">⚠</span>
        <div>
          <p className="font-semibold mb-1">API unreachable</p>
          <p>Could not reach the backend at localhost:8001. Make sure uvicorn is running.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Hero header */}
      <header className="space-y-2">
        <div className="flex items-center gap-2">
          <span
            className="badge badge-brand"
            style={{ background: 'var(--accent-soft)', border: '1px solid rgba(213,33,44,0.28)' }}
          >
            <Sparkles size={10} className="opacity-70" />
            CSL Behring · Knowledge Platform
          </span>
        </div>
        <h1 className="page-title">Dashboard</h1>
        <p className="page-desc">
          Real-time visibility into graph health, pipeline throughput, and live
          job activity. Hover any control for context, click a stat card to
          drill in, and start a new run from the quick actions below.
        </p>
      </header>

      {/* Top stats row */}
      <section className="grid grid-cols-2 xl:grid-cols-4 gap-4 stagger">
        <StatCard
          label="Entities"
          value={isLoading ? 0 : data!.total_entities}
          icon={Network}
          accent="#6366f1"
          sub="across all sources"
          href="/graph"
        />
        <StatCard
          label="Relationships"
          value={isLoading ? 0 : data!.total_relationships}
          icon={GitBranch}
          accent="#8b5cf6"
          sub="extracted edges"
          href="/graph"
        />
        <StatCard
          label="Entity Types"
          value={isLoading ? 0 : Object.keys(data!.entity_type_counts).length}
          icon={Shapes}
          accent="#ec4899"
          sub="unique node labels"
        />
        <StatCard
          label="Relation Types"
          value={isLoading ? 0 : Object.keys(data!.relationship_type_counts).length}
          icon={Link2}
          accent="#10b981"
          sub="unique edge labels"
        />
      </section>

      <MetricsPanel />

      <section className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Breakdown
            title="Entity Type Distribution"
            counts={data?.entity_type_counts ?? {}}
          />
          <Breakdown
            title="Relationship Type Distribution"
            counts={data?.relationship_type_counts ?? {}}
          />
        </div>
        <RecentJobsPanel />
      </section>

      <section className="grid grid-cols-1 sm:grid-cols-3 gap-4 stagger">
        <ActionTile
          href="/process"
          icon={Zap}
          title="Process a Document"
          desc="Paste a URL or text → live LLM extraction"
          accent="#d5212c"
        />
        <ActionTile
          href="/ingest"
          icon={Database}
          title="Ingest Sources"
          desc="Open Targets · PubMed · Web Crawl"
          accent="#f59e0b"
        />
        <ActionTile
          href="/documents"
          icon={Layers}
          title="Job History"
          desc="Replay any run with full stage timeline"
          accent="#15803d"
        />
      </section>
    </div>
  );
}
