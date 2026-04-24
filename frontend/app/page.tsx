'use client';

import Link from 'next/link';
import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  ArrowUpRight,
  Cpu,
  Database,
  Flame,
  GitBranch,
  Layers,
  Link2,
  Network,
  Shapes,
  Sparkles,
  Star,
  Trophy,
  TrendingUp,
  Zap,
} from 'lucide-react';
import { getCurationAudit, getGraphStats, getMetrics, listJobs } from '@/lib/api';
import AnimatedNumber from '@/components/AnimatedNumber';

const PALETTE = [
  '#d5212c',  // CSL red leads
  '#f59e0b',
  '#58cc02',  // duolingo green
  '#1cb0f6',
  '#8b5cf6',
  '#ec4899',
  '#0e7490',
  '#64748b',
];

/* ─────────────────────────────────────────────────────────────────
   Chunky stat tile — Duolingo "lesson node" feel
   ───────────────────────────────────────────────────────────────── */
function StatCard({
  label, value, icon: Icon, accent, sub, href,
}: {
  label: string; value: number | string; icon: any; accent: string; sub?: string; href?: string;
}) {
  const inner = (
    <div className="card-chunky p-5 flex flex-col gap-3 relative overflow-hidden h-full group">
      <div
        className="absolute -top-10 -right-10 h-28 w-28 rounded-full opacity-30 blur-2xl pointer-events-none"
        style={{ background: accent }}
      />
      <div className="flex items-center justify-between relative">
        <span className="field-label !mb-0">{label}</span>
        <div
          className="h-11 w-11 rounded-2xl flex items-center justify-center"
          style={{
            background: `linear-gradient(135deg, ${accent}26, ${accent}14)`,
            border: `2px solid ${accent}33`,
            boxShadow: `0 3px 0 ${accent}33`,
          }}
        >
          <Icon size={18} style={{ color: accent }} strokeWidth={2.4} />
        </div>
      </div>
      <div className="flex items-end gap-2 relative">
        <p
          className="text-[36px] font-black tracking-tight tabular-nums leading-none"
          style={{ color: 'var(--text-primary)' }}
        >
          {typeof value === 'number' ? <AnimatedNumber value={value} /> : value}
        </p>
        {href && (
          <ArrowUpRight
            size={16}
            className="mb-1.5 opacity-0 group-hover:opacity-70 transition-opacity"
            style={{ color: accent }}
            strokeWidth={2.6}
          />
        )}
      </div>
      {sub && (
        <p className="text-[12px] font-semibold relative" style={{ color: 'var(--text-muted)' }}>
          {sub}
        </p>
      )}
    </div>
  );
  if (href) return <Link href={href} className="block h-full">{inner}</Link>;
  return inner;
}

/* ─────────────────────────────────────────────────────────────────
   Distribution breakdown — chunkier bars
   ───────────────────────────────────────────────────────────────── */
function Breakdown({ title, counts }: { title: string; counts: Record<string, number> }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const total = entries.reduce((s, [, v]) => s + v, 0);

  return (
    <div className="card p-5">
      <div className="flex items-center gap-2 mb-5">
        <div
          className="h-9 w-9 rounded-xl flex items-center justify-center"
          style={{
            background: 'var(--accent-muted)',
            border: '2px solid rgba(213,33,44,0.20)',
            boxShadow: '0 2px 0 rgba(213,33,44,0.20)',
          }}
        >
          <TrendingUp size={15} style={{ color: 'var(--accent)' }} strokeWidth={2.6} />
        </div>
        <h2 className="text-[15px] font-extrabold" style={{ color: 'var(--text-primary)' }}>
          {title}
        </h2>
      </div>
      {entries.length === 0 ? (
        <p className="text-sm font-semibold" style={{ color: 'var(--text-muted)' }}>
          No data yet — kick off a job to start filling this in!
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
                      className="h-2.5 w-2.5 rounded-full"
                      style={{ background: color, boxShadow: `0 0 0 3px ${color}22` }}
                    />
                    <span className="text-[13px] font-bold" style={{ color: 'var(--text-primary)' }}>
                      {name}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs tabular-nums font-bold" style={{ color: 'var(--text-muted)' }}>
                      {pct}%
                    </span>
                    <span className="text-xs font-extrabold tabular-nums" style={{ color: 'var(--text-secondary)' }}>
                      {value.toLocaleString()}
                    </span>
                  </div>
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${pct}%`,
                      backgroundImage: `linear-gradient(90deg, ${color} 0%, ${color}cc 100%)`,
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
      <div className="card p-5">
        <div className="skeleton h-5 w-44 mb-5" />
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="skeleton h-24" />
          ))}
        </div>
      </div>
    );
  }

  const { llm, embedding, pipeline, cache_sizes } = data;
  const cells = [
    { label: 'LLM Calls',      value: llm.calls,        sub: `${llm.cache_hits} cached`,                         icon: Cpu,      accent: '#d5212c' },
    { label: 'Total Tokens',   value: llm.total_tokens, sub: `${llm.prompt_tokens.toLocaleString()} prompt · ${llm.completion_tokens.toLocaleString()} completion`, icon: Sparkles, accent: '#f59e0b' },
    { label: 'Avg Latency',    value: llm.avg_latency_ms, suffix: ' ms', sub: 'per non-cached call',             icon: Activity, accent: '#1cb0f6' },
    { label: 'Cache Hit Rate', value: Math.round(llm.cache_hit_rate * 100), suffix: '%', sub: `${cache_sizes.dedup_entries} dedup entries`, icon: Zap, accent: '#58cc02' },
    { label: 'Embeddings',     value: embedding.calls,  sub: `${Math.round(embedding.cache_hit_rate * 100)}% hit rate`, icon: Network, accent: '#8b5cf6' },
    { label: 'Documents',      value: pipeline.documents_processed, sub: `${pipeline.chunks_processed.toLocaleString()} chunks`, icon: Layers, accent: '#ec4899' },
  ];

  return (
    <div className="card relative overflow-hidden">
      <div
        className="absolute top-0 inset-x-0 h-[5px] bg-animated"
        style={{ background: 'var(--grad-celebration)' }}
      />
      <div className="p-5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div
            className="h-11 w-11 rounded-2xl flex items-center justify-center"
            style={{
              background: 'linear-gradient(135deg, var(--accent-soft), var(--bg-muted))',
              border: '2px solid rgba(213,33,44,0.20)',
              boxShadow: '0 3px 0 rgba(213,33,44,0.20)',
            }}
          >
            <Cpu size={18} style={{ color: 'var(--accent)' }} strokeWidth={2.4} />
          </div>
          <div>
            <h2 className="text-[16px] font-extrabold" style={{ color: 'var(--text-primary)' }}>
              Pipeline Performance
            </h2>
            <p className="text-[11px] font-bold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
              Live · refreshes every 5s
            </p>
          </div>
        </div>
        <span className="badge badge-neutral tabular-nums" title="Time since the API process started">
          uptime {formatUptime(data.uptime_seconds)}
        </span>
      </div>
      <div className="px-5 pb-5">
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 stagger">
          {cells.map((c) => (
            <div
              key={c.label}
              className="rounded-2xl p-4 relative overflow-hidden"
              style={{
                background: `linear-gradient(135deg, ${c.accent}0d, transparent)`,
                border: `2px solid ${c.accent}25`,
                boxShadow: `0 3px 0 ${c.accent}25`,
              }}
            >
              <div className="flex items-center justify-between mb-2">
                <p className="text-[10px] uppercase tracking-widest font-extrabold" style={{ color: 'var(--text-muted)' }}>
                  {c.label}
                </p>
                <c.icon size={14} style={{ color: c.accent, opacity: 0.8 }} strokeWidth={2.6} />
              </div>
              <p className="text-[24px] font-black tabular-nums leading-none" style={{ color: 'var(--text-primary)' }}>
                <AnimatedNumber value={typeof c.value === 'number' ? c.value : 0} />
                {c.suffix && <span className="text-base font-extrabold opacity-70">{c.suffix}</span>}
              </p>
              {c.sub && (
                <p className="text-[10.5px] mt-1.5 font-semibold truncate" style={{ color: 'var(--text-secondary)' }} title={c.sub}>
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
   Daily curation streak — derived from the audit log
   ───────────────────────────────────────────────────────────────── */
function StreakCard() {
  const { data } = useQuery({
    queryKey: ['curation-audit-stats'],
    queryFn: () => getCurationAudit(500),
    refetchInterval: 15000,
  });

  const stats = (() => {
    if (!data?.items?.length) return { today: 0, week: 0, success: 0 };
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
    return { today, week, success };
  })();

  const goal = 10;
  const pct = Math.min(100, Math.round((stats.today / goal) * 100));

  return (
    <div className="card p-5 h-full">
      <div className="flex items-center gap-2 mb-3">
        <div
          className="h-9 w-9 rounded-xl flex items-center justify-center"
          style={{
            background: 'linear-gradient(135deg, #ff7a59, #ff3d00)',
            border: '2px solid rgba(255,61,0,0.40)',
            boxShadow: '0 3px 0 #b91c00',
          }}
        >
          <Flame size={16} className="text-white" strokeWidth={2.6} />
        </div>
        <div>
          <p className="text-[15px] font-extrabold" style={{ color: 'var(--text-primary)' }}>
            Daily curation
          </p>
          <p className="text-[10.5px] font-bold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
            Last 24 hours
          </p>
        </div>
      </div>
      <div className="flex items-baseline gap-2 mb-3">
        <p className="text-[40px] font-black leading-none" style={{ color: 'var(--text-primary)' }}>
          <AnimatedNumber value={stats.today} />
        </p>
        <span className="text-[13px] font-bold" style={{ color: 'var(--text-muted)' }}>
          / {goal} reviewed
        </span>
      </div>
      <div className="progress-bar mb-3">
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="flex items-center gap-2 flex-wrap">
        <span className="badge badge-xp">
          <Star size={11} fill="#5a4400" /> +{stats.success} XP
        </span>
        <span className="badge badge-info">
          {stats.week} this week
        </span>
      </div>
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
    <div className="card p-5 h-full flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div
            className="h-9 w-9 rounded-xl flex items-center justify-center"
            style={{
              background: 'var(--accent-muted)',
              border: '2px solid rgba(213,33,44,0.20)',
              boxShadow: '0 3px 0 rgba(213,33,44,0.20)',
            }}
          >
            <Layers size={15} style={{ color: 'var(--accent)' }} strokeWidth={2.6} />
          </div>
          <h2 className="text-[15px] font-extrabold" style={{ color: 'var(--text-primary)' }}>
            Recent Quests
          </h2>
        </div>
        <Link
          href="/documents"
          className="text-[11px] font-extrabold uppercase tracking-wider flex items-center gap-1 group"
          style={{ color: 'var(--accent)' }}
        >
          See all
          <ArrowUpRight
            size={12}
            className="transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5"
            strokeWidth={3}
          />
        </Link>
      </div>
      {!data || data.length === 0 ? (
        <div
          className="flex-1 flex flex-col items-center justify-center text-center gap-3 py-8"
          style={{ color: 'var(--text-muted)' }}
        >
          <span className="empty-icon bouncy"><Trophy size={26} strokeWidth={2.4} /></span>
          <p className="text-[13px] font-bold" style={{ color: 'var(--text-secondary)' }}>
            No jobs yet
          </p>
          <p className="text-[11px] font-semibold">
            Start a Process or Ingest to begin your knowledge quest!
          </p>
        </div>
      ) : (
        <ul className="space-y-2 stagger">
          {data.map((j) => (
            <li
              key={j.job_id}
              className="rounded-xl px-3 py-2.5 flex items-center justify-between transition-colors"
              style={{
                background: 'var(--bg-muted)',
                border: '2px solid transparent',
              }}
            >
              <div className="min-w-0 flex items-center gap-2">
                <StatusDot status={j.status} animated={j.status === 'running'} />
                <div className="min-w-0">
                  <p className="text-[12.5px] font-extrabold truncate" style={{ color: 'var(--text-primary)' }}>
                    <span className="font-mono opacity-60">{j.job_id.slice(0, 7)}</span>
                    <span className="mx-1.5 opacity-40">·</span>
                    {j.kind}
                  </p>
                  <p className="text-[10.5px] truncate font-semibold" style={{ color: 'var(--text-muted)' }}>
                    {j.message ?? j.current_stage ?? '—'}
                  </p>
                </div>
              </div>
              <span className="text-[11px] font-extrabold tabular-nums" style={{ color: 'var(--text-secondary)' }}>
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
    <span className="relative inline-flex h-2.5 w-2.5">
      {animated && (
        <span
          className="absolute inline-flex h-full w-full rounded-full opacity-60 pulse-soft"
          style={{ background: color }}
        />
      )}
      <span className="relative inline-flex h-2.5 w-2.5 rounded-full" style={{ background: color }} />
    </span>
  );
}

function formatUptime(s: number): string {
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}

/* ─────────────────────────────────────────────────────────────────
   Quick action tile — chunky lesson-node style
   ───────────────────────────────────────────────────────────────── */
function ActionTile({
  href, icon: Icon, title, desc, accent,
}: {
  href: string; icon: any; title: string; desc: string; accent: string;
}) {
  return (
    <Link href={href} className="group block">
      <div className="card-chunky p-5 flex items-center gap-3 relative overflow-hidden h-full">
        <div
          className="absolute -bottom-10 -right-10 h-24 w-24 rounded-full opacity-30 blur-2xl pointer-events-none transition-opacity group-hover:opacity-60"
          style={{ background: accent }}
        />
        <div
          className="h-12 w-12 rounded-2xl flex items-center justify-center shrink-0 relative"
          style={{
            background: `linear-gradient(135deg, ${accent}33, ${accent}14)`,
            border: `2px solid ${accent}40`,
            boxShadow: `0 3px 0 ${accent}40`,
          }}
        >
          <Icon size={20} style={{ color: accent }} strokeWidth={2.6} />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-[14.5px] font-extrabold flex items-center gap-1.5" style={{ color: 'var(--text-primary)' }}>
            {title}
            <ArrowUpRight
              size={14}
              className="opacity-0 transition-all group-hover:opacity-70 group-hover:translate-x-0.5 group-hover:-translate-y-0.5"
              style={{ color: accent }}
              strokeWidth={2.8}
            />
          </p>
          <p className="text-[11.5px] font-semibold mt-0.5" style={{ color: 'var(--text-muted)' }}>
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
        style={{
          color: 'var(--danger)',
          background: 'var(--danger-muted)',
          borderColor: 'rgba(234,43,43,0.30)',
        }}
      >
        <span className="mt-0.5">⚠</span>
        <div>
          <p className="font-extrabold mb-1">API unreachable</p>
          <p className="font-semibold">Could not reach the backend at localhost:8001. Make sure uvicorn is running.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Hero header */}
      <header className="space-y-3">
        <div className="flex items-center gap-2 flex-wrap">
          <span
            className="badge badge-brand"
            style={{ background: 'var(--accent-soft)', border: '2px solid rgba(213,33,44,0.30)' }}
          >
            <Sparkles size={11} className="opacity-80" />
            Welcome back
          </span>
          <span className="badge badge-xp pop-in">
            <Star size={11} fill="#5a4400" /> Level up your graph
          </span>
        </div>
        <h1 className="page-title">Knowledge Quest</h1>
        <p className="page-desc">
          Build, review, and verify your knowledge graph. Numbers tick on every
          refetch — hit the quick actions below to keep your streak going!
        </p>
      </header>

      {/* Top stats */}
      <section className="grid grid-cols-2 xl:grid-cols-4 gap-4 stagger">
        <StatCard label="Entities" value={isLoading ? 0 : data!.total_entities} icon={Network} accent="#d5212c" sub="across all sources" href="/graph" />
        <StatCard label="Relationships" value={isLoading ? 0 : data!.total_relationships} icon={GitBranch} accent="#f59e0b" sub="extracted edges" href="/graph" />
        <StatCard label="Entity Types" value={isLoading ? 0 : Object.keys(data!.entity_type_counts).length} icon={Shapes} accent="#58cc02" sub="unique node labels" />
        <StatCard label="Relation Types" value={isLoading ? 0 : Object.keys(data!.relationship_type_counts).length} icon={Link2} accent="#1cb0f6" sub="unique edge labels" />
      </section>

      <MetricsPanel />

      <section className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Breakdown title="Entity Types" counts={data?.entity_type_counts ?? {}} />
          <Breakdown title="Relationship Types" counts={data?.relationship_type_counts ?? {}} />
        </div>
        <div className="grid grid-cols-1 gap-4">
          <StreakCard />
          <RecentJobsPanel />
        </div>
      </section>

      <section className="grid grid-cols-1 sm:grid-cols-3 gap-4 stagger">
        <ActionTile href="/process"   icon={Zap}      title="Process a Document" desc="Paste a URL or text → live extraction"     accent="#d5212c" />
        <ActionTile href="/ingest"    icon={Database} title="Ingest Sources"     desc="Open Targets · PubMed · Web Crawl"        accent="#f59e0b" />
        <ActionTile href="/documents" icon={Layers}   title="Replay a Quest"     desc="Inspect any past run with stage timeline"  accent="#58cc02" />
      </section>
    </div>
  );
}
