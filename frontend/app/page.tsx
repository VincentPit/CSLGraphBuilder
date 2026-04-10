'use client';
import { useQuery } from '@tanstack/react-query';
import { getGraphStats } from '@/lib/api';
import { Network, GitBranch, Shapes, Link2, Activity, TrendingUp } from 'lucide-react';

const COLORS = ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4', '#f97316', '#64748b'];

function StatCard({ label, value, icon: Icon, accent }: { label: string; value: number; icon: any; accent: string }) {
  return (
    <div className="card p-5 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="field-label !mb-0">{label}</span>
        <div className="h-8 w-8 rounded-lg flex items-center justify-center" style={{ background: `${accent}12` }}>
          <Icon size={15} style={{ color: accent }} />
        </div>
      </div>
      <p className="text-[28px] font-semibold tracking-tight tabular-nums" style={{ color: 'var(--text-primary)' }}>
        {value.toLocaleString()}
      </p>
    </div>
  );
}

function Breakdown({ title, counts }: { title: string; counts: Record<string, number> }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 10);
  const total = entries.reduce((s, [, v]) => s + v, 0);

  return (
    <div className="card p-5">
      <div className="flex items-center gap-2 mb-5">
        <TrendingUp size={14} style={{ color: 'var(--accent)' }} />
        <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{title}</h2>
      </div>
      <div className="space-y-3">
        {entries.length === 0 ? (
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No data yet.</p>
        ) : entries.map(([name, value], i) => {
          const pct = total > 0 ? Math.round((value / total) * 100) : 0;
          const color = COLORS[i % COLORS.length];
          return (
            <div key={name}>
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full" style={{ background: color }} />
                  <span className="text-[13px] font-medium" style={{ color: 'var(--text-primary)' }}>{name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs tabular-nums" style={{ color: 'var(--text-muted)' }}>{pct}%</span>
                  <span className="text-xs font-medium tabular-nums" style={{ color: 'var(--text-secondary)' }}>
                    {value.toLocaleString()}
                  </span>
                </div>
              </div>
              <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-muted)' }}>
                <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function Dashboard() {
  const { data, isLoading, error } = useQuery({ queryKey: ['stats'], queryFn: getGraphStats });

  if (isLoading) {
    return (
      <div className="h-64 flex items-center justify-center gap-2" style={{ color: 'var(--text-muted)' }}>
        <Activity size={16} className="animate-pulse" />
        <span className="text-sm">Loading dashboard…</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-5 text-sm" style={{ color: 'var(--danger)', background: 'var(--danger-muted)' }}>
        Could not reach API. Check backend status on port 8001.
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <header>
        <h1 className="page-title">Dashboard</h1>
        <p className="page-desc">Monitor graph size, type distribution, and pipeline activity.</p>
      </header>

      <section className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <StatCard label="Entities" value={data!.total_entities} icon={Network} accent="#6366f1" />
        <StatCard label="Relationships" value={data!.total_relationships} icon={GitBranch} accent="#8b5cf6" />
        <StatCard label="Entity Types" value={Object.keys(data!.entity_type_counts).length} icon={Shapes} accent="#ec4899" />
        <StatCard label="Relation Types" value={Object.keys(data!.relationship_type_counts).length} icon={Link2} accent="#10b981" />
      </section>

      <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Breakdown title="Entity Type Distribution" counts={data!.entity_type_counts} />
        <Breakdown title="Relationship Type Distribution" counts={data!.relationship_type_counts} />
      </section>
    </div>
  );
}
