'use client';
import { useQuery } from '@tanstack/react-query';
import { getGraphStats } from '@/lib/api';
import { Network, GitBranch, Shapes, Link2, Activity } from 'lucide-react';

function StatCard({ label, value, icon: Icon }: { label: string; value: number; icon: any }) {
  return (
    <div className="surface p-5">
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs uppercase tracking-wide text-slate-500 font-semibold">{label}</p>
        <div className="h-8 w-8 rounded-md bg-slate-100 border border-slate-200 flex items-center justify-center">
          <Icon size={15} className="text-slate-600" />
        </div>
      </div>
      <p className="text-3xl font-semibold text-slate-900 tabular-nums">{value.toLocaleString()}</p>
    </div>
  );
}

function Breakdown({ title, counts }: { title: string; counts: Record<string, number> }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const total = entries.reduce((sum, [, value]) => sum + value, 0);

  return (
    <div className="surface p-5">
      <h2 className="text-slate-900 font-semibold mb-4">{title}</h2>
      <div className="space-y-3">
        {entries.length === 0 ? <p className="text-sm text-slate-500">No data yet.</p> : entries.map(([name, value]) => {
          const pct = total > 0 ? Math.round((value / total) * 100) : 0;
          return (
            <div key={name}>
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-slate-700 truncate pr-2">{name}</span>
                <span className="text-slate-500 tabular-nums">{value.toLocaleString()}</span>
              </div>
              <div className="h-1.5 rounded-full bg-slate-200 overflow-hidden">
                <div className="h-full rounded-full bg-slate-500" style={{ width: `${pct}%` }} />
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
      <div className="h-64 flex items-center justify-center text-slate-500 text-sm gap-2">
        <Activity size={16} className="animate-pulse" /> Loading dashboard
      </div>
    );
  }

  if (error) {
    return <div className="surface p-6 text-sm text-rose-700">Could not reach API. Check backend status on port 8001.</div>;
  }

  return (
    <div className="space-y-8">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900">System Dashboard</h1>
        <p className="text-slate-600 max-w-2xl">Monitor graph size, type distribution, and ingestion progress.</p>
      </header>

      <section className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <StatCard label="Entities" value={data!.total_entities} icon={Network} />
        <StatCard label="Relationships" value={data!.total_relationships} icon={GitBranch} />
        <StatCard label="Entity Types" value={Object.keys(data!.entity_type_counts).length} icon={Shapes} />
        <StatCard label="Relation Types" value={Object.keys(data!.relationship_type_counts).length} icon={Link2} />
      </section>

      <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Breakdown title="Entity Type Distribution" counts={data!.entity_type_counts} />
        <Breakdown title="Relationship Type Distribution" counts={data!.relationship_type_counts} />
      </section>
    </div>
  );
}
