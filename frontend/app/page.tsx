'use client';
import { useQuery } from '@tanstack/react-query';
import { getGraphStats } from '@/lib/api';
import { Network, GitBranch, TrendingUp, Activity, BarChart3 } from 'lucide-react';

function StatCard({ label, value, icon: Icon, gradient }: { label: string; value: number; icon: any; gradient: string }) {
  return (
    <div className="group bg-gradient-to-b from-[#0f1829] to-[#0d1526] border border-white/[0.06] rounded-2xl p-6 hover:border-white/[0.1] transition-all duration-300">
      <div className="flex items-start justify-between mb-5">
        <p className="text-[13px] font-medium text-slate-400">{label}</p>
        <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${gradient} shadow-lg`}>
          <Icon size={17} className="text-white" />
        </div>
      </div>
      <p className="text-4xl font-extrabold text-white tabular-nums tracking-tight">{value.toLocaleString()}</p>
    </div>
  );
}

function TypeBreakdown({ title, counts, accentFrom, accentTo }: { title: string; counts: Record<string, number>; accentFrom: string; accentTo: string }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const total = entries.reduce((s, [, v]) => s + v, 0);
  return (
    <div className="bg-gradient-to-b from-[#0f1829] to-[#0d1526] border border-white/[0.06] rounded-2xl p-6">
      <h2 className="font-bold text-white text-[15px] mb-5">{title}</h2>
      {entries.length === 0 ? (
        <p className="text-slate-600 text-sm">No data yet</p>
      ) : (
        <ul className="space-y-3.5">
          {entries.map(([k, v]) => {
            const pct = total > 0 ? Math.round((v / total) * 100) : 0;
            return (
              <li key={k}>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-slate-300 font-medium truncate mr-2">{k}</span>
                  <div className="flex items-center gap-2 shrink-0">
                    <span className="text-slate-600 font-mono text-[11px]">{pct}%</span>
                    <span className="text-slate-400 font-mono font-semibold">{v.toLocaleString()}</span>
                  </div>
                </div>
                <div className="w-full h-1.5 bg-white/[0.04] rounded-full overflow-hidden">
                  <div className={`h-full rounded-full bg-gradient-to-r ${accentFrom} ${accentTo}`} style={{ width: `${pct}%` }} />
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

export default function Dashboard() {
  const { data, isLoading, error } = useQuery({ queryKey: ['stats'], queryFn: getGraphStats });

  if (isLoading) return (
    <div className="flex items-center justify-center h-64">
      <div className="flex gap-2.5 items-center text-slate-500 text-sm"><Activity size={16} className="animate-pulse" /> Loading stats…</div>
    </div>
  );
  if (error) return (
    <div className="flex items-center justify-center h-64">
      <div className="text-red-300 text-sm bg-red-500/[0.08] border border-red-500/15 rounded-xl px-6 py-4 max-w-md text-center">
        <p className="font-semibold mb-1">Cannot reach API</p>
        <p className="text-red-400/60 text-xs">Is the server running on port 8001?</p>
      </div>
    </div>
  );

  return (
    <div className="space-y-10 max-w-5xl">
      {/* Header */}
      <div className="space-y-3">
        <div className="flex items-center gap-2.5">
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-indigo-500/20 to-sky-500/10 border border-indigo-500/10 flex items-center justify-center">
            <BarChart3 size={18} className="text-indigo-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">System Dashboard</h1>
            <p className="text-xs text-slate-500 font-medium">Real-time knowledge graph overview</p>
          </div>
        </div>
        <p className="text-sm text-slate-400 leading-relaxed max-w-xl">
          Use the sidebar to ingest new data, run extractions, verify relationships, or visualize the full graph.
        </p>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
        <StatCard label="Total Entities"  value={data!.total_entities}                                  icon={Network}    gradient="bg-gradient-to-br from-indigo-600 to-indigo-500 shadow-indigo-600/20" />
        <StatCard label="Relationships"   value={data!.total_relationships}                             icon={GitBranch}  gradient="bg-gradient-to-br from-sky-600 to-sky-500 shadow-sky-600/20" />
        <StatCard label="Entity Types"    value={Object.keys(data!.entity_type_counts).length}          icon={TrendingUp} gradient="bg-gradient-to-br from-violet-600 to-violet-500 shadow-violet-600/20" />
        <StatCard label="Relation Types"  value={Object.keys(data!.relationship_type_counts).length}    icon={Activity}   gradient="bg-gradient-to-br from-emerald-600 to-emerald-500 shadow-emerald-600/20" />
      </div>

      {/* Breakdowns */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <TypeBreakdown title="Entity Types"       counts={data!.entity_type_counts}       accentFrom="from-indigo-500" accentTo="to-violet-500" />
        <TypeBreakdown title="Relationship Types" counts={data!.relationship_type_counts} accentFrom="from-sky-500"    accentTo="to-cyan-400" />
      </div>
    </div>
  );
}
