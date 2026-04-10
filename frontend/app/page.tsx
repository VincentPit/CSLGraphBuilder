'use client';
import { useQuery } from '@tanstack/react-query';
import { getGraphStats } from '@/lib/api';

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
      <p className="text-slate-400 text-sm">{label}</p>
      <p className="text-3xl font-bold text-sky-400 mt-1">{value.toLocaleString()}</p>
    </div>
  );
}

export default function Dashboard() {
  const { data, isLoading, error } = useQuery({ queryKey: ['stats'], queryFn: getGraphStats });

  if (isLoading) return <p className="text-slate-400">Loading stats…</p>;
  if (error) return <p className="text-red-400">Could not reach API — is the server running?</p>;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Dashboard</h1>
      <div className="grid grid-cols-2 gap-4 max-w-lg">
        <StatCard label="Entities" value={data!.total_entities} />
        <StatCard label="Relationships" value={data!.total_relationships} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-3xl">
        <TypeBreakdown title="Entity Types" counts={data!.entity_type_counts} />
        <TypeBreakdown title="Relationship Types" counts={data!.relationship_type_counts} />
      </div>
    </div>
  );
}

function TypeBreakdown({ title, counts }: { title: string; counts: Record<string, number> }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  return (
    <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
      <h2 className="font-semibold mb-3">{title}</h2>
      {entries.length === 0 ? (
        <p className="text-slate-500 text-sm">None yet</p>
      ) : (
        <ul className="space-y-1 text-sm">
          {entries.map(([k, v]) => (
            <li key={k} className="flex justify-between">
              <span className="text-slate-300">{k}</span>
              <span className="text-sky-400 font-mono">{v}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
