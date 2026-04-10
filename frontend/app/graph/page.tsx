'use client';
import { useQuery } from '@tanstack/react-query';
import { getEntities, getRelationships } from '@/lib/api';
import dynamic from 'next/dynamic';
import { useMemo } from 'react';
import { Activity } from 'lucide-react';

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), { ssr: false });

const TYPE_COLORS: Record<string, string> = {
  GENE:     '#818cf8',
  PROTEIN:  '#34d399',
  DISEASE:  '#f87171',
  DRUG:     '#fb923c',
  PATHWAY:  '#a78bfa',
  COMPOUND: '#fbbf24',
};

function entityColor(type: string): string {
  return TYPE_COLORS[type.toUpperCase()] ?? '#475569';
}

export default function GraphPage() {
  const { data: entData, isLoading: entLoading } = useQuery({ queryKey: ['entities'], queryFn: () => getEntities({ limit: 500 }) });
  const { data: relData, isLoading: relLoading } = useQuery({ queryKey: ['relationships'], queryFn: () => getRelationships({ limit: 1000 }) });

  const graphData = useMemo(() => ({
    nodes: (entData?.items ?? []).map((e) => ({ id: e.id, name: e.name, type: e.entity_type, color: entityColor(e.entity_type) })),
    links: (relData?.items ?? []).map((r) => ({ source: r.source_entity_id, target: r.target_entity_id, label: r.relationship_type })),
  }), [entData, relData]);

  const loading = entLoading || relLoading;
  const legend = Object.entries(TYPE_COLORS);

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Knowledge Graph</h1>
          <p className="mt-1 text-sm text-slate-400">Interactive force-directed layout. Drag to pan, scroll to zoom, hover nodes for details.</p>
        </div>
        {!loading && (
          <div className="flex gap-4 text-sm shrink-0">
            <div className="text-center">
              <p className="text-xl font-bold text-indigo-400">{graphData.nodes.length}</p>
              <p className="text-xs text-slate-500">nodes</p>
            </div>
            <div className="text-center">
              <p className="text-xl font-bold text-sky-400">{graphData.links.length}</p>
              <p className="text-xs text-slate-500">edges</p>
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-wrap gap-2">
        {legend.map(([type, color]) => (
          <div key={type} className="flex items-center gap-1.5 px-2.5 py-1 bg-[#0d1526] border border-slate-800 rounded-full text-xs text-slate-400">
            <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
            {type.charAt(0) + type.slice(1).toLowerCase()}
          </div>
        ))}
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-[600px] bg-[#0d1526] border border-slate-800 rounded-2xl">
          <div className="flex items-center gap-2 text-slate-500 text-sm"><Activity size={16} className="animate-pulse" /> Loading graph…</div>
        </div>
      ) : (
        <div className="rounded-2xl overflow-hidden border border-slate-800 bg-[#050c1a]">
          <ForceGraph2D
            graphData={graphData}
            nodeLabel={(n: any) => `${n.name} (${n.type})`}
            nodeColor={(n: any) => n.color}
            linkLabel={(l: any) => l.label}
            linkColor={() => '#1e293b'}
            nodeRelSize={5}
            width={typeof window !== 'undefined' ? window.innerWidth - 320 : 800}
            height={600}
            backgroundColor="#050c1a"
          />
        </div>
      )}
    </div>
  );
}
