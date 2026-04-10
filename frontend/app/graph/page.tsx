'use client';
import { useQuery } from '@tanstack/react-query';
import { getEntities, getRelationships } from '@/lib/api';
import dynamic from 'next/dynamic';
import { useMemo } from 'react';
import { Activity } from 'lucide-react';

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), { ssr: false });

const TYPE_COLORS: Record<string, string> = {
  GENE: '#2563eb',
  PROTEIN: '#0f766e',
  DISEASE: '#b45309',
  DRUG: '#7c3aed',
  PATHWAY: '#0369a1',
  COMPOUND: '#475569',
};

function entityColor(type: string): string {
  return TYPE_COLORS[type.toUpperCase()] ?? '#64748b';
}

export default function GraphPage() {
  const { data: entData, isLoading: entLoading } = useQuery({ queryKey: ['entities'], queryFn: () => getEntities({ limit: 500 }) });
  const { data: relData, isLoading: relLoading } = useQuery({ queryKey: ['relationships'], queryFn: () => getRelationships({ limit: 1000 }) });

  const graphData = useMemo(() => ({
    nodes: (entData?.items ?? []).map((e) => ({ id: e.id, name: e.name, type: e.entity_type, color: entityColor(e.entity_type) })),
    links: (relData?.items ?? []).map((r) => ({ source: r.source_entity_id, target: r.target_entity_id, label: r.relationship_type })),
  }), [entData, relData]);

  const loading = entLoading || relLoading;

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">Knowledge Graph</h1>
        <p className="text-slate-600">Interactive graph view. Drag to move, scroll to zoom, hover nodes for labels.</p>
      </header>

      {!loading && (
        <div className="flex flex-wrap gap-2">
          {Object.entries(TYPE_COLORS).map(([type, color]) => (
            <span key={type} className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white border border-[#d0d7de] text-xs text-slate-700">
              <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
              {type}
            </span>
          ))}
        </div>
      )}

      <div className="surface overflow-hidden">
        {loading ? (
          <div className="h-[620px] flex items-center justify-center text-slate-600 text-sm gap-2">
            <Activity size={16} className="animate-pulse" /> Loading graph
          </div>
        ) : (
          <ForceGraph2D
            graphData={graphData}
            nodeLabel={(n: any) => `${n.name} (${n.type})`}
            nodeColor={(n: any) => n.color}
            linkLabel={(l: any) => l.label}
            linkColor={() => '#94a3b8'}
            nodeRelSize={5}
            width={typeof window !== 'undefined' ? Math.max(window.innerWidth - 380, 700) : 900}
            height={620}
            backgroundColor="#ffffff"
          />
        )}
      </div>
    </div>
  );
}
