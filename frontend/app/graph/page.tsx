'use client';
import { useQuery } from '@tanstack/react-query';
import { getEntities, getRelationships } from '@/lib/api';
import dynamic from 'next/dynamic';
import { useMemo } from 'react';

// react-force-graph-2d uses browser APIs — load client-only
const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), { ssr: false });

export default function GraphPage() {
  const { data: entData } = useQuery({
    queryKey: ['entities'],
    queryFn: () => getEntities({ limit: 500 }),
  });
  const { data: relData } = useQuery({
    queryKey: ['relationships'],
    queryFn: () => getRelationships({ limit: 1000 }),
  });

  const graphData = useMemo(() => {
    const nodes = (entData?.items ?? []).map((e) => ({
      id: e.id,
      name: e.name,
      type: e.entity_type,
      color: entityColor(e.entity_type),
    }));
    const links = (relData?.items ?? []).map((r) => ({
      source: r.source_entity_id,
      target: r.target_entity_id,
      label: r.relationship_type,
    }));
    return { nodes, links };
  }, [entData, relData]);

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Knowledge Graph</h1>
      <p className="text-slate-400 text-sm">
        {graphData.nodes.length} nodes · {graphData.links.length} edges
      </p>
      <div className="rounded-xl overflow-hidden border border-slate-700 bg-slate-900">
        <ForceGraph2D
          graphData={graphData}
          nodeLabel={(n: any) => `${n.name} (${n.type})`}
          nodeColor={(n: any) => n.color}
          linkLabel={(l: any) => l.label}
          nodeRelSize={6}
          width={typeof window !== 'undefined' ? window.innerWidth - 280 : 800}
          height={600}
          backgroundColor="#0f172a"
        />
      </div>
    </div>
  );
}

const TYPE_COLORS: Record<string, string> = {
  GENE: '#38bdf8',
  PROTEIN: '#34d399',
  DISEASE: '#f87171',
  DRUG: '#fb923c',
  PATHWAY: '#a78bfa',
  COMPOUND: '#fbbf24',
};
function entityColor(type: string): string {
  return TYPE_COLORS[type.toUpperCase()] ?? '#94a3b8';
}
