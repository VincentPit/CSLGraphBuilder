'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  GitBranch,
  Maximize2,
  Network,
  Search,
  Sparkles,
  X,
} from 'lucide-react';
import { getEntities, getRelationships, type Entity, type Relationship } from '@/lib/api';

// `next/dynamic` returns a plain LoadableComponent — refs passed to it are
// dropped (and React warns). Sidestep the issue by accepting an `fgRef` prop
// instead of React's reserved `ref`, and assigning the underlying graph
// instance via the inner `<ForceGraph2D ref={...} />` ourselves.
type ForceGraphProps = Record<string, any> & {
  fgRef?: { current: any | null };
};

const ForceGraph2D = dynamic<ForceGraphProps>(
  () =>
    import('react-force-graph-2d').then((mod) => {
      const RFG = mod.default;
      function Wrapped({ fgRef, ...rest }: ForceGraphProps) {
        return <RFG ref={fgRef as any} {...rest} />;
      }
      return Wrapped;
    }),
  { ssr: false }
);

// Entity-type → colour map. CSL Behring red is reserved for DISEASE because
// disease is the central concept of biomedical knowledge graphs and most
// users land here looking for disease-centric subgraphs. Genes use a steel
// blue, drugs amber, etc. so each category is instantly distinguishable.
const TYPE_COLORS: Record<string, string> = {
  DISEASE:   '#d5212c',  // CSL red
  GENE:      '#1d4ed8',  // steel blue
  PROTEIN:   '#0e7490',  // teal
  DRUG:      '#f59e0b',  // amber
  PATHWAY:   '#7c3aed',  // violet
  COMPOUND:  '#475569',  // slate
  CONCEPT:   '#0891b2',  // cyan
  ORGANISM:  '#15803d',  // green
};

const TYPE_DESCRIPTIONS: Record<string, string> = {
  DISEASE:  'Conditions, disorders, syndromes',
  GENE:     'Coding sequences and genomic loci',
  PROTEIN:  'Translated products, enzymes, receptors',
  DRUG:     'Therapeutics, compounds with known activity',
  PATHWAY:  'Biological pathways and cascades',
  COMPOUND: 'General chemical compounds',
  CONCEPT:  'Other biomedical concepts',
  ORGANISM: 'Species, strains, model organisms',
};

function entityColor(type: string): string {
  return TYPE_COLORS[type.toUpperCase()] ?? '#94a3b8';
}

export default function GraphPage() {
  const { data: entData, isLoading: entLoading } = useQuery({
    queryKey: ['entities'],
    queryFn: () => getEntities({ limit: 500 }),
  });
  const { data: relData, isLoading: relLoading } = useQuery({
    queryKey: ['relationships'],
    queryFn: () => getRelationships({ limit: 1000 }),
  });

  const [search, setSearch] = useState('');
  const [activeTypes, setActiveTypes] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState<Entity | null>(null);
  const [width, setWidth] = useState(900);
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<any>(null);

  // Track container width for the force graph to size to its parent
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) setWidth(e.contentRect.width);
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // All entity types found in the loaded graph (for the legend rows).
  const allTypes = useMemo(() => {
    const t = new Set<string>();
    (entData?.items ?? []).forEach((e) => t.add(e.entity_type.toUpperCase()));
    return Array.from(t).sort();
  }, [entData]);

  // Count nodes per type — feeds the count column in the legend.
  const typeCounts = useMemo(() => {
    const m: Record<string, number> = {};
    (entData?.items ?? []).forEach((e) => {
      const k = e.entity_type.toUpperCase();
      m[k] = (m[k] ?? 0) + 1;
    });
    return m;
  }, [entData]);

  const filteredEntities = useMemo(() => {
    const list: Entity[] = entData?.items ?? [];
    const q = search.trim().toLowerCase();
    return list.filter((e) => {
      if (activeTypes.size > 0 && !activeTypes.has(e.entity_type.toUpperCase())) return false;
      if (q && !e.name.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [entData, search, activeTypes]);

  const visibleIds = useMemo(
    () => new Set(filteredEntities.map((e) => e.id)),
    [filteredEntities]
  );

  const filteredRels = useMemo(() => {
    const list: Relationship[] = relData?.items ?? [];
    return list.filter(
      (r) => visibleIds.has(r.source_entity_id) && visibleIds.has(r.target_entity_id)
    );
  }, [relData, visibleIds]);

  const graphData = useMemo(
    () => ({
      nodes: filteredEntities.map((e) => ({
        id: e.id,
        name: e.name,
        type: e.entity_type,
        color: entityColor(e.entity_type),
        raw: e,
      })),
      links: filteredRels.map((r) => ({
        source: r.source_entity_id,
        target: r.target_entity_id,
        label: r.relationship_type,
      })),
    }),
    [filteredEntities, filteredRels]
  );

  const loading = entLoading || relLoading;

  function toggleType(t: string) {
    setActiveTypes((prev) => {
      const next = new Set(prev);
      if (next.has(t)) next.delete(t);
      else next.add(t);
      return next;
    });
  }

  function recenter() {
    fgRef.current?.zoomToFit?.(400, 60);
  }

  return (
    <div className="space-y-6">
      <header className="flex items-end justify-between flex-wrap gap-3">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span
              className="badge badge-brand"
              style={{ background: 'var(--accent-soft)', border: '1px solid rgba(213,33,44,0.25)' }}
            >
              <Sparkles size={10} className="opacity-70" />
              Live · Interactive
            </span>
          </div>
          <h1 className="page-title">Knowledge Graph</h1>
          <p className="page-desc">
            Drag nodes to rearrange, scroll to zoom, click to inspect.
            Each colour represents an entity type — see the legend on the right.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="card px-3 py-1.5 flex items-center gap-1.5" title="Visible nodes">
            <Network size={12} style={{ color: 'var(--accent)' }} />
            <span className="text-xs font-semibold tabular-nums">
              {filteredEntities.length}
            </span>
            <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
              nodes
            </span>
          </div>
          <div className="card px-3 py-1.5 flex items-center gap-1.5" title="Visible edges">
            <GitBranch size={12} style={{ color: 'var(--accent)' }} />
            <span className="text-xs font-semibold tabular-nums">
              {filteredRels.length}
            </span>
            <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
              edges
            </span>
          </div>
          <button
            onClick={recenter}
            className="btn-ghost"
            title="Fit the entire graph into the viewport"
          >
            <Maximize2 size={13} />
            Recenter
          </button>
        </div>
      </header>

      {/* Search bar (filter chips moved into the Legend panel) */}
      <div className="card flex items-center gap-2 px-3 py-2 max-w-2xl">
        <Search size={14} style={{ color: 'var(--text-muted)' }} />
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search entities by name (e.g. BRCA1, hemophilia)…"
          className="bg-transparent outline-none text-sm flex-1"
        />
        {search && (
          <button
            onClick={() => setSearch('')}
            className="text-xs text-slate-400 hover:text-slate-700"
            title="Clear search"
          >
            <X size={13} />
          </button>
        )}
        {activeTypes.size > 0 && (
          <button
            onClick={() => setActiveTypes(new Set())}
            className="text-[11px] font-semibold px-2 py-1 rounded-md"
            style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}
            title="Show all entity types"
          >
            Reset filter ({activeTypes.size})
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        <div className="lg:col-span-3 card overflow-hidden relative" ref={containerRef}>
          {loading ? (
            <div
              className="h-[640px] flex items-center justify-center text-sm gap-2"
              style={{ color: 'var(--text-muted)' }}
            >
              <Activity size={16} className="animate-pulse" /> Loading graph
            </div>
          ) : (
            <ForceGraph2D
              fgRef={fgRef}
              graphData={graphData}
              nodeLabel={(n: any) => `${n.name} (${n.type})`}
              nodeColor={(n: any) => n.color}
              nodeRelSize={5}
              linkColor={() => 'rgba(148,163,184,0.45)'}
              linkLabel={(l: any) => l.label}
              linkDirectionalArrowLength={3}
              linkDirectionalArrowRelPos={1}
              linkWidth={(l: any) => {
                if (!selected) return 1;
                return l.source.id === selected.id || l.target.id === selected.id ? 2.5 : 0.5;
              }}
              nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
                const isSelected = selected?.id === node.id;
                const r = isSelected ? 7 : 5;
                ctx.beginPath();
                ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
                ctx.fillStyle = node.color;
                ctx.fill();
                if (isSelected) {
                  ctx.lineWidth = 2;
                  ctx.strokeStyle = 'rgba(213,33,44,0.90)';
                  ctx.stroke();
                  // Subtle CSL-red glow
                  ctx.beginPath();
                  ctx.arc(node.x, node.y, r + 4, 0, 2 * Math.PI);
                  ctx.strokeStyle = 'rgba(213,33,44,0.25)';
                  ctx.lineWidth = 4;
                  ctx.stroke();
                }
                if (globalScale > 1.3 || isSelected) {
                  ctx.font = `${isSelected ? 11 : 9}px Inter, sans-serif`;
                  ctx.fillStyle = isSelected ? '#0f172a' : 'rgba(15,23,42,0.65)';
                  ctx.textAlign = 'center';
                  ctx.textBaseline = 'top';
                  ctx.fillText(node.name, node.x, node.y + r + 2);
                }
              }}
              onNodeClick={(n: any) => setSelected(n.raw)}
              onBackgroundClick={() => setSelected(null)}
              width={Math.max(width, 400)}
              height={640}
              backgroundColor="#fafbff"
              cooldownTicks={120}
            />
          )}
        </div>

        {/* Right column: Legend (always visible) + Inspector */}
        <div className="space-y-4 lg:sticky lg:top-4 h-fit">
          {/* ── Legend ───────────────────────────────────────────────── */}
          <div className="card p-4">
            <div className="flex items-center justify-between mb-3">
              <p
                className="text-[10px] uppercase tracking-wider font-semibold"
                style={{ color: 'var(--text-muted)' }}
              >
                Legend
              </p>
              <span
                className="help-icon"
                title="Click a row to filter the graph to that type. Click again to remove the filter."
              >
                ?
              </span>
            </div>
            <ul className="space-y-1">
              {allTypes.map((t) => {
                const filterActive = activeTypes.size > 0;
                const isOn = !filterActive || activeTypes.has(t);
                const color = entityColor(t);
                const count = typeCounts[t] ?? 0;
                return (
                  <li key={t}>
                    <button
                      onClick={() => toggleType(t)}
                      className="w-full text-left px-2 py-1.5 rounded-md flex items-center gap-2 transition-all"
                      style={{
                        background: filterActive && isOn ? `${color}10` : 'transparent',
                        border: `1px solid ${filterActive && isOn ? `${color}33` : 'transparent'}`,
                        opacity: isOn ? 1 : 0.4,
                      }}
                      title={
                        !filterActive
                          ? `Click to show only ${t} (${TYPE_DESCRIPTIONS[t] ?? ''})`
                          : isOn
                            ? `${t} is shown — click to hide`
                            : `${t} is hidden — click to show`
                      }
                    >
                      <span
                        className="h-2.5 w-2.5 rounded-full shrink-0"
                        style={{
                          background: color,
                          boxShadow: isOn ? `0 0 0 3px ${color}22` : 'none',
                        }}
                      />
                      <span
                        className="text-[12.5px] font-semibold flex-1"
                        style={{ color: isOn ? 'var(--text-primary)' : 'var(--text-muted)' }}
                      >
                        {t}
                      </span>
                      <span
                        className="text-[11px] tabular-nums font-semibold"
                        style={{ color: 'var(--text-muted)' }}
                      >
                        {count}
                      </span>
                    </button>
                    {TYPE_DESCRIPTIONS[t] && (
                      <p
                        className="text-[10.5px] pl-7 -mt-0.5 mb-1"
                        style={{ color: 'var(--text-muted)' }}
                      >
                        {TYPE_DESCRIPTIONS[t]}
                      </p>
                    )}
                  </li>
                );
              })}
            </ul>
            {allTypes.length === 0 && (
              <p className="text-[12px]" style={{ color: 'var(--text-muted)' }}>
                Graph is empty.
              </p>
            )}
          </div>

          {/* ── Inspector ────────────────────────────────────────────── */}
          <div className="card p-4">
          {selected ? (
            <div className="space-y-3 fade-up">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <p className="text-[10px] uppercase tracking-wider font-semibold" style={{ color: 'var(--text-muted)' }}>
                    Selected entity
                  </p>
                  <p className="text-base font-semibold mt-0.5" style={{ color: 'var(--text-primary)' }}>
                    {selected.name}
                  </p>
                </div>
                <button
                  onClick={() => setSelected(null)}
                  className="text-slate-400 hover:text-slate-600"
                  title="Clear"
                >
                  <X size={14} />
                </button>
              </div>
              <span
                className="badge"
                style={{
                  background: `${entityColor(selected.entity_type)}15`,
                  color: entityColor(selected.entity_type),
                  border: `1px solid ${entityColor(selected.entity_type)}40`,
                }}
              >
                {selected.entity_type}
              </span>
              {selected.description && (
                <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                  {selected.description}
                </p>
              )}
              {selected.source_document_ids?.length > 0 && (
                <div>
                  <p className="text-[10px] uppercase tracking-wider font-semibold mb-1" style={{ color: 'var(--text-muted)' }}>
                    Sources
                  </p>
                  <p className="text-[11px] font-mono opacity-70">
                    {selected.source_document_ids.length} document{selected.source_document_ids.length === 1 ? '' : 's'}
                  </p>
                </div>
              )}
              <div>
                <p className="text-[10px] uppercase tracking-wider font-semibold mb-1.5" style={{ color: 'var(--text-muted)' }}>
                  Connected via
                </p>
                <ul className="space-y-1 max-h-48 overflow-y-auto">
                  {filteredRels
                    .filter((r) => r.source_entity_id === selected.id || r.target_entity_id === selected.id)
                    .slice(0, 12)
                    .map((r) => {
                      const otherId =
                        r.source_entity_id === selected.id ? r.target_entity_id : r.source_entity_id;
                      const other = entData?.items.find((e) => e.id === otherId);
                      return (
                        <li
                          key={r.id}
                          className="text-[11.5px] px-2 py-1.5 rounded-md flex items-center gap-1.5"
                          style={{ background: 'var(--bg-muted)' }}
                        >
                          <span className="opacity-60">{r.relationship_type}</span>
                          <span className="font-medium truncate" style={{ color: 'var(--text-primary)' }}>
                            {other?.name ?? otherId.slice(0, 8)}
                          </span>
                        </li>
                      );
                    })}
                </ul>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <span className="empty-icon">
                <Network size={20} />
              </span>
              <p className="text-[12.5px] font-medium" style={{ color: 'var(--text-secondary)' }}>
                Click a node to inspect
              </p>
              <p className="text-[10.5px]">
                Type, description, source documents, and the entity's
                neighbours will appear here.
              </p>
            </div>
          )}
          </div>
        </div>
      </div>
    </div>
  );
}
