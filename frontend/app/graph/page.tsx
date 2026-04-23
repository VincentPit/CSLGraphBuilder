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
  // Anchor position of the floating inspector — pixels relative to the
  // canvas container. Re-computed on every animation frame while a node
  // is selected so the popover stays glued to the node through drag /
  // pan / zoom / simulation tick.
  const [popoverPos, setPopoverPos] = useState<{ x: number; y: number } | null>(null);
  const [width, setWidth] = useState(900);
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<any>(null);
  // Holds the actual force-graph node object (with live x/y) — distinct
  // from the React `selected` Entity (which is the persisted record).
  const selectedNodeRef = useRef<any>(null);

  // Track container width for the force graph to size to its parent
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) setWidth(e.contentRect.width);
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // Glue the popover to the selected node. The force-graph instance gives
  // us a `graph2ScreenCoords(x, y)` helper that returns canvas-space
  // pixels. We poll it on every animation frame — cheaper than wiring up
  // every onZoom / onPan / onTick / onDrag callback the library exposes,
  // and immune to any we forget. Loop tears down the moment `selected`
  // clears.
  useEffect(() => {
    if (!selected || !selectedNodeRef.current) {
      setPopoverPos(null);
      return;
    }
    let raf = 0;
    let alive = true;
    const tick = () => {
      if (!alive) return;
      const fg = fgRef.current;
      const node = selectedNodeRef.current;
      if (fg?.graph2ScreenCoords && typeof node?.x === 'number') {
        const p = fg.graph2ScreenCoords(node.x, node.y);
        setPopoverPos({ x: p.x, y: p.y });
      }
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => {
      alive = false;
      if (raf) cancelAnimationFrame(raf);
    };
  }, [selected]);

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
              onNodeClick={(n: any) => {
                selectedNodeRef.current = n;
                setSelected(n.raw);
              }}
              onBackgroundClick={() => {
                selectedNodeRef.current = null;
                setSelected(null);
              }}
              width={Math.max(width, 400)}
              height={640}
              backgroundColor="#fafbff"
              cooldownTicks={120}
            />
          )}

          {/* Floating inspector — anchored to the selected node, follows
              it through pan / zoom / drag. Constrained inside the canvas
              container so it can't escape into the legend column. */}
          {selected && popoverPos && (
            <FloatingInspector
              entity={selected}
              anchor={popoverPos}
              container={containerRef.current}
              relationships={filteredRels}
              entitiesById={entData?.items}
              onClose={() => {
                selectedNodeRef.current = null;
                setSelected(null);
              }}
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

          {/* Inspector lives inline only when nothing is selected — it
              becomes a hint card. Once a node is clicked, the popover
              floats over the canvas instead. */}
          {!selected && (
            <div className="card p-4">
              <div className="empty-state">
                <span className="empty-icon">
                  <Network size={20} />
                </span>
                <p className="text-[12.5px] font-medium" style={{ color: 'var(--text-secondary)' }}>
                  Click a node to inspect
                </p>
                <p className="text-[10.5px]">
                  A details card will pop up next to the node showing its
                  type, description, sources, and neighbours.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────
   FloatingInspector — small popover anchored to a graph node.

   - Absolutely positioned inside the canvas card (which is `relative`).
   - On mount + every layout pass, measures itself and clamps placement
     so it stays inside the container: prefers right-of-node, flips
     left if it would overflow; same logic vertically.
   - Visual: thin connector line + dot pointing back at the node so
     the relationship between popover and node stays clear under heavy
     panning.
   ───────────────────────────────────────────────────────────────── */
function FloatingInspector({
  entity,
  anchor,
  container,
  relationships,
  entitiesById,
  onClose,
}: {
  entity: Entity;
  anchor: { x: number; y: number };
  container: HTMLElement | null;
  relationships: Relationship[];
  entitiesById?: Entity[];
  onClose: () => void;
}) {
  const popRef = useRef<HTMLDivElement>(null);
  const [placement, setPlacement] = useState<{
    left: number;
    top: number;
    side: 'right' | 'left';
  }>({ left: anchor.x + 18, top: anchor.y - 60, side: 'right' });

  // Reposition every time the anchor moves (we re-render via the rAF
  // loop in the parent).
  useEffect(() => {
    const pop = popRef.current;
    if (!pop || !container) return;
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    const pw = pop.offsetWidth;
    const ph = pop.offsetHeight;
    const margin = 8;
    const offset = 18;

    // Prefer right of the node; flip if it would overflow.
    let side: 'right' | 'left' = 'right';
    let left = anchor.x + offset;
    if (left + pw + margin > cw) {
      side = 'left';
      left = anchor.x - offset - pw;
    }
    // Clamp within container horizontally as a final safety net.
    left = Math.max(margin, Math.min(left, cw - pw - margin));

    // Vertically centre on the node, then clamp.
    let top = anchor.y - ph / 2;
    top = Math.max(margin, Math.min(top, ch - ph - margin));

    setPlacement({ left, top, side });
  }, [anchor.x, anchor.y, container, entity.id]);

  const color = entityColorLocal(entity.entity_type);
  const neighbours = relationships
    .filter((r) => r.source_entity_id === entity.id || r.target_entity_id === entity.id)
    .slice(0, 8);

  return (
    <>
      {/* Connector line: thin red dotted segment from node to popover */}
      <svg
        className="absolute pointer-events-none"
        style={{ left: 0, top: 0, width: '100%', height: '100%' }}
      >
        <line
          x1={anchor.x}
          y1={anchor.y}
          x2={placement.side === 'right' ? placement.left : placement.left + 280}
          y2={placement.top + 24}
          stroke="rgba(213,33,44,0.45)"
          strokeWidth={1.5}
          strokeDasharray="3 3"
        />
        <circle cx={anchor.x} cy={anchor.y} r={4} fill="rgba(213,33,44,0.85)" />
      </svg>

      <div
        ref={popRef}
        className="absolute card p-3.5 fade-up"
        style={{
          left: `${placement.left}px`,
          top: `${placement.top}px`,
          width: 280,
          zIndex: 20,
          boxShadow: '0 12px 30px -6px rgba(15,23,42,0.18), 0 4px 10px -4px rgba(15,23,42,0.10)',
        }}
        // Background-click to close shouldn't fire when clicking inside.
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-2 mb-2">
          <div className="min-w-0">
            <p
              className="text-[10px] uppercase tracking-wider font-semibold"
              style={{ color: 'var(--text-muted)' }}
            >
              Selected
            </p>
            <p
              className="text-[15px] font-semibold leading-tight truncate"
              style={{ color: 'var(--text-primary)' }}
              title={entity.name}
            >
              {entity.name}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-700 -m-1 p-1 rounded"
            title="Close (or click background)"
          >
            <X size={14} />
          </button>
        </div>

        <span
          className="badge mb-2 inline-flex"
          style={{
            background: `${color}15`,
            color,
            border: `1px solid ${color}40`,
          }}
        >
          {entity.entity_type}
        </span>

        {entity.description && (
          <p
            className="text-[11.5px] leading-relaxed mb-2 line-clamp-3"
            style={{ color: 'var(--text-secondary)' }}
          >
            {entity.description}
          </p>
        )}

        {entity.source_document_ids?.length > 0 && (
          <p
            className="text-[10.5px] mb-2"
            style={{ color: 'var(--text-muted)' }}
          >
            {entity.source_document_ids.length} source document{entity.source_document_ids.length === 1 ? '' : 's'}
          </p>
        )}

        {neighbours.length > 0 && (
          <div>
            <p
              className="text-[10px] uppercase tracking-wider font-semibold mb-1"
              style={{ color: 'var(--text-muted)' }}
            >
              {neighbours.length} connection{neighbours.length === 1 ? '' : 's'}
            </p>
            <ul className="space-y-1 max-h-32 overflow-y-auto">
              {neighbours.map((r) => {
                const otherId =
                  r.source_entity_id === entity.id
                    ? r.target_entity_id
                    : r.source_entity_id;
                const other = entitiesById?.find((e) => e.id === otherId);
                return (
                  <li
                    key={r.id}
                    className="text-[11px] px-2 py-1 rounded-md flex items-center gap-1.5 truncate"
                    style={{ background: 'var(--bg-muted)' }}
                  >
                    <span className="opacity-60 shrink-0">
                      {r.relationship_type}
                    </span>
                    <span
                      className="font-medium truncate"
                      style={{ color: 'var(--text-primary)' }}
                    >
                      {other?.name ?? otherId.slice(0, 8)}
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>
        )}
      </div>
    </>
  );
}

// Local copy so the popover doesn't depend on hoisting order.
function entityColorLocal(type: string): string {
  return TYPE_COLORS[type.toUpperCase()] ?? '#94a3b8';
}
