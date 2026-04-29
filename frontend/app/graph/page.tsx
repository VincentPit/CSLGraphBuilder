'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import { useQueries, useQuery } from '@tanstack/react-query';
import {
  Activity,
  GitBranch,
  Maximize2,
  Network,
  Search,
  Sparkles,
  X,
} from 'lucide-react';
import {
  getEntityTypes,
  getSubgraph,
  type Entity,
  type Relationship,
} from '@/lib/api';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { LoadingState } from '@/components/ui/LoadingState';

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

// Shape-per-type for the graph nodes. Color carries the same signal but
// shape adds a second redundant channel that survives at low zoom (when
// dots are too small for the eye to compare colors reliably). Defaults
// to circle for any type without an explicit mapping.
type NodeShape = 'circle' | 'square' | 'triangle' | 'diamond';
const TYPE_SHAPES: Record<string, NodeShape> = {
  DISEASE: 'triangle',
  DRUG: 'square',
  COMPOUND: 'square',
  PATHWAY: 'diamond',
  GENE: 'circle',
  PROTEIN: 'circle',
};
function shapeFor(type: string): NodeShape {
  return TYPE_SHAPES[type.toUpperCase()] ?? 'circle';
}

/** Tiny inline SVG matching the canvas shape — used in the legend so
 *  the swatch next to each type matches the dot drawn in the graph. */
function LegendShape({ type, color }: { type: string; color: string }) {
  const shape = shapeFor(type);
  const stroke = 'rgba(15,23,42,0.45)';
  const common = { fill: color, stroke, strokeWidth: 1 } as const;
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
      {shape === 'square' && <rect x="1" y="1" width="10" height="10" {...common} />}
      {shape === 'triangle' && <polygon points="6,1 11,10.5 1,10.5" {...common} />}
      {shape === 'diamond' && <polygon points="6,1 11,6 6,11 1,6" {...common} />}
      {shape === 'circle' && <circle cx="6" cy="6" r="5" {...common} />}
    </svg>
  );
}

/** Trace the outline of a typed node onto the canvas at (x,y) with
 *  half-extent r. Caller handles fill + stroke after calling this. */
function tracePath(
  ctx: CanvasRenderingContext2D,
  shape: NodeShape,
  x: number,
  y: number,
  r: number,
) {
  ctx.beginPath();
  switch (shape) {
    case 'square':
      ctx.rect(x - r, y - r, 2 * r, 2 * r);
      break;
    case 'triangle': {
      // Equilateral-ish triangle, point up.
      const h = r * 1.15;
      ctx.moveTo(x, y - h);
      ctx.lineTo(x - r, y + h * 0.55);
      ctx.lineTo(x + r, y + h * 0.55);
      ctx.closePath();
      break;
    }
    case 'diamond':
      ctx.moveTo(x, y - r);
      ctx.lineTo(x + r, y);
      ctx.lineTo(x, y + r);
      ctx.lineTo(x - r, y);
      ctx.closePath();
      break;
    case 'circle':
    default:
      ctx.arc(x, y, r, 0, 2 * Math.PI);
  }
}

// Order in which entity types are fetched. The first few are the most
// useful to land on screen quickly for biomedical browsing, so they
// run first in the per-type fan-out. Anything not in this list is
// appended at the end so unknown / future EntityType values still get
// loaded.
const TYPE_FETCH_PRIORITY = [
  'DISEASE',
  'GENE',
  'DRUG',
  'PROTEIN',
  'PATHWAY',
  'COMPOUND',
  'CONCEPT',
  'ORGANISM',
];

// Types we never seed from. Mirrors the historical default of the
// /graph/subgraph endpoint — Document entities are GWAS studies and
// clutter the visualisation. Kept here too so the per-type fan-out
// doesn't fire a wasted request for them.
const EXCLUDED_SEED_TYPES = new Set(['DOCUMENT']);

export default function GraphPage() {
  // ── Progressive loading ────────────────────────────────────────────
  // Instead of one big subgraph fetch (which would leave the canvas
  // blank until the full ~hundreds-of-nodes payload finished), we fan
  // out one request **per entity type**. Each request returns a self-
  // consistent slice (seeds of that type + their edges + cross-type
  // neighbours) so the graph stays coherent as batches arrive.
  //
  // Step 1 — pull the enum of valid entity types from the backend.
  // This is an instant lookup (just enum values, no DB scan) and
  // avoids hard-coding the list in two places.
  const { data: typeList } = useQuery({
    queryKey: ['entity-types'],
    queryFn: getEntityTypes,
    staleTime: 5 * 60 * 1000, // enum doesn't change between deploys
  });

  // Order types so the high-signal ones render first. We restrict to
  // the biomedical priority list rather than fetching every value in
  // the EntityType enum — most non-biomedical legacy types (Person,
  // Organization, …) are empty in practice, and an extra request per
  // empty type still triggers a full backend scan. Future biomedical
  // additions should be added to TYPE_FETCH_PRIORITY explicitly.
  const orderedTypes = useMemo(() => {
    if (!typeList) return [] as string[];
    const upper = new Set(typeList.map((t) => t.toUpperCase()));
    return TYPE_FETCH_PRIORITY.filter(
      (t) => upper.has(t) && !EXCLUDED_SEED_TYPES.has(t),
    );
  }, [typeList]);

  // Step 2 — fan out one /graph/subgraph request per type. React Query
  // will dispatch them in parallel; whichever returns first paints
  // first. We pass the original enum-cased value (not uppercased) so
  // the backend's case-insensitive match works regardless of which
  // EntityType uses TitleCase vs UPPERCASE in the enum.
  const typeQueries = useQueries({
    queries: orderedTypes.map((upperType) => {
      // Recover the original-cased name from typeList for the request
      // string. Backend is case-insensitive but this keeps cache keys
      // consistent with the enum the user sees in /types/entities.
      const original =
        typeList?.find((t) => t.toUpperCase() === upperType) ?? upperType;
      return {
        queryKey: ['subgraph', 'by-type', original],
        queryFn: () =>
          getSubgraph({
            per_type_limit: 50,
            exclude_types: 'Document',
            include_types: original,
            max_neighbors: 2000,
          }),
        staleTime: 30 * 1000,
      };
    }),
  });

  // Step 3 — accumulate completed batches into deduped Maps. A node or
  // edge that appears in multiple per-type slices (e.g. a PROTEIN that
  // links to both a GENE and a DISEASE) gets kept once. Memoized on
  // the *successful query identities* rather than the queries array
  // itself so the rAF popover loop in this component (which causes
  // re-renders ~60×/sec) doesn't rebuild these every frame.
  const completedSignature = typeQueries
    .map((q) => (q.data ? `${q.data.entities.length}:${q.data.relationships.length}` : '-'))
    .join('|');

  const { entityList, relationshipList, totalEntities, totalRelationships } = useMemo(() => {
    const entMap = new Map<string, Entity>();
    const relMap = new Map<string, Relationship>();
    let totalEnt = 0;
    let totalRel = 0;
    for (const q of typeQueries) {
      const slice = q.data;
      if (!slice) continue;
      for (const e of slice.entities) entMap.set(e.id, e);
      for (const r of slice.relationships) relMap.set(r.id, r);
      // total_* is the same for every slice (it's the global count,
      // not per-type), so the last non-zero wins. Take the max so
      // mid-load mismatches don't flicker the badge downward.
      totalEnt = Math.max(totalEnt, slice.total_entities);
      totalRel = Math.max(totalRel, slice.total_relationships);
    }
    return {
      entityList: Array.from(entMap.values()),
      relationshipList: Array.from(relMap.values()),
      totalEntities: totalEnt,
      totalRelationships: totalRel,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [completedSignature]);

  // Adapter shape so the rest of this file (which was written against
  // entData.items / relData.items) keeps working unchanged.
  const entData = useMemo(
    () => ({ items: entityList, total: totalEntities }),
    [entityList, totalEntities],
  );
  const relData = useMemo(
    () => ({ items: relationshipList, total: totalRelationships }),
    [relationshipList, totalRelationships],
  );

  // Progressive loading state. We only block the canvas with the full-
  // page LoadingState until the *first* batch lands; after that, the
  // graph is rendered and remaining types stream in alongside a small
  // inline progress chip so users can interact with what's there.
  const completedTypes = typeQueries.filter((q) => q.isSuccess).length;
  const totalTypes = typeQueries.length;
  const initialLoad = totalTypes === 0 || (completedTypes === 0 && entityList.length === 0);
  const stillStreaming = totalTypes > 0 && completedTypes < totalTypes;
  const entLoading = initialLoad;
  const relLoading = initialLoad;

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
    let lastX: number | null = null;
    let lastY: number | null = null;
    const tick = () => {
      if (!alive) return;
      const fg = fgRef.current;
      const node = selectedNodeRef.current;
      if (fg?.graph2ScreenCoords && typeof node?.x === 'number') {
        const p = fg.graph2ScreenCoords(node.x, node.y);
        // Only propagate when the screen position changed by ≥0.5px;
        // a per-frame setState with a fresh object would otherwise
        // re-render the page 60×/sec for nothing. First tick (lastX
        // null) always publishes so the inspector mounts.
        if (
          lastX === null ||
          Math.abs(p.x - lastX) > 0.5 ||
          Math.abs(p.y - (lastY as number)) > 0.5
        ) {
          lastX = p.x;
          lastY = p.y;
          setPopoverPos({ x: p.x, y: p.y });
        }
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

  // Force-directed layout tuning. Two competing goals:
  //   - Pull disconnected clusters close enough that the canvas isn't
  //     mostly empty space.
  //   - Keep nodes inside a cluster from overlapping.
  // Charge needs to be strong enough to counteract the link force on a
  // ~300-node, densely-connected graph; -50 was too weak and let every
  // hub collapse into a tight ball.
  //
  // Re-key on node/link counts (not the graphData object reference),
  // so the rAF popover loop can churn React state without dragging the
  // simulation back to alpha=1 every frame.
  const nodeCount = graphData.nodes.length;
  const linkCount = graphData.links.length;
  useEffect(() => {
    if (!nodeCount) return;
    const fg = fgRef.current;
    if (!fg?.d3Force) return;
    const charge = fg.d3Force('charge');
    const link = fg.d3Force('link');
    if (charge?.strength) charge.strength(-180);   // default ~-30; needed to spread dense hubs
    if (link?.distance) link.distance(60);          // default ~30; room for labels
    // Collision: enforce a minimum gap so the bigger shapes from the
    // canvas redesign don't visually pile up. d3 isn't imported here,
    // so we register a small custom per-tick force that pushes any
    // pair closer than ``minDist`` apart proportional to alpha. The
    // force reads nodes off the live d3 simulation (passed in via
    // initialize) rather than a captured React array, so it stays
    // correct after filter changes without re-registering.
    const minDist = 28;
    let simNodes: any[] = [];
    const collide = (alpha: number) => {
      for (let i = 0; i < simNodes.length; i++) {
        const ni = simNodes[i];
        for (let j = i + 1; j < simNodes.length; j++) {
          const nj = simNodes[j];
          const dx = (nj.x ?? 0) - (ni.x ?? 0);
          const dy = (nj.y ?? 0) - (ni.y ?? 0);
          const d2 = dx * dx + dy * dy;
          if (d2 < minDist * minDist && d2 > 0.0001) {
            const d = Math.sqrt(d2);
            const push = ((minDist - d) / d) * 0.5 * alpha;
            ni.x -= dx * push;
            ni.y -= dy * push;
            nj.x += dx * push;
            nj.y += dy * push;
          }
        }
      }
    };
    (collide as any).initialize = (nodes: any[]) => {
      simNodes = nodes;
    };
    fg.d3Force('collide', collide);
    fg.d3ReheatSimulation?.();
  }, [nodeCount, linkCount]);

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
    <div className="space-y-8 lg:space-y-10">
      <header className="flex items-end justify-between flex-wrap gap-4">
        <div className="space-y-3">
          <Badge tone="brand">
            <Sparkles size={10} className="opacity-70" aria-hidden="true" />
            Live · Interactive
          </Badge>
          <h1 className="page-title">Knowledge Graph</h1>
          <p className="page-desc">
            Drag nodes to rearrange, scroll to zoom, click to inspect.
            Each colour represents an entity type — use the filter panel
            on the left to narrow the view.
          </p>
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <div className="card px-3 py-1.5 flex items-center gap-1.5" aria-label={`${filteredEntities.length} visible nodes`}>
            <Network size={12} style={{ color: 'var(--accent)' }} aria-hidden="true" />
            <span className="text-xs font-semibold tabular-nums">
              {filteredEntities.length}
            </span>
            <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
              nodes
            </span>
          </div>
          <div className="card px-3 py-1.5 flex items-center gap-1.5" aria-label={`${filteredRels.length} visible edges`}>
            <GitBranch size={12} style={{ color: 'var(--accent)' }} aria-hidden="true" />
            <span className="text-xs font-semibold tabular-nums">
              {filteredRels.length}
            </span>
            <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
              edges
            </span>
          </div>
          {/* Progressive-loader status: visible only while per-type
              batches are still arriving. The graph itself is already
              interactive at this point, so this is non-blocking. */}
          {stillStreaming && (
            <div
              className="card px-3 py-1.5 flex items-center gap-2"
              role="status"
              aria-live="polite"
              aria-label={`Loading ${totalTypes - completedTypes} more entity type${totalTypes - completedTypes === 1 ? '' : 's'}`}
            >
              <span
                className="inline-block w-2.5 h-2.5 rounded-full"
                style={{
                  background: 'var(--accent)',
                  animation: 'pulse 1.4s ease-in-out infinite',
                }}
                aria-hidden="true"
              />
              <span className="text-[11px] tabular-nums" style={{ color: 'var(--text-muted)' }}>
                Loading{' '}
                <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>
                  {completedTypes}/{totalTypes}
                </span>{' '}
                types
              </span>
            </div>
          )}
          <Button
            onClick={recenter}
            title="Fit the entire graph into the viewport"
            aria-label="Recenter graph to fit viewport"
          >
            <Maximize2 size={13} aria-hidden="true" />
            Recenter
          </Button>
        </div>
      </header>

      {/* Prominent search bar — full width, large hit area. */}
      <div
        className="flex items-center gap-3 px-5 py-4"
        style={{
          background: 'var(--bg-card)',
          border: '1.5px solid var(--border-input)',
          borderRadius: 'var(--radius-md)',
          boxShadow: 'var(--shadow-card)',
        }}
      >
        <Search size={22} style={{ color: 'var(--text-muted)' }} aria-hidden="true" />
        <label htmlFor="graph-search" className="sr-only">Search entities</label>
        <input
          id="graph-search"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search entities by name (e.g. BRCA1, hemophilia)…"
          className="bg-transparent outline-none text-[16px] font-semibold flex-1 py-1"
          style={{ color: 'var(--text-primary)' }}
        />
        {search && (
          <button
            onClick={() => setSearch('')}
            className="btn-icon"
            style={{ width: 34, height: 34 }}
            aria-label="Clear search"
          >
            <X size={15} aria-hidden="true" />
          </button>
        )}
        {activeTypes.size > 0 && (
          <button
            onClick={() => setActiveTypes(new Set())}
            className="text-[12px] font-semibold px-3 py-1.5 rounded-md whitespace-nowrap"
            style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}
            aria-label={`Reset entity-type filter (${activeTypes.size} active)`}
          >
            Reset filter ({activeTypes.size})
          </button>
        )}
      </div>

      {/* Filter on the LEFT always, graph on the RIGHT. Always
          side-by-side regardless of viewport width. */}
      <div className="flex gap-5">
        <aside
          className="w-[240px] shrink-0 sticky top-4 h-fit"
        >
          <FilterPanel
            allTypes={allTypes}
            typeCounts={typeCounts}
            activeTypes={activeTypes}
            toggleType={toggleType}
            clearFilter={() => setActiveTypes(new Set())}
          />
        </aside>

        {/* Right column: the graph itself */}
        <div className="flex-1 min-w-0 card overflow-hidden relative" ref={containerRef}>
          {loading ? (
            <div className="h-[640px] flex items-center justify-center">
              <LoadingState>Loading graph</LoadingState>
            </div>
          ) : (
            <ForceGraph2D
              fgRef={fgRef}
              graphData={graphData}
              nodeLabel={(n: any) => `${n.name} (${n.type})`}
              nodeColor={(n: any) => n.color}
              nodeRelSize={9}
              linkColor={() => 'rgba(148,163,184,0.55)'}
              linkLabel={(l: any) => l.label}
              linkDirectionalArrowLength={3}
              linkDirectionalArrowRelPos={1}
              linkWidth={(l: any) => {
                if (!selected) return 1.2;
                return l.source.id === selected.id || l.target.id === selected.id ? 2.5 : 0.5;
              }}
              nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
                const isSelected = selected?.id === node.id;
                const r = isSelected ? 11 : 9;
                const shape = shapeFor(node.type);
                // Subtle dark outline so colored shapes stay legible
                // against the off-white background.
                ctx.fillStyle = node.color;
                ctx.strokeStyle = 'rgba(15,23,42,0.45)';
                ctx.lineWidth = 1.25;
                tracePath(ctx, shape, node.x, node.y, r);
                ctx.fill();
                ctx.stroke();
                if (isSelected) {
                  // CSL-red ring + glow.
                  ctx.lineWidth = 2;
                  ctx.strokeStyle = 'rgba(213,33,44,0.90)';
                  tracePath(ctx, shape, node.x, node.y, r);
                  ctx.stroke();
                  tracePath(ctx, shape, node.x, node.y, r + 5);
                  ctx.strokeStyle = 'rgba(213,33,44,0.25)';
                  ctx.lineWidth = 4;
                  ctx.stroke();
                }
                // Labels only appear when zoomed in moderately, or for
                // the selected node. Showing every label at default zoom
                // produced an unreadable wall of overlapping text.
                if (globalScale > 1.4 || isSelected) {
                  ctx.font = `${isSelected ? 12 : 10}px Inter, sans-serif`;
                  ctx.fillStyle = isSelected ? '#0f172a' : 'rgba(15,23,42,0.7)';
                  ctx.textAlign = 'center';
                  ctx.textBaseline = 'top';
                  ctx.fillText(node.name, node.x, node.y + r + 3);
                }
              }}
              nodePointerAreaPaint={(node: any, color: string, ctx: CanvasRenderingContext2D) => {
                // Match the visible shape so click hit-testing aligns.
                ctx.fillStyle = color;
                tracePath(ctx, shapeFor(node.type), node.x, node.y, 11);
                ctx.fill();
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

      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────
   FilterPanel — checkbox-style entity-type filter panel that lives
   in the left column of the graph page.

   Behaviour mirrors the legacy legend: when no boxes are checked,
   ALL entity types are visible (no filter). Checking one or more
   boxes restricts the graph to those types.

   The colour swatch is the same one rendered in the graph for that
   entity type, so users can mentally connect rows in the panel to
   nodes in the canvas.
   ───────────────────────────────────────────────────────────────── */
function FilterPanel({
  allTypes,
  typeCounts,
  activeTypes,
  toggleType,
  clearFilter,
}: {
  allTypes: string[];
  typeCounts: Record<string, number>;
  activeTypes: Set<string>;
  toggleType: (t: string) => void;
  clearFilter: () => void;
}) {
  const filterActive = activeTypes.size > 0;
  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-3">
        <p
          className="text-[11px] uppercase tracking-wider font-extrabold"
          style={{ color: 'var(--text-secondary)' }}
        >
          Entity-type filter
        </p>
        {filterActive && (
          <button
            type="button"
            onClick={clearFilter}
            className="text-[11px] font-semibold"
            style={{ color: 'var(--accent)' }}
          >
            Clear ({activeTypes.size})
          </button>
        )}
      </div>
      <p
        className="text-[11px] mb-3"
        style={{ color: 'var(--text-muted)' }}
      >
        Tick a row to limit the graph to that type. Untick all to show every
        entity type.
      </p>
      {allTypes.length === 0 ? (
        <p className="text-[12px]" style={{ color: 'var(--text-muted)' }}>
          Graph is empty.
        </p>
      ) : (
        <ul className="space-y-1">
          {allTypes.map((t) => {
            const checked = filterActive ? activeTypes.has(t) : false;
            const isOn = !filterActive || checked;
            const color = entityColor(t);
            const count = typeCounts[t] ?? 0;
            const inputId = `filter-${t}`;
            return (
              <li key={t}>
                <label
                  htmlFor={inputId}
                  className="flex items-center gap-2.5 px-2 py-2 cursor-pointer transition-colors"
                  style={{
                    background: checked ? `${color}10` : 'transparent',
                    border: `1px solid ${checked ? `${color}33` : 'transparent'}`,
                    borderRadius: 'var(--radius-md)',
                    opacity: isOn ? 1 : 0.55,
                  }}
                  title={TYPE_DESCRIPTIONS[t] ?? t}
                >
                  <input
                    id={inputId}
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggleType(t)}
                    className="shrink-0 cursor-pointer"
                    style={{ accentColor: color, width: 14, height: 14 }}
                    aria-label={`Filter to ${t} entities`}
                  />
                  <span className="shrink-0 inline-flex" aria-hidden="true">
                    <LegendShape type={t} color={color} />
                  </span>
                  <span
                    className="text-[12.5px] font-bold flex-1 truncate"
                    style={{ color: 'var(--text-primary)' }}
                  >
                    {t}
                  </span>
                  <span
                    className="text-[11px] tabular-nums font-semibold"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    {count.toLocaleString()}
                  </span>
                </label>
              </li>
            );
          })}
        </ul>
      )}
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

  // Escape closes the inspector. Background click also closes (handled by
  // the parent page's onBackgroundClick), but keyboard users need a path.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  // Move focus into the popover on open so Tab order is sensible and
  // Escape works without first clicking inside.
  useEffect(() => {
    popRef.current?.focus();
  }, [entity.id]);

  const titleId = `graph-inspector-title-${entity.id}`;

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
        role="dialog"
        aria-modal="false"
        aria-labelledby={titleId}
        tabIndex={-1}
        className="absolute card p-3.5 fade-up focus:outline-none"
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
              id={titleId}
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
            aria-label="Close inspector (Esc)"
          >
            <X size={14} aria-hidden="true" />
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
