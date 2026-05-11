'use client';
/**
 * Per-source confidence breakdown card (§4 of docs/RAG_QA_PLAN.md).
 *
 * Shows:
 * - Numbered citation chip [n] matching the LLM's [n] markers
 * - Entity / Relationship / Chunk kind badge
 * - Multi-bar confidence (vector / bm25 / cypher) so the user can see
 *   which channels agreed instead of one rolled-up score
 * - Final fused confidence with the contributing-channels list
 * - Doc origin + chunk preview when hydration succeeded
 *
 * Channels that didn't contribute are rendered as faded "—" placeholders
 * so the layout doesn't shift between sources. That's intentional: a row
 * of bars with consistent column widths is easier to compare than a
 * variable-row card.
 */

import { ChatSource, RetrievalChannel } from '@/lib/api';
import { Bot, Database, Link as LinkIcon, Boxes } from 'lucide-react';

interface Props {
  index: number; // 1-indexed citation number
  source: ChatSource;
  cited: boolean;
}

const KIND_LABEL: Record<ChatSource['kind'], string> = {
  entity: 'Entity',
  relationship: 'Relationship',
  chunk: 'Chunk',
};

const KIND_ICON: Record<ChatSource['kind'], typeof Bot> = {
  entity: Boxes,
  relationship: LinkIcon,
  chunk: Database,
};

const CHANNEL_LABEL: Record<RetrievalChannel, string> = {
  vector_entity: 'vector',
  vector_relationship: 'vector',
  bm25: 'bm25',
  cypher: 'cypher',
};

function pct(v: number | null | undefined): string {
  if (v == null) return '—';
  return `${Math.round(v * 100)}%`;
}

function ConfidenceBar({ label, value }: { label: string; value: number | null | undefined }) {
  // Channels that didn't run render a faint placeholder so the row of
  // bars stays a stable shape across sources.
  const hasValue = value != null;
  const width = hasValue ? Math.round((value as number) * 100) : 0;
  return (
    <div className="flex items-center gap-2 min-w-0">
      <span
        className="text-[10px] font-mono tabular-nums w-12 text-right shrink-0"
        style={{ color: 'var(--text-muted)' }}
      >
        {label}
      </span>
      <div
        className="flex-1 h-1.5 rounded-full overflow-hidden"
        style={{ background: 'var(--bg-muted)' }}
      >
        <div
          className="h-full transition-all"
          style={{
            width: `${width}%`,
            background: hasValue ? 'var(--accent)' : 'transparent',
          }}
        />
      </div>
      <span
        className="text-[10px] font-mono tabular-nums w-9 text-right shrink-0"
        style={{ color: hasValue ? 'var(--text-secondary)' : 'var(--text-muted)' }}
      >
        {pct(value)}
      </span>
    </div>
  );
}

export default function SourceCard({ index, source, cited }: Props) {
  const Icon = KIND_ICON[source.kind];
  const finalPct = Math.round(source.final_confidence * 100);
  // Dedup the contributing-channel list — VECTOR_ENTITY and
  // VECTOR_RELATIONSHIP both appear as "vector" to the user, so we
  // normalise before rendering the chips.
  const channelLabels = Array.from(
    new Set(source.contributing_channels.map((c) => CHANNEL_LABEL[c])),
  );

  return (
    <div
      className="card p-4 space-y-2.5"
      style={cited ? { borderColor: 'var(--accent)' } : {}}
    >
      <div className="flex items-start gap-2.5">
        <span
          className="badge"
          style={{
            background: cited ? 'var(--accent)' : 'var(--bg-muted)',
            color: cited ? '#fff' : 'var(--text-muted)',
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          [{index}]
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <Icon size={13} style={{ color: 'var(--text-muted)' }} />
            <span
              className="text-[10px] font-bold uppercase tracking-wider"
              style={{ color: 'var(--text-muted)' }}
            >
              {KIND_LABEL[source.kind]}
            </span>
          </div>
          <p
            className="text-sm font-medium mt-0.5 truncate"
            style={{ color: 'var(--text-primary)' }}
            title={source.label}
          >
            {source.label}
          </p>
        </div>
        <span
          className="text-sm font-semibold tabular-nums shrink-0"
          style={{
            color: source.final_confidence >= 0.7
              ? 'var(--success)'
              : source.final_confidence >= 0.4
                ? 'var(--text-primary)'
                : 'var(--text-muted)',
          }}
        >
          {finalPct}%
        </span>
      </div>

      <div className="space-y-1">
        <ConfidenceBar label="vector" value={source.score_vector} />
        <ConfidenceBar label="bm25" value={source.score_bm25} />
        <ConfidenceBar label="cypher" value={source.score_cypher} />
      </div>

      {source.chunk_preview ? (
        <p
          className="text-xs italic line-clamp-3"
          style={{ color: 'var(--text-secondary)' }}
        >
          “{source.chunk_preview}”
        </p>
      ) : source.description ? (
        // Fallback for entities with no hydrated chunk (e.g. Open
        // Targets imports) — show the node's own description, which is
        // the prose the LLM grounded its answer on.
        <p
          className="text-xs line-clamp-3"
          style={{ color: 'var(--text-secondary)' }}
        >
          {source.description}
        </p>
      ) : null}

      <div
        className="flex items-center gap-2 flex-wrap text-[10px]"
        style={{ color: 'var(--text-muted)' }}
      >
        {channelLabels.map((c) => (
          <span key={c} className="badge badge-neutral text-[10px]">{c}</span>
        ))}
        {source.source_doc_id && (
          <span className="truncate" title={source.source_doc_id}>
            from <code style={{ color: 'var(--text-secondary)' }}>{source.source_doc_id}</code>
          </span>
        )}
      </div>
    </div>
  );
}
