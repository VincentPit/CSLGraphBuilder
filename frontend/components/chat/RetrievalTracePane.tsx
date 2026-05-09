'use client';
/**
 * Collapsible "show your work" panel — renders the RetrievalTrace
 * returned by /qa/ask. Mirrors the per-stage breakdown the verification
 * page already shows for verified relationships, so the visual
 * vocabulary stays consistent across the app.
 */

import { useState } from 'react';
import { ChevronDown, ChevronUp, Cog } from 'lucide-react';
import { ChannelTrace, RetrievalTrace } from '@/lib/api';

interface Props {
  trace: RetrievalTrace;
}

const CHANNEL_NAMES: Record<ChannelTrace['channel'], string> = {
  vector_entity: 'vector · entities',
  vector_relationship: 'vector · relationships',
  bm25: 'bm25 · fulltext',
  cypher: 'cypher · 1-hop',
};

export default function RetrievalTracePane({ trace }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{ border: '1px solid var(--border-subtle)' }}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full px-3 py-2 flex items-center justify-between text-xs hover:bg-[var(--bg-muted)] transition"
        style={{ color: 'var(--text-secondary)' }}
      >
        <span className="inline-flex items-center gap-2">
          <Cog size={12} />
          <span className="font-medium">Retrieval trace</span>
          <span style={{ color: 'var(--text-muted)' }}>
            · {trace.channels.length} channel{trace.channels.length === 1 ? '' : 's'}
            {' · '}
            {trace.final_top_k} kept
            {' · '}
            {trace.total_latency_ms}ms
          </span>
        </span>
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>

      {open && (
        <div
          className="px-3 py-3 space-y-3 text-xs"
          style={{ background: 'var(--bg-muted)' }}
        >
          {trace.extracted_terms.length > 0 && (
            <div>
              <p
                className="text-[10px] font-bold uppercase tracking-widest mb-1"
                style={{ color: 'var(--text-muted)' }}
              >
                Extracted terms
              </p>
              <div className="flex flex-wrap gap-1">
                {trace.extracted_terms.map((t, i) => (
                  <span key={i} className="badge badge-neutral">{t}</span>
                ))}
              </div>
            </div>
          )}

          <div>
            <p
              className="text-[10px] font-bold uppercase tracking-widest mb-1"
              style={{ color: 'var(--text-muted)' }}
            >
              Channels
            </p>
            <div className="space-y-1">
              {trace.channels.map((c, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between rounded px-2 py-1"
                  style={{
                    background: 'var(--bg-card)',
                    border: '1px solid var(--border-subtle)',
                  }}
                >
                  <span style={{ color: 'var(--text-primary)' }}>
                    {CHANNEL_NAMES[c.channel] ?? c.channel}
                  </span>
                  <span
                    className="font-mono tabular-nums"
                    style={{
                      color: c.error ? 'var(--danger)' : 'var(--text-muted)',
                    }}
                  >
                    {c.error
                      ? c.error
                      : `${c.hits} hit${c.hits === 1 ? '' : 's'} · ${c.latency_ms}ms`}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div
            className="grid grid-cols-3 gap-2 text-center"
            style={{ color: 'var(--text-secondary)' }}
          >
            <div className="rounded p-2" style={{ background: 'var(--bg-card)' }}>
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                fused
              </p>
              <p className="font-semibold tabular-nums">{trace.rrf_top_n}</p>
            </div>
            <div className="rounded p-2" style={{ background: 'var(--bg-card)' }}>
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                kept
              </p>
              <p className="font-semibold tabular-nums">{trace.final_top_k}</p>
            </div>
            <div className="rounded p-2" style={{ background: 'var(--bg-card)' }}>
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                hydrated
              </p>
              <p className="font-semibold tabular-nums">{trace.hydrated_chunks}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
