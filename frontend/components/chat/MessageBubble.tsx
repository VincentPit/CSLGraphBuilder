'use client';
/**
 * One chat exchange: the user query bubble + the assistant answer.
 *
 * The assistant answer renders citation markers like [1] [2] as inline
 * chips that scroll the corresponding SourceCard into view. The chips
 * carry their own styling so users can tell at a glance which sources
 * the bot actually used vs which it just retrieved.
 */

import { AskResponse, ChatSource, sendChatFeedback } from '@/lib/api';
import { useState } from 'react';
import { ThumbsUp, ThumbsDown, User, Bot, Loader2 } from 'lucide-react';
import SourceCard from './SourceCard';
import RetrievalTracePane from './RetrievalTracePane';

interface Props {
  query: string;
  /** ``null`` before retrieval completes; a partial ``AskResponse`` (empty
   *  ``answer`` accreting via SSE deltas, ``turn_id`` not yet assigned)
   *  while streaming; the full response once ``done`` lands. */
  response: AskResponse | null;
  /** When true, render a "thinking…" state instead of the answer body.
   *  (No retrieval result yet — nothing to show.) */
  pending?: boolean;
  /** When true, the answer is mid-stream: sources are already populated,
   *  the answer text is growing, ``cited_source_indices`` + ``turn_id``
   *  + ``latency_ms`` aren't final yet. */
  streaming?: boolean;
  /** Optional override of where citations scroll to (for embedded views). */
  scrollToSource?: (sourceId: string) => void;
}

/** Split an answer into [text, [n], text, [m], …] tokens for rendering. */
function tokenise(answer: string): Array<{ kind: 'text' | 'cite'; value: string }> {
  const out: Array<{ kind: 'text' | 'cite'; value: string }> = [];
  const re = /\[(\d+)\]/g;
  let lastIdx = 0;
  let match: RegExpExecArray | null;
  while ((match = re.exec(answer)) !== null) {
    if (match.index > lastIdx) {
      out.push({ kind: 'text', value: answer.slice(lastIdx, match.index) });
    }
    out.push({ kind: 'cite', value: match[1] });
    lastIdx = re.lastIndex;
  }
  if (lastIdx < answer.length) {
    out.push({ kind: 'text', value: answer.slice(lastIdx) });
  }
  return out;
}

function CitationChip({
  index,
  source,
  cited,
  onClick,
}: {
  index: number;
  source?: ChatSource;
  cited: boolean;
  onClick: () => void;
}) {
  const label = source?.label ?? 'unknown';
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      className="inline-flex items-center justify-center mx-0.5 px-1.5 py-px text-[10px] font-semibold rounded transition"
      style={{
        background: cited ? 'var(--accent)' : 'var(--bg-muted)',
        color: cited ? '#fff' : 'var(--text-secondary)',
        verticalAlign: 'baseline',
        minWidth: '20px',
      }}
    >
      [{index}]
    </button>
  );
}

function FeedbackButtons({ turnId }: { turnId: string }) {
  // Local-only state — feedback is fire-and-forget; we don't need to
  // re-render the whole turn or invalidate any query.
  const [given, setGiven] = useState<-1 | 0 | 1 | null>(null);
  const [busy, setBusy] = useState(false);

  async function send(rating: -1 | 1) {
    if (busy) return;
    setBusy(true);
    try {
      await sendChatFeedback(turnId, { rating });
      setGiven(rating);
    } catch {
      // Silent failure — feedback isn't load-bearing.
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex items-center gap-1">
      <button
        type="button"
        onClick={() => send(1)}
        disabled={busy || given === 1}
        className="p-1 rounded transition hover:bg-[var(--bg-muted)] disabled:opacity-50"
        title={given === 1 ? 'Thanks!' : 'This was helpful'}
      >
        <ThumbsUp
          size={13}
          style={{ color: given === 1 ? 'var(--success)' : 'var(--text-muted)' }}
        />
      </button>
      <button
        type="button"
        onClick={() => send(-1)}
        disabled={busy || given === -1}
        className="p-1 rounded transition hover:bg-[var(--bg-muted)] disabled:opacity-50"
        title={given === -1 ? 'Recorded.' : 'This was not helpful'}
      >
        <ThumbsDown
          size={13}
          style={{ color: given === -1 ? 'var(--danger)' : 'var(--text-muted)' }}
        />
      </button>
    </div>
  );
}

export default function MessageBubble({ query, response, pending, streaming, scrollToSource }: Props) {
  const sources = response?.sources ?? [];
  const cited = new Set(response?.cited_source_indices ?? []);
  // DOM ids for the source cards must be unique across the whole thread,
  // otherwise getElementById(`source-1`) always resolves to the FIRST
  // turn's card. Namespace by turn id so a `[1]` chip on turn 3 scrolls
  // to turn 3's source #1, not turn 1's.
  const domNs = response?.turn_id ?? 'pending';
  const sourceDomId = (n: number) => `source-${domNs}-${n}`;

  // Source rows split into "cited in the answer" vs "retrieved but not
  // cited". By default we only render the cited ones — the rest are a
  // noisy tail (rerank still keeps low-relevance hits to give the LLM
  // options) so they go behind a toggle. On a refusal nothing is cited,
  // so the default view is empty (just the header + a "show N" toggle).
  const rows = sources.map((s, i) => ({ s, n: i + 1 }));
  const citedRows = rows.filter((r) => cited.has(r.n));
  const uncitedRows = rows.filter((r) => !cited.has(r.n));
  const [sourcesExpanded, setSourcesExpanded] = useState(false);
  const visibleRows = sourcesExpanded ? rows : citedRows;

  function handleCitationClick(idx: number) {
    if (scrollToSource && sources[idx - 1]) {
      scrollToSource(sources[idx - 1].id);
      return;
    }
    const el = document.getElementById(sourceDomId(idx));
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  const tokens = response ? tokenise(response.answer) : [];

  return (
    <div className="space-y-3">
      {/* User query bubble */}
      <div className="flex gap-3">
        <div
          className="shrink-0 mt-0.5 flex items-center justify-center h-7 w-7 rounded-full"
          style={{ background: 'var(--bg-muted)' }}
          aria-hidden="true"
        >
          <User size={14} style={{ color: 'var(--text-secondary)' }} />
        </div>
        <div
          className="flex-1 rounded-lg px-4 py-2.5 text-sm whitespace-pre-wrap break-words"
          style={{
            background: 'var(--bg-muted)',
            color: 'var(--text-primary)',
          }}
        >
          {query}
        </div>
      </div>

      {/* Assistant answer bubble */}
      <div className="flex gap-3">
        <div
          className="shrink-0 mt-0.5 flex items-center justify-center h-7 w-7 rounded-full"
          style={{ background: 'var(--accent)' }}
          aria-hidden="true"
        >
          <Bot size={14} color="#fff" />
        </div>
        <div className="flex-1 space-y-3 min-w-0">
          <div
            className="rounded-lg px-4 py-3 text-sm whitespace-pre-wrap break-words"
            style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border-default)',
              color: 'var(--text-primary)',
            }}
          >
            {pending && (
              <span
                className="inline-flex items-center gap-2"
                style={{ color: 'var(--text-muted)' }}
              >
                <Loader2 size={14} className="animate-spin" /> thinking…
              </span>
            )}
            {!pending && response && (
              <>
                {tokens.map((tok, i) =>
                  tok.kind === 'text' ? (
                    <span key={i}>{tok.value}</span>
                  ) : (
                    <CitationChip
                      key={i}
                      index={Number(tok.value)}
                      source={sources[Number(tok.value) - 1]}
                      cited={cited.has(Number(tok.value))}
                      onClick={() => handleCitationClick(Number(tok.value))}
                    />
                  ),
                )}
                {/* Streaming: blinking caret after the partial answer, or a
                    "generating…" hint if the first delta hasn't landed yet
                    (retrieval finished but the LLM hasn't emitted a token). */}
                {streaming &&
                  (tokens.length === 0 ? (
                    <span
                      className="inline-flex items-center gap-2"
                      style={{ color: 'var(--text-muted)' }}
                    >
                      <Loader2 size={14} className="animate-spin" /> generating…
                    </span>
                  ) : (
                    <span className="animate-pulse" aria-hidden="true">
                      ▍
                    </span>
                  ))}
              </>
            )}
          </div>

          {response && (
            <div
              className="flex items-center justify-between text-[10px]"
              style={{ color: 'var(--text-muted)' }}
            >
              <span>
                {response.sources.length} source{response.sources.length === 1 ? '' : 's'}
                {streaming ? (
                  ' · streaming…'
                ) : (
                  <>
                    {' · '}
                    {response.latency_ms}ms
                    {response.request_id && ` · ${response.request_id}`}
                  </>
                )}
              </span>
              {/* Feedback needs a real turn id — only available once the
                  stream's `done` event has assigned one. */}
              {response.turn_id && <FeedbackButtons turnId={response.turn_id} />}
            </div>
          )}

          {response && sources.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <p
                  className="text-[10px] font-bold uppercase tracking-widest"
                  style={{ color: 'var(--text-muted)' }}
                >
                  {citedRows.length > 0 ? 'Cited sources' : 'Retrieved sources'}
                </p>
                {uncitedRows.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setSourcesExpanded((v) => !v)}
                    className="text-[10px] hover:underline shrink-0"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    {sourcesExpanded
                      ? 'Hide retrieved'
                      : citedRows.length > 0
                        ? `+ ${uncitedRows.length} more retrieved`
                        : `Show ${uncitedRows.length} retrieved sources`}
                  </button>
                )}
              </div>
              {visibleRows.length > 0 && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-2">
                  {visibleRows.map(({ s, n }) => (
                    <div key={s.id + ':' + n} id={sourceDomId(n)}>
                      <SourceCard index={n} source={s} cited={cited.has(n)} />
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {response && (
            <RetrievalTracePane trace={response.retrieval_trace} />
          )}
        </div>
      </div>
    </div>
  );
}
