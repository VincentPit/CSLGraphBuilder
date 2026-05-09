'use client';
/**
 * /chat — biomedical knowledge-graph chatbot UI (P12 of docs/RAG_QA_PLAN.md).
 *
 * Single-pane layout (sidebar + thread + composer). Each turn renders
 * the user query, the assistant answer with citation chips, a grid of
 * SourceCards with per-channel confidence bars, and a collapsible
 * retrieval trace. Sessions persist in Neo4j and re-load on click.
 *
 * Design choices:
 * - Active session id lives in component state, not the URL. Deep-linking
 *   to specific sessions is a follow-up; for v1 navigating away resets.
 * - We keep prior turns + the in-flight one in a single `turns` array
 *   so the thread renders identically whether the data came from
 *   GET /qa/sessions/{id} (existing) or POST /qa/ask (new).
 * - The "thinking…" pending state is a synthetic turn with `response: null`
 *   that swaps in for the real response when /qa/ask resolves.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Loader2, Send, MessageSquare, Sparkles } from 'lucide-react';

import {
  AskResponse,
  ChannelTrace,
  ChatSession,
  ChatTurn,
  RetrievalTrace,
  askQuestion,
  formatApiError,
  getChatSession,
  getChatUser,
  listChatSessions,
} from '@/lib/api';
import {
  ChatIdentity,
  getStoredIdentity,
  onIdentityChange,
  setStoredIdentity,
} from '@/lib/identity';
import IdentityPrompt from '@/components/chat/IdentityPrompt';
import MessageBubble from '@/components/chat/MessageBubble';
import SessionSidebar from '@/components/chat/SessionSidebar';

interface UITurn {
  query: string;
  /** ``null`` while the answer is in flight. */
  response: AskResponse | null;
  /** Local-only id used as the React list key for in-flight turns. */
  localId: string;
}

/** Inflate a persisted ChatTurn (from GET /qa/sessions/{id}) into the
 *  same shape POST /qa/ask returns, so MessageBubble doesn't care which
 *  side of the wall the data came from. */
function turnToUITurn(t: ChatTurn): UITurn {
  // Persisted turns don't currently round-trip the full source list +
  // retrieval trace. We synthesise an empty trace so MessageBubble can
  // still render the answer + citation chips; sources will appear empty
  // for replayed turns until we extend the schema.
  const trace: RetrievalTrace = {
    query: t.user_query,
    extracted_terms: [],
    channels: [] as ChannelTrace[],
    rrf_top_n: 0,
    final_top_k: 0,
    hydrated_chunks: 0,
    total_latency_ms: t.latency_ms,
  };
  const response: AskResponse = {
    session_id: t.session_id,
    turn_id: t.id,
    answer: t.llm_answer,
    sources: [],
    cited_source_indices: [],
    retrieval_trace: trace,
    request_id: t.request_id ?? null,
    latency_ms: t.latency_ms,
  };
  return { query: t.user_query, response, localId: t.id };
}

export default function ChatPage() {
  const qc = useQueryClient();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const [turns, setTurns] = useState<UITurn[]>([]);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const threadEndRef = useRef<HTMLDivElement | null>(null);

  // Identity. ``identityChecked`` distinguishes "still hydrating from
  // localStorage on the client" (don't render anything yet) from
  // "definitely no identity, show the prompt".
  const [identity, setIdentity] = useState<ChatIdentity | null>(null);
  const [identityChecked, setIdentityChecked] = useState(false);

  // Hydrate identity once on mount + listen for storage events so a
  // sign-in or sign-out in another tab updates this one.
  useEffect(() => {
    const stored = getStoredIdentity();
    if (stored) {
      // Validate the stored id against the backend — the user node may
      // have been deleted out from under us; if so we drop the stale
      // identity and re-prompt rather than 401-looping every request.
      getChatUser(stored.id)
        .then((u) => {
          const validated: ChatIdentity = { id: u.id, displayName: u.display_name };
          setStoredIdentity(validated);
          setIdentity(validated);
        })
        .catch(() => {
          setStoredIdentity(null);
          setIdentity(null);
        })
        .finally(() => setIdentityChecked(true));
    } else {
      setIdentityChecked(true);
    }
    return onIdentityChange((next) => setIdentity(next));
  }, []);

  function handleIdentitySet(next: ChatIdentity) {
    setIdentity(next);
    setIdentityChecked(true);
    // The sidebar list is keyed on the *current* user via the X-User-Id
    // header; refresh it now that the header will be set on subsequent
    // calls.
    qc.invalidateQueries({ queryKey: ['chat-sessions'] });
  }

  function handleClearIdentity() {
    setStoredIdentity(null);
    setIdentity(null);
    setSessionId(null);
    setTurns([]);
    qc.invalidateQueries({ queryKey: ['chat-sessions'] });
  }

  // Sidebar list — refetched after each /ask + after deletes.
  // Disabled while we don't yet have an identity so we don't list
  // anonymous sessions to a soon-to-be-identified user.
  const { data: sessionList, isLoading: sessionsLoading } = useQuery({
    queryKey: ['chat-sessions', identity?.id ?? 'anon'],
    queryFn: () => listChatSessions({ limit: 50 }),
    enabled: identityChecked && identity !== null,
  });
  const sessions: ChatSession[] = sessionList?.sessions ?? [];

  // When the user picks a session from the sidebar, inflate persisted
  // turns. Skipped while sending so we don't overwrite the in-flight
  // turn with stale data from the server.
  useEffect(() => {
    if (!sessionId || sending) return;
    let cancelled = false;
    (async () => {
      try {
        const data = await getChatSession(sessionId);
        if (!cancelled) {
          setTurns(data.turns.map(turnToUITurn));
          setError(null);
        }
      } catch (err) {
        if (!cancelled) setError(formatApiError(err, 'Could not load session'));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sessionId, sending]);

  // Auto-scroll on new content.
  useEffect(() => {
    threadEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [turns]);

  function handleNew() {
    setSessionId(null);
    setTurns([]);
    setError(null);
    setDraft('');
  }

  function handleSelect(id: string) {
    setSessionId(id);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const q = draft.trim();
    if (!q || sending) return;

    const localId = `local-${Date.now()}`;
    const pending: UITurn = { query: q, response: null, localId };
    setTurns((prev) => [...prev, pending]);
    setDraft('');
    setSending(true);
    setError(null);

    try {
      const resp = await askQuestion({
        query: q,
        session_id: sessionId ?? undefined,
      });
      // Replace the pending turn with the real response.
      setTurns((prev) =>
        prev.map((t) => (t.localId === localId ? { ...t, response: resp } : t)),
      );
      // First turn → adopt the new session id. Subsequent turns reuse it.
      if (!sessionId) setSessionId(resp.session_id);
      // Refresh the sidebar (turn_count + last_active_at moved).
      qc.invalidateQueries({ queryKey: ['chat-sessions'] });
    } catch (err) {
      setError(formatApiError(err, 'Could not get an answer'));
      // Drop the pending bubble so the user can retry without an empty turn.
      setTurns((prev) => prev.filter((t) => t.localId !== localId));
    } finally {
      setSending(false);
    }
  }

  function handleDeleted(id: string) {
    if (id === sessionId) {
      setSessionId(null);
      setTurns([]);
    }
    qc.invalidateQueries({ queryKey: ['chat-sessions'] });
  }

  const showWelcome = useMemo(() => turns.length === 0 && !sessionId, [turns, sessionId]);
  const needsIdentity = identityChecked && identity === null;

  return (
    <div className="space-y-6 max-w-[1280px] mx-auto">
      <header className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-3">
        <div>
          <h1 className="page-title">Chat</h1>
          <p className="page-desc">
            Ask the knowledge graph in natural language. Every answer cites the
            entities, relationships, and source chunks it used, with per-channel
            confidence so you can see how it got there.
          </p>
        </div>
        {identity && (
          <div
            className="flex items-center gap-2 text-xs"
            style={{ color: 'var(--text-muted)' }}
          >
            <span>
              Signed in as{' '}
              <span style={{ color: 'var(--text-primary)' }}>
                {identity.displayName}
              </span>
            </span>
            <button
              type="button"
              onClick={handleClearIdentity}
              className="btn-ghost text-xs"
              title="Forget this browser identity"
            >
              Sign out
            </button>
          </div>
        )}
      </header>

      <div
        className="flex flex-col lg:flex-row rounded-xl overflow-hidden"
        style={{
          background: 'var(--bg-card)',
          border: '1px solid var(--border-default)',
          minHeight: 'min(calc(100vh - 14rem), 720px)',
        }}
      >
        {/* Hide the sidebar until identity is set so an anonymous reader
            can't kick off retrieval queries without registering first. */}
        {identity && (
          <SessionSidebar
            sessions={sessions}
            loading={sessionsLoading}
            activeSessionId={sessionId}
            onSelect={handleSelect}
            onNew={handleNew}
            onDeleted={handleDeleted}
          />
        )}

        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 overflow-y-auto p-5 space-y-6">
            {needsIdentity && (
              <IdentityPrompt onIdentitySet={handleIdentitySet} />
            )}
            {!needsIdentity && showWelcome && (
              <WelcomeCard onUseExample={(q) => setDraft(q)} />
            )}

            {!showWelcome && turns.length === 0 && (
              <div
                className="flex flex-col items-center justify-center text-sm py-16"
                style={{ color: 'var(--text-muted)' }}
              >
                <MessageSquare size={28} className="mb-2 opacity-50" />
                Pick a session from the sidebar or start a new chat.
              </div>
            )}

            {turns.map((t) => (
              <MessageBubble
                key={t.localId}
                query={t.query}
                response={t.response}
                pending={t.response === null}
              />
            ))}

            {error && (
              <div
                className="card p-3 text-xs"
                style={{ color: 'var(--danger)', borderColor: 'var(--danger)' }}
              >
                {error}
              </div>
            )}

            <div ref={threadEndRef} />
          </div>

          {identity && (
            <form
              onSubmit={handleSubmit}
              className="border-t p-3"
              style={{ borderColor: 'var(--border-subtle)' }}
            >
              <div className="flex items-end gap-2">
                <textarea
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => {
                    // Enter sends; Shift+Enter inserts a newline. Mirrors
                    // the verification page's textarea ergonomics.
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSubmit(e as unknown as React.FormEvent);
                    }
                  }}
                  placeholder="Ask about a gene, drug, disease, or relationship in the graph…"
                  rows={2}
                  disabled={sending}
                  className="input flex-1 resize-none"
                  style={{ minHeight: '52px', maxHeight: '160px' }}
                />
                <button
                  type="submit"
                  disabled={sending || !draft.trim()}
                  className="btn-primary"
                  aria-label="Send"
                >
                  {sending ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <Send size={14} />
                  )}
                  <span className="hidden sm:inline">Send</span>
                </button>
              </div>
              <p
                className="text-[10px] mt-1.5"
                style={{ color: 'var(--text-muted)' }}
              >
                Enter to send · Shift+Enter for newline · Citations like [1] in the answer link to the matching source card.
              </p>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}

function WelcomeCard({ onUseExample }: { onUseExample: (q: string) => void }) {
  // Examples are wired to clicks, not auto-submitted, so the user can
  // tweak before sending.
  const examples = [
    'What does Imatinib target?',
    'Tell me about BCR-ABL.',
    'Which drugs treat chronic myeloid leukaemia?',
  ];
  return (
    <div
      className="card p-6 space-y-4"
      style={{ background: 'var(--accent-soft)', borderColor: 'var(--accent-muted)' }}
    >
      <div className="flex items-center gap-2">
        <Sparkles size={16} style={{ color: 'var(--accent)' }} />
        <p
          className="text-sm font-bold"
          style={{ color: 'var(--text-primary)' }}
        >
          Ask the graph
        </p>
      </div>
      <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
        The chatbot retrieves relevant entities, relationships, and source
        chunks via a hybrid graph-and-vector pipeline, then answers using only
        what it found. Try one of these to start:
      </p>
      <div className="flex flex-wrap gap-2">
        {examples.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => onUseExample(q)}
            className="btn-ghost text-xs"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
