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
  askQuestionStream,
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
import DebugPane from '@/components/chat/DebugPane';
import IdentityPrompt from '@/components/chat/IdentityPrompt';
import MessageBubble from '@/components/chat/MessageBubble';
import { MutationCard, isMutationToolCall } from '@/components/chat/MutationCard';
import SessionSidebar from '@/components/chat/SessionSidebar';

interface UITurn {
  query: string;
  /** ``null`` until retrieval completes; a partial ``AskResponse`` (empty
   *  ``answer`` growing via SSE deltas, no ``turn_id`` yet) while the
   *  answer streams; the full response once the stream's ``done`` lands. */
  response: AskResponse | null;
  /** True while SSE deltas are still arriving for this turn. */
  streaming?: boolean;
  /** Local-only id used as the React list key for in-flight turns. */
  localId: string;
}

/** Inflate a persisted ChatTurn (from GET /qa/sessions/{id}) into the
 *  same shape POST /qa/ask returns, so MessageBubble doesn't care which
 *  side of the wall the data came from.
 *
 *  Turns persisted by QAService now carry a `retrieval_snapshot` in
 *  their metadata (compact source list + cited indices + trace), so a
 *  reopened session renders the same source cards + trace pane as a
 *  live ask. Older turns predating that snapshot fall back to an empty
 *  trace — the answer text + citation markers still render. */
function turnToUITurn(t: ChatTurn): UITurn {
  const snap = t.metadata?.retrieval_snapshot;
  const trace: RetrievalTrace = snap?.retrieval_trace ?? {
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
    sources: snap?.sources ?? [],
    cited_source_indices: snap?.cited_source_indices ?? [],
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
  // The in-flight SSE stream (POST /qa/ask/stream), so we can cancel it
  // on unmount, on a new submit, or when the user navigates sessions.
  const streamRef = useRef<AbortController | null>(null);

  function abortInFlight() {
    if (streamRef.current) {
      streamRef.current.abort();
      streamRef.current = null;
      setSending(false);
    }
  }
  // Cancel any open stream when the page unmounts.
  useEffect(() => () => streamRef.current?.abort(), []);

  // Identity. ``identityChecked`` distinguishes "still hydrating from
  // localStorage on the client" (don't render anything yet) from
  // "definitely no identity, show the prompt".
  const [identity, setIdentity] = useState<ChatIdentity | null>(null);
  const [identityChecked, setIdentityChecked] = useState(false);

  // Tool-use opt-ins (P9 / P10). Off by default — production traffic
  // answers most questions from the upfront retrieval alone, and the
  // agentic loop adds latency + LLM cost. Persisted to localStorage so
  // the SME's last choice survives refreshes.
  const [enableTools, setEnableTools] = useState<boolean>(false);
  const [enableMutations, setEnableMutations] = useState<boolean>(false);
  useEffect(() => {
    if (typeof window === 'undefined') return;
    setEnableTools(window.localStorage.getItem('chat:enableTools') === '1');
    setEnableMutations(window.localStorage.getItem('chat:enableMutations') === '1');
  }, []);
  function persistToggle(key: 'enableTools' | 'enableMutations', value: boolean) {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(`chat:${key}`, value ? '1' : '0');
  }

  // Debug pane (§8.5) — controlled by ``?debug=1`` query param. Done as
  // a one-time check on mount rather than a route subscription because
  // toggling it mid-session shouldn't change the rendering retroactively.
  const [debugMode, setDebugMode] = useState<boolean>(false);
  useEffect(() => {
    if (typeof window === 'undefined') return;
    setDebugMode(new URLSearchParams(window.location.search).get('debug') === '1');
  }, []);

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
  //
  // CRUCIAL: only reload when the session id change came from
  // ``handleSelect`` (sidebar click). After the first ask in a new
  // chat we call ``setSessionId(resp.session_id)`` ourselves — at that
  // point the in-state turns already hold the FULL response (sources +
  // retrieval trace), which is strictly richer than what
  // ``GET /qa/sessions/{id}`` returns (persisted turns don't currently
  // round-trip sources/trace). Reloading here would clobber that with
  // empty-trace stubs — that's the "0 sources / 0 channels" bug.
  const sidebarPickRef = useRef<string | null>(null);
  useEffect(() => {
    if (!sessionId || sending) return;
    if (sidebarPickRef.current !== sessionId) return;
    sidebarPickRef.current = null;
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
    abortInFlight();
    setSessionId(null);
    setTurns([]);
    setError(null);
    setDraft('');
  }

  function handleSelect(id: string) {
    abortInFlight();
    // Mark this id as a sidebar pick so the load effect knows to
    // re-fetch its turns (vs a session id we minted ourselves on an ask).
    sidebarPickRef.current = id;
    setSessionId(id);
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const q = draft.trim();
    if (!q || sending) return;

    const localId = `local-${Date.now()}`;
    const pending: UITurn = { query: q, response: null, localId };
    setTurns((prev) => [...prev, pending]);
    setDraft('');
    setSending(true);
    setError(null);
    streamRef.current?.abort();

    // Mutate just this turn in the list.
    const update = (fn: (t: UITurn) => UITurn) =>
      setTurns((prev) => prev.map((t) => (t.localId === localId ? fn(t) : t)));
    // Capture the session id at submit time — used for the partial
    // response's session_id until `done` carries the real one.
    const startSessionId = sessionId;

    // Holds the controller for *this* stream so the callbacks can detect
    // that they've been superseded (a new submit) or cancelled (session
    // switch / unmount) and bail without touching unrelated state.
    let controller: AbortController;
    const isCurrent = () => streamRef.current === controller;

    controller = askQuestionStream(
      {
        query: q,
        session_id: startSessionId ?? undefined,
        enable_tools: enableTools || undefined,
        enable_mutations: enableMutations || undefined,
      },
      {
        // Retrieval done → swap the "thinking…" pending turn for a partial
        // response holding the sources + traces. The answer fills in via
        // deltas; turn_id / cited indices / latency arrive with `done`.
        onRetrieval: (d) => {
          if (!isCurrent()) return;
          update((t) => ({
            ...t,
            streaming: true,
            response: {
              session_id: startSessionId ?? '',
              turn_id: '',
              answer: '',
              sources: d.sources,
              cited_source_indices: [],
              retrieval_trace: d.retrieval_trace,
              memory_trace: d.memory_trace ?? null,
              request_id: null,
              latency_ms: 0,
              tool_calls: [],
            },
          }));
        },
        onToolCall: (call) => {
          if (!isCurrent()) return;
          update((t) =>
            t.response
              ? {
                  ...t,
                  response: {
                    ...t.response,
                    tool_calls: [...(t.response.tool_calls ?? []), call],
                  },
                }
              : t,
          );
        },
        onDelta: (text) => {
          if (!isCurrent()) return;
          update((t) =>
            t.response
              ? { ...t, response: { ...t.response, answer: t.response.answer + text } }
              : t,
          );
        },
        onDone: (d) => {
          if (!isCurrent()) return;
          update((t) =>
            t.response
              ? {
                  ...t,
                  streaming: false,
                  response: {
                    ...t.response,
                    session_id: d.session_id,
                    turn_id: d.turn_id,
                    answer: d.answer,
                    cited_source_indices: d.cited_source_indices,
                    faithfulness: d.faithfulness ?? null,
                    tool_calls: d.tool_calls ?? t.response.tool_calls ?? [],
                    request_id: d.request_id ?? null,
                    latency_ms: d.latency_ms,
                  },
                }
              : t,
          );
          if (!startSessionId) setSessionId(d.session_id);
          qc.invalidateQueries({ queryKey: ['chat-sessions'] });
          setSending(false);
          streamRef.current = null;
        },
        onError: (msg) => {
          if (!isCurrent()) return;
          // The SSE client already normalised the error to a readable
          // string (HTTP `detail`, transport error, or server `error` event).
          setError(msg || 'Could not get an answer');
          // Drop the pending bubble so the user can retry without an empty turn.
          setTurns((prev) => prev.filter((t) => t.localId !== localId));
          setSending(false);
          streamRef.current = null;
        },
      },
    );
    streamRef.current = controller;
  }

  function handleDeleted(id: string) {
    if (id === sessionId) {
      abortInFlight();
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

            {turns.map((t) => {
              const mutationCalls = (t.response?.tool_calls ?? []).filter(
                isMutationToolCall,
              );
              return (
                <div key={t.localId}>
                  <MessageBubble
                    query={t.query}
                    response={t.response}
                    pending={t.response === null}
                    streaming={t.streaming}
                  />
                  {mutationCalls.map((call, i) => (
                    <MutationCard key={`mut-${t.localId}-${i}`} call={call} />
                  ))}
                  {debugMode && t.response && !t.streaming && (
                    <DebugPane response={t.response} forceOpen={false} />
                  )}
                </div>
              );
            })}

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
              <div className="flex items-center justify-between gap-2 mt-1.5">
                <p
                  className="text-[10px]"
                  style={{ color: 'var(--text-muted)' }}
                >
                  Enter to send · Shift+Enter for newline · Citations like [1] in the answer link to the matching source card.
                </p>
                <div className="flex items-center gap-3 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                  <label className="inline-flex items-center gap-1 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={enableTools}
                      onChange={(e) => {
                        setEnableTools(e.target.checked);
                        persistToggle('enableTools', e.target.checked);
                      }}
                      className="w-3 h-3"
                    />
                    <span title="Let the LLM call search_graph / get_entity / verify_claim before answering">
                      Tools
                    </span>
                  </label>
                  <label className="inline-flex items-center gap-1 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={enableMutations}
                      onChange={(e) => {
                        setEnableMutations(e.target.checked);
                        persistToggle('enableMutations', e.target.checked);
                      }}
                      className="w-3 h-3"
                    />
                    <span title="Let the LLM propose graph mutations — each call queues a proposal in /curation, nothing applies automatically">
                      Mutations
                    </span>
                  </label>
                </div>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}

function WelcomeCard({ onUseExample }: { onUseExample: (q: string) => void }) {
  // Examples are wired to clicks, not auto-submitted, so the user can
  // tweak before sending. These map onto Open-Targets-ingested entities
  // that carry a real description (EGFR, KRAS, Parkinson's disease) so
  // the bot has citable prose even though those nodes have no source
  // chunks. Topics absent from the corpus get an honest "I cannot find
  // this" refusal.
  const examples = [
    'What is EGFR?',
    'Tell me about KRAS.',
    "What is Parkinson's disease?",
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
