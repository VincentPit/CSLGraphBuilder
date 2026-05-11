# RAG_QA — chat latency fixes

Companion to [`RAG_QA_PLAN.md`](RAG_QA_PLAN.md) and [`RAG_QA_FOLLOWUPS.md`](RAG_QA_FOLLOWUPS.md). All P0–P14 phases plus the FOLLOWUPS items have shipped; this doc tracks a focused round of latency / perceived-latency work on the `/chat` flow. Ordered by leverage — do them top-down, each is independent.

## Status

| # | Item | Type | Effort | Status |
|---|---|---|---|---|
| 1 | Stream the answer in `/chat` (switch to `POST /qa/ask/stream`) | Engineering — perceived latency | ~½ day | ✅ done |
| 2 | Background the rolling-summary regen | Engineering — real latency | ~2 h | ✅ done |
| 3 | Don't block retrieval on the query embedding | Engineering — real latency | ~2 h | ✅ done |
| — | `max_tokens` trimming | Investigated — **no-op**, see §4 | — | ✋ won't do |

Rough budget today for a warm, no-tools ask (anecdotal, gpt-4o-mini):

```
embed query        ~150 ms   ─┐ serial, before the gather
retrieval (gather) ~300 ms   ─┘ (BM25/Cypher channels idle-wait on embed)
+ summary regen    ~600 ms-2 s   only on turn ≥ 5 of a session (double-LLM)
LLM answer         ~2-4 s        first token ~2-4 s in, then ~100-200 tok
faithfulness       ~50 ms        lexical-only
─────────────────────────────
total              ~2.5-7 s, all of it "thinking…" with no visible progress
```

After §1–§3: first visible token in ~400–600 ms (retrieval result paints even sooner), summary regen off the critical path, ~150 ms shaved off retrieval.

---

## 1. Stream the answer in `/chat`

**Leverage: highest.** Doesn't make the LLM faster — cuts *perceived* latency ~3–4×. First token lands in ~400–600 ms (retrieval `phase`/`retrieval` events even sooner) instead of staring at "thinking…" for 2–4 s while the whole `AskResponse` is built. Backend already done — `POST /qa/ask/stream` shipped in P11 ([`api/routers/qa.py:249`](../api/routers/qa.py#L249)) and works with tool-use (FOLLOWUPS §3, Option A: agentic loop runs to completion, then the answer streams as one `delta`). This is a pure frontend change: an SSE client + incremental render.

### Event contract (already implemented server-side)

Each SSE frame is `event: <name>` + `data: <single-line JSON>`. Sequence per turn (see [`qa_service.ask_stream`](../src/graphbuilder/core/retrieval/qa_service.py#L543)):

| event | data | when |
|---|---|---|
| `phase` | `{ phase: "retrieving", request_id }` | immediately |
| `retrieval` | `{ sources[], retrieval_trace, memory_trace, intent }` | retrieval + memory done |
| `phase` | `{ phase: "tools" }` | only if `enable_tools`/`enable_mutations` |
| `tool_call` | one `ToolCallRecord` dict | repeated, only with tools |
| `phase` | `{ phase: "generating" }` | before the answer |
| `delta` | `{ text }` | repeated; token chunk(s). With tools: a single chunk = the agentic loop's final answer |
| `done` | `{ session_id, turn_id, answer, cited_source_indices, faithfulness, tool_calls, request_id, latency_ms }` | stream end |
| `error` | `{ message, kind }` | in place of `done` on failure; stop reading |

`kind` ∈ `session_not_found` | `retrieval_failed` | `llm_failed` | `internal_error`.

### 1a. SSE client — `frontend/lib/api.ts` (~1.5 h)

`EventSource` can't set headers (we need `X-API-Key` + `X-User-Id`), and `/documents/jobs/{id}/stream` sidesteps that with an `?api_key=` query param ([`getJobStreamUrl`](../frontend/lib/api.ts#L198)) — but `/qa/ask/stream` is a `POST` with a JSON body, so `EventSource` is out entirely. Use `fetch()` + `ReadableStream` and parse SSE frames by hand. Add:

```ts
export interface AskStreamHandlers {
  onPhase?(phase: string, requestId?: string): void;
  onRetrieval?(d: { sources: ChatSource[]; retrieval_trace: RetrievalTrace;
                    memory_trace?: MemoryTrace | null; intent?: string | null }): void;
  onToolCall?(call: ToolCall): void;
  onDelta?(text: string): void;          // append to the running answer
  onDone?(d: AskStreamDone): void;
  onError?(message: string, kind: string): void;
}

export interface AskStreamDone {
  session_id: string; turn_id: string; answer: string;
  cited_source_indices: number[]; faithfulness?: FaithfulnessResult | null;
  tool_calls?: ToolCall[]; request_id?: string | null; latency_ms: number;
}

// Returns an AbortController so the caller can cancel an in-flight stream.
export function askQuestionStream(
  body: { query: string; session_id?: string; user_id?: string | null;
          top_k?: number; enable_tools?: boolean; enable_mutations?: boolean;
          model?: string | null },
  handlers: AskStreamHandlers,
): AbortController;
```

Implementation notes:
- `fetch(`${BASE_URL}/qa/ask/stream`, { method: 'POST', headers: { 'Content-Type': 'application/json', ...(API_KEY && {'X-API-Key': API_KEY}), ...(identity && {'X-User-Id': identity.id}) }, body: JSON.stringify(body), signal })`.
- Read `res.body!.getReader()`, decode with `TextDecoder`, buffer, split on `\n\n` for complete frames. Each frame: lines starting `event:` and `data:`. `JSON.parse` the `data:` payload.
- Non-2xx response → call `onError` with the parsed `detail` (reuse `formatApiError` shape) and bail.
- Network drop mid-stream → `onError(err.message, 'network')`.
- Don't reuse the axios `apiClient` here — axios doesn't expose the stream body in the browser. Keep the manual `X-API-Key`/`X-User-Id` header construction in sync with the interceptor in [`api.ts:18`](../frontend/lib/api.ts#L18).

### 1b. Incremental render — `MessageBubble.tsx` (~1 h)

`MessageBubble` already takes `response: AskResponse | null` + `pending`. Add a third state: streaming. Two options —

- **Option A (minimal):** keep `UITurn.response` as the source of truth; on each `onDelta` mutate a partial `AskResponse` (sources from `onRetrieval`, `answer` accreting from deltas, `cited_source_indices` empty until `onDone`). `MessageBubble` renders it the same way; `tokenise()` already handles a partial `[1]` gracefully (an unterminated `[1` just stays as text until the `]` arrives in the next chunk — verify, but the regex is `/\[(\d+)\]/g` so a half-written marker renders literally for a frame, acceptable).
- Add a `streaming?: boolean` prop so the bubble can show a blinking cursor / keep the answer area from collapsing, and suppress the sources/trace panes until `onRetrieval` has fired (they render fine once `sources` is populated — `onRetrieval` arrives *before* the first `delta`, so in practice sources paint first).

Citation chips: `cited_source_indices` only arrives in `done`. Until then, render every `[n]` chip in the un-cited style; on `done`, swap to the cited highlight. Cheap re-render, no flicker worth worrying about.

### 1c. Wire it in `page.tsx` — `handleSubmit` (~1 h)

Replace the `await askQuestion(...)` call in [`handleSubmit`](../frontend/app/chat/page.tsx#L234) with `askQuestionStream`. The pending `UITurn` is already there; mutate it in place via `setTurns`:

- `onPhase('retrieving')` → no-op (or swap spinner copy).
- `onRetrieval(d)` → set `turn.response = { ...skeleton, sources: d.sources, retrieval_trace: d.retrieval_trace, memory_trace: d.memory_trace, answer: '', cited_source_indices: [], latency_ms: 0 }`.
- `onPhase('tools')` / `onToolCall(c)` → push into `turn.response.tool_calls` so `MutationCard` can render live.
- `onDelta(text)` → `turn.response.answer += text`.
- `onDone(d)` → fill in `turn_id`, `cited_source_indices`, `faithfulness`, `latency_ms`, final `answer`; `if (!sessionId) setSessionId(d.session_id)`; `qc.invalidateQueries(['chat-sessions'])`; `setSending(false)`.
- `onError(msg, kind)` → `setError`; drop the pending turn (same as today's `catch`); `setSending(false)`.
- Keep the `AbortController`; abort it in a `useEffect` cleanup if the component unmounts mid-stream, and on a new submit.

**Keep `askQuestion` (non-streaming) around** — `turnToUITurn` for reopened sessions doesn't change, and a fallback path (or the eval harness) may still want the one-shot call.

### 1d. Gotchas
- `sse-starlette` sends periodic `: ping` comment frames by default — the SSE parser must skip lines starting with `:`. (Confirm whether the existing `EventSourceResponse` config disables pings; `/documents/jobs` stream handling in the frontend is the reference.)
- Proxy/buffering: `EventSourceResponse` sets `X-Accel-Buffering: no` and `Cache-Control: no-cache` — fine for the local `start-local.sh` setup; note it if we ever put nginx in front.
- React 18 batching: deltas can arrive faster than paint. `setTurns` per delta is fine (React coalesces), but if it's janky, batch deltas on a `requestAnimationFrame`.
- The `DebugPane` (`?debug=1`) reads `t.response` — it'll light up once `done` lands; no change needed.

### 1e. Tests
- Frontend: a small unit test for the SSE frame parser (feed it a hand-written multi-frame string, assert the handler calls). Playwright/e2e is overkill for v1.
- Backend: `POST /qa/ask/stream` already covered (P11 + FOLLOWUPS §3). No change.

---

## 2. Background the rolling-summary regen

**Leverage: high on turn ≥ 5 of a session, zero before that.** Today, once a session has more older turns than `working_memory_turns` (3), every `/ask` that crosses a new turn count calls the summariser LLM *before* the answer LLM — a serial double-LLM hit ([`memory.py:211`](../src/graphbuilder/core/retrieval/memory.py#L211), `_maybe_refresh_summary`, invoked inside `MemoryService.build`, which `QAService` `await`s in the `asyncio.gather` before generation). That's the "ongoing-conversation latency cliff": ~600 ms–2 s added to a turn that already costs 2–4 s.

### Approach: serve last turn's cached summary, refresh in the background

`_maybe_refresh_summary` already detects staleness via the `[summary covers N turn(s)]` marker and short-circuits when fresh. Change the *stale* branch: instead of awaiting the LLM, return the **cached (stale) summary** immediately and fire the regen as `asyncio.create_task(...)`. The next turn picks up the fresh one. A single-turn-stale summary is harmless — it just omits the most recent turn, which is *in working memory verbatim* anyway, so the LLM loses nothing.

Concretely:
- Add `MemoryConfig.background_summary_refresh: bool = True` so it can be disabled (eval/tests want determinism).
- In `_maybe_refresh_summary`, when `cached` doesn't match `marker_for` and `background_summary_refresh` and there's a running loop:
  - `task = asyncio.create_task(self._regenerate_and_persist(session_id, older_turns))` — extract the LLM-call + truncate + `update_session_summary` body into `_regenerate_and_persist`.
  - `task.add_done_callback` → log on exception (mirror `QAService._on_persona_refresh_done` at [`qa_service.py:1035`](../src/graphbuilder/core/retrieval/qa_service.py#L1035)).
  - Return `(cached, False)` — `summary_regenerated=False` in the trace is honest (this turn used the cached one).
  - **First-ever summary** (`cached == ""`): there's nothing to serve. Two choices: (a) block once, as today, or (b) serve `_fallback_summary(older_turns)` (deterministic concat) synchronously and let the background task replace it. (b) keeps even the first cliff off the path; go with (b) but cap it at `max_summary_chars`.
- `RuntimeError` (no running loop, e.g. service used from a sync context) → fall back to the current synchronous path.

### Concurrency / correctness
- Two overlapping `/ask` calls on the same session could both spawn a regen. Harmless — last write wins, `update_session_summary` is idempotent-ish, and the marker makes a redundant regen cheap to detect next time. If we want to be tidy, a per-session `asyncio.Lock` or an in-flight set on `MemoryService`, but it's not load-bearing for a single-instance API. Note it, don't build it for v1.
- The background task holds a reference to `older_turns` (already-loaded `ConversationTurn` objects) — no repo re-read needed, no staleness window beyond what we accept.
- `MemoryService` is currently described as "stateless" in its docstring — a background task + optional lock makes it mildly stateful. Update the docstring.

### Tests
- `tests/unit/test_qa_service.py` / a memory test: assert that with a stale marker and `background_summary_refresh=True`, `build()` returns the cached summary *and* schedules a task (spy on `asyncio.create_task` or assert `update_session_summary` is eventually called via `await asyncio.sleep(0)`).
- Assert `background_summary_refresh=False` preserves today's synchronous behaviour (the eval harness relies on it).

---

## 3. Don't block retrieval on the query embedding

**Leverage: ~150 ms off every turn.** `QAService` calls `await self._embed_query(query)` ([`qa_service.py:384`](../src/graphbuilder/core/retrieval/qa_service.py#L384) for `ask`, [`:576`](../src/graphbuilder/core/retrieval/qa_service.py#L576) for `ask_stream`) *before* the `asyncio.gather(retrieve, memory.build)`. So ~150 ms of embedding runs serially in front of everything — but most of the downstream work doesn't need it yet:

- **BM25 channel** ([`channels.py:234`](../src/graphbuilder/core/retrieval/channels.py#L234)) — takes `query_embedding` but never reads it. Pure lexical.
- **Cypher channel** ([`channels.py:326`](../src/graphbuilder/core/retrieval/channels.py#L326)) — *uses* the embedding for one of its anchor sources (vector-anchored seeds, line ~448), but works fine on `terms` alone and only short-circuits when *both* terms and embedding are empty. It can do term extraction + the term-anchored fetch without waiting.
- **Vector channel** ([`channels.py:90`](../src/graphbuilder/core/retrieval/channels.py#L90)) — needs it immediately; no embedding ⇒ no hits.
- **`memory.build`** — working-memory window + rolling-summary build need nothing. Only **episodic recall** ([`memory.py:308`](../src/graphbuilder/core/retrieval/memory.py#L308)) needs the embedding.

So the embedding gates *only* the vector channel (hard), the Cypher channel's vector-anchor sub-step (soft), and episodic recall (hard). Everything else can be in flight while the embedder runs.

### Approach: pass a *future* for the embedding, not the value

Make `_embed_query` start eagerly (as a task) and have the consumers `await` it only at the point they actually need it:

- In `QAService.ask` / `ask_stream`: `embed_task = asyncio.create_task(self._embed_query(query))` instead of `query_embedding = await self._embed_query(query)`. Then `asyncio.gather(self._orch.retrieve(..., query_embedding=embed_task), self._memory.build(..., query_embedding=embed_task))`.
- **Cleanest seam:** change the `query_embedding` parameter type on `RetrievalOrchestrator.retrieve`, `MemoryService.build`, and each channel's `run(...)` to accept *either* `Optional[List[float]]` *or* `Awaitable[Optional[List[float]]]`. Resolve it lazily, as late as possible:
  - `orchestrator.retrieve` / `_run_channels`: pass the awaitable straight through to all three channel coros and `gather` them. Inside `VectorChannel.run`, first line: `query_embedding = await query_embedding if isawaitable(query_embedding) else query_embedding` — it then blocks only itself. Inside `CypherChannel.run`, do the resolve *after* term extraction and *before* the vector-anchor fetch, so the term-anchored work overlaps the embedder. `Bm25Channel.run` ignores it entirely, so it runs flat-out from t=0. Net: BM25 + Cypher's term path + the embedder all run concurrently; vector + Cypher's vector-anchor path join when the embedding lands.
  - `memory.build`: resolve the awaitable only inside the `if enable_episodic_recall` branch (skip recall if it resolves to `None`). Working memory + summary build don't touch it.
  - Add a tiny helper `async def _resolve(x): return await x if inspect.isawaitable(x) else x` (or inline it) so the `isawaitable` dance isn't copy-pasted four times.
- Persist path: `_append_turn` needs the resolved `query_embedding` value (it's stored on the turn). By the time we reach `_append_turn` the task is long done — `await embed_task` again is free (returns the cached result, doesn't re-run). Keep a local `query_embedding = await embed_task` right after the gather and use it for the rest of the function.

### Caveats
- If embedding fails, `_embed_query` already returns `None` and swallows the exception — the future resolves to `None`, every consumer handles `None` today. Good.
- Don't let the embedding task become an orphaned exception: `_embed_query` never raises (try/except inside), so `create_task` won't leak an un-retrieved exception. Still, add it to a local var and `await` it on the persist path so it's definitely consumed.
- This is a smaller, fiddlier change than §1/§2 for ~150 ms — do it last. If threading the awaitable through the channels gets hairy, an acceptable 80%-version: `embed_task = create_task(...)` in `QAService`, pass the awaitable only to `memory.build` (resolved lazily in the episodic branch), but still `await embed_task` before `orch.retrieve`. That overlaps memory's working+summary build with the embedder; retrieval still waits. Half the win, a tenth the risk.

### Tests
- Existing retrieval/memory tests pass a concrete `query_embedding` — keep that working (the `isawaitable` check is a no-op for a list).
- Add one test passing an `asyncio.Future`/coro as `query_embedding` and asserting the result is identical.

---

## 4. `max_tokens` trimming — investigated, won't do

Considered lowering the `max_tokens=1024` cap on the QA generate calls ([`qa_service.py:1156`](../src/graphbuilder/core/retrieval/qa_service.py#L1156) and the streaming/agentic siblings). **No effect on latency.** gpt-4o-mini (and every modern chat model) generates until it emits a stop token — latency tracks *tokens actually generated*, not the cap. QA answers are ~100–200 tokens (the system prompt asks for 2–5 sentences). The 1024 cap only bounds a pathological runaway; lowering it risks truncating a legitimately long "tell me about…" answer mid-sentence for zero speedup. Leave it.

(If we ever switch to a provider/model that pre-allocates a KV-cache sized to `max_tokens`, revisit — but OpenAI doesn't, and neither does the local-LLM path as configured.)

---

## Sequencing

1. **§1 first** — biggest UX win, fully contained in the frontend, backend already shipped. Ship it alone.
2. **§2 next** — small, server-only, removes the worst real-latency outlier (long sessions). Independent of §1.
3. **§3 last** — smallest payoff, fiddliest diff. Do the 80%-version if the `_run_channels` restructure looks risky.

Each lands as its own PR. No schema changes. No new deps (the SSE parser is hand-rolled `fetch` + `TextDecoder`).
