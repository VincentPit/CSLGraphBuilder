# RAG_QA — post-P14 follow-ups

Companion plan to [`RAG_QA_PLAN.md`](RAG_QA_PLAN.md). All 15 numbered phases (P0–P14) have shipped; this document tracks the remaining engineering work + open product decisions called out in `RAG_QA_PLAN.md` §13 (deferred row) and §14 (open questions).

## Status

| Item | Type | Status |
|---|---|---|
| §1. `MutationCard` + `DebugPane` in `/chat` | Engineering — deferred from P12 | ✅ shipped 2026-05-10 |
| §2. Q2–Q8 product decisions (`RAG_QA_PLAN.md` §14) | Product | ✅ resolved 2026-05-10 |
| §3. `/qa/ask/stream` × tool-use combo | Engineering — eval-driven | ✅ shipped 2026-05-10 (Option A) |
| §4. P14 polish (manual refresh, cross-session episodic recall) | Engineering — optional | deferred |

---

## 1. `+ MutationCard / DebugPane` in `/chat`

The last `deferred` row in `RAG_QA_PLAN.md` §13. P9/P10/P14 ship the data on `AskResponse`; the React `/chat` page ignores most of it. Scope: render mutating-tool proposals as inline cards and a developer pane that surfaces traces + faithfulness verdicts.

### 1a. Plumbing — extend `AskResponse` on the frontend (~30 min)

- [`frontend/lib/api.ts:542`](../frontend/lib/api.ts#L542) — add to `AskResponse`:
  - `tool_calls?: ToolCall[]`
  - `faithfulness?: FaithfulnessResult | null`
  - `cited_source_indices?: number[]`
- New types mirror [`api/schemas/qa.py`](../api/schemas/qa.py): `ToolCall`, `FaithfulnessResult`, `ClaimVerification`.
- Add `enableTools?: boolean`, `enableMutations?: boolean` to the `askQuestion` request body; thread through to the POST.
- Add API client wrappers:
  - `applyProposal(proposalId, notes?)` → `POST /qa/proposals/{id}/apply`
  - `rejectProposal(proposalId, reason?)` → `POST /qa/proposals/{id}/reject`
  - `listProposals(status?)` → `GET /qa/proposals`

### 1b. `MutationCard.tsx` — §7.4 focused subgraph preview (~½ day)

New component in [`frontend/components/chat/`](../frontend/components/chat/). Rendered inline in the thread alongside the assistant bubble whenever a turn's `tool_calls[]` contains a mutating tool (use the `is_mutation` set from [`src/graphbuilder/core/retrieval/mutation_tools.py`](../src/graphbuilder/core/retrieval/mutation_tools.py)).

**Card body:**
- Header: tool name + one-line plain-English summary (e.g. *"Propose new entity: BRCA1 (Gene)"*).
- Args panel: pretty-printed JSON, sensitive ids truncated.
- Diff preview:
  - `update_entity` / `merge_entities`: fetch current entity via existing `/graph` API, render before/after side-by-side.
  - `propose_*`: render the proposed payload only.
  - **Out of scope for v1:** full Cytoscape subgraph preview from §7.4 — too much UI surface; defer to v2.
- Status pill + actions: *"Pending review"*, *"View in /curation"* link to the existing curator page, inline *"Approve"* / *"Reject"* buttons calling `applyProposal` / `rejectProposal`. Any authenticated user is a curator for now (§14.6 lightweight auth).

**Empty/error states:** when `record.error` is present, render a red card with the error text and a hint (e.g. *"Mutating tools are not enabled for this request"*).

### 1c. `DebugPane.tsx` — §8.5 developer debug pane (~½ day)

New collapsible pane, default closed. Toggle via `?debug=1` query param + a keyboard shortcut (`⌘.`) so production users never see it.

**Per-turn dump:**
- `request_id` (copy-on-click)
- `intent` + chosen retrieval profile from `retrieval_trace.intent`
- Memory trace: working turns, summary chars, episodic hit + score, `summary_regenerated`
- Tool-calls table: name / args / latency / result-or-error
- Faithfulness verdicts: per-claim chip (✓ supported / ⚠ borderline / ✗ unsupported) with the matched chunk + lexical score. Clicking a chip highlights the corresponding citation in the bubble.

**Reuse:** existing [`RetrievalTracePane`](../frontend/components/chat/RetrievalTracePane.tsx) is wrapped as a collapsible inside `DebugPane` — same JSON shape, no need to duplicate.

### 1d. Wire-up in [`frontend/app/chat/page.tsx:202`](../frontend/app/chat/page.tsx#L202) (~1 hr)

- On response, push `tool_calls` + `faithfulness` onto the rendered turn alongside `sources`.
- Render `<MutationCard />` per mutating record; `<DebugPane />` once per turn when `?debug=1` is set.
- Add `enableTools` / `enableMutations` toggles to the composer (Settings cog), default **off**, persisted to `localStorage`.

### 1e. Tests

- Snapshot test for `MutationCard` against each tool kind (`propose_entity`, `merge_entities`, `soft_delete_*`, error case).
- One Playwright e2e (if the harness exists in [`frontend/`](../frontend/)): toggle "enable mutations" → ask a mutating question → expect a `MutationCard` with the right proposal id → click *"View in /curation"* → land on the existing curator page.

**Total estimate: ~1.5 days for v1.**

---

## 2. Open product decisions (`RAG_QA_PLAN.md` §14) — ✅ resolved 2026-05-10

| # | Resolution | Code |
|---|---|---|
| **Q2** | gpt-4o-mini default for QA flow; per-request `AskRequest.model` override; ingestion stays on gpt-4o. New `QA_LLM_MODEL_NAME` env var + `qa_model_name` field on `LLMConfiguration`. | [`settings.py:88`](../src/graphbuilder/infrastructure/config/settings.py#L88), [`llm_service.py`](../src/graphbuilder/infrastructure/services/llm_service.py) (`model` kwarg on `generate_text` / `generate_text_stream` / `generate_with_tools`), [`qa_service.py:_resolve_model`](../src/graphbuilder/core/retrieval/qa_service.py), [`api/schemas/qa.py`](../api/schemas/qa.py) `AskRequest.model`. Tests: `test_qa_service.py::test_ask_*_model_*`. |
| **Q3** | Keep `cross-encoder/ms-marco-MiniLM-L-6-v2`. No code change. | Documentation only — `RAG_QA_PLAN.md` §14 Q3. |
| **Q5** | Mine the curation queue. New script emits a YAML draft for SME edit. | [`scripts/seed_gold_from_curation.py`](../scripts/seed_gold_from_curation.py). |
| **Q7** | Soft-delete only in v1. No code change. | Documentation only — `RAG_QA_PLAN.md` §14 Q7. |
| **Q8** | No retention policy in v1; persist indefinitely. No code change. | Documentation only — `RAG_QA_PLAN.md` §14 Q8. |

---

## 3. `/qa/ask/stream` × tool-use combo

Surfaced as an open item in the P11 refinements: `/qa/ask/stream` silently ignores `enable_tools` / `enable_mutations` ([`api/routers/qa.py:273`](../api/routers/qa.py#L273)). Clients that want both must fall back to `/qa/ask`.

### Option A — block on agentic loop, stream the final answer (~½ day)

Run the agentic loop to completion inside `ask_stream`, *then* stream the final answer tokens. Loses the "watch it think" UX but is straightforward and doesn't depend on provider capability.

**Event sequence:**
```
phase(retrieving) → retrieval → phase(tools) → tool_call × N → tool_result × N
                  → phase(generating) → delta × N → done
```

### Option B — interleave tool events mid-stream (~2 days)

Emit `tool_call` events as the LLM invokes tools and `tool_result` events as the dispatcher returns, between deltas. Frontend renders tool activity inline. Matches Claude.ai behaviour.

**Dependency:** needs LLM-provider streaming-with-tools support. Verify in [`llm_factory`](../src/graphbuilder/infrastructure/services/llm_factory.py) before committing — OpenAI's current API supports it; older fakes don't.

**Recommendation:** Option A for v1, Option B as a follow-up once frontend tool-call rendering is solid (§1b lands the rendering first).

---

## 4. P14 polish (low-priority)

### 4a. Manual persona-refresh endpoint (~30 min)

`POST /users/{id}/persona/refresh` calls `SemanticMemoryService.refresh_persona(force=True)`. Useful for SME/admin testing without round-tripping through `DELETE /qa/sessions/{id}`.

### 4b. Cross-session episodic recall (~½ day)

Wire the `semantic_embedding` already persisted on `User.metadata` into a new `vector_search_turns(session_id=None, user_id=...)` path so cross-session queries like *"what did I ask about Imatinib last week?"* work.

**Needs:** a new Neo4j vector index on `User.semantic_embedding` (mirrors `turn_query_vector`).

**Skip until** users actually ask for this — v1 ships persona text in the system prompt, which covers most of the value.

---

## Suggested order

1. **Q2 + Q3 + Q5 + Q7 + Q8** — answer in one round; mostly trivial to implement after decisions land.
2. **§1 MutationCard + DebugPane** — closes the last `deferred` row in `RAG_QA_PLAN.md` §13.
3. **§3 Option A** — makes `/qa/ask/stream` feature-complete vs `/qa/ask`.
4. **§4** — defer indefinitely unless usage demands.
