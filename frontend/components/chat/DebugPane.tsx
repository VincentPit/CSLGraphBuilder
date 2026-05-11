'use client';
/**
 * DebugPane — developer-only collapsible inspector for a single turn
 * (§8.5 of docs/RAG_QA_PLAN.md).
 *
 * Rendered when the chat page is loaded with ``?debug=1`` (or the
 * caller passes ``forceOpen``). Surfaces the per-turn fields the
 * production UI omits because they're noise for end users but
 * essential for diagnosing a bad answer:
 *
 * - ``request_id`` (copy-on-click)
 * - intent + chosen retrieval profile (from ``retrieval_trace.intent``)
 * - memory trace (working / summary / episodic)
 * - tool-calls table (name / latency / result-or-error)
 * - faithfulness verdicts (per-claim chips with the lexical score)
 *
 * Reuses the existing ``RetrievalTracePane`` for the channel breakdown,
 * so we don't duplicate that view here.
 */

import { useState } from 'react';
import {
  AlertTriangle,
  Check,
  ChevronDown,
  ChevronUp,
  Clipboard,
  Cog,
  Database,
  Sparkles,
} from 'lucide-react';

import {
  AskResponse,
  ClaimVerification,
  FaithfulnessResult,
  MemoryTrace,
  ToolCall,
} from '@/lib/api';

interface Props {
  response: AskResponse;
  /** Override the auto-detection of ``?debug=1``. */
  forceOpen?: boolean;
}

function copyToClipboard(text: string) {
  if (typeof navigator !== 'undefined' && navigator.clipboard) {
    void navigator.clipboard.writeText(text);
  }
}

export default function DebugPane({ response, forceOpen }: Props) {
  const [open, setOpen] = useState(Boolean(forceOpen));
  const memoryTrace = response.memory_trace ?? null;
  const toolCalls = response.tool_calls ?? [];
  const faithfulness = response.faithfulness ?? null;
  // retrieval_trace.intent is added by the QAService when intent
  // routing fires; on retrieval_override paths it can be null.
  const intent = (response.retrieval_trace as { intent?: string | null } | undefined)
    ?.intent ?? null;

  return (
    <div
      className="rounded-lg overflow-hidden mt-2"
      style={{ border: '1px dashed var(--border-subtle)' }}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full px-3 py-2 flex items-center justify-between text-xs hover:bg-[var(--bg-muted)] transition"
        style={{ color: 'var(--text-secondary)' }}
      >
        <span className="inline-flex items-center gap-2">
          <Cog size={12} />
          <span className="font-medium">Debug pane</span>
          <span style={{ color: 'var(--text-muted)' }}>
            · {toolCalls.length} tool call{toolCalls.length === 1 ? '' : 's'}
            {' · faithfulness '}
            {faithfulness?.overall_score != null
              ? faithfulness.overall_score.toFixed(2)
              : 'n/a'}
            {' · '}
            {response.latency_ms}ms
          </span>
        </span>
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>

      {open && (
        <div
          className="px-3 py-3 space-y-4 text-xs"
          style={{ background: 'var(--bg-muted)' }}
        >
          <RequestMetadata
            requestId={response.request_id}
            intent={intent}
            latencyMs={response.latency_ms}
          />
          {memoryTrace && <MemorySection trace={memoryTrace} />}
          {toolCalls.length > 0 && <ToolCallsSection calls={toolCalls} />}
          {faithfulness && <FaithfulnessSection result={faithfulness} />}
        </div>
      )}
    </div>
  );
}

function SectionHeader({ icon: Icon, label }: { icon: typeof Cog; label: string }) {
  return (
    <p
      className="text-[10px] font-bold uppercase tracking-widest mb-1 inline-flex items-center gap-1"
      style={{ color: 'var(--text-muted)' }}
    >
      <Icon size={10} /> {label}
    </p>
  );
}

function RequestMetadata({
  requestId,
  intent,
  latencyMs,
}: {
  requestId?: string | null;
  intent: string | null;
  latencyMs: number;
}) {
  return (
    <div>
      <SectionHeader icon={Cog} label="Request" />
      <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1">
        <dt className="text-muted-foreground">request_id</dt>
        <dd className="flex items-center gap-1">
          <code className="font-mono">{requestId ?? 'n/a'}</code>
          {requestId && (
            <button
              type="button"
              onClick={() => copyToClipboard(requestId)}
              className="text-muted-foreground hover:text-current"
              aria-label="Copy request id"
            >
              <Clipboard size={11} />
            </button>
          )}
        </dd>
        <dt className="text-muted-foreground">intent</dt>
        <dd>{intent ?? '(override — routing bypassed)'}</dd>
        <dt className="text-muted-foreground">latency_ms</dt>
        <dd>{latencyMs}</dd>
      </dl>
    </div>
  );
}

function MemorySection({ trace }: { trace: MemoryTrace }) {
  return (
    <div>
      <SectionHeader icon={Database} label="Memory" />
      <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1">
        <dt className="text-muted-foreground">working_turns</dt>
        <dd>{trace.working_turns}</dd>
        <dt className="text-muted-foreground">summary_chars</dt>
        <dd>
          {trace.summary_chars}
          {trace.summary_regenerated && (
            <span className="ml-1 text-amber-700 dark:text-amber-300">(refreshed)</span>
          )}
        </dd>
        <dt className="text-muted-foreground">episodic</dt>
        <dd>
          {trace.episodic_hit
            ? (
                <span>
                  hit {trace.episodic_hit.turn_id.slice(0, 12)}… (sim&nbsp;
                  {trace.episodic_hit.score.toFixed(2)})
                </span>
              )
            : <span className="text-muted-foreground">no hit</span>}
        </dd>
      </dl>
    </div>
  );
}

function ToolCallsSection({ calls }: { calls: ToolCall[] }) {
  return (
    <div>
      <SectionHeader icon={Sparkles} label="Tool calls" />
      <table className="w-full text-[11px]">
        <thead>
          <tr className="text-left text-muted-foreground">
            <th className="font-normal pr-2">tool</th>
            <th className="font-normal pr-2">latency</th>
            <th className="font-normal">result / error</th>
          </tr>
        </thead>
        <tbody>
          {calls.map((c, i) => (
            <tr key={i} className="align-top border-t border-dashed">
              <td className="font-mono pr-2 py-1">{c.tool}</td>
              <td className="pr-2 py-1">{c.latency_ms}ms</td>
              <td className="py-1 font-mono break-all">
                {c.error ? (
                  <span className="text-red-600 dark:text-red-400">
                    error: {c.error}
                  </span>
                ) : (
                  <span className="text-muted-foreground">
                    {summariseToolResult(c.result)}
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function summariseToolResult(result: ToolCall['result']): string {
  if (!result) return '(empty)';
  // Keep the shape readable — don't dump 10kB of search hits inline.
  const text = JSON.stringify(result);
  return text.length > 200 ? `${text.slice(0, 199)}…` : text;
}

function FaithfulnessSection({ result }: { result: FaithfulnessResult }) {
  if (result.claims.length === 0) {
    return (
      <div>
        <SectionHeader icon={Check} label="Faithfulness" />
        <p className="text-muted-foreground">
          No cited claims to score (refusal / no citations).
        </p>
      </div>
    );
  }
  return (
    <div>
      <SectionHeader icon={Check} label="Faithfulness" />
      <p className="mb-1">
        overall&nbsp;
        <span className="font-mono">
          {result.overall_score != null ? result.overall_score.toFixed(2) : 'n/a'}
        </span>
        {result.failed_claims > 0 && (
          <span className="ml-2 text-red-600 dark:text-red-400">
            {result.failed_claims} failed
          </span>
        )}
      </p>
      <ul className="space-y-1">
        {result.claims.map((claim, i) => (
          <li key={i}>
            <ClaimChip claim={claim} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function ClaimChip({ claim }: { claim: ClaimVerification }) {
  const verdict = (claim.verdict || 'borderline').toLowerCase();
  const palette: Record<string, string> = {
    supported: 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200',
    borderline: 'bg-amber-100 dark:bg-amber-900 text-amber-800 dark:text-amber-200',
    unsupported: 'bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200',
  };
  const cls = palette[verdict] ?? palette.borderline;
  const Icon = verdict === 'supported' ? Check : verdict === 'unsupported' ? AlertTriangle : Sparkles;
  return (
    <div className="flex items-start gap-1">
      <span
        className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium shrink-0 ${cls}`}
      >
        <Icon className="w-3 h-3" /> {claim.score.toFixed(2)}
      </span>
      <span className="text-xs flex-1">{claim.claim_text}</span>
    </div>
  );
}
