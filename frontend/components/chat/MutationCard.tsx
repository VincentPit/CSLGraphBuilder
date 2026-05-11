'use client';
/**
 * MutationCard — inline card for a chatbot-proposed graph mutation (§7 of
 * docs/RAG_QA_PLAN.md, P10).
 *
 * Rendered alongside the assistant bubble whenever a turn's
 * ``tool_calls[]`` contains a mutating tool (``propose_entity`` /
 * ``propose_relationship`` / ``update_entity`` / ``merge_entities`` /
 * ``soft_delete_entity`` / ``soft_delete_relationship``).
 *
 * v1 surface:
 * - Tool name + plain-English summary header
 * - Pretty-printed args (sensitive ids truncated)
 * - Status pill (pending / approved / rejected / errored)
 * - Curator actions: Approve / Reject inline buttons, plus a "View in
 *   /curation" deep-link for the full workflow
 *
 * Deferred to v2: the full Cytoscape-based focused-subgraph preview from
 * §7.4 (too much UI surface). The pending → approved flow still works;
 * the SME just gets a diff-only view, not a graph diff.
 */

import type { ReactElement } from 'react';
import { useState } from 'react';
import {
  AlertTriangle,
  Check,
  Clock,
  ExternalLink,
  GitPullRequestArrow,
  X,
} from 'lucide-react';

import {
  applyProposal as applyProposalApi,
  formatApiError,
  rejectProposal as rejectProposalApi,
  ToolCall,
} from '@/lib/api';

/** Tools that actually mutate the graph — must stay in sync with
 *  ``src/graphbuilder/core/retrieval/mutation_tools.py::is_mutation``. */
export const MUTATING_TOOLS = new Set([
  'propose_entity',
  'propose_relationship',
  'update_entity',
  'merge_entities',
  'soft_delete_entity',
  'soft_delete_relationship',
]);

export function isMutationToolCall(tc: ToolCall): boolean {
  return MUTATING_TOOLS.has(tc.tool);
}

type CardStatus = 'pending' | 'approved' | 'rejected' | 'errored';

function summariseTool(tool: string, args: Record<string, unknown>): string {
  // Args shapes mirror the mutation_tools.py Pydantic schemas.
  switch (tool) {
    case 'propose_entity': {
      const name = (args.name as string) ?? '(unnamed)';
      const kind = (args.entity_type as string) ?? 'Entity';
      return `Propose new entity: ${name} (${kind})`;
    }
    case 'propose_relationship': {
      const source = (args.source_entity_id as string) ?? '?';
      const target = (args.target_entity_id as string) ?? '?';
      const rel = (args.relationship_type as string) ?? 'RELATES';
      return `Propose relationship: ${source} —[${rel}]→ ${target}`;
    }
    case 'update_entity': {
      const id = (args.entity_id as string) ?? '?';
      const updates = Object.keys((args.updates as Record<string, unknown>) ?? {});
      return `Update entity ${id} (${updates.length} field${updates.length === 1 ? '' : 's'})`;
    }
    case 'merge_entities': {
      const keep = (args.keep_entity_id as string) ?? '?';
      const drop = (args.drop_entity_id as string) ?? '?';
      return `Merge entity ${drop} → ${keep}`;
    }
    case 'soft_delete_entity':
      return `Soft-delete entity ${(args.entity_id as string) ?? '?'}`;
    case 'soft_delete_relationship':
      return `Soft-delete relationship ${(args.relationship_id as string) ?? '?'}`;
    default:
      return tool;
  }
}

function truncate(text: string, max = 56): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function renderArgsTable(args: Record<string, unknown>): ReactElement {
  const entries = Object.entries(args);
  if (entries.length === 0) {
    return <div className="text-xs text-muted-foreground italic">(no args)</div>;
  }
  return (
    <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 text-xs">
      {entries.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="font-mono text-muted-foreground">{k}</dt>
          <dd className="font-mono break-all">
            {typeof v === 'string' ? truncate(v, 80) : JSON.stringify(v)}
          </dd>
        </div>
      ))}
    </dl>
  );
}

interface Props {
  call: ToolCall;
  /** When omitted, the curator actions don't render — useful for replay
   *  views where we don't want to re-decide an already-decided proposal. */
  curationUrl?: string;
}

export function MutationCard({ call, curationUrl = '/curation' }: Props) {
  // The backend records the proposal id in ``result.proposal_id`` on
  // success and the error string in ``error`` on failure. We mirror that
  // back into the UI status.
  const errored = Boolean(call.error);
  const initialProposalId = errored
    ? null
    : ((call.result?.proposal_id as string | undefined) ?? null);

  const [status, setStatus] = useState<CardStatus>(
    errored ? 'errored' : 'pending',
  );
  const [busy, setBusy] = useState<'apply' | 'reject' | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const proposalId = initialProposalId;

  async function onApprove(): Promise<void> {
    if (!proposalId) return;
    setBusy('apply');
    setActionError(null);
    try {
      const res = await applyProposalApi(proposalId);
      if (res.error) {
        setActionError(res.error);
        // Backend keeps status="approved" even on apply failure so the
        // SME can retry. Reflect that — don't snap back to "pending".
        setStatus('approved');
      } else {
        setStatus('approved');
      }
    } catch (err) {
      setActionError(formatApiError(err, 'apply failed'));
    } finally {
      setBusy(null);
    }
  }

  async function onReject(): Promise<void> {
    if (!proposalId) return;
    setBusy('reject');
    setActionError(null);
    try {
      await rejectProposalApi(proposalId);
      setStatus('rejected');
    } catch (err) {
      setActionError(formatApiError(err, 'reject failed'));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="rounded-md border border-amber-300/50 bg-amber-50/40 dark:bg-amber-950/20 px-3 py-2 my-2 text-sm">
      <div className="flex items-start gap-2">
        <GitPullRequestArrow
          className="w-4 h-4 mt-0.5 text-amber-600 dark:text-amber-400 shrink-0"
          aria-hidden
        />
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <span className="font-medium">
              {summariseTool(call.tool, call.args)}
            </span>
            <StatusPill status={status} />
          </div>

          <div className="mt-2">{renderArgsTable(call.args)}</div>

          {errored && call.error && (
            <div className="mt-2 text-xs text-red-700 dark:text-red-300 flex items-start gap-1">
              <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
              <span>{call.error}</span>
            </div>
          )}

          {actionError && (
            <div className="mt-2 text-xs text-red-700 dark:text-red-300">
              {actionError}
            </div>
          )}

          {proposalId && (
            <div className="mt-2 flex items-center gap-3 text-xs">
              <span className="text-muted-foreground">
                proposal&nbsp;
                <code className="font-mono">{truncate(proposalId, 28)}</code>
              </span>
              <a
                href={curationUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-0.5 text-blue-600 dark:text-blue-400 hover:underline"
              >
                View in /curation <ExternalLink className="w-3 h-3" />
              </a>
              {status === 'pending' && (
                <div className="ml-auto flex gap-2">
                  <button
                    type="button"
                    onClick={onReject}
                    disabled={busy !== null}
                    className="inline-flex items-center gap-1 rounded border px-2 py-0.5 text-xs hover:bg-red-50 dark:hover:bg-red-950 disabled:opacity-50"
                  >
                    <X className="w-3 h-3" /> Reject
                  </button>
                  <button
                    type="button"
                    onClick={onApprove}
                    disabled={busy !== null}
                    className="inline-flex items-center gap-1 rounded bg-green-600 text-white px-2 py-0.5 text-xs hover:bg-green-700 disabled:opacity-50"
                  >
                    <Check className="w-3 h-3" />
                    {busy === 'apply' ? 'Applying…' : 'Approve'}
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatusPill({ status }: { status: CardStatus }) {
  const styles: Record<CardStatus, { cls: string; label: string; Icon: typeof Clock }> = {
    pending: {
      cls: 'bg-amber-100 dark:bg-amber-900 text-amber-700 dark:text-amber-200',
      label: 'pending review',
      Icon: Clock,
    },
    approved: {
      cls: 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-200',
      label: 'approved',
      Icon: Check,
    },
    rejected: {
      cls: 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-200',
      label: 'rejected',
      Icon: X,
    },
    errored: {
      cls: 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-200',
      label: 'errored',
      Icon: AlertTriangle,
    },
  };
  const { cls, label, Icon } = styles[status];
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium ${cls}`}
    >
      <Icon className="w-3 h-3" /> {label}
    </span>
  );
}
