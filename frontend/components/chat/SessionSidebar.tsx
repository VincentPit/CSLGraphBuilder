'use client';
/**
 * Sidebar with past sessions. Click to load, X to delete, "+ New" to start
 * fresh. Anonymous-only for v1 — when auth lands the user_id filter is
 * already in place on the backend.
 */

import { ChatSession, deleteChatSession } from '@/lib/api';
import { Plus, MessageSquare, Trash2, Loader2 } from 'lucide-react';
import { useState } from 'react';

interface Props {
  sessions: ChatSession[];
  loading: boolean;
  activeSessionId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  /** Called after a successful delete so the parent can refetch. */
  onDeleted: (id: string) => void;
}

function relativeTime(iso: string): string {
  const then = new Date(iso).getTime();
  const now = Date.now();
  const s = Math.max(0, Math.floor((now - then) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export default function SessionSidebar({
  sessions,
  loading,
  activeSessionId,
  onSelect,
  onNew,
  onDeleted,
}: Props) {
  const [deletingId, setDeletingId] = useState<string | null>(null);

  async function handleDelete(e: React.MouseEvent, id: string) {
    e.stopPropagation();
    setDeletingId(id);
    try {
      await deleteChatSession(id);
      onDeleted(id);
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <aside
      className="w-full lg:w-64 shrink-0 lg:max-h-[calc(100vh-8rem)] flex flex-col"
      style={{ borderRight: '1px solid var(--border-subtle)' }}
    >
      <div className="p-3">
        <button
          type="button"
          onClick={onNew}
          className="btn-primary w-full justify-center text-sm"
        >
          <Plus size={14} /> New chat
        </button>
      </div>

      <p
        className="px-4 mb-1.5 text-[10px] font-extrabold uppercase tracking-widest"
        style={{ color: 'var(--text-muted)' }}
      >
        Past sessions
      </p>

      <div className="flex-1 overflow-y-auto px-2 pb-3 space-y-1">
        {loading && (
          <div
            className="flex items-center gap-2 px-3 py-2 text-xs"
            style={{ color: 'var(--text-muted)' }}
          >
            <Loader2 size={12} className="animate-spin" /> Loading…
          </div>
        )}
        {!loading && sessions.length === 0 && (
          <p
            className="px-3 py-2 text-xs"
            style={{ color: 'var(--text-muted)' }}
          >
            No past sessions yet.
          </p>
        )}
        {sessions.map((s) => {
          const active = s.id === activeSessionId;
          const title = s.title || s.summary || `Session · ${s.id.slice(8, 16)}`;
          return (
            <div
              key={s.id}
              role="button"
              tabIndex={0}
              onClick={() => onSelect(s.id)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onSelect(s.id);
                }
              }}
              className="group flex items-center gap-2 px-2.5 py-2 rounded-md cursor-pointer transition"
              style={{
                background: active ? 'var(--accent-muted)' : 'transparent',
                color: active ? 'var(--text-primary)' : 'var(--text-secondary)',
              }}
              onMouseEnter={(e) => {
                if (!active) e.currentTarget.style.background = 'var(--bg-muted)';
              }}
              onMouseLeave={(e) => {
                if (!active) e.currentTarget.style.background = 'transparent';
              }}
            >
              <MessageSquare
                size={13}
                style={{ color: active ? 'var(--accent)' : 'var(--text-muted)' }}
              />
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium truncate" title={title}>
                  {title}
                </p>
                <p
                  className="text-[10px]"
                  style={{ color: 'var(--text-muted)' }}
                >
                  {s.turn_count} turn{s.turn_count === 1 ? '' : 's'} ·{' '}
                  {relativeTime(s.last_active_at)}
                </p>
              </div>
              <button
                type="button"
                onClick={(e) => handleDelete(e, s.id)}
                className="opacity-0 group-hover:opacity-100 p-1 rounded transition hover:bg-[var(--danger-muted)] focus:opacity-100"
                aria-label={`Delete session ${title}`}
                title="Delete session"
                disabled={deletingId === s.id}
              >
                {deletingId === s.id ? (
                  <Loader2 size={11} className="animate-spin" />
                ) : (
                  <Trash2 size={11} style={{ color: 'var(--text-muted)' }} />
                )}
              </button>
            </div>
          );
        })}
      </div>
    </aside>
  );
}
