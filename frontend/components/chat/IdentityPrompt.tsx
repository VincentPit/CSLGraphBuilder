'use client';
/**
 * Inline first-visit identity card. Replaces the welcome card on /chat
 * until the browser has a registered user_id stashed in localStorage.
 *
 * No password, no email — just a free-text display name. The UX is
 * "name your seat in the room" rather than "log in".
 */

import { useState } from 'react';
import { Loader2, Sparkles, UserCircle } from 'lucide-react';
import { ChatIdentity, setStoredIdentity } from '@/lib/identity';
import { formatApiError, registerChatUser } from '@/lib/api';

interface Props {
  onIdentitySet: (identity: ChatIdentity) => void;
}

export default function IdentityPrompt({ onIdentitySet }: Props) {
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed || busy) return;
    setBusy(true);
    setError(null);
    try {
      const user = await registerChatUser({ display_name: trimmed });
      const identity: ChatIdentity = { id: user.id, displayName: user.display_name };
      setStoredIdentity(identity);
      onIdentitySet(identity);
    } catch (err) {
      setError(formatApiError(err, 'Could not register'));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="card p-6 space-y-4"
      style={{
        background: 'var(--accent-soft)',
        borderColor: 'var(--accent-muted)',
      }}
    >
      <div className="flex items-center gap-2">
        <Sparkles size={16} style={{ color: 'var(--accent)' }} />
        <p className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
          Hi! What should we call you?
        </p>
      </div>
      <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
        Pick a display name so your chat history stays in your own bucket
        (instead of mixed in with everyone else who's poking at this
        backend). Stored locally in your browser — no password, no
        email, swappable any time.
      </p>
      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <div className="relative">
          <UserCircle
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2"
            style={{ color: 'var(--text-muted)' }}
          />
          <input
            type="text"
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Stephen"
            maxLength={80}
            disabled={busy}
            className="input w-full pl-9"
          />
        </div>
        {error && (
          <p className="text-xs" style={{ color: 'var(--danger)' }}>
            {error}
          </p>
        )}
        <button
          type="submit"
          disabled={busy || !name.trim()}
          className="btn-primary self-start"
        >
          {busy ? <Loader2 size={14} className="animate-spin" /> : null}
          Continue
        </button>
      </form>
    </div>
  );
}
