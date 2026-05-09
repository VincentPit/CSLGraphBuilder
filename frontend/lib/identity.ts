/**
 * Lightweight browser-identity helpers for the chatbot.
 *
 * Persists the current user's ``id`` + ``display_name`` in
 * localStorage so:
 *   - the sidebar shows only this user's sessions, and
 *   - every /qa/* request carries an ``X-User-Id`` header.
 *
 * This is *not* authentication — anyone can clear localStorage and
 * register a new identity, or paste someone else's id into devtools.
 * It's the lightweight option from §14.1 of docs/RAG_QA_PLAN.md
 * (revised 2026-05-09); swap for real auth in a follow-up.
 *
 * SSR-safety: every public function returns ``null`` / a no-op when
 * ``window`` is undefined so the same module imports cleanly into
 * Next.js server components.
 */

export interface ChatIdentity {
  id: string;
  displayName: string;
}

const STORAGE_KEY = 'graphbuilder.chat.identity.v1';

function isBrowser(): boolean {
  return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';
}

/** Read the persisted identity, or ``null`` if none / on the server. */
export function getStoredIdentity(): ChatIdentity | null {
  if (!isBrowser()) return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (
      parsed &&
      typeof parsed.id === 'string' &&
      typeof parsed.displayName === 'string'
    ) {
      return { id: parsed.id, displayName: parsed.displayName };
    }
  } catch {
    // Corrupted localStorage — drop it so the user re-registers cleanly.
  }
  return null;
}

/** Replace the persisted identity. Pass ``null`` to clear. */
export function setStoredIdentity(identity: ChatIdentity | null): void {
  if (!isBrowser()) return;
  if (identity === null) {
    window.localStorage.removeItem(STORAGE_KEY);
    return;
  }
  window.localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({ id: identity.id, displayName: identity.displayName }),
  );
}

/** Storage event listener — lets multiple tabs share an identity
 *  switch without each one needing its own sign-in flow. */
export function onIdentityChange(cb: (next: ChatIdentity | null) => void): () => void {
  if (!isBrowser()) return () => undefined;
  const handler = (e: StorageEvent) => {
    if (e.key !== STORAGE_KEY) return;
    cb(getStoredIdentity());
  };
  window.addEventListener('storage', handler);
  return () => window.removeEventListener('storage', handler);
}
