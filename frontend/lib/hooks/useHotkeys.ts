'use client';

import { useEffect } from 'react';

export type HotkeyMap = Record<string, (e: KeyboardEvent) => void>;

/**
 * Bind a map of single-key shortcuts to the document. Skips events that
 * originated from a typing context (input / textarea / contenteditable)
 * unless the bound key is a modifier-style keystroke that wouldn't
 * conflict with text entry (Escape, ?).
 *
 * Keys are matched on `e.key` (case-insensitive). The map should look
 * like: `{ j: nextItem, k: prevItem, '?': openHelp }`.
 *
 * Pass `enabled: false` to short-circuit (useful while a modal is open).
 */
export function useHotkeys(map: HotkeyMap, enabled = true) {
  useEffect(() => {
    if (!enabled) return;
    function isTyping(target: EventTarget | null): boolean {
      if (!(target instanceof HTMLElement)) return false;
      const tag = target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
      if (target.isContentEditable) return true;
      return false;
    }
    function handler(e: KeyboardEvent) {
      // Allow Escape from anywhere — even inputs.
      const k = e.key.toLowerCase();
      if (k !== 'escape' && isTyping(e.target)) return;
      // Don't fire if any modifier other than Shift is held.
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const handler =
        map[k] ?? map[e.key] ?? (e.shiftKey && e.key === '?' ? map['?'] : undefined);
      if (handler) {
        e.preventDefault();
        handler(e);
      }
    }
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [map, enabled]);
}
