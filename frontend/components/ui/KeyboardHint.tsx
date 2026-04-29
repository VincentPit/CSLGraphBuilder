'use client';

import { Keyboard } from 'lucide-react';
import type { ReactNode } from 'react';

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd
      className="inline-flex items-center justify-center min-w-[20px] h-[20px] px-[5px] rounded-md text-[11px] font-bold tabular-nums"
      style={{
        background: 'var(--bg-card)',
        color: 'var(--text-primary)',
        border: '1.5px solid var(--border-default)',
        boxShadow: '0 1.5px 0 var(--border-default)',
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
      }}
    >
      {children}
    </kbd>
  );
}

/**
 * Footer chip that surfaces "Keyboard mode active — press ? for help".
 * Renders nothing on touch-only devices to avoid noise.
 */
export function KeyboardHint({
  onShowHelp,
  className,
}: {
  onShowHelp?: () => void;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onShowHelp}
      className={['inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold uppercase tracking-wider transition-colors', className]
        .filter(Boolean).join(' ')}
      style={{
        background: 'var(--bg-muted)',
        color: 'var(--text-secondary)',
        border: '1.5px solid var(--border-default)',
      }}
      aria-label="Show keyboard shortcuts"
    >
      <Keyboard size={11} aria-hidden="true" />
      Keyboard mode · press <Kbd>?</Kbd>
    </button>
  );
}

export type Shortcut = {
  keys: string[];
  description: string;
};

export function ShortcutList({ shortcuts }: { shortcuts: Shortcut[] }) {
  return (
    <dl className="space-y-2">
      {shortcuts.map((s, i) => (
        <div key={i} className="flex items-center justify-between gap-4 text-[13px]">
          <dt style={{ color: 'var(--text-secondary)' }}>{s.description}</dt>
          <dd className="flex items-center gap-1">
            {s.keys.map((k, j) => (
              <Kbd key={j}>{k}</Kbd>
            ))}
          </dd>
        </div>
      ))}
    </dl>
  );
}
