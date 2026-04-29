'use client';

import { AlertCircle, X } from 'lucide-react';
import type { ReactNode } from 'react';

/**
 * Inline error / alert banner. Use for action errors (e.g. mutation
 * failures) — rendered with role=alert so assistive tech announces it.
 *
 * Pass `onDismiss` to render a close button.
 */
export function ErrorBanner({
  title,
  children,
  onDismiss,
  className,
}: {
  title?: string;
  children: ReactNode;
  onDismiss?: () => void;
  className?: string;
}) {
  return (
    <div
      role="alert"
      className={[
        'card p-3 flex items-start gap-2.5 text-[13px]',
        className,
      ].filter(Boolean).join(' ')}
      style={{
        background: 'var(--danger-soft)',
        borderColor: 'rgba(234,43,43,0.35)',
      }}
    >
      <AlertCircle
        size={18}
        strokeWidth={2.4}
        className="shrink-0 mt-0.5"
        style={{ color: 'var(--danger)' }}
        aria-hidden="true"
      />
      <div className="flex-1 min-w-0">
        {title && (
          <p className="font-bold leading-tight" style={{ color: 'var(--danger-shadow)' }}>
            {title}
          </p>
        )}
        <div
          className={title ? 'mt-0.5' : ''}
          style={{ color: 'var(--text-primary)' }}
        >
          {children}
        </div>
      </div>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          className="btn-icon shrink-0"
          style={{ width: 28, height: 28 }}
          aria-label="Dismiss error"
        >
          <X size={14} aria-hidden="true" />
        </button>
      )}
    </div>
  );
}
