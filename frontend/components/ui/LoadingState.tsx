'use client';

import { Loader2 } from 'lucide-react';
import type { ReactNode } from 'react';

type Variant = 'spinner' | 'skeleton' | 'inline';

export function LoadingState({
  variant = 'spinner',
  label = 'Loading…',
  className,
  children,
}: {
  variant?: Variant;
  label?: string;
  className?: string;
  children?: ReactNode;
}) {
  if (variant === 'skeleton') {
    return (
      <div
        role="status"
        aria-label={label}
        aria-busy="true"
        className={['skeleton', className].filter(Boolean).join(' ')}
      >
        <span className="sr-only">{label}</span>
      </div>
    );
  }
  if (variant === 'inline') {
    return (
      <span
        role="status"
        aria-busy="true"
        className={[
          'inline-flex items-center gap-2 text-[13px] font-semibold',
          className,
        ].filter(Boolean).join(' ')}
        style={{ color: 'var(--text-muted)' }}
      >
        <Loader2 size={14} className="animate-spin" aria-hidden="true" />
        {children ?? label}
      </span>
    );
  }
  return (
    <div
      role="status"
      aria-busy="true"
      className={['flex items-center justify-center gap-2 p-6 text-[13px] font-semibold', className]
        .filter(Boolean)
        .join(' ')}
      style={{ color: 'var(--text-muted)' }}
    >
      <Loader2 size={16} className="animate-spin" aria-hidden="true" />
      {children ?? label}
    </div>
  );
}

/** Convenience grid of skeleton blocks — used inside cards while data loads. */
export function SkeletonGrid({
  count = 6,
  cols = 'grid-cols-2 md:grid-cols-3',
  blockClassName = 'h-24',
}: {
  count?: number;
  cols?: string;
  blockClassName?: string;
}) {
  return (
    <div className={`grid ${cols} gap-3`} role="status" aria-busy="true" aria-label="Loading">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className={`skeleton ${blockClassName}`} />
      ))}
    </div>
  );
}
