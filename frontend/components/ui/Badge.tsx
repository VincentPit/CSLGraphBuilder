'use client';

import type { CSSProperties, ReactNode } from 'react';

type Tone = 'success' | 'danger' | 'warning' | 'info' | 'brand' | 'neutral' | 'xp' | 'streak';

const TONE_CLASS: Record<Tone, string> = {
  success: 'badge-success',
  danger:  'badge-danger',
  warning: 'badge-warning',
  info:    'badge-info',
  brand:   'badge-brand',
  neutral: 'badge-neutral',
  xp:      'badge-xp',
  streak:  'badge-streak',
};

export function Badge({
  tone = 'neutral',
  className,
  style,
  children,
  title,
}: {
  tone?: Tone;
  className?: string;
  style?: CSSProperties;
  children: ReactNode;
  title?: string;
}) {
  const cls = ['badge', TONE_CLASS[tone], className].filter(Boolean).join(' ');
  return (
    <span className={cls} style={style} title={title}>
      {children}
    </span>
  );
}

/** Verification status badge — shared between Curation and Verification pages. */
const STATUS_TONE: Record<string, { tone: Tone; label: string }> = {
  verified:   { tone: 'success', label: 'Verified' },
  rejected:   { tone: 'danger',  label: 'Rejected' },
  flagged:    { tone: 'warning', label: 'Flagged'  },
  unverified: { tone: 'info',    label: 'Unverified' },
};

export function StatusBadge({ status }: { status: string }) {
  const cfg = STATUS_TONE[status];
  if (!cfg) return <Badge tone="neutral">{status}</Badge>;
  return <Badge tone={cfg.tone}>{cfg.label}</Badge>;
}

/** Coloured chip for a biomedical entity type (DISEASE, GENE, …). */
export function TypeChip({
  type,
  color,
}: {
  type?: string | null;
  color: string;
}) {
  if (!type) return null;
  return (
    <span
      className="badge"
      style={{
        background: `${color}1a`,
        color,
        border: `1px solid ${color}55`,
      }}
    >
      {type}
    </span>
  );
}
