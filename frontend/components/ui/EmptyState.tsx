'use client';

import type { LucideIcon } from 'lucide-react';
import type { ReactNode } from 'react';

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  bouncy = false,
  className,
}: {
  icon: LucideIcon;
  title: string;
  description?: ReactNode;
  action?: ReactNode;
  bouncy?: boolean;
  className?: string;
}) {
  return (
    <div className={`empty-state ${className ?? ''}`}>
      <span
        className={`empty-icon ${bouncy ? 'bouncy' : ''}`}
        aria-hidden="true"
      >
        <Icon size={20} />
      </span>
      <p
        className="text-[13px] font-semibold"
        style={{ color: 'var(--text-secondary)' }}
      >
        {title}
      </p>
      {description && (
        <p className="text-[12px] max-w-sm leading-relaxed">{description}</p>
      )}
      {action && <div className="pt-1">{action}</div>}
    </div>
  );
}
