'use client';

import Link from 'next/link';
import { ArrowUpRight, type LucideIcon } from 'lucide-react';
import AnimatedNumber from '../AnimatedNumber';

export function StatCard({
  label,
  value,
  icon: Icon,
  accent,
  sub,
  href,
  format,
}: {
  label: string;
  value: number | string;
  icon: LucideIcon;
  /** CSS colour (any valid color string) for the icon chip + decoration. */
  accent: string;
  sub?: string;
  href?: string;
  format?: (n: number) => string;
}) {
  const inner = (
    <div className="card-chunky p-7 sm:p-8 flex flex-col gap-3 relative h-full group">
      {/* Decorative blur in its own clipped layer so it can't leak
          outside the card — content above stays unclipped. */}
      <div
        className="absolute inset-0 overflow-hidden pointer-events-none"
        style={{ borderRadius: 'inherit' }}
        aria-hidden="true"
      >
        <div
          className="absolute -top-10 -right-10 h-28 w-28 rounded-full opacity-25 blur-2xl"
          style={{ background: accent }}
        />
      </div>

      <div className="flex items-center justify-between gap-2 relative min-w-0">
        <span className="field-label !mb-0 truncate min-w-0">{label}</span>
        <div
          className="h-10 w-10 flex items-center justify-center shrink-0"
          style={{
            background: `linear-gradient(135deg, ${accent}26, ${accent}14)`,
            border: `1.5px solid ${accent}33`,
            boxShadow: `0 2px 0 ${accent}33`,
            borderRadius: 'var(--radius-md)',
            color: accent,
          }}
        >
          <Icon size={18} strokeWidth={2.4} aria-hidden="true" />
        </div>
      </div>
      <div className="flex items-end gap-2 relative min-w-0">
        <p
          className="text-[28px] sm:text-[32px] lg:text-[36px] font-black tracking-tight tabular-nums leading-none truncate min-w-0"
          style={{ color: 'var(--text-primary)' }}
        >
          {typeof value === 'number'
            ? <AnimatedNumber value={value} format={format} />
            : value}
        </p>
        {href && (
          <ArrowUpRight
            size={16}
            className="mb-1.5 opacity-0 group-hover:opacity-70 transition-opacity shrink-0"
            style={{ color: accent }}
            strokeWidth={2.4}
            aria-hidden="true"
          />
        )}
      </div>
      {sub && (
        <p
          className="text-[12px] font-semibold relative line-clamp-2"
          style={{ color: 'var(--text-muted)' }}
        >
          {sub}
        </p>
      )}
    </div>
  );

  if (href) {
    return (
      <Link href={href} className="block h-full" aria-label={`${label}: open details`}>
        {inner}
      </Link>
    );
  }
  return inner;
}
