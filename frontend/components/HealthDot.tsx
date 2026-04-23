'use client';

import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api';

/**
 * Tiny live indicator: green pulsing dot when the API is reachable,
 * red when it isn't. Used in the sidebar footer.
 */
export default function HealthDot() {
  const { data, isError } = useQuery({
    queryKey: ['health-ping'],
    queryFn: () => apiClient.get('/health').then((r) => r.data),
    refetchInterval: 5000,
    retry: false,
  });

  const ok = !!data && !isError;
  const color = ok ? 'var(--success)' : 'var(--danger)';
  const label = ok ? 'API connected' : 'API unreachable';

  return (
    <div className="flex items-center gap-2">
      <span className="relative inline-flex h-2 w-2">
        {ok && (
          <span
            className="absolute inline-flex h-full w-full rounded-full opacity-75 pulse-soft"
            style={{ background: color }}
          />
        )}
        <span
          className="relative inline-flex h-2 w-2 rounded-full"
          style={{ background: color }}
        />
      </span>
      <span
        className="text-[10px] font-medium uppercase tracking-wider"
        style={{ color: ok ? 'rgba(255,255,255,0.55)' : '#fca5a5' }}
      >
        {label}
      </span>
    </div>
  );
}
