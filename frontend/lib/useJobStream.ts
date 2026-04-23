'use client';

import { useEffect, useRef, useState } from 'react';
import { Job, getJob, getJobStreamUrl } from '@/lib/api';

/**
 * Subscribe to a job via the SSE stream and fall back to polling on
 * EventSource errors. Returns the live ``job`` snapshot.
 *
 * The hook is safe to mount with ``jobId === null`` — it does nothing
 * until a real id arrives.
 */
export function useJobStream(jobId: string | null): Job | null {
  const [job, setJob] = useState<Job | null>(null);
  const esRef = useRef<EventSource | null>(null);
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    setJob(null);
    if (!jobId) return;

    let cancelled = false;
    const cleanup = () => {
      cancelled = true;
      esRef.current?.close();
      if (pollRef.current) clearTimeout(pollRef.current);
    };

    function startPolling() {
      const tick = async () => {
        if (cancelled || !jobId) return;
        try {
          const fresh = await getJob(jobId);
          setJob(fresh);
          if (
            fresh.status === 'completed' ||
            fresh.status === 'failed' ||
            fresh.status === 'cancelled'
          ) {
            return;
          }
        } catch {
          /* swallow — keep polling */
        }
        pollRef.current = setTimeout(tick, 1500);
      };
      tick();
    }

    try {
      const es = new EventSource(getJobStreamUrl(jobId));
      esRef.current = es;
      const handleSnapshot = (evt: MessageEvent) => {
        try {
          const snap = JSON.parse(evt.data) as Job;
          setJob(snap);
        } catch {
          /* ignore malformed event */
        }
      };
      es.addEventListener('progress', handleSnapshot as EventListener);
      es.addEventListener('done', (evt: MessageEvent) => {
        handleSnapshot(evt);
        es.close();
      });
      es.onmessage = handleSnapshot;
      es.onerror = () => {
        es.close();
        startPolling();
      };
    } catch {
      startPolling();
    }

    return cleanup;
  }, [jobId]);

  return job;
}
