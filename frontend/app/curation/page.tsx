'use client';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getCurationQueue, submitCurationEvents, CurationQueueItem } from '@/lib/api';
import { useState } from 'react';
import { CheckCircle2, XCircle, Loader2 } from 'lucide-react';

const STATUS_BADGE: Record<string, string> = {
  rejected: 'badge badge-danger',
  flagged: 'badge badge-warning',
  unverified: 'badge badge-neutral',
};

export default function CurationPage() {
  const qc = useQueryClient();
  const [filter, setFilter] = useState('');

  const { data, isLoading } = useQuery({
    queryKey: ['curation-queue', filter],
    queryFn: () => getCurationQueue({ status: filter || undefined, limit: 200 }),
  });

  const mutation = useMutation({
    mutationFn: submitCurationEvents,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['curation-queue'] }),
  });

  const filters = [
    { value: '', label: 'All' },
    { value: 'unverified', label: 'Unverified' },
    { value: 'flagged', label: 'Flagged' },
    { value: 'rejected', label: 'Rejected' },
  ];

  const act = (item: CurationQueueItem, action: 'approve' | 'reject') => {
    const event = item.type === 'entity' ? { entity_id: item.id, action } : { relationship_id: item.id, action };
    mutation.mutate([event as any]);
  };

  return (
    <div className="space-y-8">
      <header>
        <h1 className="page-title">Curation Queue</h1>
        <p className="page-desc">Review extracted items before they are promoted to trusted graph content.</p>
      </header>

      <div className="flex flex-wrap gap-2">
        {filters.map(({ value, label }) => (
          <button
            key={value}
            onClick={() => setFilter(value)}
            className={`btn-ghost ${filter === value ? 'active' : ''}`}
            style={filter === value ? { background: 'var(--accent-muted)', color: 'var(--accent)', borderColor: 'var(--accent)' } : {}}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="card overflow-hidden">
        {isLoading ? (
          <div className="p-6 text-sm flex items-center gap-2" style={{ color: 'var(--text-muted)' }}><Loader2 size={14} className="animate-spin" /> Loading queue</div>
        ) : (data?.items.length ?? 0) === 0 ? (
          <div className="p-8 text-sm" style={{ color: 'var(--text-muted)' }}>No items in queue.</div>
        ) : (
          <table className="w-full text-sm">
            <thead style={{ background: 'var(--bg-muted)', borderBottom: '1px solid var(--border)' }}>
              <tr>
                {['Type', 'Name / Identifier', 'Status', 'Notes', 'Actions'].map((h) => (
                  <th key={h} className="text-left px-4 py-3 text-xs uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data!.items.map((item) => (
                <tr key={item.id} className="hover:bg-[var(--bg-muted)]" style={{ borderBottom: '1px solid var(--border)' }}>
                  <td className="px-4 py-3 capitalize" style={{ color: 'var(--text-secondary)' }}>{item.type}</td>
                  <td className="px-4 py-3" style={{ color: 'var(--text-primary)' }}>{item.name ?? item.relationship_type ?? item.id}</td>
                  <td className="px-4 py-3">
                    <span className={STATUS_BADGE[item.verification_status] ?? STATUS_BADGE.unverified}>
                      {item.verification_status}
                    </span>
                  </td>
                  <td className="px-4 py-3" style={{ color: 'var(--text-muted)' }}>{item.notes ?? '—'}</td>
                  <td className="px-4 py-3">
                    <div className="flex gap-2">
                      <button onClick={() => act(item, 'approve')} className="badge badge-success cursor-pointer hover:opacity-80 transition">
                        <CheckCircle2 size={12} /> Approve
                      </button>
                      <button onClick={() => act(item, 'reject')} className="badge badge-danger cursor-pointer hover:opacity-80 transition">
                        <XCircle size={12} /> Reject
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
