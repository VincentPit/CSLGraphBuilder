'use client';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getCurationQueue, submitCurationEvents, CurationQueueItem } from '@/lib/api';
import { useState } from 'react';
import { CheckCircle2, XCircle, Loader2 } from 'lucide-react';

const STATUS_STYLE: Record<string, string> = {
  rejected: 'text-rose-700 border-rose-200 bg-rose-50',
  flagged: 'text-amber-700 border-amber-200 bg-amber-50',
  unverified: 'text-slate-700 border-slate-200 bg-slate-100',
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
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">Curation Queue</h1>
        <p className="text-slate-600">Review extracted items before they are promoted to trusted graph content.</p>
      </header>

      <div className="flex flex-wrap gap-2">
        {filters.map(({ value, label }) => (
          <button
            key={value}
            onClick={() => setFilter(value)}
            className={`px-3 py-1.5 rounded-md text-xs font-medium border transition ${filter === value ? 'bg-[#ddf4ff] text-[#0969da] border-[#54aeff]' : 'bg-white text-slate-700 border-[#d0d7de] hover:border-[#afb8c1]'}`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="surface overflow-hidden">
        {isLoading ? (
          <div className="p-6 text-sm text-slate-600 flex items-center gap-2"><Loader2 size={14} className="animate-spin" /> Loading queue</div>
        ) : (data?.items.length ?? 0) === 0 ? (
          <div className="p-8 text-slate-600 text-sm">No items in queue.</div>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-slate-50 border-b border-slate-200">
              <tr>
                <th className="text-left px-4 py-3 text-slate-500 text-xs uppercase tracking-wide">Type</th>
                <th className="text-left px-4 py-3 text-slate-500 text-xs uppercase tracking-wide">Name / Identifier</th>
                <th className="text-left px-4 py-3 text-slate-500 text-xs uppercase tracking-wide">Status</th>
                <th className="text-left px-4 py-3 text-slate-500 text-xs uppercase tracking-wide">Notes</th>
                <th className="text-left px-4 py-3 text-slate-500 text-xs uppercase tracking-wide">Actions</th>
              </tr>
            </thead>
            <tbody>
              {data!.items.map((item) => (
                <tr key={item.id} className="border-b border-slate-100 hover:bg-slate-50">
                  <td className="px-4 py-3 text-slate-700 capitalize">{item.type}</td>
                  <td className="px-4 py-3 text-slate-900">{item.name ?? item.relationship_type ?? item.id}</td>
                  <td className="px-4 py-3">
                    <span className={`text-xs px-2 py-1 rounded border ${STATUS_STYLE[item.verification_status] ?? STATUS_STYLE.unverified}`}>
                      {item.verification_status}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-slate-600">{item.notes ?? '—'}</td>
                  <td className="px-4 py-3">
                    <div className="flex gap-2">
                      <button onClick={() => act(item, 'approve')} className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md bg-[#dafbe1] text-[#1a7f37] border border-[#a2e5b5] hover:bg-[#c5f3d2] text-xs font-medium">
                        <CheckCircle2 size={12} /> Approve
                      </button>
                      <button onClick={() => act(item, 'reject')} className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md bg-[#ffebe9] text-[#cf222e] border border-[#ffc1ba] hover:bg-[#ffd8d3] text-xs font-medium">
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
