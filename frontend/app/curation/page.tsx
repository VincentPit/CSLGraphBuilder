'use client';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getCurationQueue, submitCurationEvents, CurationQueueItem } from '@/lib/api';
import { useState } from 'react';

const STATUS_COLOR: Record<string, string> = {
  rejected: 'text-red-400',
  flagged: 'text-yellow-400',
  unverified: 'text-slate-400',
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

  function act(item: CurationQueueItem, action: 'approve' | 'reject') {
    const event = item.type === 'entity'
      ? { entity_id: item.id, action }
      : { relationship_id: item.id, action };
    mutation.mutate([event as any]);
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Curation Queue</h1>

      <div className="flex gap-2">
        {['', 'rejected', 'flagged', 'unverified'].map((s) => (
          <button key={s} onClick={() => setFilter(s)}
            className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${filter === s ? 'bg-sky-600 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}>
            {s || 'All'}
          </button>
        ))}
      </div>

      {isLoading ? (
        <p className="text-slate-400">Loading…</p>
      ) : (data?.items.length ?? 0) === 0 ? (
        <p className="text-slate-500">Queue is empty.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-400 border-b border-slate-700">
                <th className="pb-2 pr-4">Type</th>
                <th className="pb-2 pr-4">Name / Type</th>
                <th className="pb-2 pr-4">Status</th>
                <th className="pb-2 pr-4">Notes</th>
                <th className="pb-2">Actions</th>
              </tr>
            </thead>
            <tbody>
              {data!.items.map((item) => (
                <tr key={item.id} className="border-b border-slate-800 hover:bg-slate-800/50">
                  <td className="py-2 pr-4 text-slate-400 capitalize">{item.type}</td>
                  <td className="py-2 pr-4 text-slate-200">{item.name ?? item.relationship_type ?? item.id}</td>
                  <td className={`py-2 pr-4 font-medium ${STATUS_COLOR[item.verification_status] ?? ''}`}>
                    {item.verification_status}
                  </td>
                  <td className="py-2 pr-4 text-slate-500 text-xs max-w-xs truncate">{item.notes}</td>
                  <td className="py-2 flex gap-2">
                    <Btn onClick={() => act(item, 'approve')} color="green">Approve</Btn>
                    <Btn onClick={() => act(item, 'reject')} color="red">Reject</Btn>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Btn({ onClick, color, children }: { onClick: () => void; color: 'green' | 'red'; children: React.ReactNode }) {
  const cls = color === 'green' ? 'bg-emerald-700 hover:bg-emerald-600' : 'bg-red-700 hover:bg-red-600';
  return <button onClick={onClick} className={`${cls} text-white text-xs px-2 py-1 rounded transition-colors`}>{children}</button>;
}
