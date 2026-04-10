'use client';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getCurationQueue, submitCurationEvents, CurationQueueItem } from '@/lib/api';
import { useState } from 'react';
import { CheckCircle2, XCircle, Loader2, ClipboardList } from 'lucide-react';

const STATUS_STYLE: Record<string, string> = {
  rejected:   'text-red-400 bg-red-500/[0.08] border-red-500/15',
  flagged:    'text-amber-400 bg-amber-500/[0.08] border-amber-500/15',
  unverified: 'text-slate-400 bg-white/[0.03] border-white/[0.06]',
};

export default function CurationPage() {
  const qc = useQueryClient();
  const [filter, setFilter] = useState('');
  const filters = [
    { value: '',           label: 'All' },
    { value: 'unverified', label: 'Unverified' },
    { value: 'flagged',    label: 'Flagged' },
    { value: 'rejected',   label: 'Rejected' },
  ];

  const { data, isLoading } = useQuery({
    queryKey: ['curation-queue', filter],
    queryFn: () => getCurationQueue({ status: filter || undefined, limit: 200 }),
  });

  const mutation = useMutation({
    mutationFn: submitCurationEvents,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['curation-queue'] }),
  });

  function act(item: CurationQueueItem, action: 'approve' | 'reject') {
    const event = item.type === 'entity' ? { entity_id: item.id, action } : { relationship_id: item.id, action };
    mutation.mutate([event as any]);
  }

  return (
    <div className="max-w-5xl space-y-10">
      {/* Header */}
      <div className="space-y-3">
        <div className="flex items-center gap-2.5">
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-amber-500/20 to-orange-500/10 border border-amber-500/10 flex items-center justify-center">
            <ClipboardList size={18} className="text-amber-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Curation Queue</h1>
            <p className="text-xs text-slate-500 font-medium">Review & approve extracted data</p>
          </div>
        </div>
        <p className="text-sm text-slate-400 leading-relaxed max-w-xl">
          Review entities and relationships flagged for human verification. Approve or reject each entry to curate the trusted graph.
        </p>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-2">
        <span className="text-[11px] text-slate-600 font-semibold uppercase tracking-wider mr-1">Filter</span>
        {filters.map(({ value, label }) => (
          <button key={value} onClick={() => setFilter(value)}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
              filter === value
                ? 'bg-gradient-to-r from-indigo-600 to-indigo-500 text-white shadow-lg shadow-indigo-600/15'
                : 'bg-white/[0.03] border border-white/[0.06] text-slate-500 hover:text-slate-300 hover:border-white/[0.1]'
            }`}>
            {label}
          </button>
        ))}
      </div>

      {/* Content */}
      {isLoading ? (
        <div className="flex items-center gap-2.5 text-slate-500 text-sm"><Loader2 size={14} className="animate-spin"/>Loading queue…</div>
      ) : (data?.items.length ?? 0) === 0 ? (
        <div className="text-center py-20 bg-gradient-to-b from-[#0f1829] to-[#0d1526] border border-white/[0.06] rounded-2xl">
          <CheckCircle2 size={36} className="text-emerald-500/60 mx-auto mb-4" />
          <p className="text-white font-semibold text-[15px]">Queue is empty</p>
          <p className="text-slate-500 text-sm mt-1.5">No items match this filter. Great work!</p>
        </div>
      ) : (
        <div className="bg-gradient-to-b from-[#0f1829] to-[#0d1526] border border-white/[0.06] rounded-2xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/[0.04]">
                <th className="text-left text-[10px] font-bold text-slate-600 uppercase tracking-widest px-6 py-4">Type</th>
                <th className="text-left text-[10px] font-bold text-slate-600 uppercase tracking-widest px-6 py-4">Name / ID</th>
                <th className="text-left text-[10px] font-bold text-slate-600 uppercase tracking-widest px-6 py-4">Status</th>
                <th className="text-left text-[10px] font-bold text-slate-600 uppercase tracking-widest px-6 py-4">Notes</th>
                <th className="text-left text-[10px] font-bold text-slate-600 uppercase tracking-widest px-6 py-4">Action</th>
              </tr>
            </thead>
            <tbody>
              {data!.items.map((item, i) => (
                <tr key={item.id} className="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors">
                  <td className="px-6 py-4">
                    <span className="capitalize text-[11px] font-semibold bg-white/[0.04] border border-white/[0.06] text-slate-400 px-2.5 py-1 rounded-md">{item.type}</span>
                  </td>
                  <td className="px-6 py-4 text-slate-200 font-medium">{item.name ?? item.relationship_type ?? item.id}</td>
                  <td className="px-6 py-4">
                    <span className={`text-[11px] font-semibold px-2.5 py-1 rounded-md border ${STATUS_STYLE[item.verification_status] ?? STATUS_STYLE.unverified}`}>
                      {item.verification_status}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-slate-600 text-xs max-w-xs truncate">{item.notes ?? '—'}</td>
                  <td className="px-6 py-4">
                    <div className="flex gap-2">
                      <button onClick={() => act(item, 'approve')} className="flex items-center gap-1.5 bg-emerald-500/[0.1] hover:bg-emerald-500/[0.2] text-emerald-400 border border-emerald-500/20 text-[11px] font-semibold px-3 py-1.5 rounded-lg transition-all">
                        <CheckCircle2 size={11}/> Approve
                      </button>
                      <button onClick={() => act(item, 'reject')} className="flex items-center gap-1.5 bg-red-500/[0.1] hover:bg-red-500/[0.2] text-red-400 border border-red-500/20 text-[11px] font-semibold px-3 py-1.5 rounded-lg transition-all">
                        <XCircle size={11}/> Reject
                      </button>
                    </div>
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
