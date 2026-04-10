'use client';
import { useState } from 'react';
import { runVerification, VerificationReport } from '@/lib/api';

export default function VerificationPage() {
  const [useText, setUseText] = useState(true);
  const [useEmbed, setUseEmbed] = useState(false);
  const [useLLM, setUseLLM] = useState(false);
  const [threshold, setThreshold] = useState('0.6');
  const [report, setReport] = useState<VerificationReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  async function handleRun(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true); setError(null); setReport(null);
    try {
      const r = await runVerification({
        use_text_match: useText,
        use_embedding: useEmbed,
        use_llm: useLLM,
        confidence_threshold: Number(threshold),
      });
      setReport(r);
    } catch (err: any) {
      setError(err.response?.data?.detail ?? err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-4xl space-y-6">
      <h1 className="text-2xl font-bold">Relationship Verification</h1>

      <form onSubmit={handleRun} className="bg-slate-800 rounded-xl p-5 border border-slate-700 space-y-4">
        <p className="text-sm text-slate-400 font-medium">Verifier stages to run:</p>
        <div className="flex gap-6">
          <Check label="Text Match" checked={useText} onChange={setUseText} />
          <Check label="Embedding" checked={useEmbed} onChange={setUseEmbed} />
          <Check label="LLM" checked={useLLM} onChange={setUseLLM} />
        </div>
        <div className="max-w-xs">
          <label className="block text-sm text-slate-400 mb-1">Confidence Threshold (0–1)</label>
          <input type="number" step="0.05" min="0" max="1" value={threshold} onChange={(e) => setThreshold(e.target.value)}
            className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-sky-500" />
        </div>
        <button type="submit" disabled={loading} className="bg-sky-600 hover:bg-sky-500 disabled:opacity-40 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors">
          {loading ? 'Running…' : 'Run Verification'}
        </button>
      </form>

      {error && <p className="text-red-400 text-sm">{error}</p>}

      {report && (
        <div className="space-y-4">
          <div className="grid grid-cols-4 gap-3">
            {[['Total', report.total], ['Verified', report.verified], ['Rejected', report.rejected], ['Unverified', report.unverified]].map(([l, v]) => (
              <div key={l as string} className="bg-slate-800 rounded-lg p-3 border border-slate-700 text-center">
                <p className="text-xs text-slate-400">{l}</p>
                <p className="text-2xl font-bold text-sky-400">{v}</p>
              </div>
            ))}
          </div>

          <div className="space-y-2">
            {report.entries.map((entry) => (
              <div key={entry.relationship_id} className="bg-slate-800 rounded-lg border border-slate-700 overflow-hidden">
                <button onClick={() => setExpanded(expanded === entry.relationship_id ? null : entry.relationship_id)}
                  className="w-full flex justify-between items-center px-4 py-3 text-sm text-left hover:bg-slate-700 transition-colors">
                  <span className="font-mono text-slate-300 truncate">{entry.relationship_id}</span>
                  <span className={`ml-4 font-medium ${entry.final_status === 'verified' ? 'text-green-400' : entry.final_status === 'rejected' ? 'text-red-400' : 'text-yellow-400'}`}>
                    {entry.final_status} ({(entry.overall_confidence * 100).toFixed(0)}%)
                  </span>
                </button>
                {expanded === entry.relationship_id && (
                  <div className="px-4 pb-3 border-t border-slate-700 space-y-1">
                    {entry.stages.map((s, i) => (
                      <div key={i} className="flex gap-3 text-xs text-slate-400">
                        <span className="w-24">{s.stage}</span>
                        <span className={s.status === 'verified' ? 'text-green-400' : 'text-red-400'}>{s.status}</span>
                        {s.confidence != null && <span>{(s.confidence * 100).toFixed(0)}%</span>}
                        {s.reason && <span className="text-slate-500">{s.reason}</span>}
                      </div>
                    ))}
                    {entry.message && <p className="text-xs text-slate-500 mt-1">{entry.message}</p>}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Check({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 text-sm cursor-pointer">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="accent-sky-500" />
      <span className="text-slate-300">{label}</span>
    </label>
  );
}
