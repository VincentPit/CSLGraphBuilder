'use client';
import { useState } from 'react';
import { runVerification, VerificationReport } from '@/lib/api';
import { ShieldCheck, ChevronDown, ChevronUp, Loader2, ShieldAlert } from 'lucide-react';

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
      setReport(await runVerification({ use_text_match: useText, use_embedding: useEmbed, use_llm: useLLM, confidence_threshold: Number(threshold) }));
    } catch (err: any) {
      setError(err.response?.data?.detail ?? err.message);
    } finally { setLoading(false); }
  }

  return (
    <div className="max-w-4xl space-y-10">
      {/* Header */}
      <div className="space-y-3">
        <div className="flex items-center gap-2.5">
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-violet-500/20 to-purple-500/10 border border-violet-500/10 flex items-center justify-center">
            <ShieldAlert size={18} className="text-violet-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Verify Relationships</h1>
            <p className="text-xs text-slate-500 font-medium">Automated confidence scoring</p>
          </div>
        </div>
        <p className="text-sm text-slate-400 leading-relaxed max-w-xl">
          Run automated checks on unverified relationships. Choose verification stages and a confidence threshold to classify results.
        </p>
      </div>

      {/* Config form */}
      <form onSubmit={handleRun} className="bg-gradient-to-b from-[#0f1829] to-[#0d1526] border border-white/[0.06] rounded-2xl p-8 space-y-8 shadow-xl shadow-black/20">
        <div>
          <p className="text-[11px] font-bold text-slate-600 uppercase tracking-widest mb-4">Verification Stages</p>
          <div className="grid grid-cols-3 gap-3">
            <ToggleCard label="Text Match" desc="Exact string matching against source text" checked={useText} onChange={setUseText} />
            <ToggleCard label="Embedding" desc="Semantic similarity via vector distance" checked={useEmbed} onChange={setUseEmbed} />
            <ToggleCard label="LLM Review" desc="Language model logical validation" checked={useLLM} onChange={setUseLLM} />
          </div>
        </div>
        <div className="max-w-xs">
          <label className="block text-[11px] font-bold text-slate-600 uppercase tracking-widest mb-3">Confidence Threshold</label>
          <p className="text-[11px] text-slate-500 mb-3">Relationships scoring above this value pass verification.</p>
          <div className="flex items-center gap-4">
            <input type="range" min="0" max="1" step="0.05" value={threshold} onChange={(e) => setThreshold(e.target.value)}
              className="flex-1 accent-indigo-500 h-1.5" />
            <span className="text-indigo-400 font-mono font-bold text-base w-12 text-right">{threshold}</span>
          </div>
        </div>
        <button type="submit" disabled={loading || (!useText && !useEmbed && !useLLM)}
          className="group w-full flex items-center justify-center gap-2.5 bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 disabled:opacity-30 disabled:cursor-not-allowed text-white text-sm font-semibold px-6 py-3.5 rounded-xl transition-all duration-200 shadow-lg shadow-indigo-600/20">
          {loading ? <><Loader2 size={15} className="animate-spin"/>Running…</> : <><ShieldCheck size={15}/>Run Verification</>}
        </button>
      </form>

      {error && (
        <div className="text-red-300 text-sm bg-red-500/[0.08] border border-red-500/15 rounded-xl px-5 py-4">{error}</div>
      )}

      {/* Results */}
      {report && (
        <div className="space-y-6">
          <div className="grid grid-cols-4 gap-4">
            {[
              { label: 'Total',      value: report.total,      color: 'text-white',        gradient: 'from-slate-500/10 to-slate-500/5' },
              { label: 'Verified',   value: report.verified,   color: 'text-emerald-400',  gradient: 'from-emerald-500/10 to-emerald-500/5' },
              { label: 'Rejected',   value: report.rejected,   color: 'text-red-400',      gradient: 'from-red-500/10 to-red-500/5' },
              { label: 'Unverified', value: report.unverified, color: 'text-amber-400',    gradient: 'from-amber-500/10 to-amber-500/5' },
            ].map(({ label, value, color, gradient }) => (
              <div key={label} className={`bg-gradient-to-b ${gradient} border border-white/[0.06] rounded-2xl p-5 text-center`}>
                <p className="text-[10px] font-bold text-slate-600 uppercase tracking-widest mb-2">{label}</p>
                <p className={`text-3xl font-extrabold ${color}`}>{value}</p>
              </div>
            ))}
          </div>

          <p className="text-[11px] font-bold text-slate-600 uppercase tracking-widest">Relationship Details</p>
          <div className="space-y-2">
            {report.entries.map((entry) => {
              const isOpen = expanded === entry.relationship_id;
              const statusColor = entry.final_status === 'verified' ? 'text-emerald-400' : entry.final_status === 'rejected' ? 'text-red-400' : 'text-amber-400';
              return (
                <div key={entry.relationship_id} className="bg-gradient-to-b from-[#0f1829] to-[#0d1526] border border-white/[0.06] rounded-xl overflow-hidden">
                  <button onClick={() => setExpanded(isOpen ? null : entry.relationship_id)}
                    className="w-full flex justify-between items-center px-5 py-3.5 text-sm text-left hover:bg-white/[0.02] transition-colors">
                    <span className="font-mono text-slate-500 text-xs truncate">{entry.relationship_id}</span>
                    <div className="flex items-center gap-3 ml-4 shrink-0">
                      <span className={`font-bold text-xs ${statusColor}`}>{entry.final_status}</span>
                      <span className="text-slate-600 font-mono text-xs">{(entry.overall_confidence * 100).toFixed(0)}%</span>
                      {isOpen ? <ChevronUp size={14} className="text-slate-600"/> : <ChevronDown size={14} className="text-slate-600"/>}
                    </div>
                  </button>
                  {isOpen && (
                    <div className="px-5 pb-4 pt-2 border-t border-white/[0.04] space-y-2">
                      {entry.stages.map((s, i) => (
                        <div key={i} className="flex items-center gap-3 text-xs">
                          <span className="w-24 text-slate-500 shrink-0 font-medium">{s.stage}</span>
                          <span className={`font-semibold ${s.status === 'verified' ? 'text-emerald-400' : 'text-red-400'}`}>{s.status}</span>
                          {s.confidence != null && <span className="text-slate-600 font-mono">{(s.confidence * 100).toFixed(0)}%</span>}
                          {s.reason && <span className="text-slate-600 truncate">{s.reason}</span>}
                        </div>
                      ))}
                      {entry.message && <p className="text-xs text-slate-600 mt-2">{entry.message}</p>}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function ToggleCard({ label, desc, checked, onChange }: { label: string; desc: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className={`flex flex-col gap-1.5 p-4 rounded-xl border cursor-pointer transition-all duration-200 ${
      checked ? 'bg-indigo-500/[0.08] border-indigo-500/25' : 'bg-white/[0.02] border-white/[0.06] hover:border-white/[0.1]'
    }`}>
      <div className="flex items-center justify-between">
        <span className={`text-sm font-semibold ${checked ? 'text-indigo-300' : 'text-slate-300'}`}>{label}</span>
        <div className={`w-4 h-4 rounded border-2 flex items-center justify-center transition-all ${checked ? 'bg-indigo-500 border-indigo-500' : 'border-slate-600'}`}>
          {checked && <svg viewBox="0 0 10 8" className="w-2.5 h-2 fill-white"><path d="M1 4l2.5 2.5L9 1"/></svg>}
        </div>
      </div>
      <span className="text-[10px] text-slate-500 leading-snug">{desc}</span>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="sr-only" />
    </label>
  );
}
